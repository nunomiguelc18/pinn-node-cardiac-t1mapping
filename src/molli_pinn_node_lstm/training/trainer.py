from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from molli_pinn_node_lstm.training import PINNLoss
from molli_pinn_node_lstm.utils import molli_signal_model

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """
    Configuration for training and validation.
        Members
        -------
        epochs:
            Number of training epochs (min 1).
        num_acquisitions:
            Number of acquisitions (timepoints) to sample per forward pass. Must be in [3, max_acquisitions].
        max_acquisitions:
            Total number of acquisitions available in each sample (e.g., 11 for MOLLI 3(3)5 ).
        p_full_seq:
            Probability of using the full acquisition sequence instead of a random subset.
        p_interpolation:
            Probability of resynthesizing the full MOLLI readouts from pmap parameters (if enabled).
        interpolate_readouts:
            If True, occasionally replace readouts with model-based interpolated readouts.
        val_mc_samples:
            Number of Monte-Carlo masks to average over during validation.
    """

    epochs: int
    num_acquisitions: int
    max_acquisitions: int

    p_full_seq: float = 0.1
    p_interpolation: float = 0.1
    interpolate_readouts: bool = False
    val_mc_samples: int = 1

    def __post_init__(self) -> None:
        self.epochs = max(1, int(self.epochs))
        self.val_mc_samples = max(1, int(self.val_mc_samples))

        if not (3 <= int(self.num_acquisitions) <= int(self.max_acquisitions)):
            raise ValueError("num_acquisitions must be within {3, ... , max_acquisitions}.")

        self.num_acquisitions = int(self.num_acquisitions)
        self.max_acquisitions = int(self.max_acquisitions)

        self.p_interpolation = float(np.clip(self.p_interpolation, 0.0, 1.0))
        self.p_full_seq = float(np.clip(self.p_full_seq, 0.0, 1.0))

        if self.interpolate_readouts and self.p_interpolation <= 0.0:
            LOGGER.warning(
                "interpolate_readouts=True but p_interpolation=0; interpolation will never run."
            )


class Trainer(nn.Module):
    """
    Training orchestrator for our Accelerated Physics-Informed Framework for MOLLI signal recovery
    
    This class orchestrates:
      - epoch-wise training and validation loops
      - acquisition subsampling via boolean masks
      - optional interpolation-based augmentation of readouts
      - TensorBoard scalar logging
      - checkpointing (latest + best)

    The trainer expects batches in dictionary form with:

      - "volume": torch.Tensor, 
        MOLLI signal acquisitions.

      - "tvec": torch.Tensor
        Inversion times / elapsed time after inversion pulse.
      
      - "pmap": torch.Tensor
        model parameters (c, k, T1*) estimated by least-squares fitting on
        S(t) = c * (1 - k * exp( -t / T1* )), e.g via Levenberg–Marquardt algorithm.

      - "molli_t1_ref": torch.Tensor. 
        Reference T1 derived from fitted parameters using T1 = T1* (k - 1).

    """
    def __init__(
        self,
        trainer_cfg: TrainerConfig,
        model: nn.Module,
        optimizer: AdamW,
        scheduler: ExponentialLR,
        tensorboard_logger: SummaryWriter,
        molli_loss: PINNLoss,
        tvec_normalization_const: float,
        signal_normalization_const: float,
        save_ckpt_dir: pathlib.Path,
        device: str,
    ) -> None:
        """
        Attributes
        ----------
        trainer_cfg: TrainerConfig
            Training configuration dataclass defining epochs, acquisition sampling, etc...
        model: nn.Module
            PyTorch module that consumes `volume` and `tvec` and returns a predicted pmap (3-parameters signal model).
        optimizer: AdamW
            AdamW optimizer used to update `model` parameters.
        scheduler: ExponentialLR
            Exponential learning-rate scheduler stepped once per epoch after validation.
        tensorboard_logger:
            TensorBoard SummaryWriter used to log training/validation losses and LR.
        molli_loss: PINNLoss
            Physics-informed loss for the 3-parameter MOLLI recovery model.
        tvec_normalization_const:
            Divisor applied to tvec and T1 reference (e.g., 1000 converts ms -> s).
        signal_normalization_const:
            Divisor applied to signal readouts.
        save_ckpt_dir:
            Directory where checkpoints and weights are written. The trainer writes:
              - latest_checkpoint.pt (every epoch)
              - best_checkpoint.pt (when validation improves)
              - best_weights.pt (when validation improves)
        device:
            Target device string (e.g., "cpu", "cuda", "cuda:0"). All tensors in the training
            step are moved to this device.

        """
        super().__init__()

        self.trainer_cfg = trainer_cfg
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tensorboard_logger = tensorboard_logger
        self.molli_loss = molli_loss

        self.tvec_normalization_const = float(tvec_normalization_const)
        self.signal_normalization_const = float(signal_normalization_const)
        self.device = device

        self.best_valid_loss = float("inf")

        self.save_ckpt_dir = pathlib.Path(save_ckpt_dir)

        self.train_history: Dict[str, List[float]] = {
            "train/total_loss": [],
            "train/T1": [],
            "train/S(t)": [],
            "train/dS(t)/dt": [],
            "lr": [],
        }
        self.val_history: Dict[str, List[float]] = {
            "val/total_loss": [],
            "val/T1": [],
            "val/S(t)": [],
            "val/dS(t)/dt": [],
        }

        val_masks = torch.stack(self._collate_validation_masks(), dim=0).to(torch.bool)
        self.register_buffer("val_MC_masks", val_masks)

    def fit(self, training_set: DataLoader, validation_set: DataLoader) -> None:
        """Run the full training pipeline.

        This method performs (for each epoch):
          1) one training epoch
          2) one validation epoch
          3) scheduler step
          4) TensorBoard logging (train metrics, val metrics, lr)
          5) checkpoint writing:
             - "latest_checkpoint.pt" every epoch
             - "best_checkpoint.pt" and "best_weights.pt" when validation improves

        Parameters
        ----------
        training_set:
            DataLoader yielding training batches as dictionaries. Each batch must contain at least:
            - volume: Tensor of shape (B, ..., T) where last dim is acquisitions/timepoints
            - tvec: Tensor of shape (T,)
            - molli_t1_ref: Tensor of shape (B, ...) or (B, 1)
            - pmap: Tensor of shape (B, 3)
        validation_set:
            DataLoader yielding validation batches with the same structure as `training_set`.

        Returns
        -------
        None
        """

        epoch_pbar = tqdm(
            range(1, self.trainer_cfg.epochs + 1),
            total = self.trainer_cfg.epochs,
            desc="Epoch",
            position=0,
            dynamic_ncols=True,
        )

        for epoch in epoch_pbar:
            
            train_loss = self._run_training(training_set)
            val_loss = self._run_validation(validation_set)

            self.scheduler.step()
            lr = float(self.optimizer.param_groups[0]["lr"])

            self.train_history["lr"].append(lr)

            for k, v in train_loss.items():
                self.train_history[f"train/{k}"].append(float(v))
                self.tensorboard_logger.add_scalar(f"train/{k}", float(v), epoch)

            for k, v in val_loss.items():
                self.val_history[f"val/{k}"].append(float(v))
                self.tensorboard_logger.add_scalar(f"val/{k}", float(v), epoch)

            self.tensorboard_logger.add_scalar("lr", lr, epoch)

            epoch_pbar.set_postfix(
                train_total=float(train_loss["total_loss"]),
                val_total=float(val_loss["total_loss"]),
                lr=lr,
            )

            self._save_checkpoint(
            path=self.save_ckpt_dir / "latest_checkpoint.pt",
            best_valid_loss=float(val_loss["total_loss"]),
            epoch=epoch,
            extra={"train_loss": train_loss, "val_loss": val_loss},
        )

            if float(val_loss["total_loss"]) < self.best_valid_loss:
                self.best_valid_loss = float(val_loss["total_loss"])
                self._save_checkpoint(
                    path = self.save_ckpt_dir / "best_checkpoint.pt",
                    best_valid_loss = self.best_valid_loss,
                    epoch=epoch,
                    extra={"train_loss": train_loss, "val_loss": val_loss},
                )
                self._save_model_weights(
                    self.save_ckpt_dir / "best_weights.pt",
                    extra={"epoch": epoch, "best_valid_loss": self.best_valid_loss},
                )

    def _run_training(self, training_set: DataLoader) -> Dict[str, float]:
        self.model.train()
        sums = {"total_loss": 0.0, "T1": 0.0, "S(t)": 0.0, "dS(t)/dt": 0.0}
        n_samples = 0

        batch_pbar = tqdm(
            training_set,
            desc=f"Training",
            leave=False,
            position=1,
            dynamic_ncols=True,
        )

        for batch in batch_pbar:
            batch = self._move_batch_to_device(batch)

            if self.trainer_cfg.interpolate_readouts and (random.random() <= self.trainer_cfg.p_interpolation):
                batch.update(self._interpolate_molli_readouts(**batch))

            bool_mask = self._sample_random_indices()
            tmp_vol = batch["volume"][..., bool_mask]
            tmp_tvec = batch["tvec"][bool_mask]

            b = int(tmp_vol.shape[0])
            n_samples += b

            self.optimizer.zero_grad(set_to_none=True)
            pmap_hat = self.model(volume=tmp_vol, tvec=tmp_tvec)

            loss_dict = self.molli_loss.compute_loss(
                pmap_ref=batch["pmap"],
                T1_ref=batch["molli_t1_ref"],
                pmap_hat=pmap_hat,
            )

            loss_dict["total_loss"].backward()
            self.optimizer.step()

            for k, v in loss_dict.items():
                sums[k] += float(v.detach()) * b

            file_progr = getattr(training_set.dataset, "file_progress", None)
            current, total = file_progr
            batch_pbar.set_postfix(file_progress = f"{current}/{total}", train_total=float(loss_dict["total_loss"].detach()))

        return {k: v / max(1, n_samples) for k, v in sums.items()}

    @torch.no_grad()
    def _run_validation(self, validation_set: DataLoader) -> Dict[str, float]:
        self.model.eval()
        sums = {"total_loss": 0.0, "T1": 0.0, "S(t)": 0.0, "dS(t)/dt": 0.0}
        n_samples = 0

        m = int(self.val_MC_masks.shape[0])

        batch_pbar = tqdm(
            validation_set,
            desc=f"Validation",
            leave=False,
            position=2,
            dynamic_ncols=True,
        )

        for batch in batch_pbar:
            batch = self._move_batch_to_device(batch)
            b = int(batch["volume"].shape[0])
            n_samples += b

            mc_sums = {"total_loss": 0.0, "T1": 0.0, "S(t)": 0.0, "dS(t)/dt": 0.0}

            for bool_mask in self.val_MC_masks:
                tmp_vol = batch["volume"][..., bool_mask]
                tmp_tvec = batch["tvec"][bool_mask]

                pmap_hat = self.model(volume=tmp_vol, tvec=tmp_tvec)
                loss_dict = self.molli_loss.compute_loss(
                    pmap_ref=batch["pmap"],
                    T1_ref=batch["molli_t1_ref"],
                    pmap_hat=pmap_hat,
                )
                for k, v in loss_dict.items():
                    mc_sums[k] += float(v.detach())

            for k in mc_sums:
                mc_sums[k] /= max(1, m)
                sums[k] += mc_sums[k] * b

            file_progr = getattr(validation_set.dataset, "file_progress")
            current, total = file_progr
            batch_pbar.set_postfix(file_progress = f"{current}/{total}", val_total=mc_sums["total_loss"])

        return {k: v / max(1, n_samples) for k, v in sums.items()}

    def _interpolate_molli_readouts(self, tvec: torch.Tensor, pmap: torch.Tensor, **kwargs):
        """ Augment MOLLI readouts from pmap parameters on a jittered time grid."""
        pmap_dict = {"C": pmap[:, 0], "K": pmap[:, 1], "T1_star": pmap[:, 2]}
        time_jitter = torch.empty_like(tvec).uniform_(-1.0, 1.0) * 0.2
        tvec_grid = torch.sort(torch.abs(tvec + time_jitter)).values
        denorm_tvec_grid = tvec_grid * self.tvec_normalization_const
        interp_volume = molli_signal_model.signal_recovery(tvec=denorm_tvec_grid, **pmap_dict)
        interp_volume /= self.signal_normalization_const
        return {"volume": interp_volume.T, "tvec": tvec_grid}

    def _collate_validation_masks(self) -> List[torch.Tensor]:
        """Pre-sample Monte-Carlo boolean masks used to average validation metrics."""
        LOGGER.info(
            f"Collecting {self.trainer_cfg.val_mc_samples} random masks for Monte-carlo sampling on {self.trainer_cfg.num_acquisitions} out of {self.trainer_cfg.max_acquisitions} MOLLI acquisitions.",
        )
        return [self._sample_random_indices() for _ in range(self.trainer_cfg.val_mc_samples)]

    def _sample_random_indices(self) -> torch.Tensor:
        """Sample a boolean mask over acquisitions with MOLLI-ish constraints."""
        n = int(self.trainer_cfg.max_acquisitions)
        k = int(self.trainer_cfg.num_acquisitions)

        if random.random() <= self.trainer_cfg.p_full_seq or k >= n:
            return torch.ones(n, dtype=torch.bool, device=self.device)

        mask = np.zeros(n, dtype=bool)

        first = np.random.choice(np.arange(0, min(3, n)), 1, replace=False)
        mid_pool = np.arange(3, max(3, n - 3))
        last = np.random.choice(np.arange(max(0, n - 3), n), 1, replace=False)

        if mid_pool.size >= (k - 2):
            mid = np.random.choice(mid_pool, k - 2, replace=False)
            idx = np.sort(np.concatenate([first, mid, last]))
        else:
            idx = np.sort(np.random.choice(np.arange(n), k, replace=False))

        mask[idx] = True
        return torch.as_tensor(mask, dtype=torch.bool, device=self.device)

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors onto the training device."""
        device = self.device
        return {
            "volume": batch["volume"].to(device, non_blocking=True),
            "tvec": batch["tvec"].to(device, non_blocking=True),
            "molli_t1_ref": batch["molli_t1_ref"].to(device, non_blocking=True),
            "pmap": batch["pmap"].to(device, non_blocking=True),
        }
    
    def _save_checkpoint(
        self,
        path: pathlib.Path,
        best_valid_loss : float,
        epoch: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a full training checkpoint (model state + optimizer + scheduler + optional metadata)."""

        ckpt = {
            "epoch": int(epoch),
            "best_valid_loss":  best_valid_loss,
            "trainer_cfg": self.trainer_cfg.__dict__,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "extra": dict(extra) if extra is not None else {},
        }
        torch.save(ckpt, path)


    def load_checkpoint(
        self,
        path: pathlib.Path,
        map_location: str | torch.device,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        strict: bool = True,
    ) -> None:
        
        """Load a checkpoint from disk into the trainer.

        This restores:
          - model weights
          - (optionally) optimizer state
          - (optionally) scheduler state
          - best validation loss value (if present)

        Parameters
        ----------
        path:
            Path to a checkpoint file.
        map_location:
            Passed to `torch.load` to remap storages (e.g. "cpu" or torch.device("cuda:0")).
        load_optimizer:
            If True and optimizer state exists, loads optimizer state_dict.
        load_scheduler:
            If True and scheduler state exists, loads scheduler state_dict.
        strict:
            Passed to `model.load_state_dict`. If False, allows partial loads.

        Returns
        -------
        None

        Notes
        -----
        This method does not automatically move the trainer/model to a device.
        Device placement is controlled by your `map_location` and where the module lives.
        """

        ckpt = torch.load(path, map_location=map_location)

        self.model.load_state_dict(ckpt["model_state"], strict=strict)

        if load_optimizer and "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])

        if load_scheduler and self.scheduler is not None and "scheduler_state" in ckpt:
            sch_state = ckpt["scheduler_state"]
            if sch_state is not None:
                self.scheduler.load_state_dict(sch_state)

        if "best_valid_loss" in ckpt and ckpt["best_valid_loss"] is not None:
            self.best_valid_loss = float(ckpt["best_valid_loss"])



    def _save_model_weights(
        self,
        path: pathlib.Path,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model weights only (state_dict) plus some metadata."""
        payload = {
            "model_state": self.model.state_dict(),
            "trainer_cfg": dict(self.trainer_cfg.__dict__),
            "extra": dict(extra) if extra is not None else {},
        }
        torch.save(payload, path)
    



    