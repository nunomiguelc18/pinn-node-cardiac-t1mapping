import argparse
import logging
import pathlib
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from molli_pinn_node_lstm.utils import load_config, dataloader
from molli_pinn_node_lstm.training import Trainer, TrainerConfig, PINNLoss
from molli_pinn_node_lstm.node_lstm import MOLLINeuralODELSTM
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger(__name__)


def checkpoint_path(path: str) -> pathlib.Path:
    """
    Argparse type helper that validates a checkpoint file path.

    Parameters
    ----------
    path : str
        Path to a checkpoint file. Must end with ``.pt``.

    Returns
    -------
    pathlib.Path
        The validated checkpoint path.

    Raises
    ------
    argparse.ArgumentTypeError
        If `path` does not end with ``.pt``.
    """
    p = pathlib.Path(path)
    if p.suffix != ".pt":
        raise argparse.ArgumentTypeError(f"{path} must be a .pt checkpoint file")
    return p


def cli_parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to YAML file with training configs.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Train a baseline model (full sequence) before finetuning on sparse acquisitions.",
    )
    parser.add_argument(
        "--resume-training",
        type=checkpoint_path,
        default=None,
        help="Path to a .pt checkpoint to resume training from.",
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="molli_baseline_best_weights.pt",
        help="Filename of baseline weights/checkpoint to load when not training baseline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed Python, NumPy and PyTorch RNGs for reproducibility.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="./runs",
        help="Root directory for runs (logs, checkpoints).",
    )
    return parser.parse_args()


def set_rng_state_seed(
    seed: int, deterministic: bool = True, strict: bool = False
) -> None:
    """
    Seed Python, NumPy and PyTorch RNGs to improve experiment reproducibility.

    This sets seeds for:
    - Python's built-in ``random`` module
    - NumPy (CPU)
    - PyTorch (CPU) RNG
    - PyTorch (CUDA) RNG (all devices if CUDA is available)

    When ``deterministic=True``, cuDNN is configured to prefer deterministic
    algorithms and autotuning is disabled (``cudnn.benchmark=False``).

    Parameters
    ----------
    seed : int
        Seed value used to initialize the pseudo-random number generators.
        Use the same seed across runs to make stochastic operations (e.g., weight
        initialization, dropout) more repeatable. Default used in this project's
        experiments is 1234.
    deterministic : bool, default=True
        If True, configure cuDNN to use deterministic algorithms when available.
        This can improve reproducibility but may reduce performance.
    strict : bool, default=False
        If True, call ``torch.use_deterministic_algorithms(True)`` so PyTorch will
        raise an error when a known nondeterministic operation is used.

    Returns
    -------
    None

    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False

    if strict:
        torch.use_deterministic_algorithms(True)


def configure_optimizer(
    model: torch.nn.Module,
    lr: float,
    gamma: float,
    weight_decay: float,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Create the optimizer and learning-rate scheduler."""
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)
    return optim, scheduler


def build_dataloaders(
    config: Dict[str, Any],
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build training and validation dataloaders from config.

    Notes
    -----
    This uses `batch_size=None` because the dataset yields pre-batched samples.

    """
    data_cfg = config["data"]
    settings = data_cfg["settings"]

    train_ds = dataloader.MOLLIDataset(
        folder_path=data_cfg["training_folder_path"],
        shuffle=True,
        drop_last=True,
        **settings,
    )
    val_ds = dataloader.MOLLIDataset(
        folder_path=data_cfg["validation_folder_path"],
        shuffle=False,
        drop_last=False,
        **settings,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=None, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=None, pin_memory=True)
    return train_loader, val_loader


def train(args: argparse.Namespace) -> None:
    """
    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI args from `cli_parse_args` function.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If resume checkpoint or baseline weights are requested but missing.
    """
    set_rng_state_seed(args.seed, deterministic=True, strict=False)

    config = load_config.load_yaml_config(args.config_path)

    runs_dir = pathlib.Path(args.runs_dir)
    run_name = config.get("name")
    if not run_name:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = runs_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    save_log_dir = save_dir / "tensorboard"
    save_log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_logger = SummaryWriter(log_dir=str(save_log_dir))

    save_ckpt_dir = save_dir / "ckpt"
    save_ckpt_dir.mkdir(parents=True, exist_ok=True)
    load_config.dump_config(config, save_path=save_ckpt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Running training on {device}")
    LOGGER.info(f"Config path: {args.config_path}")
    LOGGER.info(f"Run directory: {save_dir}")

    train_loader, val_loader = build_dataloaders(config)

    data_settings = config["data"]["settings"]
    tvec_norm = data_settings["tvec_norm"]
    signal_norm = data_settings["signal_norm"]

    trainer_cfg = TrainerConfig(**config["training"]["configs"])

    if args.baseline:
        max_seq_acquisitions = int(config["training"]["configs"]["max_acquisitions"])
        LOGGER.info(
            f"Training baseline: setting num_acquisitions={max_seq_acquisitions} and val_mc_samples=1"
        )
        trainer_cfg.num_acquisitions = max_seq_acquisitions
        trainer_cfg.val_mc_samples = (
            1  # No need to run MC sampling if baseline runs on full sequence
        )

    model = MOLLINeuralODELSTM(**config["node_lstm"])

    optimizer, scheduler = configure_optimizer(
        model=model, **config["training"]["optimizer"]
    )
    pinn_loss = PINNLoss(tvec_norm=tvec_norm, signal_norm=signal_norm)

    trainer = Trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tensorboard_logger=tensorboard_logger,
        molli_loss=pinn_loss,
        tvec_norm=tvec_norm,
        signal_norm=signal_norm,
        save_ckpt_dir=save_ckpt_dir,
        device=str(device),
        baseline=bool(args.baseline),
    )

    if args.resume_training is not None:
        ckpt_path = args.resume_training
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        LOGGER.info(f"Resuming training from checkpoint: {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)

    elif not args.baseline:
        baseline_name = args.baseline_name
        if not baseline_name.endswith(".pt"):
            baseline_name = f"{baseline_name}.pt"
        baseline_path = save_ckpt_dir / baseline_name
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Baseline weights not found: {baseline_path}\n"
                f"Run baseline first (--baseline) or provide the correct --baseline-name."
            )
        LOGGER.info(f"Loading baseline weights: {baseline_path}")
        trainer.load_model_state_dict(baseline_path)

    try:
        trainer.fit(training_set=train_loader, validation_set=val_loader)
        LOGGER.info("Training Completed.")
    finally:
        tensorboard_logger.close()


def main() -> None:
    args = cli_parse_args()
    train(args)


if __name__ == "__main__":
    main()
