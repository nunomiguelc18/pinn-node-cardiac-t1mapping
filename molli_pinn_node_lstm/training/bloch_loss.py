import torch
from abc import ABC, abstractmethod
import torch.nn.functional as F
from molli_pinn_node_lstm.utils import molli_signal_model
from typing import Dict

class SignalRecoveryLoss(ABC):
    @abstractmethod
    def compute_loss(self, *args, **kwargs)-> Dict[str, torch.Tensor]:
        pass

class PINNLoss(SignalRecoveryLoss):
    """
    Physics-informed loss for the 3-parameter MOLLI recovery model.

    This loss combines three terms computed from MOLLI parameters (C, K, T1*):
      1) T1 supervision:      MSE(T1_hat, T1_ref), with T1_hat = T1* (K - 1)
      2) Signal consistency:  MSE(S_hat(t), S_ref(t)) over a dense time grid
      3) Dynamics consistency (PINN): MSE(dS_hat/dt, dS_ref/dt) over the same grid,
         scaled by `pinn_lambda` (derivatives near t≈0 can be much larger in magnitude).

    Attributes
    ----------
    tvec_normalization_const:
        Divisor applied to tvec and T1 reference (e.g., 1000 converts ms -> s).
    signal_normalization_const:
        Divisor applied to signal readouts.
    t_end_recovery:
        End of the high-resolution (recovery) portion of the time grid, in seconds.
    t_end:
        End of the full time grid (recovery + saturation), in seconds. Must be > t_end_recovery.
    n_grid_samples:
        Total number of samples in the time grid.
    p_recovery:
        Fraction of samples allocated to the recovery region. The remainder is allocated to the
        saturation region.
    pinn_lambda:
        Weight applied to the derivative consistency term.

    Notes
    -----
    The derivative target uses a chain-rule scaling:
        gamma = tvec_normalization_const / signal_normalization_const
    to keep dS/dt terms comparable when time and signal are represented in normalized units.

   
    The split is clamped to guarantee at least 2 samples in both recovery and saturation regions.
    """
    def __init__(self, tvec_normalization_const: float, signal_normalization_const: float, 
                 t_end_recovery: float = 2.0, t_end: float = 4.0, n_grid_samples: int = 1000, p_recovery: float = 0.75, pinn_lambda: float = 1e-2):
        if t_end <= t_end_recovery:
            raise ValueError("Invalid condition: t_end <= t_end_recovery!")
        if n_grid_samples < 4:
            raise ValueError("n_grid_samples must be >= 4 to allocate at least 2 samples to both recovery and saturation.")
        if not (0.0 <= p_recovery <= 1.0):
            raise ValueError("p_recovery must be in [0, 1].")


        self.chain_rule_gamma = float(tvec_normalization_const / signal_normalization_const)
        self.tvec_normalization_const = float(tvec_normalization_const)
        self.signal_normalization_const = float(signal_normalization_const)
        self.t_end_recovery = float(t_end_recovery)
        self.t_end = float(t_end)
        self.n_grid_samples = int(n_grid_samples)
        self.p_recovery = float(p_recovery)
        self.pinn_lambda = float(pinn_lambda)

        n_recovery = int(n_grid_samples * p_recovery)
        n_recovery = max(2, min(n_recovery, n_grid_samples - 2))
        n_saturation = n_grid_samples - n_recovery

        saturation_gap = 1e-3 #To avoid repeating the same t at boundaries
        recovery_grid = torch.linspace(0.0, t_end_recovery, n_recovery)
        saturation_grid = torch.linspace(t_end_recovery + saturation_gap, t_end, n_saturation)
        self.tvec_grid_cpu = torch.cat([recovery_grid, saturation_grid], dim=0)
        

    
    def compute_loss(
        self,
        pmap_ref: Dict[str, torch.Tensor],
        T1_ref: torch.Tensor,
        pmap_hat: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss terms for a batch of MOLLI parameter predictions.

        Parameters
        ----------
        pmap_ref:
            Reference MOLLI parameters as a dict with keys {"C", "K", "T1_star"}.

        T1_ref:
            Reference T1 values (e.g., offline fit).

        pmap_hat:
            Predicted MOLLI parameters as a dict with keys {"C", "K", "T1_star"}.

        Returns
        -------
        Dict[str, torch.Tensor]
            - "total_loss": sum of all terms
            - "T1":         (MSE) supervised T1
            - "S(t)":       (MSE) signal reconstruction on the time grid
            - "dS(t)/dt":   (MSE) derivative consistency term; already scaled by `pinn_lambda`
        """

        anchor = pmap_hat["T1_star"] # ensure we cast to the same device, dtype as our model 
        t = self.tvec_grid_cpu.to(device=anchor.device, dtype=anchor.dtype)

        T1_hat = molli_signal_model.t1_from_apparent(K=pmap_hat["K"], T1_star=pmap_hat["T1_star"])
        T1_loss = F.mse_loss(T1_hat, T1_ref, reduction="mean")

        signal_rec_hat = molli_signal_model.signal_recovery(tvec=t, **pmap_hat)
        signal_rec_ref = molli_signal_model.signal_recovery(tvec=t, **pmap_ref)
        signal_rec_loss = F.mse_loss(signal_rec_hat, signal_rec_ref, reduction="mean")

        temp_dynamics_hat = molli_signal_model.ds_dt(tvec=t, **pmap_hat)
        denorm_tvec_grid = t * self.tvec_normalization_const
        temp_dynamics_ref = molli_signal_model.ds_dt(tvec=denorm_tvec_grid, **pmap_ref) * self.chain_rule_gamma
        temp_dynamics_loss = F.mse_loss(temp_dynamics_hat, temp_dynamics_ref, reduction="mean") * self.pinn_lambda

        total_loss = T1_loss + signal_rec_loss + temp_dynamics_loss
        return {"total_loss": total_loss, "T1": T1_loss, "S(t)": signal_rec_loss, "dS(t)/dt": temp_dynamics_loss}