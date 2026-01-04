import torch

def signal_recovery(
    t: torch.Tensor,
    C: torch.Tensor,
    K: torch.Tensor,
    T1_star: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    MOLLI 3-parameter signal model:
        S(t) = C * (1 - K * exp(-t / T1*))

    Shapes broadcast naturally (e.g., t can be (n,) and parameters (batch, 1)).
    """
    denom = T1_star + eps
    return C * (1.0 - K * torch.exp(-t / denom))


def ds_dt(
    t: torch.Tensor,
    C: torch.Tensor,
    K: torch.Tensor,
    T1_star: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Time derivative of the MOLLI 3-parameter model:
        dS/dt = C*K*exp(-t/T1*) / T1*
    """
    denom = T1_star + eps
    return C * K * torch.exp(-t / denom) / denom


def t1_from_apparent(K: torch.Tensor, T1_star: torch.Tensor) -> torch.Tensor:
    """Compute apparent-to-true T1 mapping: T1 = T1* * (K - 1)."""
    return T1_star * (K - 1.0)