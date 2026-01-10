import torch

def _broadcast(tvec, C, K , T1_star):
    if tvec.ndim == 1:
        tvec = tvec[:, None]
    if (C.ndim == 2) and (C.shape[-1] == 1):
        C = C.T
    if (K.ndim == 2) and (K.shape[-1] == 1):
        K = K.T
    if (T1_star.ndim == 2) and (T1_star.shape[-1] == 1):
        T1_star = T1_star.T
    return tvec, C, K, T1_star

def signal_recovery(
    tvec: torch.Tensor,
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
    (tvec, C, K , T1_star) = _broadcast(tvec,C,K,T1_star)
    # if 
    denom = T1_star + eps
    return C * (1.0 - K * torch.exp(-tvec / denom))


def ds_dt(
    tvec: torch.Tensor,
    C: torch.Tensor,
    K: torch.Tensor,
    T1_star: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Time derivative of the MOLLI 3-parameter model:
        dS/dt = C*K*exp(-t/T1*) / T1*
    """
    (tvec, C, K , T1_star) = _broadcast(tvec,C,K,T1_star)
    denom = T1_star + eps
    return C * K * torch.exp(-tvec / denom) / denom


def t1_from_apparent(K: torch.Tensor, T1_star: torch.Tensor) -> torch.Tensor:
    """Compute apparent-to-true T1 mapping: T1 = T1* * (K - 1)."""
    return T1_star * (K - 1.0)