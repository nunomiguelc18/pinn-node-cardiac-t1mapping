import torch
import torch.nn as nn
from torchdyn.numerics import odeint
from typing import Dict, Tuple, Optional

class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.normalization = nn.LayerNorm(out_dim)
        self.activation =  nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class ModelParametersProjection(nn.Module):
    """Projection head that maps a latent state to signal recovery model 3-parameters ( C , K , T1*).

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent input state.
    output_dim : int, default=3
        Number of parameters to predict.

    Notes
    -----
    A Softplus is applied to ensure parameters are physically coherent.
    """

    def __init__(self, latent_dim: int, output_dim: int = 3) -> None:
        super().__init__()
        first_block_out_dims = max(1,latent_dim // 2)
        second_block_out_dims = max(1, latent_dim // 4)
        self.net = nn.Sequential(
            MLPBlock(latent_dim, first_block_out_dims),
            MLPBlock(first_block_out_dims, second_block_out_dims),
            nn.Linear(second_block_out_dims, output_dim),
        )
        self.softplus = nn.Softplus(beta=0.5) # beta = 0.5 to make it smoother in final estimates.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute positive parameter predictions.

        Parameters
        ----------
        x : torch.Tensor
            Latent features of shape (batch, latent_dim).

        Returns
        -------
        torch.Tensor
            Positive parameters of shape (batch, output_dim).
        """
        x = self.net(x)
        x = self.softplus(x)
        return x

class NODEFunc(nn.Module):
    """ (t, h) -> dh/dt signature required by `odeint`."""

    def __init__(self, f_node: nn.Module) -> None:
        super().__init__()
        self.f_node = f_node

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Evaluate dh/dt.

        Parameters
        ----------
        t : torch.Tensor
            Scalar tensor representing the current integration time.
        h : torch.Tensor
            Hidden state tensor of shape (batch, hidden_size).

        Returns
        -------
        torch.Tensor
            Time derivative `dh/dt` of shape (batch, hidden_size).
        """
        batch = h.shape[0]
        t_col = t.to(device=h.device, dtype=h.dtype).reshape(1, 1).expand(batch, 1)
        dh_dt = self.f_node(torch.cat([h, t_col], dim=1))
        return dh_dt


class NODELSTMCell(nn.Module):
    """One NODE-LSTM cell.

    At each step:
      1) Apply an `nn.LSTMCell` update producing a candidate hidden state.
      2) Integrate that hidden state using a neural ODE from `t0` to `t1`.

    Parameters
    ----------
    input_size : int
        Number of input features per timestep.
    hidden_size : int
        Hidden size for all layers.
    solver : str
        ODE solver name passed to `torchdyn.numerics.odeint`.
    rtol : float
        Relative tolerance for the solver.
    atol : float
        Absolute tolerance for the solver.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        solver: str,
        rtol: float,
        atol: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        self.lstm = nn.LSTMCell(input_size, hidden_size)

        self.f_node = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.odefunc = NODEFunc(self.f_node)

    def forward(self, x_t: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor], t0: torch.Tensor, t1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one ODE-LSTM update.

        Parameters
        ----------
        x_t : torch.Tensor
            Input at current step, shape (batch, input_size).
        hx : tuple[torch.Tensor, torch.Tensor]
            Previous hidden and cell states `(h, c)`, each shape (batch, hidden_size).
        t0 : torch.Tensor
            Start time of the integration interval (scalar tensor).
        t1 : torch.Tensor
            End time of the integration interval (scalar tensor).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Next hidden and cell states `(h_next, c_next)`
        """
        h, c = hx
        h_tilde, c_next = self.lstm(x_t, (h, c))

        t_span = torch.stack([t0, t1]).to(device=h.device, dtype=h.dtype)

        _, traj = odeint(
            self.odefunc,
            h_tilde,
            t_span=t_span,
            solver=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        h_next = traj[-1]
        return h_next, c_next


class NeuralODELSTM(nn.Module):
    """
    Parameters
    ----------
    input_size : int
        Number of input features per timestep.
    hidden_size : int
        Hidden size for all layers.
    num_layers : int,
        Number of stacked ODE-LSTM layers.
    batch_first : bool
        If True, expects inputs as (batch, seq, features). Otherwise (seq, batch, features), as standard in PyTorch LSTM nn.module.
    solver : str
        ODE solver name passed to `torchdyn.numerics.odeint`.
    rtol : float
        Relative tolerance for the solver.
    atol : float
        Absolute tolerance for the solver.

    Notes
    -----
    This implementation expects `tvec` to be a shared time 1D vector .
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool,
        solver: str,
        rtol: float,
        atol: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        self.cells = nn.ModuleList(
            [
                NODELSTMCell(
                    input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    solver=solver,
                    rtol=rtol,
                    atol=atol,
                )
                for i in range(num_layers)
            ]
        )

    def init_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)
        return h0, c0

    def forward(
        self,
        x: torch.Tensor,
        tvec: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the ODE-LSTM over a sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape:
              - (batch, seq_len, input_size) if `batch_first=True`
              - (seq_len, batch, input_size) otherwise
        tvec : torch.Tensor
            Strictly increasing time vector of shape (seq_len,).
        hx : tuple[torch.Tensor, torch.Tensor], optional
            Initial states `(h0, c0)` each of shape (num_layers, batch, hidden_size).

        Returns
        -------
        y : torch.Tensor
            Output sequence of shape:
              - (batch, seq_len, hidden_size) if `batch_first=True`
              - (seq_len, batch, hidden_size) otherwise
        (h_n, c_n) : tuple[torch.Tensor, torch.Tensor]
            Final hidden and cell states, each shape (num_layers, batch, hidden_size).
        """
        if tvec.ndim != 1:
            raise ValueError(f"tvec must have shape (seq_len,), got {tuple(tvec.shape)}")

        if self.batch_first:
            x = x.transpose(0, 1)  # (seq, batch, feat)

        seq_len, batch, _ = x.shape
        device, dtype = x.device, x.dtype

        if hx is None:
            h, c = self.init_hidden(batch, device=device, dtype=dtype)
        else:
            h, c = hx

        outputs = []
        t0 = tvec.new_tensor(0.0)

        for step in range(seq_len):
            t1 = tvec[step]
            layer_input = x[step]

            h_next_layers = []
            c_next_layers = []

            for layer_idx, layer in enumerate(self.cells):
                h_l, c_l = h[layer_idx], c[layer_idx]
                h_l_next, c_l_next = layer(layer_input, (h_l, c_l), t0=t0, t1=t1)
                layer_input = h_l_next
                h_next_layers.append(h_l_next)
                c_next_layers.append(c_l_next)

            h = torch.stack(h_next_layers, dim=0)
            c = torch.stack(c_next_layers, dim=0)

            outputs.append(layer_input)
            t0 = t1

        y = torch.stack(outputs, dim=0)

        if self.batch_first:
            y = y.transpose(0, 1)

        return y, (h, c)


class MOLLINeuralODELSTM(nn.Module):
    """
    Physics-informed continuous-time LSTM-ODE model for cardiac T1 mapping from sparse MOLLI readouts.

    This module is designed around a framework combining:
      (1) a continuous-time recurrent backbone (LSTM-ODE) that handles irregular inversion times, and
      (2) a decoder that predicts parameters of the MOLLI 3-parameter signal model.

    Attributes
    ----------
    embedding_dim : int, default=128
        Dimension of the input embedding for (volume, time) pairs.
    lstm_hidden_size : int, default=128
        Hidden size for the ODE-LSTM.
    solver : str, default="dopri5"
        Black-box ODE solver name.
    rtol : float, default=1e-3
        Relative tolerance for ODE integration.
    atol : float, default=1e-3
        Absolute tolerance for ODE integration.

    Architecture details
    --------------------

    1) Pairwise feature construction
      Each timestep i is represented as a 2D feature:
          x_i = [volume_i, t_i]
      giving a tensor of shape (batch, seq_len, 2).

    2) Embedding
      A learnable linear layer maps R^2 -> R^embedding_dim for each timestep:
          e_i = Embedding(x_i)
      producing (batch, seq_len, embedding_dim).

    3) Continuous-time LSTM-ODE encoding
      The embedded sequence is processed by an ODE-LSTM backbone (continuous-time LSTM-ODE).
      At each timestep:
        1) an LSTMCell performs the discrete gating update producing a candidate hidden state h_tilde,
        2) h_tilde is evolved continuously over the interval [t_{i-1}, t_i] by integrating a neural ODE:
              dh/dt = f_theta(h, t)
      while the cell state remains discrete for stable gradient propagation. The output is a continuous-time
      encoding of the relaxation trajectory.

    4) Parameter decoding
      The final hidden state is decoded to three positive parameters:
          params = (c, k, T1*)
      using a small MLP + Softplus. A constant offset is applied to k for normalization purposes.

    """

    def __init__(
        self,
        embedding_dim: int = 128,
        lstm_hidden_size: int = 128,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-3,
        batch_first: bool = True,
        num_layers: int = 1
    ) -> None:
        super().__init__()

        self.embedding = nn.Linear(2, embedding_dim)

        self.ode_lstm = NeuralODELSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            solver=solver,
            rtol=rtol,
            atol=atol,
        )

        self.bloch_head = ModelParametersProjection(latent_dim=lstm_hidden_size, output_dim=3)
        self.register_buffer("k_offset", torch.tensor(2.0))


    def forward(self, volume: torch.Tensor, tvec: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Estimate MOLLI signal model parameters from a signal sequence and its time vector.

        Parameters
        ----------
        volume : torch.Tensor
            per-voxel signal sequence (e.g., MOLLI signal intensities)
        tvec: torch.Tensor
            inversion time / time elapsed vector for the sequence

        Returns
        ---------
        Dict[str, torch.Tensor]
            A dictionary wrapping the signal model parameters estimates ( C , K , T1*)  
        """
        if volume.ndim != 2:
            raise ValueError(f"volume must have shape (batch, seq_len); got {tuple(volume.shape)}")
        if tvec.ndim != 1:
            raise ValueError(f"tvec must have shape (seq_len,); got {tuple(tvec.shape)}")
        if volume.shape[1] != tvec.shape[0]:
            raise ValueError(
                f"volume seq_len ({volume.shape[1]}) must match tvec length ({tvec.shape[0]})"
            )
        
        vol = volume[:, :, None]
        t_broadcasted = tvec[None, :, None].expand(vol.shape)
        x_paired = torch.cat((vol, t_broadcasted), dim=-1)
        x_emb = self.embedding(x_paired)

        _, (h_n, _) = self.ode_lstm(x_emb, tvec)

        last_hidden = h_n[-1] 
        params = self.bloch_head(last_hidden) 

        C = params[:, [0]]
        K = params[:, [1]] + self.k_offset
        T1_star = params[:, [2]]
        return {"C" : C, "K" : K, "T1_star": T1_star}