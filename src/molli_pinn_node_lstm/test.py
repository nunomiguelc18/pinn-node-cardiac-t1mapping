import argparse
import logging
import pathlib
import random
from typing import Tuple

import numpy as np
import torch

from molli_pinn_node_lstm.utils import load_config, dataloader, molli_signal_model
from molli_pinn_node_lstm.node_lstm import MOLLINeuralODELSTM
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger(__name__)


def handle_T1_results(
    mean_map: np.ndarray,
    sd_map: np.ndarray,
    molli_t1_ref: np.ndarray,
    output_folder: pathlib.Path,
    file_name: str,
) -> None:
    """Matplotlib qualitative figures plotting."""

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(18, 4), constrained_layout=True)

    for ax in axs:
        ax.set_axis_off()

    mean_T1_hat = axs[0].imshow(mean_map, vmin=0, vmax=1500, cmap="jet")
    axs[0].set_title(r"(Mean $\hat{T_1}$)")

    sd_map_T1_hat = axs[1].imshow(sd_map, vmin=0, vmax=100, cmap="jet")
    axs[1].set_title(r"(SD $\hat{T_1}$)")

    mean_T1 = axs[2].imshow(molli_t1_ref, vmin=0, vmax=1500, cmap="jet")
    axs[2].set_title("$ T_1 $")

    diff = np.abs(mean_map - molli_t1_ref)
    diff_im = axs[3].imshow(diff, vmin=0, vmax=100, cmap="jet")
    axs[3].set_title(r"$ \vert \hat{T_1} - T_1 \vert $")

    fig.colorbar(mean_T1_hat, ax=axs[0], location="right", label="T1 [ms]")
    fig.colorbar(sd_map_T1_hat, ax=axs[1], location="right", label="T1 [ms]")
    fig.colorbar(mean_T1, ax=axs[2], location="right", label="T1 [ms]")
    fig.colorbar(diff_im, ax=axs[3], location="right", label="T1 [ms]")

    fig.savefig(output_folder / f"{file_name}.png", dpi=300)
    plt.close(fig)


def set_rng_state_seed(
    seed: int, deterministic: bool = True, strict: bool = False
) -> None:
    """Seed Python, NumPy and PyTorch RNGs to improve reproducibility."""
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


def preprocess(
    volume: np.ndarray, tvec: np.ndarray, signal_norm: float, tvec_norm: float, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess MOLLI data to ensure the correct shapes and normalization."""
    if volume.ndim != 3:
        raise ValueError(f"Expected volume shape (H, W, T); got {volume.shape}")
    if tvec.ndim != 1:
        raise ValueError(f"Expected tvec shape (T,); got {tvec.shape}")
    if volume.shape[-1] != tvec.size:
        raise ValueError(
            f"Volume T ({volume.shape[-1]}) must match len(tvec) ({tvec.size})"
        )

    order = np.argsort(tvec)
    sorted_tvec = tvec[order]
    vol = volume[:, :, order]

    vol = (vol / signal_norm).astype(np.float32, copy=False)
    sorted_tvec = (sorted_tvec / tvec_norm).astype(np.float32, copy=False)

    h, w, t = vol.shape
    vol = np.reshape(vol, (h * w, t))
    return vol, sorted_tvec


def checkpoint_path(path: str) -> pathlib.Path:
    """Helper to validate a checkpoint file path."""
    p = pathlib.Path(path)
    if p.suffix != ".pt":
        raise argparse.ArgumentTypeError(f"{path} must be a .pt checkpoint file")
    return p


def cli_parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config-path", type=str, required=True, help="Path to YAML config file."
    )
    parser.add_argument(
        "--testing-dir",
        type=pathlib.Path,
        required=True,
        help="Directory with .mat files for testing.",
    )
    parser.add_argument(
        "--load-state-dict",
        type=checkpoint_path,
        required=True,
        help="Path to model weights for inference.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./simulations",
        help="Directory to save inference results.",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Seed for reproducibility."
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        required=True,
        help="Number of Monte Carlo samples for each figure.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=8192,
        help="Batch size for inference.",
    )
    return parser.parse_args()


def mask_noisy_signals(volume: np.ndarray) -> np.ndarray:
    """Return boolean mask (H*W,) of voxels to mask out noisy readouts."""
    h, w, t = volume.shape
    vol = volume.reshape(h * w, t)
    max_abs = np.max(np.abs(vol), axis=-1)
    noise = np.isfinite(max_abs)
    mask = (max_abs < 25) | (max_abs > 600) | ~noise
    return mask


def sample_random_indices(num_acquisitions: int, max_acquisitions: int) -> np.ndarray:
    """Return a boolean mask of length max_acquisitions."""
    n = int(max_acquisitions)
    k = int(num_acquisitions)
    if k >= n:
        return np.ones(n, dtype=bool)

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
    return mask


@torch.inference_mode()
def polarity_recovery(model, vol, tvec, batch_size, device):
    n_voxels, molli_acq = vol.shape
    pmap = np.zeros((n_voxels, 3))
    for b_enum in range(0, n_voxels, batch_size):
        tmp_batch_vol = vol[b_enum : b_enum + batch_size, ...].copy()
        cache_polarity_loss = np.full(
            shape=(tmp_batch_vol.shape[0],), fill_value=np.inf
        )
        tmp_pmap = np.zeros(shape=(tmp_batch_vol.shape[0], 3))
        for i in range(molli_acq):
            tmp_polarity_inv = tmp_batch_vol.copy()
            tmp_polarity_inv[:, :i] *= -1.0
            tmp_polarity_inv = torch.from_numpy(tmp_polarity_inv).to(device)

            tmp_pmap_hat = model(volume=tmp_polarity_inv, tvec=tvec)
            signal_rec_hat = molli_signal_model.signal_recovery(
                tvec=tvec, **tmp_pmap_hat
            )
            loss = F.mse_loss(tmp_polarity_inv, signal_rec_hat, reduction="none")
            mean_loss = torch.mean(loss, dim=1).cpu().numpy()

            flag_closer_null_index = cache_polarity_loss > mean_loss
            cache_polarity_loss[flag_closer_null_index] = mean_loss[
                flag_closer_null_index
            ]
            C, K, T1_star = (
                tmp_pmap_hat["C"].cpu().numpy(),
                tmp_pmap_hat["K"].cpu().numpy(),
                tmp_pmap_hat["T1_star"].cpu().numpy(),
            )
            pack_pmap = np.concatenate((C, K, T1_star), axis=-1)
            tmp_pmap[flag_closer_null_index, ...] = pack_pmap[
                flag_closer_null_index, ...
            ]

        if tmp_batch_vol.shape[0] == batch_size:
            pmap[b_enum : b_enum + batch_size, ...] = tmp_pmap
        else:
            r = n_voxels % batch_size
            pmap[-r:, ...] = tmp_pmap

        del (
            tmp_polarity_inv,
            tmp_pmap_hat,
            signal_rec_hat,
            loss,
        )  # just to avoid memory overhead
        torch.cuda.empty_cache()

    return pmap


def test(args: argparse.Namespace) -> None:
    """
    Run the model inference on testing data.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed from the CLI.
    """
    set_rng_state_seed(args.seed, deterministic=True, strict=False)

    config = load_config.load_yaml_config(args.config_path)
    max_molli_acquisitions = int(config["training"]["configs"]["max_acquisitions"])
    molli_acquisitions = int(config["training"]["configs"]["num_acquisitions"])

    inference_dir = pathlib.Path(args.save_dir)
    run_name = config.get("name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    save_dir = inference_dir / f"molli_{molli_acquisitions}_{run_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Running Inference on {device}")
    LOGGER.info(f"Config path: {args.config_path}")
    LOGGER.info(f"Save directory: {save_dir}")

    model = MOLLINeuralODELSTM(**config["node_lstm"]).to(device)

    LOGGER.info(f"Loading state dict from {str(args.load_state_dict)}")
    payload = torch.load(args.load_state_dict, map_location=device)
    state_dict = payload["model_state"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    LOGGER.info("Loaded Weights successfully")

    file_list = list(sorted(args.testing_dir.glob("*.mat")))
    testing_pbar = tqdm(file_list, desc="Testing", dynamic_ncols=True, position=0)
    for mat_file_path in testing_pbar:
        LOGGER.info(f"Loading {mat_file_path.name}")
        mat_file = dataloader.MOLLIDataset.read_loadmat(mat_file_path)
        filter_mask = mask_noisy_signals(volume=mat_file["volume"])
        molli_t1_ref = mat_file["molli_t1_ref"]
        h, w = molli_t1_ref.shape
        molli_t1_ref = molli_t1_ref.flatten()
        molli_t1_ref[filter_mask, ...] *= 0
        vol, tvec = preprocess(
            **mat_file,
            signal_norm=config["data"]["settings"]["signal_norm"],
            tvec_norm=config["data"]["settings"]["tvec_norm"],
        )

        simulations_tracker = []
        for simulation in range(args.mc_samples):
            LOGGER.info(f"Evaluating Simulation {simulation}/{args.mc_samples}.")
            random_mask = sample_random_indices(
                num_acquisitions=molli_acquisitions,
                max_acquisitions=max_molli_acquisitions,
            )
            tmp_vol = vol[..., random_mask].copy()
            tmp_tvec = torch.from_numpy(tvec[random_mask]).to(device)
            pmap_hat = polarity_recovery(
                model=model,
                vol=tmp_vol,
                tvec=tmp_tvec,
                batch_size=args.batch_size,
                device=device,
            )

            T1_output = molli_signal_model.t1_from_apparent(
                K=pmap_hat[..., 1], T1_star=pmap_hat[..., 2]
            )
            T1_output *= config["data"]["settings"]["tvec_norm"]
            simulations_tracker.append(T1_output[..., None])

        MC_estimates = np.concatenate(simulations_tracker, axis=-1)
        MC_estimates[filter_mask, ...] *= 0
        mean_map = np.mean(MC_estimates, axis=-1)
        sd_map = np.std(MC_estimates, axis=-1)

        molli_t1_ref = np.reshape(molli_t1_ref, (h, w))
        mean_map = np.reshape(mean_map, (h, w))
        sd_map = np.reshape(sd_map, (h, w))
        handle_T1_results(
            mean_map=mean_map,
            sd_map=sd_map,
            molli_t1_ref=molli_t1_ref,
            output_folder=save_dir,
            file_name=mat_file_path.stem,
        )


def main() -> None:
    """
    Main function to execute the testing process.
    """
    args = cli_parse_args()
    test(args)


if __name__ == "__main__":
    main()
