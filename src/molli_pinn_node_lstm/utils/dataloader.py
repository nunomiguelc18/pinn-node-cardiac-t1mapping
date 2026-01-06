import pathlib
from typing import Dict, Tuple, Union, Iterator
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from scipy.io import loadmat


class MOLLIDataset(IterableDataset):
    """
    Stream MOLLI acquisitions stored as MATLAB (.mat) files and yield voxel-wise batches.

    Each `.mat` file is expected to contain a single 2D+t acquisition:
      - volume:      (H, W, T)
      - tvec:        (T,)
      - null_index:  (H, W) polarity transition index per voxel
      - pmap_mse:    (H, W, 3)
      - T1:          (H, W)

    Pipeline:
      1) load a file
      2) sort by tvec and apply polarity correction using `null_index`
      3) flatten spatial dims -> (H*W, T)
      4) mask invalid voxels and normalize
      5) yield batches:
           signals: (B, T), t1_ref: (B, 1), pmap: (B, 3), tvec: (T,)

    Notes
    -----
    This dataset already yields batches; set `torch.utils.data.DataLoader(batch_size=None)`.
    Single-worker only: use DataLoader(num_workers=0). With num_workers>0, IterableDataset
    instances can duplicate data unless worker sharding is implemented.

    Parameters
    ----------
    folder_path:
        Directory containing `.mat` files.
    batch_size:
        Number of voxels per yielded batch.
    drop_last:
        If True, drop the final smaller batch per file.
    shuffle:
        If True, deterministically shuffles (per epoch):
            1) the order of `.mat` files, and
            2) the voxel order within each file (before batching).
        Determinism is controlled by `base_seed + epoch` (see `set_epoch`).
    tvec_normalization_const:
        Divisor applied to tvec and T1 reference (e.g., 1000 converts ms -> s).
    signal_normalization_const:
        Divisor applied to signal readouts.
    base_seed:
        Base seed for deterministic shuffling.
    """

    def __init__(
        self,
        folder_path: Union[str, pathlib.Path],
        batch_size: int = 128,
        drop_last: bool = False,
        shuffle: bool = False,
        tvec_normalization_const: Union[int, float] = 1000,
        signal_normalization_const: Union[int, float] = 270,
        base_seed: int = 1234,
        **kwargs,
    ):
        folder = pathlib.Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"{folder} directory does not exist!")
        if not folder.is_dir():
            raise NotADirectoryError(f"{folder} is not a directory!")

        self.mat_file_paths = sorted(folder.glob("*.mat"))
        if not self.mat_file_paths:
            raise ValueError(f"No .mat files found in: {folder}")

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)


        self.tvec_normalization_const = float(tvec_normalization_const)
        self.signal_normalization_const = float(signal_normalization_const)
        if self.tvec_normalization_const <= 0:
            raise ValueError("tvec_normalization_const must be > 0")
        if self.signal_normalization_const <= 0:
            raise ValueError("signal_normalization_const must be > 0")

        self.base_seed = int(base_seed)
        self.shuffle = bool(shuffle)
        self._epoch = 0
        _ = kwargs # make ruff happy

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch in your training loop."""
        self.epoch = epoch

    def preprocess(
        self, mat_file: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply polarity correction, flatten, mask invalid voxels, and normalize.

        Returns
        -------
        volume: (N, T)
        tvec: (T,)
        molli_t1_ref: (N, 1)
        pmap: (N, 3)
        """
        volume = mat_file["volume"]

        # to avoid mutating the original loaded arrays
        tvec = mat_file["tvec"].copy() 
        null_index = mat_file["null_index"].copy()

        pmap = mat_file["pmap"]
        molli_t1_ref = mat_file["molli_t1_ref"]

        tvec, volume = self._sort_molli_by_polarity(volume, tvec, null_index)

        if not np.all(np.diff(tvec) > 0):
            raise ValueError("tvec is not strictly increasing after sorting.")

        volume, pmap, molli_t1_ref = self._flatten_volume(
            volume=volume, molli_t1_ref=molli_t1_ref, pmap=pmap
        )

        max_abs = np.max(np.abs(volume), axis=1)
        finite = np.isfinite(max_abs)

        mask_t1 = (molli_t1_ref[:, 0] > 20) & (molli_t1_ref[:, 0] < 3000)

        t1_star = pmap[:, -1] # Filter noisy param estimates that may pass other masking thresholds 
        mask_pmap = (t1_star > 20) & (t1_star < 4000)

        mask_signal = (max_abs > 25) & (max_abs < 600)

        keep = finite & mask_signal & mask_pmap & mask_t1

        volume = (volume[keep] / self.signal_normalization_const).astype(np.float32, copy=False)
        molli_t1_ref = (molli_t1_ref[keep] / self.tvec_normalization_const).astype(np.float32, copy=False)
        tvec = (tvec / self.tvec_normalization_const).astype(np.float32, copy=False)
        pmap = pmap[keep].astype(np.float32, copy=False)

        return volume, tvec, molli_t1_ref, pmap

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if get_worker_info() is not None:
            raise RuntimeError("MOLLIDataset is single-worker only. Use DataLoader(num_workers=0).")

        file_list = list(self.mat_file_paths)
        rng = np.random.default_rng(self.base_seed + self.epoch)

        if self.shuffle:
            rng.shuffle(file_list)

        for mat_file_path in file_list:
            mat_file = self.read_loadmat(mat_file_path)
            volume, tvec, molli_t1_ref, pmap = self.preprocess(mat_file)
            if volume.shape[0] == 0:
                continue

            tvec_t = torch.from_numpy(np.ascontiguousarray(tvec))
            volume_t = torch.from_numpy(np.ascontiguousarray(volume))
            t1_t = torch.from_numpy(np.ascontiguousarray(molli_t1_ref))
            pmap_t = torch.from_numpy(np.ascontiguousarray(pmap))

            perm = rng.permutation(volume_t.shape[0]) if self.shuffle else np.arange(volume_t.shape[0])
            end = perm.size - (perm.size % self.batch_size) if self.drop_last else perm.size

            for start in range(0, end, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch = {"volume": volume_t[idx], "molli_t1_ref" : t1_t[idx], "pmap" : pmap_t[idx], "tvec" : tvec_t}
                yield batch

    @staticmethod
    def read_loadmat(path: Union[str, pathlib.Path]) -> Dict[str, np.ndarray]:
        """
        Load required arrays from a MOLLI `.mat` file.

        Required fields in the MATLAB file
        ----------------------------------
        pmap_mse:
            (H, W, 3) model parameters (c, k, T1*) estimated by least-squares fitting on
            S(t) = c * (1 - k * exp( -t / T1* )), e.g via Levenberg–Marquardt algorithm.
        volume:
            (H, W, T) MOLLI signal acquisitions.
        tvec:
            (T,) inversion times / elapsed time after inversion pulse.
        null_index:
            (H, W) polarity transition index per voxel.
        T1:
            (H, W) reference T1 derived from fitted parameters using T1 = T1* (k - 1).
        """
        path = pathlib.Path(path)
        load_data = loadmat(path)

        try:
            pmap = load_data["pmap_mse"].astype(np.float32)
            volume = load_data["volume"].astype(np.float32)
            tvec = np.asarray(load_data["tvec"]).astype(np.float32).flatten()
            null_index = np.asarray(load_data["null_index"]).astype(np.uint8)
            molli_t1_ref = np.asarray(load_data["T1"]).astype(np.float32)
        except KeyError as e:
            raise KeyError(f"Missing key {e} in {path}") from e

        return {
            "volume": volume,
            "tvec": tvec,
            "pmap": pmap,
            "null_index": null_index,
            "molli_t1_ref": molli_t1_ref,
        }

    @staticmethod
    def _sort_molli_by_polarity(
        volume: np.ndarray, tvec: np.ndarray, null_index: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sort readouts by tvec and apply polarity correction based on null_index.

        Polarity correction: for each voxel, samples with index < null_index are inverted.

        Returns
        -------
        sorted_tvec: (T,)
        volume_corrected: (H, W, T)
        """
        if volume.ndim != 3:
            raise ValueError(f"Expected volume shape (H, W, T); got {volume.shape}")
        if tvec.ndim != 1:
            raise ValueError(f"Expected tvec shape (T,); got {tvec.shape}")
        if volume.shape[-1] != tvec.size:
            raise ValueError(
                f"volume T ({volume.shape[-1]}) must match len(tvec) ({tvec.size})"
            )
        if null_index.shape != volume.shape[:2]:
            raise ValueError(
                f"null_index shape {null_index.shape} must match (H,W) {volume.shape[:2]}"
            )

        null_index = np.clip(null_index, 0, tvec.size).astype(null_index.dtype, copy=False)
        order = np.argsort(tvec)
        sorted_tvec = tvec[order]
        vol = volume[:, :, order]

        readout_idx = np.arange(tvec.size, dtype=null_index.dtype)[None, None, :]

        invert_mask = readout_idx < null_index[:, :, None]

        volume_corrected = np.where(invert_mask, -vol, vol)
        return sorted_tvec, volume_corrected

    @staticmethod
    def _flatten_volume(
        volume: np.ndarray, molli_t1_ref: np.ndarray, pmap: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten 2D+t arrays into voxel-wise rows (H*W, ...)."""
        if pmap.ndim != 3 or pmap.shape[-1] != 3:
            raise ValueError(f"Expected pmap shape (H, W, 3); got {pmap.shape}")
        if volume.shape[:2] != pmap.shape[:2]:
            raise ValueError("pmap must match H,W of volume")
        if molli_t1_ref.shape != volume.shape[:2]:
            raise ValueError("T1 ref must match H,W of volume")

        h, w, t = volume.shape
        volume = volume.reshape(h * w, t)
        pmap = pmap.reshape(h * w, 3)
        molli_t1_ref = molli_t1_ref.reshape(h * w, 1)
        return volume, pmap, molli_t1_ref
