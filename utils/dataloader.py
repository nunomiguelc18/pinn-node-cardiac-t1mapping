from torch.utils.data import IterableDataset
from scipy.io import loadmat
import pathlib
from typing import Dict, Tuple, Union
import numpy as np
import torch

class LoadMOLLIDataset(IterableDataset):
    def __init__(self,
                 folder_path: str,
                 validation: bool = False,
                 tvec_normalization_const : Union[int,float] = 1000,
                 signal_normalization_const: Union[int,float] = 270, 
                 base_seed: int = 1234,
                 **kwargs

                 ):
        self.mat_files_paths = sorted(list(pathlib.Path(folder_path).glob('*.mat')))
        self.validation = validation
        self.tvec_normalization_const = tvec_normalization_const
        self.signal_normalization_const = signal_normalization_const
        self.epoch = 0
        self.base_seed = base_seed

    def set_epoch(self, epoch:int) -> None:
        self.epoch = epoch

    def sort_molli_by_polarity(self, volume: np.ndarray, tvec: np.ndarray, null_index:np.ndarray, 
                               **kwargs):
        I=np.argsort(tvec)
        sorted_tvec = np.sort(tvec)
        
        #Invert signal points according to null_index 
        mask_indices = np.arange(tvec.shape[0])[np.newaxis, np.newaxis, :] < null_index[:,:,np.newaxis]
        signals_wpr = np.where(mask_indices, -1*volume[:,:,I], volume[:,:,I])
                
        return torch.from_numpy(sorted_tvec), torch.from_numpy(signals_wpr)
    
    def data_preprocess(self, volume: torch.Tensor, tvec: torch.Tensor, molli_t1_ref: torch.Tensor, pmap: torch.Tensor) :
        max_signal_readouts = np.max(abs(volume), axis=1)
        mask_t1_ref = (molli_t1_ref < 3000) & (molli_t1_ref > 20)
        mask_pmap =  (np.max(pmap,axis=-1) < 3000) & (np.max(pmap,axis=-1) > 20)
        non_nan_mask = ~np.isnan(max_signal_readouts)
        mask_signal_readouts = (max_signal_readouts> 25 ) & (max_signal_readouts< 600)
        
        masking_intersection = mask_signal_readouts & non_nan_mask & mask_pmap & mask_t1_ref
        
        volume = volume[masking_intersection] / self.signal_normalization_const
        molli_t1_ref = molli_t1_ref[masking_intersection] / self.tvec_normalization_const
        tvec /= self.tvec_normalization_const
        
        return volume, molli_t1_ref, tvec
    
    def flatten_volume(self, volume: np.ndarray, pmap: np.ndarray, molli_t1_ref: np.ndarray):
        assert min(volume.shape) == volume.shape[-1], "First two axis in volume must be Height/Width" 
        assert min(pmap.shape) == pmap.shape[-1], "First two axis in pmap must be Height/Width" 
        assert pmap.shape[-1] == 3 # (c, k ,T1*) parameters
        h,w = volume.shape[:2]
        molli_readouts = volume.shape[-1]

        volume = volume.reshape((h*w, molli_readouts))
        pmap = pmap.reshape((h*w,3))
        molli_t1_ref = molli_t1_ref.reshape((h*w,1))
        return volume, pmap, molli_t1_ref

    def __iter__(self):
        _epoch = self.epoch if not self.validation else 0 #make it consistent across epochs during validation
        rng = np.random.default_rng(_epoch + self.base_seed)
        file_list = self.mat_files_paths.copy()
        rng.shuffle(file_list)
        
        for mat_file in file_list:
            load_mat_file = self.read_loadmat(mat_file)
            tvec, volume = self.sort_molli_by_polarity(**load_mat_file)
            molli_t1_ref = torch.from_numpy(load_mat_file['molli_t1_ref'])
            pmap = torch.from_numpy(load_mat_file["pmap"])

            volume, pmap, molli_t1_ref = self.flatten_volume(volume=volume,pmap=pmap,molli_t1_ref=molli_t1_ref)
            tvec = tvec.squeeze(0)

            volume, tvec, pmap, molli_t1_ref = self.data_preprocess(volume=volume,tvec=tvec,molli_t1_ref=molli_t1_ref,pmap=pmap)
            for i in range(volume.shape[0]):
                yield volume[i], molli_t1_ref[i], pmap[i], tvec
        #
    


    @staticmethod
    def read_loadmat(path: pathlib.Path) -> Dict[str, np.ndarray]:
        """ A simple encapsulation of loadmat and data retrieval. """
        load_data = loadmat(path) 
        #Levenberg - Marquedat Least-Square Fitting 3-parameter estimation for 
        # S(t) = C (1 - kexp(-t/T1*))
        pmap = load_data["pmap_mse"].astype(np.float32)
        volume = load_data["volume"].astype(np.float32)
        tvec = load_data["tvec"].squeeze().astype(np.float32)
        null_index = load_data["null_index"].astype(np.uint8)
        molli_t1_ref = load_data['T1'] # Extracted from T1* in pmap, using T1 = T1* (k-1)
        return dict(volume=volume,
                    tvec=tvec,
                    pmap=pmap,
                    null_index=null_index,
                    molli_t1_ref = molli_t1_ref,
                )
