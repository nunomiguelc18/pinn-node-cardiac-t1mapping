from __future__ import annotations

import torch.nn as nn
from typing import Dict
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW
import numpy as np
import random
from molli_pinn_node_lstm.utils import molli_signal_model
from torch.utils.data import DataLoader

class Trainer(nn.Module):
    def __init__(self, model: nn.Module, 
                 optimizer: AdamW,
                 scheduler: ExponentialLR, 
                 training_set: DataLoader,
                 validation_set: DataLoader,
                 num_acquisitions: int, 
                 max_acquisitions: int, 
                 epochs: int, 
                 interpolate_readouts: bool, 
                 p_interpolation: float,
                 p_full_seq: float, 
                 validation: bool, 
                 tvec_normalization_const: float, 
                 signal_normalization_const: float):
        super().__init__()
        if 0 < num_acquisitions <= max_acquisitions:
            raise ValueError("Number of points must lie within the set of integers {1,2,..max_num_points}.")
        if epochs <= 0:
            raise ValueError("Number of Epochs must be > 0.")
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_set = training_set
        self.validaiton_set = validation_set
        self.num_points = int(num_acquisitions)
        self.max_acquisitions = int(max_acquisitions)
        self.p_full_seq = np.clip(p_full_seq,0,1)
        self.epochs = int(epochs)
        self.interpolate_readouts = bool(interpolate_readouts)
        self.p_interpolation = p_interpolation if interpolate_readouts else 0.0
        self.validation = bool(validation)
        self.tvec_normalization_const = float(tvec_normalization_const)
        self.signal_normalization_const = float(signal_normalization_const)

    def run_training(self):
        pass

    def run_validation(self):
        pass
        


    # def _interpolate_readouts(tvec, bloch_parameters):
    #     time_jitter = torch.empty(tvec.shape, device=tvec.device).uniform_(-1,1)*0.2 #Add some random jitter to tvec
    #     tvec_linspace = torch.sort(abs(tvec + time_jitter)).values #avoid negative time jitters
    #     bloch_based_signals = bloch_equation(tvec_linspace*1000, bloch_parameters[:,0].unsqueeze(-1),bloch_parameters[:,1].unsqueeze(-1), bloch_parameters[:,2].unsqueeze(-1))
    #     bloch_based_signals /= 272
    #     return tvec_linspace, bloch_based_signals



    # def generate_random_mask(num_points, epoch):
    #     if random.random() <= 0.1:
    #         return torch.ones(11, dtype=bool)
        
    #     mask = torch.zeros(11, dtype=bool)
    #     fix_first_readout = np.random.choice(range(3),1)
    #     remaining_indices = np.random.choice(range(3,8), num_points-2, replace=False)
    #     last_readout = np.random.choice(range(8,11), 1, replace=False)
    #     random_indices=np.sort(np.concatenate((fix_first_readout,remaining_indices, last_readout)))
    #     mask[random_indices] = True
    #     return mask