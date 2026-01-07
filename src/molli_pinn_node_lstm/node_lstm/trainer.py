from __future__ import annotations
import logging

LOGGER = logging.getLogger(__name__)
#Collate masks for validation so they are fixed for MC dropring !! we cna remove that, w eju colate
# and assume a for loop in validation


# Is this trainer fine? keep nn.Module? how to save checkpoints? of optimzer and scheduler? stand what other methods to keep? it's fine to import like this ? the dataloader?
# Do we save this Trainer module? maybe move it to training folder?
# We need Loggers as well? 
# What to use as buffers? anything? cna we keep the past training info? so that we can continue? 
# Should we pass the training-set and vlaidaiton set like this? is it ok?
# Implement the rest
# Implement the loggers using tensorboard

import torch.nn as nn
from typing import Dict, List, Union
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW
import numpy as np
import random
from molli_pinn_node_lstm.utils import molli_signal_model
from torch.utils.data import DataLoader
from collections import defaultdict
from molli_pinn_node_lstm.utils import molli_signal_model
from molli_pinn_node_lstm.training import SignalRecoveryLoss

class Trainer(nn.Module):
    def __init__(self, model: nn.Module, 
                 optimizer: AdamW,
                 scheduler: ExponentialLR,
                 molli_loss: SignalRecoveryLoss, 
                 training_set: DataLoader,
                 validation_set: DataLoader,
                 num_acquisitions: int, 
                 max_acquisitions: int, 
                 epochs: int, 
                 interpolate_readouts: bool, 
                 p_interpolation: float,
                 p_full_seq: float, 
                 validation: bool, 
                 val_MC_samples: int,
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
        self.molli_loss = molli_loss
        self.training_set = training_set
        self.validaiton_set = validation_set
        self.num_points = int(num_acquisitions)
        self.max_acquisitions = int(max_acquisitions)
        self.p_full_seq = np.clip(p_full_seq,0,1)
        self.epochs = int(epochs)
        self.interpolate_readouts = bool(interpolate_readouts)
        self.p_interpolation = p_interpolation if interpolate_readouts else 0.0
        self.validation = bool(validation)
        self.val_MC_samples = int(val_MC_samples)
        self.tvec_normalization_const = float(tvec_normalization_const)
        self.signal_normalization_const = float(signal_normalization_const)

        self.train_epoch_losses = defaultdict(list)
        self.val_epoch_losses = defaultdict(list)
        self.best_valid_loss = float('inf')



    def run_training(self):
        logging.info('Training Epoch Started')
        for epoch in range(self.epochs):
            self.model.train()
            tmp_train_loss = defaultdict(list)
            logging.info(f"Training on Epoch {epoch +1 :4d}/{self.epochs}")
            for _, batch in self.training_set:
                self.optimizer.zero_grad()
                batch = self._to_device(**batch)
                
                if random.random() <= 0.1:
                    tmp_acquisitions = self._interpolate_molli_readouts(**batch)
                    batch.update(tmp_acquisitions)

                mask = self._generate_random_mask(MC_sampling=False)
                tmp_vol, tmp_tvec = batch['volume'][:, mask], batch['tvec'][mask]

                pmap_hat = self.model(volume = tmp_vol, tvec = tmp_tvec)
                
                train_loss = self.molli_loss.compute_loss(pmap_ref = batch['pmap'], T1_ref = batch['molli_t1_ref'], pmap_hat = pmap_hat)
                
                total_loss = train_loss['total_loss']
                total_loss.backward()
                self.optimizer.step()

                #self.update_loss_stats(ref_dict = tmp_train_loss, updates_dict=train_loss)

    @torch.no_grad()
    def run_validation(self):
        logging.info(f"Running Validation")
        self.model.eval()
        tmp_val_loss= defaultdict(list)
        for _, batch in self.validaiton_set:
            batch = self._to_device(**batch)
            

            random_mask = self._generate_random_mask(MC_sampling=True)
            
            # Run Monte-Carlo sampling 
            for mask in random_mask:
                tmp_vol, tmp_tvec = batch['volume'][:, mask], batch['tvec'][random_mask]
                pmap_hat = self.model(volume = tmp_vol, tvec = tmp_tvec)
                val_loss = self.molli_loss.compute_loss(pmap_ref = batch['pmap'], T1_ref = batch['molli_t1_ref'], pmap_hat = pmap_hat)
                


    def _interpolate_molli_readouts(self, tvec: torch.Tensor, pmap: torch.Tensor, **kwargs):
        time_jitter = torch.empty(tvec.shape, device=tvec.device).uniform_(-1,1)*0.2 #Add some random jitter to tvec
        tvec_grid = torch.sort(torch.abs(tvec + time_jitter)) #avoid negative time jitters
        denorm_tvec_grid = tvec_grid * self.tvec_normalization_const
        interp_volume = molli_signal_model(t=denorm_tvec_grid, **pmap)
        interp_volume /= self.signal_normalization_const
        return {'volume' : interp_volume, 'tvec' : tvec_grid}

    def _generate_random_mask(self, MC_sampling = False):
        if not MC_sampling:
            return self._random_sampling()
        stacked_masks = []
        for _ in range(self.val_MC_samples):
            mask = self._random_sampling()
            stacked_masks.append(mask)

        return stacked_masks

    def _random_sampling(self):
        if random.random() <= 0.1:
            return torch.ones(11, dtype=bool)
        
        mask = torch.zeros(11, dtype=bool)
        fix_first_readout = np.random.choice(range(3),1)
        remaining_indices = np.random.choice(range(3,8), self.num_points-2, replace=False)
        last_readout = np.random.choice(range(8,11), 1, replace=False)
        random_indices=np.sort(np.concatenate((fix_first_readout,remaining_indices, last_readout)))
        mask[random_indices] = True
        return mask
    

    
    @staticmethod
    def _to_device(volume: torch.Tensor, molli_t1_ref: torch.Tensor, pmap: torch.Tensor, tvec: torch.Tensor, device: str = 'cpu'):
        return dict(volume=volume.to(device),tvec=tvec.to(device),molli_t1_ref=molli_t1_ref.to(device),
        pmap=pmap.to(device))

    # @staticmethod    
    # def update_loss_stats(ref_dict: Union[defaultdict[List], Dict[str, torch.Tensor]], updates_dict: Dict[str, torch.Tensor])
    #     for key,value in updates_dict.items():
    #         for _ in value:
    #             ref_dict[key].append(value.pop().item())


# def train(
#     input_model, optimizer, device, training_data, num_signal_points, lambda_parameter,epoch):
#     epoch_loss = 0
#     epoch_loss1 = 0
#     epoch_loss2 = 0
#     epoch_loss3 = 0
#     input_model.train()
#     Finished = False
#     num_batches = 0
#     logging.info('Training Epoch Started')
#     while not Finished:
#         iter_file, Finished, _, is_Native = next(training_data)
#         for _, batch in enumerate(iter_file):
#             optimizer.zero_grad()
#             tvec_readouts = batch[0][:,11:].to(device)
#             signal_readouts = batch[0][:,:11].to(device)
#             bloch_parameters = batch[2].to(device)
#             T1_offline = batch[1].float().to(device)

#             if random.random() <= 0.1:
#                 tvec_readouts, signal_readouts = interpolate_readouts(tvec_readouts,bloch_parameters)
    
#             random_mask = generate_random_mask(num_signal_points, epoch)
    

            
#             estimated_bloch_parameters = input_model(signal_readouts[:,random_mask].float(),tvec_readouts[:,random_mask].float())
#             loss, loss1, loss2, loss3 = custom_loss(estimated_bloch_parameters,T1_offline,bloch_parameters,tvec_readouts[0], is_Native)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#             epoch_loss1 += loss1.item()
#             epoch_loss2 += loss2.item()
#             epoch_loss3 += loss3.item()
#             num_batches += 1
#     avg_loss = epoch_loss / num_batches
#     avg_loss1 = epoch_loss1 / num_batches
#     avg_loss2 = epoch_loss2 / num_batches
#     avg_loss3 = epoch_loss3 / num_batches
#     return avg_loss, avg_loss1, avg_loss2, avg_loss3

# def evaluate(input_model, device, validation_data, num_signal_points, lambda_parameter,epoch):
#     epoch_loss = 0
#     Finished = False
#     num_batches = 0
#     input_model.eval()
#     logging.info('Validation Started')
#     #fixed_mask = np.ones(11,dtype=bool)
#     # fixed_mask = np.zeros(11,dtype=bool)
#     # if num_signal_points == 5:
#     #     fixed_mask[[1,3,5,6,8]] = True
#     # elif num_signal_points == 4:
#     #     fixed_mask[[1,3,6,8]] = True
#     # elif num_signal_points == 3:
#     #     fixed_mask[[1,4,8]] = True
#     fixed_mask = generate_random_mask(num_signal_points, epoch)
#     with torch.no_grad():
#         while not Finished:
#             iter_file, Finished, LL, is_Native = next(validation_data)
#             for _, batch in enumerate(iter_file):
#                 tvec_readouts = batch[0][:,11:].to(device)
#                 signal_readouts = batch[0][:,:11].to(device)
#                 T1_offline = batch[1].float().to(device)
#                 bloch_parameters = batch[2].to(device)

#                 estimated_bloch_parameters = input_model(signal_readouts[:,fixed_mask].float(),tvec_readouts[:,fixed_mask].float())
#                 loss, _, _, _ = custom_loss(estimated_bloch_parameters,T1_offline,bloch_parameters,tvec_readouts[0], is_Native)

#                 epoch_loss += loss.item()
#                 num_batches += 1
#     avg_loss = epoch_loss / num_batches
#     return avg_loss


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