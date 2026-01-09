import numpy as np
import torch
import logging
import yaml
import random
import torch.nn as nn
import torch
from molli_pinn_node_lstm.utils import load_config, dataloader
from molli_pinn_node_lstm.training import Trainer, TrainerConfig, PINNLoss
from molli_pinn_node_lstm.node_lstm import MOLLINeuralODELSTM
from torch.utils.tensorboard import SummaryWriter


#TODO make sure seeds proliferate or we re use this set rng state somewhere else
# tvec_norm_const renaming -> move in TRainer to the TRainerCFG ! will require some refactoring ofc.
# training docstrings
# argparse for testing ! baseline, continue training pretty much
# testing code
# if testing works, time to train models
# test with the trained models
# If it works, then finally time to write the readme and docs and deploy
# before then, run ruff for linting !
# find a new job ! (Hopefulyl something nice is watining for me :( )

LOGGER = logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")



def set_rng_state_seed(seed: int, deterministic : bool = True, strict: bool = False) -> None:
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



def configure_optimizer(model, lr: float, gamma: float, weight_decay: float):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)
    return optim, scheduler

def main(rng_seed : int = 1234) -> None:
    train_molli_dataset = dataloader.MOLLIDataset(folder_path=r"C:\Users\tiago\Desktop\PINN_Transformer\Imphys_Clinical_Data\Training_low", shuffle=True, batch_size = 1024)
    train_dataloader = torch.utils.data.DataLoader(train_molli_dataset,batch_size=None, pin_memory=True)
    import pathlib
    save_dir = pathlib.Path(r"C:\Users\tiago\Desktop\pinn-node-cardiac-t1mapping\src\molli_pinn_node_lstm\runs\test")
    save_log_dir = (save_dir / "tensorboard" )
    save_log_dir.mkdir(exist_ok=True)
    tensorboard_logger = SummaryWriter(log_dir = save_log_dir)

    save_ckpt_dir = (save_dir / "ckpt" )
    save_ckpt_dir.mkdir(exist_ok=True)

    val_molli_dataset = dataloader.MOLLIDataset(folder_path=r"C:\Users\tiago\Desktop\PINN_Transformer\Imphys_Clinical_Data\Validation_low")
    val_dataloader = torch.utils.data.DataLoader(val_molli_dataset,batch_size=None, pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    node_lstm = MOLLINeuralODELSTM()
    optimizer, scheduler = configure_optimizer(model=node_lstm, lr=1e-4, gamma = 0.95, weight_decay= 0.01)
    trainer_cfg = TrainerConfig(epochs = 10,num_acquisitions= 5, max_acquisitions= 11)
    pinn_loss = PINNLoss(tvec_normalization_const=1000, signal_normalization_const= 272)
    trainer = Trainer(trainer_cfg=trainer_cfg,model=node_lstm,optimizer=optimizer,scheduler=scheduler,tensorboard_logger=tensorboard_logger,
                      molli_loss=pinn_loss,tvec_normalization_const=1000,signal_normalization_const=252,
                      save_ckpt_dir=save_ckpt_dir, device=device)
    trainer.fit(training_set=train_dataloader, validation_set=val_dataloader)

if __name__ == "__main__":
    main()
    