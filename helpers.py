import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import os

# A obj handling scheduling of loss term weights
# such as the beta term weighting the KL loss
# and/or delta term weighting the property loss
class LossWeightScheduler():
    def __init__(self, 
                 val_min:float, 
                 val_max:float, 
                 steps:int,
                 method:str='linear'):
        if method not in ['linear']:
            raise ValueError(f'Invalid {method=} for LossWeightScheduler, only linear is supported.')
        logging.warning(f"{steps=} assumed the same for beta and delta")

        self.method  = method
        self.val_min = val_min
        self.val_max = val_max
        self.steps   = steps-1 # ensure we reach val_max during last epoch
        self.val     = val_min

    def state_dict(self):
        state_dict = {
            "method": self.method,
            "val_min": self.val_min,
            "val_max": self.val_max,
            "steps": self.steps,
            "val": self.val
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.method  = state_dict["method"]
        self.val_min = state_dict["val_min"]
        self.val_max = state_dict["val_max"]
        self.steps   = state_dict["steps"]
        self.val     = state_dict["val"]

    def linear_schedule_step(self, epoch:int) -> float:
        return self.val_min + (self.val_max - self.val_min) * (epoch / self.steps)

    def schedule_step(self, epoch:int)->float:
        if self.method == 'linear':
            return self.linear_schedule_step(epoch)

    def update(self, epoch):
        self.val  = min( self.val_max, self.schedule_step(epoch+1) )

    def get_val(self):
        return self.val
    

def checkpoint_model(model:nn.Module,
                     optim:torch.optim.Optimizer,
                     annealer:LossWeightScheduler|None,
                     epoch:int,
                     training_loss:float,
                     validation_loss:float,
                     path:str,
                     save_every:int=10,
                     save_suffix:str=""):
    if epoch % save_every == 0:
        # save_to = path + f"{model.name}-epoch{epoch}{save_suffix}.pt"
        save_to = path + f"epoch-{epoch}.pt"
        logging.info(f"saving checkpoint at epoch {epoch} to {save_to}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "annealer_state_dict": annealer.state_dict() if annealer is not None else None,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        }, save_to)

def load_checkpoint(path:str,
                    model:nn.Module,
                    optim:torch.optim.Optimizer|None=None,
                    annealer:LossWeightScheduler|None=None,
                    device:str="cpu")->tuple[int,float,float]:
    """
    loads a checkpoint file and returns the epoch, training_loss, validation_loss
    loads objects in-place

    inputs:
        path : path to .pt file, 
        model: the pytorch module for the model architecture
        optimizer: object of torch.optim.Optimizer or None (default)
        annealer:  object of LossWeightScheduler   or None (default)
        device  :  str, either "cpu" or "gpu"
    outputs
        epoch, 
        training_loss, 
        validation_loss
    """

    # check file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file {path} not found")

    if device=="cpu":
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optim is not None:
        optim.load_state_dict(checkpoint["optim_state_dict"])
    if annealer is not None:
        annealer.load_state_dict(checkpoint["annealer_state_dict"])
    
    epoch = checkpoint["epoch"]
    training_loss   = checkpoint[   "training_loss"]
    validation_loss = checkpoint[ "validation_loss"]
    return epoch, training_loss, validation_loss


def make_save_dir(save_dir:str, model_name:str):
    """
    makes directories for saving to. It will look like this:
    save_dir/
    --model_name/
    ----model_args.pkl
    ----losses.pkl 
    ----checkpoints/
    inputs:
        save_dir: str, path to directory where dir tree will be created
        model_name: str, name of model
    
    outputs:
        model_dir: str, path to directory where model args + losses will be saved
        chkpt_dir: str, path to directory where checkpoints will be saved
    """
    model_dir = os.path.join(save_dir, model_name) + "/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir) # will create intermediate dirs if needed

    chkpt_dir = os.path.join(model_dir, "checkpoints") + "/"
    if not os.path.exists(chkpt_dir):
        os.mkdir(chkpt_dir)

    return model_dir, chkpt_dir

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, nn.Embedding):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.RNN):
        torch.nn.init.xavier_uniform_(m.weight_ih_l0)
        torch.nn.init.xavier_uniform_(m.weight_hh_l0)
        m.bias_ih_l0.data.zero_()
        m.bias_hh_l0.data.zero_()
    elif type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
            elif "bias" in param:
                m._parameters[param].data.zero_()
    elif type(m) == nn.Conv1d:
        for param in m._parameters:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
            elif "bias" in param:
                m._parameters[param].data.zero_()


if __name__=="__main__":

    EPOCHS = 11
    BETA_INIT = 1e-8
    BETA_FINA = 5e-2
    betaSchedule = LossWeightScheduler(val_min=BETA_INIT, val_max=BETA_FINA, steps=EPOCHS)

    for epoch in range(EPOCHS):
        print(f"{epoch=}","beta =", betaSchedule.get_val())
        betaSchedule.update(epoch)