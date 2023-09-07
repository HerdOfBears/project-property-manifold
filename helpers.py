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
        save_to = path + f"{model.name}-epoch{epoch}{save_suffix}.pt"
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
                    optim:torch.optim.Optimizer,
                    annealer:LossWeightScheduler|None)->tuple[int,float,float]:
    """
    takes path to .pt file, a model and optimizer obj, and loads the checkpoint
    optionally takes an annealer object.
    loads objects in-place
    returns epoch, training_loss, validation_loss
    """

    # check file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file {path} not found")

    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint["model_state_dict"])
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
    os.mkdir(model_dir)

    chkpt_dir = os.path.join(model_dir, "checkpoints") + "/"
    os.mkdir(chkpt_dir)

    return model_dir, chkpt_dir

if __name__=="__main__":

    EPOCHS = 11
    BETA_INIT = 1e-8
    BETA_FINA = 5e-2
    betaSchedule = LossWeightScheduler(val_min=BETA_INIT, val_max=BETA_FINA, steps=EPOCHS)

    for epoch in range(EPOCHS):
        print(f"{epoch=}","beta =", betaSchedule.get_val())
        betaSchedule.update(epoch)