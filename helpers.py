import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

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

    def linear_schedule_step(self, epoch:int) -> float:
        return self.val_min + (self.val_max - self.val_min) * (epoch / self.steps)

    def schedule_step(self, epoch:int)->float:
        if self.method == 'linear':
            return self.linear_schedule_step(epoch)

    def update(self, epoch):
        self.val  = min( self.val_max, self.schedule_step(epoch+1) )

    def get_val(self):
        return self.val
    

if __name__=="__main__":

    EPOCHS = 11
    BETA_INIT = 1e-8
    BETA_FINA = 5e-2
    betaSchedule = LossWeightScheduler(val_min=BETA_INIT, val_max=BETA_FINA, steps=EPOCHS)

    for epoch in range(EPOCHS):
        print(f"{epoch=}","beta =", betaSchedule.get_val())
        betaSchedule.update(epoch)