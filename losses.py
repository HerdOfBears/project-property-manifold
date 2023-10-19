import torch
import torch.nn as nn
import torch.nn.functional as F

import logging


class OracleVAELoss(nn.Module):

    def __init__(self, beta:float=1.0, delta:float=1.0, padding_idx:int=0, use_pp:bool=True, reduction="mean", pp_mode:str="regression"):
        """
        Loss function for the VAE.
        Inputs:
            beta:   weight for the KL loss term
            delta:  weight for the property loss term
            use_pp: boolean whether to use the property predictor
            reduction:  how to reduce the loss, either "mean" or "sum"
            pp_mode:    how to treat the property prediction task, either "regression" or "classification"

        """
        super(OracleVAELoss, self).__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError(f"Invalid {reduction=} for OracleVAELoss, only 'mean' and 'sum' are supported.")
        if pp_mode not in ["regression", "classification"]:
            raise ValueError(f"Invalid {pp_mode=} for OracleVAELoss, only 'regression' and 'classification' are supported.")
        
        self.beta = beta
        self.delta = delta
        self.use_pp = use_pp
        self.reduction = reduction
        self.padding_idx = padding_idx
        self.pp_mode = pp_mode

    def forward(self, 
                logits:torch.Tensor, 
                xtrue:torch.Tensor, 
                mu:torch.Tensor, 
                logvar:torch.Tensor, 
                yhat:torch.Tensor|None=None, 
                ytrue:torch.Tensor|None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        computes the losses for the VAE: 
            reconstruction loss, KL loss, property prediction loss

        inputs:
            logits:     output of the decoder, (batch_size, max_length, alphabet_size)
            xtrue:      true input sequences,  (batch_size, max_length)
            mu:         mean of the latent variable,         (batch_size, d_latent)
            logvar:     log variance of the latent variable, (batch_size, d_latent)
            yhat:       predicted property values,           (batch_size, n_properties)
            ytrue:      true property values,                (batch_size, n_properties)
        
        outputs:
            BCE:        reconstruction loss
            KL:         KL loss
            loss_pp:    property prediction loss OR None
        """
        BCE, KL, loss_pp = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        # compute reconstruction loss
        BCE = F.cross_entropy(
            logits,
            xtrue,
            reduction=self.reduction,
            ignore_index=self.padding_idx
        )

        # compute Kullback-Leibler divergence
        KL = torch.mean(-0.5*torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)

        # compute property prediction loss if existing
        if self.use_pp:
            if self.pp_mode == "regression":
                loss_pp = F.mse_loss(
                    yhat[ ~ytrue.isnan()], 
                    ytrue[~ytrue.isnan()], 
                    reduction=self.reduction)
            else: # classification loss
                loss_pp = F.cross_entropy(yhat, ytrue, reduction=self.reduction)
        
        return BCE, KL, loss_pp