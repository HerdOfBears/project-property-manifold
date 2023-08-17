import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import os
import logging


class GRUBlock(nn.Module):
    def __init__(self, d_input, d_model, num_layers=1, dropout=0.0):
        """
        GRU block with batchnorm
        """
        super(GRUBlock, self).__init__()
        self.gru_cell = nn.GRU(
                            d_input, 
                            d_model, 
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True
        )
        self.bn  = nn.BatchNorm1d(d_model)

    def forward(self, x, h0=None):
        _, hlast = self.gru(x, h0)
        out = self.bn(hlast)
        return out


class RNNVae(nn.Module):
    def __init__(self, 
                 d_input, 
                 d_model, 
                 d_latent, 
                 num_layers=2, 
                 dropout=0.0, 
                 use_pp=False,
                 generator=None):
        """
        RNN VAE with GRU encoder and decoder
        """
        super(RNNVae, self).__init__()
        self.name = f"rnn-vae-dLatent{d_latent}-dModel{d_model}-numGRUlayers{num_layers}"
        self.d_input  = d_input
        self.d_model  = d_model
        self.d_latent = d_latent
        self.num_layers = num_layers # (for rnn cells)
        self.dropout = dropout
        self.generator = generator
        
        self.padding_idx = 0
        logging.warning(f"padding idx is assumed to be {self.padding_idx} and is ignored in loss function")        
        
        self.use_pp = use_pp # boolean whether to use property predictor
        self.output_losses = True # to work with training_loop() in train.py

        self.embd = nn.Embedding(d_input, d_model)
        self.dec_embd = nn.Embedding(d_input, d_model)

        # encoder
        # self.gru_enc = GRUBlock(d_model, d_model, num_layers, dropout)
        self.gru_enc = nn.GRU(
                            d_model, 
                            d_model, 
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True
        )
        
        self.ln_gru_enc = nn.LayerNorm(d_model)
        
        # full connected layer leading to mu and logvar
        self.fc = nn.Linear(d_model, 2*d_latent)
        
        # property predictor
        if use_pp:
            self.fc_pp = nn.Sequential(
                nn.Linear(d_latent, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        # decoder
        self.fc_invproj = nn.Linear(d_latent, d_model)
        # self.gru_dec = GRUBlock(d_model, d_model, num_layers, dropout)
        self.gru_dec = nn.GRU(
                            d_model, 
                            d_model, 
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True
        )
        self.ln_gru_dec = nn.LayerNorm(d_model)
        self.fc_out  = nn.Linear(d_model, d_input)

    def encode(self, idx, h0=None):
        """
        Encode a sequence of one-hot vectors into a latent vector
        """

        if h0 is None:
            h0 = torch.randn(self.num_layers, 
                             idx.size(0), 
                             self.d_model, 
                             generator=self.generator
            ).to(idx.device)
        _, hlast = self.gru_enc(idx, h0) # (batch_size, T, d_model) -> (batch_size, T, d_model), (n_layers, batch_size, d_model)
        
        # use the last hidden state from the last GRU layer to predict mu and logvar
        mu_logvar = self.fc(
            self.ln_gru_enc(hlast[-1])
        ) # (batch_size, d_model) -> (batch_size, 2*d_latent)
        
        mu     = mu_logvar[:, :self.d_latent ]
        logvar = mu_logvar[:,  self.d_latent:]
        return mu, logvar

    def decode(self, x, z=None):
        """
        Decode a latent vector into a sequence of indices
        x is usually start token
        z is latent vector
        """
        if z is None:
            z = torch.randn(self.num_layers, 
                             x.size(0), 
                             self.d_model, 
                             generator=self.generator
            ).to(x.device)

        outputs, hlast = self.gru_dec(x, z) # (bsz, T-1, d_model), (n_layers, bsz, d_latent) -> (bsz, T-1, d_model)
        return outputs, hlast

    def reparameterize(self, mu, logvar):
        """
        Reparameterize the latent vector
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, prop_targets):
        """
        Forward pass of the RNN VAE
        """

        # embed the sequence of indices 
        tokens = self.embd(x) # (bsz, T) -> (bsz, T, d_model) | note bsz == batch_size

        mu, logvar = self.encode(tokens) # (bsz, T, d_model) -> (bsz, d_latent)

        z = self.reparameterize(mu, logvar) # (bsz, d_latent)
        context = self.fc_invproj(z)        # (bsz, d_latent) -> (bsz, d_model)
        # z = F.relu(z)                       # (bsz, d_model)

        # embed the sequence of indices (for decoder) | note T-1 b/c we want to predict the next token
        idx = self.dec_embd(x[:,:-1]) # (bsz, T-1) -> (bsz, T-1, d_model)

        context = context.unsqueeze(1).repeat(1, idx.size(1), 1) # (bsz, d_model) -> (bsz, T-1, d_model)
        outputs, hlast = self.decode(context) # (bsz, T-1, d_model), () -> (bsz, T-1, d_model), (n_layers, T-1, d_model)
        logits = self.fc_out(
            self.ln_gru_dec(outputs)
        )              # (bsz, T-1, d_model) -> (batch_size, T-1, d_input)        
        

        # property predictor
        if self.use_pp:
            prop_pred = self.fc_pp(z) # (bsz, d_latent) -> (bsz, 1)
            loss_pp = F.mse_loss(
                prop_pred, 
                prop_targets.unsqueeze(1),
                reduction='mean'
            )
        else:
            prop_pred = None
            loss_pp = torch.tensor(0.0)
        BCE, KLD, _ = self.loss(x, logits, mu, logvar)

        return logits, prop_pred, mu, logvar, BCE, KLD, loss_pp

    def loss(self, x, x_recon, mu, logvar, beta=1.0):
        """
        Compute the loss of the RNN VAE
        """
        BCE = F.cross_entropy(
            x_recon.view(-1, self.d_input), # (bsz * (T-1), d_input) 
            x[:,1:].reshape(-1), # (bsz * (T-1)) 
            reduction='mean',
            ignore_index=self.padding_idx
        )

        KLD = -0.5 * beta* torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE, KLD, torch.tensor(0.0)