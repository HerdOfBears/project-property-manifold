import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import os
import logging

# a variational autoencoder shell for SMILES:
# - decode function
# - encode function
# - repararmeterization function
class VAESkeleton(nn.Module):
    def __init__(self,
                 encoder,
                 decoder) -> None:
        super(VAESkeleton, self).__init__()

        # ensure encoder and decoder are registered as modules
        # and backprop will update their parameters
        self.module_list = nn.ModuleList([encoder, decoder]) 

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x, last_non_pad_idxs=None) -> tuple[torch.Tensor, torch.Tensor]:
        """ 
        x:      (batch_size, T, n_embd)
        return: (batch_size, T, latent dim)
        """ 
        if isinstance(x, torch.Tensor) is False:
            raise TypeError("x must be a torch.Tensor")
        
        logging.info("VAESkeleton encode")
        
        output = self.encoder(x, last_non_pad_idxs) # (batch_size, latent_dim*2)
        latent_dim = output.shape[-1] // 2
        logging.info(f"{latent_dim=}")
        # mu, logvar = output[:, :latent_dim], output[:, latent_dim:]
        mu, logvar = output[:,:latent_dim], output[:,latent_dim:]
        return mu, logvar

    def decode(self, z:torch.Tensor) -> torch.Tensor:
        """ 
        z: (batch_size, latent_dim)
        return: (batch_size, input_dim)
        """ 
        if isinstance(z, torch.Tensor) is False:
            raise TypeError("z must be a torch.Tensor")
        logging.info(f"{z.shape=}")
        return self.decoder(z)
    
    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
        """ 
        VAE repararmeterization 'trick'
        mu:     (batch_size, latent_dim)
        logvar: (batch_size, latent_dim)
        return: (batch_size, latent_dim)
        """ 
        if isinstance(mu, torch.Tensor) is False:
            raise TypeError("mu must be a torch.Tensor")
        if isinstance(logvar, torch.Tensor) is False:
            raise TypeError("logvar must be a torch.Tensor")
        logging.info(f"{mu.shape=} | {logvar.shape=}")
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x:torch.Tensor, last_non_pad_idxs:torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
        forward pass through VAE. 
        Returns 3 things: decoded x, mu, logvar
        
        input:
            x:  array of int indices.
                (batch_size, input_dim)
            last_non_pad_idxs:  array of indices of last non-padding idxs
                                (batch_size, 1)
        return: 
            decoded x:  (batch_size,  input_dim)
            mu:         (batch_size, latent_dim)
            logvar:     (batch_size, latent_dim)
        """
        logging.info(f"VAESkeleton fwd")
        logging.info(f"{x.shape=}")
        logging.info(f"{last_non_pad_idxs.shape=}")
        if isinstance(x, torch.Tensor) is False:
            raise TypeError("x must be a torch.Tensor")
        
        if last_non_pad_idxs is not None:
            mu, logvar = self.encode(x, last_non_pad_idxs)
        else:
            mu, logvar = self.encode(x)
        
        z = self.reparameterize(mu, logvar)
        logging.info(f"in fwd {z.shape}")
        return self.decode(z), mu, logvar


class Encoder(nn.Module):
    """
    encoder base class. 
    Performs encoding
    I.e. takes array of ints, embeds them, and encodes them into latent space.
    """
    def __init__(self,
                        n_embd:int,
                    latent_dim:int=10,
                 onehot_embd=True,
                 padding_index:int=0) -> None:
        super().__init__()
        
        self.n_embd = n_embd
        self.padd_idx = padding_index

        logging.info(f"{latent_dim=}")
        latent_dim2 = latent_dim * 2

        self.rnn_cell = nn.RNN(
            n_embd, 
            latent_dim2,
            batch_first=True
        )
        self.encoder_net = nn.Sequential(
            self.rnn_cell
        )
        # self.encoder_net = nn.Sequential(
        #     nn.Linear(n_embd, 64),
        #     nn.ReLU(),
        #     nn.LazyLinear(latent_dim2)
        # )
        
    def forward(self, x:torch.Tensor, last_non_pad_idxs:torch.Tensor) -> torch.Tensor:

        # embed array of ints
        # (batch_size, T, n_embd)
        logging.info(f"Encoder fwd")

        logging.info(f"{x.shape=}")
        # encode embedded array 
        # (batch_size, T, n_embd) -> (batch_size, latent_dim)
        # output = self.encoder_net(x)
        hidden_states, last_hidden_state = self.encoder_net(x)
        logging.info(f"{last_hidden_state.shape=} | {hidden_states.shape=}")
        logging.info(f"{last_non_pad_idxs.shape=}")
        # use last non pad idxs to grab last hidden state before padding
        valid_hidden_states = hidden_states[torch.arange(0,x.shape[0]).long(), last_non_pad_idxs]
        logging.info(f"{valid_hidden_states.shape=}")

        return valid_hidden_states
        

# a decoder NOT DONE
class Decoder(nn.Module):
    def __init__(self,
                 vocab_size:int,
                 latent_dim:int,
                 num_layers:int=1,
                 batch_first:bool=True) -> None:
        super().__init__()

        self.rnn = nn.RNN(vocab_size,
                          latent_dim,
                          num_layers=num_layers,
                          batch_first=batch_first)
    
    def forward(self, 
                x:torch.Tensor, 
                context:torch.Tensor=None, 
                max_length:int=111) -> torch.Tensor:
        """
        input:
            x:          (batch_size, 1)
            context:    (batch_size, latent_dim)
            max_length: int
        return:
            decoded:    (batch_size, max_length, vocab_size)
        """
        raise NotImplementedError


# a supervised learning shell taking the VAE latent dimension as input
# and outputting a single value
# - forward function
class FeedForward(nn.Module):
    def __init__(self,
                 latent_dim:int,
                 hidden_dim:int,
                 output_dim:int) -> None:
        """
        simple feed-forward network for predicting properties 
        from latent space
        """
        super(FeedForward, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, mu:torch.Tensor) -> torch.Tensor:
        """ 
        forward pass through simple feed-forward network
        Returns estimated label
        
        input:
            mu:     the latent vector of mean values from the VAE.
                    (batch_size, latent_dim)
        return: 
            output: the estimated label
                    (batch_size, output_dim)
        """
        if isinstance(mu, torch.Tensor) is False:
            raise TypeError("z must be a torch.Tensor")

        output = F.relu(self.fc1(mu))
        output = self.fc2(output)
        return output
    

# a property-constrained variational autoencoder (pc-VAE) shell
# - init function taking VAE and FeedForward objects
# - forward function passing through VAE and using mu as input to FeedForward
class pcVAE(nn.Module):
    """ 
    a property-constrained variational autoencoder (pc-VAE) shell
    this is a VAE with a feed-forward network
    taking the VAE latent space (the means) as input
    and outputting 
        - the reconstructed molecule, 
        - estimated labels corresponding to the encoded vectors
        - mu, logvar
    """
    def __init__(self,
                 vae:VAESkeleton,
                 ff:FeedForward) -> None:

        super(pcVAE, self).__init__()

        # ensure vae and ff are registered as modules
        # so that backprop will update their parameters
        self.module_list = nn.ModuleList([vae, ff]) 

        self.vae = vae
        self.ff = ff

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
        forward pass through pc-VAE. 
        Returns 4 things: decoded x, estimated_label, mu, logvar
        
        input:
            x:  array of int indices.
                (batch_size, input_dim)
        return: 
            decoded x:      (batch_size, input_dim)
            estimated_label:(batch_size, output_dim)
            mu:             (batch_size, latent_dim)
            logvar:         (batch_size, latent_dim)
        """
        if isinstance(x, torch.Tensor) is False:
            raise TypeError("x must be a torch.Tensor")
        
        logging.info(f"pcVAE fwd")

        decoded_x, mu, logvar = self.vae(x)
        estimated_label = self.ff(mu)

        return decoded_x, estimated_label, mu, logvar


class Test(nn.Module):
    def __init__(self,
                 alphabet_size:int,
                 n_embd:int,
                 one_hot_encoding:bool=False) -> None:
        super().__init__()
        self.latent_dim = 4
        self.one_hot_encoding = one_hot_encoding
        self.padding_idx = 0 

        if not one_hot_encoding:
            self.embedding = nn.Embedding(alphabet_size, n_embd)
        else:
            n_embd = alphabet_size

        encoder = Encoder(n_embd, self.latent_dim) # does embedding, and 2*latent_dim inside
        decoder = nn.Linear(self.latent_dim, alphabet_size)

        vae = VAESkeleton(encoder, decoder)
        ff = FeedForward(self.latent_dim, 4, 1)
        self.vae = vae
        self.ff  = ff
        
        # super().__init__(vae, ff)
    
    def forward(self, 
                idx:torch.Tensor, 
                targets:torch.Tensor=None
                )->tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # idx: (batch_size, T)
        # targets: (batch_size, 1)

        # grab last non pad idxs
        last_non_pad_idxs = torch.sum(idx != self.padding_idx, dim=1) - 1
        logging.info(f"Test fwd")
        logging.info(f"{last_non_pad_idxs=}")

        if self.one_hot_encoding:
            x = F.one_hot(idx, self.alphabet_size).float() # (batch_size, T, alphabet_size)
        else:
            x = self.embedding(idx) # (batch_size, T, n_embd)
        logging.info(f"after embd {x.shape=}")
        decoded_x, mu, logvar = self.vae(x, last_non_pad_idxs) # (batch_size, T, alphabet_size), (batch_size, latent_dim), (batch_size, latent_dim)
        estimated_label = self.ff(mu) # (batch_size, 1)

        loss = None
        if targets is not None:
            logging.info(f"{decoded_x.shape=} | {idx.shape=} | {targets.shape=} | {estimated_label.shape=} ")
            loss_recon = F.cross_entropy(decoded_x, idx[:,[1]].view(-1), ignore_index=self.padding_idx)
            loss_KL    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss_property = F.mse_loss(estimated_label.view(-1), targets )
            logging.warning(f"{loss_recon=} | {loss_KL=} | {loss_property=}")
            loss = loss_recon + loss_KL + loss_property
        
        return decoded_x, estimated_label, mu, logvar, loss


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    test = Test(37,
                9)
    m1 = torch.zeros(1,3)
    m2 = torch.zeros(1,3)
    # print(test.reparameterize(m1, m2))
    print("successful construction")
    tst_input = torch.tensor([[1,6,5,3,0,0,0,0]])
    print(f"{tst_input.shape=}")
    reconstructed_x, yhat, _, _, loss = test(tst_input)
    print(reconstructed_x)
    print("==================")
    print(yhat)
    print("==================")
    print(loss)
    # print(test(torch.tensor([[0,1,2,3]])))