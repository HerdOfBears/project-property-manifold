import torch 
import torch.nn as nn
import torch.nn.functional as F

import logging

from losses import OracleVAELoss

class ConvEncoder(nn.Module):
    def __init__(self,
                 d_input:int,
                 conv_outs:list[int],
                 conv_kernels:list[int],
                 d_model:int,
                 padding:str='same',
                 dropout:float=0.0):
        """ 
        Encoder using CNNs
        Inputs:
            d_input: dimensionality of the input
            conv_outs: list of output channels for each convolutional layer
            conv_kernels: list of kernel sizes for each convolutional layer
            d_model: dimensionality of the output, done by a feedforward layer
        """
        super(ConvEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential()
        for i, (out, kernel) in enumerate(zip(conv_outs, conv_kernels)):
            if i == 0:
                self.conv_layers.append(
                    nn.Conv1d(d_input, out, kernel, padding=padding)
                )
            else:
                self.conv_layers.append(
                    nn.Conv1d(conv_outs[i-1], out, kernel, padding=padding)
                )
            self.conv_layers.append(nn.LeakyReLU()) # FLAG! what pooling should be use? 

        self.ff = nn.Linear(conv_outs[-1], d_model)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.max(x, dim=2)[0]
        x = self.ff(x)
        return x


class RNNBlock(nn.Module):
    def __init__(self,
                 d_input:int,
                 d_hidden:int,
                 d_output:int,
                 n_layers:int=1,
                 dropout:float=0.0,
                 batch_first:bool=False):
        """ 
        Block of RNNs, with LayerNorm, and a linear output layer.
        Inputs:
            d_input: dimensionality of the input
            d_hidden: dimensionality of the hidden state
            d_output: dimensionality of the output, 
                    mediated by a feedforward layer
            n_layers: number of layers in the RNN
            dropout: dropout rate
        """
        super(RNNBlock, self).__init__()
        logging.warning("RNNBlock assumes GRU cells")
        logging.warning("RNNBlock only outputs transformed hidden states")
        self.d_hidden = d_hidden
        self.n_layers = n_layers

        self.rnn = nn.GRU(d_input, 
                          d_hidden, 
                          n_layers, 
                          dropout=dropout, 
                          batch_first=batch_first
        )
        self.ln = nn.LayerNorm(d_hidden)
        self.ff = nn.Linear(d_hidden, d_output)

    def init_hidden(self, batch_size:int, g:torch.Generator)->torch.Tensor:
        return torch.randn(self.n_layers, 
                           batch_size, 
                           self.d_hidden, 
                           generator=g)

    def forward(self, x:torch.Tensor, h:torch.Tensor)->torch.Tensor:
        """outputs are transformed hidden states"""
        x, h = self.rnn(x, h)
        x = self.ln(x)
        x = self.ff(x)
        return x


class GomezBombarelli(nn.Module):
    """ 
    Gomez-Bombarelli et al. (2018)
    https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572

    Encoder uses CNNs, decoder uses RNNs

    Some details are not specified in the paper, so I made some assumptions:
    - The latent space is half the size of the hidden dimension
    - feedforward networks convert between spaces of different dimension
    - The property predictor is a feedforward network with 3 layers of 67 neurons
        "To simply shape the latent space, a smaller perceptron
          of 3 layers of 67 neurons was used for the property predictor"
    - LayerNorm is used in the RNN (this might be very different from the paper)
    - one-hot encoding was used for the input
    - convolutional layers use ReLU activations in Encoder
    - the context vector is used to initialize the hidden state of the decoder
    """
    def __init__(self, 
                 params:dict,
                 loss_fn:nn.Module,
                 generator:torch.Generator=None):
        super(GomezBombarelli, self).__init__()
        
        self.alphabet_size  = params["d_input"]
        self.d_input        = params["d_input"] 
        self.d_latent       = params["d_latent"] # half of the hidden dimension is the mean, half is the variance
        self.d_hidden       = self.d_latent*2
        self.d_rnn_hidden   = params["d_rnn_hidden"] 
        self.n_gru_layers   = params["num_gru_layers"]
        self.d_output       = params["d_output"]

        self.dropout        = params["dropout"]
        self.max_length     = params["max_length"] 
        self.padding_idx    = params["padding_idx"]
        self.padding_idx = 0
        self.masking_probability = params["masking_probability"] # probability of masking a token before giving to the decoder.
        self.output_losses  = True # to work with previous code

        self.use_pp         = params["use_pp"] # boolean for whether to use a property predictor
        self.d_pp_input     = self.d_latent
        self.d_pp_hidden    = params["d_pp_hidden"]
        self.d_pp_output    = params["d_pp_output"]
        
        self.name = f"GomezBombarelli_{round(self.d_hidden/self.d_rnn_hidden,3)}_{self.d_rnn_hidden}"
        print(f"model name: {self.name}")
        
        logging.warning("max_length is hardcoded to 150")
        logging.warning(f"hardcoded: {self.d_pp_output=}")
        logging.warning(f"assuming one-hot encoding for inputs")
        logging.warning(f"loss function ignores: {self.padding_idx=}")

        self.loss_fn = loss_fn

        self.encoder = ConvEncoder(self.d_input, [9, 9, 10], [9, 9, 11], self.d_hidden)

        self.ff_context = nn.Linear(self.d_latent, self.d_rnn_hidden) # latent variable -> context vector

        self.decoder = RNNBlock(self.d_input, 
                                self.d_rnn_hidden, 
                                self.d_output, 
                                self.n_gru_layers,
                                dropout=self.dropout,
                                batch_first=True)

        if self.use_pp:
            self.property_predictor = nn.Sequential(
                nn.Linear(self.d_latent, self.d_pp_hidden),
                nn.ReLU(),
                nn.Linear(self.d_pp_hidden, self.d_pp_output)
            )

        if generator is None:
            logging.warning("generator of seed 0 assumed")
            self.generator = torch.Generator()
            self.generator.manual_seed(0)
        else:
            self.generator = generator

    def count_parameters(self):
        tot = 0
        for p in self.parameters():
            if p.requires_grad:
                tot += p.numel()
        return tot

    def get_tensor_devices(self):
        return [p.device for p in self.parameters()]
    
    def reparameterize(self, 
                       mu:torch.Tensor, 
                       logvar:torch.Tensor, 
                       generator:torch.Generator)->torch.Tensor:
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn(std.shape, generator=generator).to(std.device)
        return mu + eps*std

    def encode(self, idx:torch.Tensor):
        # embed indices | used as input to both encoder and decoder
        B, T = idx.shape
        tok = F.one_hot(idx, self.alphabet_size).float() # (B, T) -> (B, T, d_input)

        ########
        # encode
        ########
        tok = tok.permute(0,2,1)        # (B, T, d_input) -> (B, d_input, T) | necessary for convolutions
        mu_logvar = self.encoder(tok)   # (B, d_input, T) -> (B, d_hidden)
        mu, logvar = mu_logvar[:,:self.d_latent], mu_logvar[:,self.d_latent:] # (B, d_hidden) -> (B, d_latent), (B, d_latent)
        tok = tok.permute(0,2,1)        # (B, d_input, T) -> (B, T, d_input) 

        return mu, logvar, tok

    def forward(self, idx:torch.Tensor, target_props:torch.Tensor=None):
        
        # embed indices | used as input to both encoder and decoder
        B, T = idx.shape
        # tok = F.one_hot(idx, self.alphabet_size).float() # (B, T) -> (B, T, d_input)

        # ########
        # # encode
        # ########
        # tok = tok.permute(0,2,1)        # (B, T, d_input) -> (B, d_input, T) | necessary for convolutions
        # mu_logvar = self.encoder(tok)   # (B, d_input, T) -> (B, d_hidden)
        # mu, logvar = mu_logvar[:,:self.d_latent], mu_logvar[:,self.d_latent:] # (B, d_hidden) -> (B, d_latent), (B, d_latent)
        # tok = tok.permute(0,2,1)        # (B, d_input, T) -> (B, T, d_input) 
        mu, logvar, tok = self.encode(idx)
        
        ########
        # VAE reparameterization trick
        ########
        z = self.reparameterize(mu, logvar, generator=self.generator) # (B, d_latent)

        # latent variable to context vector
        z = self.ff_context(z)  # (B,     d_latent) -> (B,    d_rnn_hidden)
        z = z.unsqueeze(0)      # (B, d_rnn_hidden) -> (1, B, d_rnn_hidden) | necessary for RNNs
        z = z.repeat(self.n_gru_layers, 1, 1) # (1, B, d_rnn_hidden) -> (n_gru_layers, B, d_rnn_hidden) 

        ########
        # decode
        ########
        # note the slice for T-1, since we want to predict the next token
        if self.masking_probability>0.0:
            mask_ = torch.randint(
                0, 2, (B, T), 
                generator=self.generator
            ).unsqueeze(2).repeat(1,1,self.d_input) # (B, T-1)
            mask_ = mask_.to(tok.device)
            masked_tok = tok.masked_fill(mask_==1, 0.0) # (B, T, d_input) -> (B, T, d_input)
            masked_tok = masked_tok[:,:-1,:] # (B, T, d_input) -> (B, T-1, d_input)
        
            logits = self.decoder(masked_tok, z) # (B, T-1, d_input) -> (B, T-1, d_output)
        else:
            logits = self.decoder(tok[:,:-1,:], z) # (B, T-1, d_input) -> (B, T-1, d_output)

        ########
        # property prediction and loss function
        # compute reconstruction loss, KL divergence, and mean-squared error
        ########
        props = None
        if self.use_pp:
            props = self.property_predictor(mu) # (B, d_latent) -> (B, d_pp_output)

            BCE, KL, MSE = self.loss_fn(
                logits.view(-1, self.alphabet_size), # (B * (T-1), d_input)
                idx[:,1:].reshape(-1),               # (B * (T-1))
                mu,
                logvar,
                props[       ~target_props.isnan()],
                target_props[~target_props.isnan()]
            )
        else:
            BCE, KL, MSE = self.loss_fn(
                logits.view(-1, self.alphabet_size), # (B * (T-1), d_input)
                idx[:,1:].reshape(-1),               # (B * (T-1))
                mu,
                logvar,
            )
        # KL  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return logits, mu, logvar, props, BCE, KL, MSE
    

if __name__ == "__main__":

    g = torch.Generator().manual_seed(42)
    model = GomezBombarelli(d_input =8,
                            d_output=8,
                            d_hidden=12,
                            d_rnn_hidden=10,
                            d_pp_hidden=10,
                            generator=g)
    
    B, T = 5, 10
    tst_idx = torch.randint(0, 8, (B, T))
    print(f"{tst_idx.shape=}")
    
    model.eval()
    print(f"{model(tst_idx)[0].shape=}")
    print(f"{model(tst_idx)[1].shape=}")
    print(f"{model(tst_idx)[2].shape=}")
