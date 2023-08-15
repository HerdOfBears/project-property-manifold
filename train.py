import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pickle as pkl

import os
import sys
import logging
import time
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# custom imports
from prepare_data import Zinc250k, Zinc250kDataset
from models import Test

def initialize_weights(m):
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)

def training_loop(
                training_data,
                model,
                optimizer,
                epoch,
                return_losses=False):
    
    if return_losses:
        losses = {"iteration": [],
                "recon": [],
                "kl": [],
                "prop": []}
    
    # set model to train mode
    model.train()
    for idx, (bch_x, bch_y) in enumerate(training_data):

        # forward
        if model.output_losses:
            recon_x, output, means_, logvars_, loss_recon, loss_kl, loss_prop = model(bch_x, bch_y)
            loss_tot = loss_recon + loss_kl + loss_prop
        else:
            recon_x, output, means_, logvars_, loss_tot = model(bch_x, bch_y)
        
        # zero gradients + backward + optimize
        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()

        if (idx % 50) == 0:
            # print(f"iter = {idx}",
            #     f"losses {round(loss_recon.item(), 4)}, {round(loss_kl.item(), 4)}, {round(loss_prop.item(), 4)}" 
            # )
            if return_losses:
                losses["iteration"].append(idx + (epoch*len(training_data)))
                losses["recon"].append(loss_recon.item())
                losses["kl"].append(loss_kl.item())
                losses["prop"].append(loss_prop.item())

    if return_losses:
        return losses

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=2)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--logging",    type=str,   default="WARNING", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    parser.add_argument("--chkpt_dir",  type=str,   default="./checkpoints")
    parser.add_argument("--chkpt_freq", type=int,   default=-1)
    parser.add_argument("--n_latent",   type=int,   default=4)
    parser.add_argument("--n_embd",     type=int,   default=10)

    args = parser.parse_args()

    LOGGING_LEVEL = args.logging
    if LOGGING_LEVEL == "DEBUG":
        logging.basicConfig(level=logging.DEBUG)
    elif LOGGING_LEVEL == "INFO":
        logging.basicConfig(level=logging.INFO)
    elif LOGGING_LEVEL == "WARNING":
        logging.basicConfig(level=logging.WARNING)
    elif LOGGING_LEVEL == "ERROR":
        logging.basicConfig(level=logging.ERROR)
    elif LOGGING_LEVEL == "CRITICAL":
        logging.basicConfig(level=logging.CRITICAL)

    #######################
    # HYPERPARAMETERS
    #######################
    N_EPOCHS = args.epochs # n times through training loader
    BATCH_SIZE = args.batch_size 
    LR = args.lr # learning rate
    CHKPT_DIR = args.chkpt_dir

    N_EMBD = args.n_embd
    N_LATENT = args.n_latent
    
    if args.chkpt_freq > 0:
        CHKPT_FREQ = args.chkpt_freq 
    else:
        CHKPT_FREQ = N_EPOCHS
    print(f"n_epochs: {N_EPOCHS}, batch_size: {BATCH_SIZE}, lr: {LR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    print(f"using device: {device}")
    print("running training script...")
    print("constructing data class")
    data = Zinc250k("./data", 
                       "250k_rndm_zinc_drugs_clean_3.csv",
                       "./data",
                       ""
    )

    print(f"max length in the dataset: {data.max_len}")
    print(f"alphabet size: {data.alphabet_size}")
    
    
    #######################
    # Set seed for replicability and construct initialization fn
    #######################
    # a lambda function is req'd here because module.apply(fn) 
    # takes a fn with only one argument but we want a defined generator
    generator = torch.Generator().manual_seed(42)
    initialize_weights_one_arg = lambda x: initialize_weights(x, generator=generator)
    
    #######################
    # create data loaders
    #######################
    # train_loader, valid_loader, test_loader = data.create_data_splits()
    train_data, valid_data, test_data, train_targets, valid_targets, test_targets = data.create_data_splits(
        generator=generator
    )

    print(f"length of train_data: {len(train_data)}")
    print(F"length of train_targets: {len(train_targets)}")

    train_data = Zinc250kDataset(train_data, train_targets)
    valid_data = Zinc250kDataset(valid_data, valid_targets) 
    test_data  = Zinc250kDataset( test_data,  test_targets)


    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )


    #######################
    # Construct model(s)
    #######################
    print("constructing Test class")
    testnn = Test(data.alphabet_size,
                  n_latent=N_LATENT,
                  n_embd=N_EMBD,
                  output_losses=True)
    
    # initialize weights
    testnn.apply(initialize_weights_one_arg)
    print("weights initialized")

    #######################
    # train
    #######################

    # send data and model(s) to device
    train_data.data    = train_data.data.to(   device)
    train_data.targets = train_data.targets.to(device)
    tst0, tst1 = next(iter(train_loader))
    print(f"tst0 device = {tst0.device}")
    print(f"tst1 device = {tst1.device}")
    testnn.to(device)

    print(f"number of parameters in model: {testnn.count_parameters()}")
    print(f"devices being used: {testnn.get_tensor_devices()}")
    # usr_input = input("continue?")
    losses = {"iteration": [],
              "recon": [],
              "kl": [],
              "prop": []}
    
    print("starting training loop")
    for epoch in range(N_EPOCHS):
        print(f"epoch {epoch}")
        t0 = time.time()

        # perform training loop
        losses_ = training_loop(train_loader, 
                    testnn, 
                    optim.SGD(testnn.parameters(), lr=LR),
                    epoch=epoch,
                    return_losses=True)
        
        print(f"epoch time: {round(time.time() - t0, 4)}s")
        
        # update losses
        for key in losses.keys():
            losses[key] += losses_[key]
        
        # save model checkpoint
        if (epoch % (CHKPT_FREQ-1)) == 0:
            logging.info(f"saving checkpoint at epoch {epoch}")
            torch.save(testnn.state_dict(), f"{CHKPT_DIR}/testnn_{epoch}.pt")

    pkl.dump(losses, open("./losses.pkl", "wb"))
