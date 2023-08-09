import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pickle as pkl

import os
import sys
import logging

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# custom imports
from prepare_data import Zinc250k, Zinc250kDataset
from models import Test

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
    # logging.basicConfig(level=logging.INFO)
    N_EPOCHS = 1
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # create data loaders
    #######################
    # train_loader, valid_loader, test_loader = data.create_data_splits()
    train_data, valid_data, test_data, train_targets, valid_targets, test_targets = data.create_data_splits()
    # train_dataloader, valid_dataloader, test_dataloader = dataset.create_data_splits()

    print(f"length of train_data: {len(train_data)}")
    print(F"length of train_targets: {len(train_targets)}")

    train_data = Zinc250kDataset(train_data, train_targets)
    valid_data = Zinc250kDataset(valid_data, valid_targets) 
    test_data  = Zinc250kDataset( test_data,  test_targets)

    generator = torch.Generator().manual_seed(42)


    train_data.to(device)


    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        generator=generator
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=32,
        shuffle=True,
        generator=generator
    )
    test_loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=True,
        generator=generator
    )


    #######################
    # Construct model(s)
    #######################
    print("constructing Test class")
    testnn = Test(data.alphabet_size,
                  9,
                  output_losses=True)
    
    #######################
    # train
    #######################

    # send data and model(s) to device
    train_data.data.to(device)
    train_data.targets.to(device)
    _, tst = next(iter(train_loader))
    print(f"tst device = {tst[0].device}")
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

        # perform training loop
        losses_ = training_loop(train_loader, 
                    testnn, 
                    optim.SGD(testnn.parameters(), lr=1e-3),
                    epoch=epoch,
                    return_losses=True)
        
        # update losses
        for key in losses.keys():
            losses[key] += losses_[key]

    pkl.dump(losses, open("./losses.pkl", "wb"))
