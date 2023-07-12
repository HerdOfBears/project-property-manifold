import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import os
import sys
import logging

from tqdm import tqdm

# custom imports
from prepare_data import Zinc250k
from models import Test

def training_loop(
                training_data,
                model,
                optimizer):
    
    # set model to train mode
    model.train()

    for idx, (bch_x, bch_y) in enumerate(tqdm(training_data)):

        # forward
        recon_x, output, means_, logvars_, loss_tot = model(bch_x, bch_y)

        
        # zero gradients + backward + optimize
        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()

        # tqdm.set_postfix(loss=round(loss_tot.item(), 4))
        print(f"loss: {loss_tot.item()}")
        if idx>5:
            break

if __name__=="__main__":
    # logging.basicConfig(level=logging.INFO)

    data = Zinc250k("./data", 
                       "250k_rndm_zinc_drugs_clean_3.csv",
                       "./data",
                       ""
    )

    print(f"max length in the dataset: {data.max_len}")
    print(f"alphabet size: {data.alphabet_size}")
    # create data loaders
    train_loader, valid_loader, test_loader = data.create_data_splits()

    print("try constructing Test class")
    testnn = Test(data.alphabet_size,
                  9)
    
    training_loop(train_loader, testnn, optim.SGD(testnn.parameters(), lr=1e-3))

