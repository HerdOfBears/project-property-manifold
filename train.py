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

from torch.utils.data import DataLoader, Dataset

# custom imports
from prepare_data import Zinc250k, Zinc250kDataset
from models import Test
from rnn_models import RNNVae
from literature_models import GomezBombarelli
from helpers import LossWeightScheduler, checkpoint_model, make_save_dir

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

def training_loop(
                training_data,
                model,
                optimizer,
                epoch,
                annealer=None,
                return_losses=False):
    
    if return_losses:
        losses = {"iteration": [],
                "recon": [],
                "kl": [],
                "prop": [],
                "epoch": []}
    
    beta = 1.0
    if annealer is not None:
        beta = annealer.get_val()

    # set model to train mode
    model.train()
    for idx, (bch_x, bch_seq_lengths, bch_y) in enumerate(training_data): # bch_x, bch_y = sequences, properties

        # forward
        if model.output_losses:
            if model.name.split("-")[0]=="rnn":
                recon_x, output_pp, means_, logvars_, loss_recon, loss_kl, loss_prop = model(bch_x, prop_targets=bch_y, sequence_lengths=bch_seq_lengths)
            else:
                recon_x, output_pp, means_, logvars_, loss_recon, loss_kl, loss_prop = model(bch_x, bch_y)

            loss_tot = loss_recon + beta*(loss_kl + loss_prop)
        else:
            recon_x, output_pp, means_, logvars_, loss_tot = model(bch_x, bch_y)
        
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
                losses["epoch"].append(epoch)

    if return_losses:
        return losses

def validation_loop(
                validation_data,
                model,
                epoch,
                annealer=None,
                return_losses=False):
    
    if return_losses:
        losses = {"iteration": [],
                "recon": [],
                "kl": [],
                "prop": [],
                "epoch": []}
    
    beta = 1.0
    if annealer is not None:
        beta = annealer.get_val()

    # set model to train mode
    model.eval()
    for idx, (bch_x, bch_seq_lengths, bch_y) in enumerate(validation_data): # bch_x, bch_y = sequences, properties

        # forward
        if model.output_losses:
            if model.name.split("-")[0]=="rnn":
                recon_x, output_pp, means_, logvars_, loss_recon, loss_kl, loss_prop = model(bch_x, prop_targets=bch_y, sequence_lengths=bch_seq_lengths)
            else:
                recon_x, output_pp, means_, logvars_, loss_recon, loss_kl, loss_prop = model(bch_x, bch_y)
            loss_tot = loss_recon + beta*(loss_kl + loss_prop)
        else:
            recon_x, output_pp, means_, logvars_, loss_tot = model(bch_x, bch_y)
        
        if (idx % 50) == 0:
            if return_losses:
                losses["iteration"].append(idx + (epoch*len(validation_data)))
                losses["recon"].append(loss_recon.item())
                losses["kl"].append(loss_kl.item())
                losses["prop"].append(loss_prop.item())
                losses["epoch"].append(epoch)

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
    parser.add_argument("--n_latent",   type=int,   default=4)
    parser.add_argument("--n_embd",     type=int,   default=10)
    parser.add_argument("--n_model",    type=int,   default=8)
    parser.add_argument("--n_hidden_prop", type=int, default=10)
    parser.add_argument("--chkpt_freq", type=int,   default=-1)
    parser.add_argument("--random_seed", type=int,  default=42)
    parser.add_argument("--drop_percent_of_labels", type=int, default=0)
    parser.add_argument("--save_dir",  type=str,   default="./runs/")
    parser.add_argument("--save_suffix", type=str, default="")

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
    DROP_PERCENT_OF_LABELS = args.drop_percent_of_labels # percentage of labels to NaN out for experiment
    # CHKPT_DIR = args.chkpt_dir #deprecated
    SAVE_DIR = args.save_dir
    SAVE_SUFFIX = args.save_suffix

    N_EMBD = args.n_embd
    N_LATENT = args.n_latent
    N_MODEL  = args.n_model
    N_HIDDEN_PROP = args.n_hidden_prop

    if args.chkpt_freq > 0:
        CHKPT_FREQ = args.chkpt_freq 
    else:
        CHKPT_FREQ = N_EPOCHS

    print(f"n_epochs: {N_EPOCHS}, batch_size: {BATCH_SIZE}, lr: {LR}")
    print(f"save_dir: {SAVE_DIR}, chkpt_freq: {CHKPT_FREQ}")
    print(f"n_latent: {N_LATENT}, n_embd: {N_EMBD}, n_hidden_prop: {N_HIDDEN_PROP}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

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
    generator = torch.Generator().manual_seed(42)
    

    #######################
    # create data loaders
    #######################
    # train_loader, valid_loader, test_loader = data.create_data_splits()
    train_data, valid_data, test_data, train_targets, valid_targets, test_targets = data.create_data_splits(
        train_size=0.7,
        valid_size=0.2,
        test_size =0.1,
        generator=generator
    )

    # always shuffle indices so that random generator is at same state each run
    shuffled_training_indices = torch.randperm(len(train_targets), generator=generator)
    if DROP_PERCENT_OF_LABELS > 0:
        print(f"dropping {DROP_PERCENT_OF_LABELS}% of labels")
        n_to_drop = int(DROP_PERCENT_OF_LABELS/100*len(train_targets))
        shuffled_training_indices = shuffled_training_indices[:n_to_drop]
        train_targets[shuffled_training_indices] = np.nan

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

    print(f"length of train_loader: {len(train_loader)}, {len(train_data)=}")
    print(f"length of valid_loader: {len(valid_loader)}, {len(valid_data)=}")
    print(f"length of test_loader: {len(test_loader)}")
    #######################
    # Construct model(s)
    #######################
    print("constructing class")
    
    model = RNNVae(data.alphabet_size,
                   N_MODEL,
                   N_LATENT,
                   num_layers=3,
                   use_pp=True,
                   generator=generator)
    # model = GomezBombarelli(d_input =data.alphabet_size,
    #                         d_output=data.alphabet_size,
    #                         use_pp=True,
    #                         generator=generator)

    MODEL_DIR, CHKPT_DIR = make_save_dir(SAVE_DIR, model.name+SAVE_SUFFIX)

    model.apply(initialize_weights)
    print("weights initialized")

    with open(MODEL_DIR + f"model_{model.name}_args{SAVE_SUFFIX}.pkl","wb") as f:
        pkl.dump(vars(args), f)

    #######################
    # train
    #######################

    # send data and model(s) to device
    train_data.data    = train_data.data.to(   device)
    train_data.targets = train_data.targets.to(device)

    valid_data.data    = valid_data.data.to(   device)
    valid_data.targets = valid_data.targets.to(device)
    
    test_data.data     = test_data.data.to(   device)
    test_data.targets  = test_data.targets.to(device)

    model.to(device)

    print(f"number of parameters in model: {model.count_parameters()}")
    print(f"device being used: {device}")
    losses = {
        "training_losses":{
            "iteration": [],
            "recon": [],
            "kl": [],
            "prop": [],
            "epoch": []
        },
        "validation_losses":{
            "iteration": [],
            "recon": [],
            "kl": [],
            "prop": [],
            "epoch": []
        },
        "n_parameters_in_model": model.count_parameters()
    }
    
    # initialize annealers
    betaSchedule = LossWeightScheduler(val_min=1e-8, val_max=5e-2, steps=N_EPOCHS)

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("starting training loop")
    seconds_per_epoch = []
    epoch_training_loss   = 0
    epoch_validation_loss = 0
    for epoch in range(N_EPOCHS):
        t0 = time.time()

        # perform training loop
        losses_ = training_loop(
                    train_loader, 
                    model, 
                    optimizer,
                    epoch=epoch,
                    annealer=betaSchedule,
                    return_losses=True
        )

        # perform validation loop
        validation_losses_ = validation_loop(
                    valid_loader,
                    model,
                    epoch=epoch,
                    annealer=betaSchedule,
                    return_losses=True
        )
        
        # collect seconds per epoch
        print(f"epoch {epoch} time taken: {round(time.time() - t0, 4)}s")
        seconds_per_epoch.append(round(time.time() - t0, 4))
        
        # update annealer(s)
        betaSchedule.update(epoch)

        # update losses
        for key in losses_.keys():
            losses["training_losses"  ][key] += losses_[key]
            losses["validation_losses"][key] += validation_losses_[key]

        # save model checkpoint
        epoch_training_loss = np.mean(losses_["recon"]) + betaSchedule.get_val()*(np.mean(losses_["kl"]) + np.mean(losses_["prop"]))
        epoch_validation_loss = np.mean(validation_losses_["recon"]) + betaSchedule.get_val()*(np.mean(validation_losses_["kl"]) + np.mean(validation_losses_["prop"]))
        checkpoint_model(
            model, 
            optimizer, 
            betaSchedule, 
            epoch, 
            epoch_training_loss,
            epoch_validation_loss, 
            path=CHKPT_DIR,
            save_every=CHKPT_FREQ-1,
            save_suffix=SAVE_SUFFIX
        )
    
    losses["seconds_per_epoch"] = seconds_per_epoch
    
    pkl.dump(losses, open(MODEL_DIR+f"losses_{model.name}{SAVE_SUFFIX}.pkl", "wb"))
