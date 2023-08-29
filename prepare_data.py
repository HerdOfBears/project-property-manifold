import torch

import pandas as pd
import numpy as np

import os
import logging 

from torch.utils.data import DataLoader, Dataset


#
class Zinc250kDataset(Dataset):
    def __init__(self, 
                 molecules, 
                 properties) -> None:
        super().__init__()

        self.data = molecules
        self.targets = torch.tensor(properties)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx].float()

# class to prepare 
# - convert to indices
# - use RDKit to compute descriptors from SMILES
# - split into train, val, test
# - save to pickle file
class Zinc250k():
    def __init__(self, 
                 data_dir, 
                 data_file, 
                 save_dir, 
                 save_file):
        # set parameters
        self.data_dir  = data_dir
        self.data_file = data_file
        self.save_dir  = save_dir
        self.save_file = save_file

        # load data, and make dictionary converting smiles to indices
        self.data = pd.read_csv(
            os.path.join(self.data_dir, self.data_file)
        )

        # remove "\n" from end of smiles if it exists
        new_smiles = []
        for smile_ in self.data['smiles'].values:
            if smile_[-1] == "\n":
                new_smiles.append(smile_[:-1])
        self.data['smiles'] = new_smiles

        # make dictionaries to convert smiles to indices and back
        smiles = self.data['smiles'].values
        self.chr_to_idx = {"<SOM>": 1, # start of molecule 
                           "<EOM>": 2,  #   end of molecule
                           "<PAD>": 0,  # padding (0 to mask out)
        }
        self.idx_to_chr = {}
        self.max_len = 0
        for smile_ in smiles:
            if len(smile_) > self.max_len:
                self.max_len = len(smile_)

            for c in smile_:
                if c not in self.chr_to_idx:
                    v_ = len(self.chr_to_idx) # this accounts for <SOM>, <EOM>, <PAD>
                    self.chr_to_idx[c ] = v_ 
                    self.idx_to_chr[v_] = c
        
        self.alphabet_size = len(self.chr_to_idx)
        logging.info(f"alphabet size: {self.alphabet_size}")
        logging.info(f"{self.chr_to_idx=}")
        encoded_data = self.encode(smiles)
        encoded_data = self.pad(encoded_data)
        self.encoded_smiles = torch.tensor(encoded_data)

    def decode(self, indices:list[list[int]])->list[str]:
        # indices: list of list of indices
        # return: list of smiles
        smiles = []
        for idx_ in indices:
            smile_ = ''.join([self.idx_to_chr[i] for i in idx_])
            smiles.append(smile_)
        return smiles
    
    def encode(self, smiles:list[str])->list[list[int]]:
        # smiles: list of smiles
        # return: list of list of indices
        indices = []
        for smile_ in smiles:
            idx_ = [self.chr_to_idx[c] for c in smile_]
            indices.append(idx_)
        return indices
    
    def pad(self, indices:list[list[int]])->list[list[int]]:
        # indices: list of list of indices
        #  return: list of list of indices
        max_len = self.max_len
        logging.info(f"max_len w/o start or end token: {max_len}")
        output = []
        for idx_ in indices:
            idx_ = [self.chr_to_idx["<SOM>"]] + idx_ + [self.chr_to_idx["<EOM>"]] # add <SOM> and <EOM>
            idx_.extend(
                [self.chr_to_idx["<PAD>"]] * ( max_len + 2 - len(idx_) ) # + 2 for <SOM> and <EOM>  
            )
            output.append(idx_)
            if len(idx_)<max_len+2:
                raise ValueError(f"error: {len(idx_)} < {max_len+2}")
        return output
    
    def create_data_splits(self, 
                        train_size:float=0.8,
                        valid_size:float=0.1,
                         test_size:float=0.1,
                        generator:torch.Generator=None,
                        verbose :bool=False):
        """
        train_size: float, fraction of data to use for training
        valid_size: float, fraction of data to use for validation
         test_size: float, fraction of data to use for testing
         generator: torch.Generator, for reproducibility
           verbose: bool, return dataset objects if True
        """

        # check that train_size, valid_size, test_size add to 1
        tot = train_size+valid_size+test_size
        if abs(tot - 1) > 1e-6:
            raise ValueError(f"train_size+valid_size+test_size={tot}, must add to 1")
        
        if generator is None:
            generator = torch.Generator().manual_seed(42)
            logging.warning(f"generator is None. Using torch.Generator().manual_seed(42)")

        # make random splits' indices
        n = len(self.data)
        train_idxs, valid_idxs, test_idxs = torch.utils.data.random_split(
            list(range(n)),
            [train_size, valid_size, test_size],
            generator=generator
        )

        # create torch dataset objects
        logging.warning(f"using logP as a hard-coded property. This should be changed to an argumnet. ")
        property_name = "logP"
        return (self.encoded_smiles[train_idxs], 
                self.encoded_smiles[valid_idxs],
                self.encoded_smiles[ test_idxs],
                self.data[property_name].values[train_idxs],
                self.data[property_name].values[valid_idxs],
                self.data[property_name].values[ test_idxs]
        )
        # train_data = Zinc250kDataset(self.encoded_smiles[train_idxs], self.data[property_name].values[train_idxs])
        # valid_data = Zinc250kDataset(self.encoded_smiles[valid_idxs], self.data[property_name].values[valid_idxs])
        # test_data  = Zinc250kDataset(self.encoded_smiles[ test_idxs], self.data[property_name].values[ test_idxs])

        # train_loader = DataLoader(
        #     train_data,
        #     batch_size=32,
        #     shuffle=True,
        #     generator=generator
        # )
        # valid_loader = DataLoader(
        #     valid_data,
        #     batch_size=32,
        #     shuffle=True,
        #     generator=generator
        # )
        # test_loader = DataLoader(
        #     test_data,
        #     batch_size=32,
        #     shuffle=True,
        #     generator=generator
        # )

        # logging.warning("datasets are hosted on CPU, not GPU")

        # if verbose:
        #     return train_loader, valid_loader, test_loader, train_data, valid_data, test_data
        # else:
        #     return train_loader, valid_loader, test_loader

    
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    dataset = Zinc250k("./data", 
                       "250k_rndm_zinc_drugs_clean_3.csv",
                       "./data",
                       ""
    )

    logging.info(dataset.data.head())
    print(f"max length in the dataset: {dataset.max_len}")
    train_data, valid_data, test_data, train_targets, valid_targets, test_targets = dataset.create_data_splits()
    # train_dataloader, valid_dataloader, test_dataloader = dataset.create_data_splits()

    print(f"length of train_data: {len(train_data)}")
    print(F"length of train_targets: {len(train_targets)}")

    train_data = Zinc250kDataset(train_data, train_targets)
    valid_data = Zinc250kDataset(valid_data, valid_targets) 
    test_data  = Zinc250kDataset( test_data,  test_targets)

    generator = torch.Generator().manual_seed(42)
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


    # print(f"length of train_data: {len(train_dataloader)}")
    # print(f"length of valid_data: {len(valid_dataloader)}")
    # print(f"length of test_data:  {len( test_dataloader)}")
    
