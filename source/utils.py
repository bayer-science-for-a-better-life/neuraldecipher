import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
import h5py
import argparse

import os
source_path = os.path.abspath(__file__)
base_path = os.path.dirname(os.path.dirname(source_path))


def str_to_bool(v: str) -> bool:
    """
    Helper function for argparse to parse strings into boolean
    :param v:
    :return:
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_train_and_test_set(ecfp_path: str,
                              test_group: int = 7,
                              random_split: bool = False) -> (dict, dict):
    """
    Created the training and validation set.
    :param ecfp_path: path to where the ecfp data is stored.
    :param test_group: for cluster split: which cluster group should be used as test group. Defaults to 7
    :param random_split: whether or not the random split should be obtained. Defaults to False.
    :return: tuple of dictionaries for training and validation set
    """
    # load smiles array
    smiles = np.load(os.path.join(base_path, "data/smiles.npy"), allow_pickle=True)

    # load cddd
    cddd_reader = h5py.File(os.path.join(base_path, "data/cddd.hdf5"), "r")
    cddd = np.array(cddd_reader.get('cddd'))
    cddd_reader.close()
    # load cluster
    cluster = np.load(os.path.join(base_path, "data/cluster.npy"), allow_pickle=True)
    # load ecfp
    ecfp = np.load(os.path.join(base_path, ecfp_path), allow_pickle=True)

    if not random_split:
        test_index = [c == test_group for c in cluster]
        test_index = np.nonzero(test_index)[0]
    else:
        np.random.seed(42)
        counts = sum([c == test_group for c in cluster])
        test_index = np.random.choice(a=np.arange(len(smiles)), size=counts, replace=False)

    train_index = np.setdiff1d(ar1=np.arange(len(smiles)), ar2=test_index, assume_unique=False)
    train_cddd = cddd[train_index]
    train_ecfp = ecfp[train_index]
    train_smiles = smiles[train_index]

    test_cddd = cddd[test_index]
    test_ecfp = ecfp[test_index]
    test_smiles = smiles[test_index]

    return {'cddd':train_cddd, 'ecfp':train_ecfp, 'smiles':train_smiles},\
           {'cddd':test_cddd, 'ecfp':test_ecfp, 'smiles':test_smiles}


class SmilesDataset(Dataset):
    '''
    Creates a torch dataset instance for training.
    Each sample stores the ecfp, cddd and smiles representation of a molecular compound.
    '''
    def __init__(self, data: dict):
        '''
        Initializes the torch dataset for training
        :param data [dict] with keys 'cddd', 'ecfp', 'smiles'
        '''
        assert isinstance(data, dict), 'Please insert a dictionary with keys <cddd>, <ecfp>, <smiles>'
        self.data = data

    def __len__(self):
        return len(self.data['smiles'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ecfp = torch.Tensor(self.data['ecfp'][idx])
        cddd = torch.Tensor(self.data['cddd'][idx])
        smiles = self.data['smiles'][idx]

        sample = {'ecfp': ecfp, 'cddd': cddd, 'smiles': smiles}

        return sample


def create_data_loaders(train_data: dict, test_data: dict,
                        batch_size: int = 512, num_workers: int = 5,
                        shuffle_train: bool = True) -> (DataLoader, DataLoader):
    """
    Creates the dataloaders for train and validation set.
    :param train_data: dictionary training set with keys <cddd>, <ecfp> and <smiles>
    :param test_data: dictionary test set with keys <cddd>, <ecfp> and <smiles>
    :param batch_size: batch size for train and test loader.
        Default: 512
    :param num_workers: number of workers for dataloader.
        Default: 5
    :param shuffle_train: boolean if training set should be shuffled.
        Default: True
    :return: train and test dataloader
    """
    train_data = SmilesDataset(train_data)
    test_data = SmilesDataset(test_data)

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=shuffle_train, num_workers=num_workers)

    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader



class EvalDataset(Dataset):

    def __init__(self, data_path):
        '''
        Initializes the torch dataset for evaluation.
        '''
        self.data = np.load(data_path)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.Tensor(self.data[idx])
        return sample


def get_eval_data(ecfp_path, smiles_path, batch_size=256, num_workers=5):

    torch_dataset = EvalDataset(os.path.join(base_path,ecfp_path))
    torch_dataloader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    smiles_data = np.load(os.path.join(base_path, smiles_path), allow_pickle=True)

    return torch_dataloader, smiles_data


class EarlyStopping:
    """Early stops the training if a metric doesn't improve after a given patience."""
    def __init__(self, patience=20, mode='min', verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time the metric has improved.
                            Default: 20
            mode (str): Whether the metric should be maximized or minimized.
            verbose (bool): If True, prints a message for each metric improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        # init values to be -infity or +infity.
        self.metric_best = -np.inf if self.mode == 'max' else np.inf
        self.delta = delta

    def __call__(self, metric_val, modelpath, model, epoch):

        if self.best_score is None:
            self.best_score = metric_val
            self.save_checkpoint(metric_val, model, modelpath, epoch)

        elif self.mode == 'min' and metric_val + self.delta > self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'max' and metric_val - self.delta < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = metric_val
            self.save_checkpoint(metric_val, model, modelpath, epoch)
            self.counter = 0

    def save_checkpoint(self, metric_val, model, modelpath, epoch):
        '''Saves model when metric has improved.'''
        if self.verbose:
            if self.mode == 'min':
                print(f'Metric decreased ({self.metric_best:.6f} --> {metric_val:.6f}).  Saving model ...')
            else:
                print(f'Metric increased ({self.metric_best:.6f} --> {metric_val:.6f}).  Saving model ...')

        save_path = os.path.join(modelpath, f'checkpoint_epoch{epoch}_metric{metric_val:.6f}.pt')
        torch.save(model.state_dict(), save_path)
        self.metric_best = metric_val
