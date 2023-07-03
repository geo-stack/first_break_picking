from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset
import os 
from typing import Tuple, List, Union
from torch import Tensor
import numpy as np
import random 
import torch 
import torchvision.transforms as t
import pandas as pd

from first_break_picking.data.data_utils import delete_ds_store, show_batch
from first_break_picking.data._datasets import (
    FirstBreakOneShot, FirstBreakDataset)

def get_predict_dataset(type_of_problem: str):
    return FirstBreakOneShot
    
    
def _get_train_dataset(problem: str):
    return FirstBreakDataset

def _split_dataset(data_dir: str,
                  val_fraction: float, 
                  ) -> Union[List[str], List[str]]:

    """
    It splits dataset into training and validation set based on the val_fraction

    Parameters
    ----------
    data_dir : str
        Directory of data files
    val_fraction : float
        Percentage of validation data [0 1]

    Returns
    -------
    train_set: List[str]
        Name of files in training set
        
    val_set: List[str]]
        Name of files in validation set
    """
    shots_name = os.listdir(data_dir)
    delete_ds_store(shots_name)

    n_data = len(shots_name)
    
    n_train = int((1 - val_fraction) * n_data)    
    
    random.shuffle(shots_name)
    
    train_set = shots_name[:n_train]
    val_set = shots_name[n_train:]
    
    return train_set, val_set


def _get_datasets(data_dir: List[str],
                  val_fraction: float,
                  problem: str) -> Union[Dataset, Dataset]:
    """
    Creates datasets based on the directory

    Parameters
    ----------
    data_dir : List[str]
        Directory of data
    val_fraction : float
        Fraction of validation set
    upsampled_size_col: int
        Number of traces in one shot after updampling

    Returns
    -------
    train_dataset: Dataset
        Training dataset
    
    val_dataset: Dataset
        Validation dataset
    """
    
    train_datasets = []
    val_datasets = []

    TrainDataset = _get_train_dataset(problem=problem)
    delete_ds_store(data_dir)

    for i, dir in enumerate(data_dir):
        train_set, val_set = _split_dataset(dir, val_fraction)

        train_datasets.append(
            TrainDataset(data_dir=dir,
                         file_names=train_set,
                         with_label=True
                         ))
                                    
        val_datasets.append(
            TrainDataset(data_dir=dir,
                         file_names=val_set,
                         with_label=True
                         ))
    
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    return train_dataset, val_dataset


def get_loaders(data_dir: List[str],
                val_fraction: float,
                batch_size: int,
                problem: str)-> Union[DataLoader, DataLoader]:
    """
    This function returns training and validation data loader
    
    Parameters
    ----------
    data_dir : str
        Data directory
    val_fraction : float
        Fraction of validation data
    batch_size : int
        Batch size
    Returns
    -------
    train_dl: DataLoader
        Training data loader
    val_dl: DataLoader
        Validation data loader
    """
    
    train_dataset, val_dataset = _get_datasets(
        data_dir, 
        val_fraction,
        problem=problem)
    
    train_dl = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(dataset=val_dataset, 
                        batch_size=batch_size, shuffle=False)
    return train_dl, val_dl


def test_data(data_dir: List[str]):
    """
    A function to test whole data. It keeps showing whole available data

    Parameters
    ----------
    data_dir : List[str]
        List of directories with data files (*.npy)
    """
    batch_size = 4

    train_dl, val_dl = get_loaders(data_dir, val_fraction=0.0,
                                    batch_size=batch_size,
                                      band_size=0,
                                      upsampled_size=256)
    print(f"There are {len(train_dl)} batches")
    for shot, pick, _ in train_dl:
        show_batch(batch_size, shot, pick)
        plt.show(block=True)


def test_trainset(data_dir: List[str]):
    """
    A function to test the loader

    Parameters
    ----------
    data_dir : List[str]
        List of directories with data files (*.npy)
    """
    BATCH_SIZE = 4
    NARROW_BAND = 0

    # This directory can have different subfolders for different projects.
    # Each project has to have a subfolder called train
    train_dl, val_dl = get_loaders(data_dir, val_fraction=0.1,
                                    batch_size=BATCH_SIZE,
                                      band_size=NARROW_BAND,
                                      upsampled_size=256)
    
    img, mask, band_mask = next(iter(train_dl))

    show_batch(BATCH_SIZE, img, mask)

    plt.show()
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    test_path = os.path.abspath(os.path.join(__file__, "../../data_files/preprocessed/amem/train"))

    test_trainset([test_path])