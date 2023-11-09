# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc.
#
# This file is part of the First-Break Picking package.
# Licensed under the terms of the MIT License.
#
# https://github.com/geo-stack/first_break_picking
# =============================================================================

from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset
import os 
from typing import Tuple, List, Union
from torch import Tensor
import numpy as np
import random 
import torch 
import torchvision.transforms as t
import pandas as pd

import first_break_picking.data.data_utils as tools# load_one_shot, show_batch

class FirstBreakOneShot(Dataset):
    """
    Custom dataset to stick all sub-shots together and create its appropriate shot

    Parameters
    ----------
    path : str
        File path
    data_info : pd.DataFrame
        A datadrame with three columns: shot_id, total_traces, number_of_subshots
    n_subshots : int
        Number of subshots
    n_time_samples : int
        Number of time samples
    n_trace : int
        Number of traces in final subshots
    validation : bool
        Defining if files have lable or not
    save_list : List[str]
        A list of shots that we want to study
    """
    def __init__(self, data_dir: str, 
                 data_info: pd.DataFrame,
                 with_label:bool,
                 file_names: List[str] = None) -> None:
        super().__init__()
        self.path = data_dir
        files = os.listdir(self.path)
        if file_names is None:
            self.files = files
        else:
            self.files = [file for file in files if file.split("_")[0] in file_names]
        self.n_data = data_info.shape[0]
        
        self.data_info = data_info
        self.files.sort()
        self.validation = with_label
        self.shots_id = list(self.data_info.index)
        self.set_transforms() 
        self.count = 0
        self.files = tools.delete_ds_store(files=self.files)

    def __len__(self):
        return self.n_data 

    def __getitem__(self, index):
        index = self.count

        main_shot_name = self.shots_id[index]
        bs = self.data_info.loc[main_shot_name][1]
        if bs>1: 
            files = [f"{main_shot_name}_{i}.npy" for i in range(bs)]
        else:  # if we don't split data
            files = [f"{main_shot_name}.npy"]
        
        shot = tools.load_one_shot(self.path, files)
        
        # shot = load_one_shot(self.path, self.files[index*self.bs:(index+1)*self.bs])
        bs = len(shot)
        
        if self.validation:
            nt, n_trace = shot[0][0].shape
            batch = torch.zeros((bs, 1, nt, n_trace))
            mask = torch.zeros((bs, 1, nt, n_trace))
            for i in range(bs):
                batch[i, 0, ...] = self._transforms(shot[i][0]) #  torch.from_numpy(shot[i][0])
                picking = shot[i][1]
                
                for j in range(n_trace):            
                    mask[i, 0, picking[j]:, j] = 1
        
            self.count += 1
            return batch, mask, main_shot_name
        else:
            nt, n_trace = shot[0].shape
            batch = torch.zeros((bs, 1, nt, n_trace))
            for i in range(bs):
                batch[i, 0, ...] = self._transforms(shot[i])  # torch.from_numpy(shot[i])
            
            self.count += 1
            return batch, main_shot_name

    def set_transforms(self) -> None:
        self._transforms = general_transform()

class FirstBreakDataset(Dataset):
    """
    Custom training/validation dataset

    Parameters
    ----------
    data_dir : str
        Directory of files
    file_names : int
        Name of shot files
    band_size : int
        Band size in case we want to have a band on picks
    """
    def __init__(self, 
                 data_dir: str,
                 file_names: List[str],
                 **kwargs
                 ) -> None:

        super().__init__()
        self.path_dir = data_dir
        self.file_names = file_names
        self.set_transforms()
        self.loaded_files = []
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        self.loaded_files.append(self.path_dir + "/" + self.file_names[index])
        
        model = np.load(self.loaded_files[-1],
                        allow_pickle=True)
        
        shot = self._transforms(model[0]).float()#.squeeze(0)

        mask = torch.zeros_like(shot)
        n_trace = model[0].shape[1]

        picking = model[1]
        picking[picking < 0] = 0  # It has negative values
        for i in range(n_trace):
            pick = picking[i]
            
            mask[0, pick:, i] = 1
        
        return shot, mask, 0

    def set_transforms(self) -> None:
        self._transforms = general_transform()
        

def general_transform():
    transforms = t.Compose([ 
                t.ToTensor(),
                t.Normalize(
                    mean = 0.5,
                    std= 0.5
                )])
    return transforms
