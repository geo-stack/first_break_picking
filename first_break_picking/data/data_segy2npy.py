# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc.
#
# This file is part of the First-Break Picking package.
# Licensed under the terms of the MIT License.
#
# https://github.com/geo-stack/first_break_picking
# =============================================================================

import numpy as np
import os 
from typing import List, Union, Tuple, Optional
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import torch 

import first_break_picking.data.data_utils as data_tools
from first_break_picking.data._datasets import general_transform
from first_break_picking.data._data_fb import preprocess_shots_fbs

def load_one_general_file(data_path: str,
                          shot_name:str,
                          label_name:str,
                          split_nt: int,
                          overlap: float,
                          time_window: List[int],
                          fbt_file_header: int,
                          fbt_time_column: int,
                          scale: bool,
                          grayscale: bool,
                          dt: float,
                          type_of_problem:str,
                          smoothing_value: float):
    
    _transformer = general_transform()
    
    sub_shots, data_info, sub_fbs = preprocess_shots_fbs(
                data_path = data_path,
                shot_name=shot_name,
                fb_name=label_name,
                split_nt = split_nt,
                overlap = overlap,
                time_window = time_window,
                fbt_file_header = None,
                fbt_time_column= fbt_time_column,
                scale = scale,
                grayscale = grayscale,
                dt=dt
                )
        
    data_info = info_dict_2_df(data_info)
    data_info = data_info.set_index("shot_id")
    
    # Because it happens for predicting all shots
    data = torch.empty((1, len(sub_shots), 1, sub_shots[0].shape[0], sub_shots[0].shape[1]))
    for i, sub in enumerate(sub_shots):
        data[0, i, 0, ...] = _transformer(sub)
            
    return data, data_info
     

def save_shots_fb(dataset_dir: str,
                split_nt: int,
                overlap: float,
                time_window: List[int],
                fbt_file_header: Union[int, None],
                fbt_time_column: int,
                dt: float,
                dir_to_save: str,
                scale: bool=True,
                grayscale: bool=True,
                shot_ext: str=".npy",
                fb_ext=None) -> pd.DataFrame:
    
    """
    Save files based on the availability of first break
    If fbt_file_header is None, it assumes that the data do not have first break label.

    Parameters
    ----------
    dataset_dir : str
        Directory to load data 
    dir_to_save : str
        Directory to save data 
    split_nt : int
        Number of traces in splited shots
    overlap : float
        Overlap for spliting shots
    time_window : List[int]
        Time window for cropping the sots in along time-direction 
    fbt_file_header : List[int or None]
        Number of lines in first break file header. If None, it assumes we don't have first break file.
    fbt_time_column : int
        Column number for arrival time in first breal file.
    scale : bool
        If you need to crop and scale the data
    grayscale : bool
        If you need to convert each shot to a gray image.
    shot_ext: str
        Extension of shot files. Default is ".npy"
    dt : float
        Temporal sampling rate
        
    Returns
    -------
    data_info : pd.DataFrame
        A dataframe containing shot_id, number of traces in each shot and number of subshots

    """
    # TODO: cehck this function
    files = os.listdir(dataset_dir)
    files = data_tools.delete_ds_store(files) 
    files_shot = [file for file in files if file.endswith(shot_ext)] 
    files_shot.sort()
    if fb_ext is None:
        fb_files = [None for file in files_shot]
    else:
        fb_files = [file for file in files if file.endswith(fb_ext)]
        fb_files.sort()
        
    data_info = {}
    for i, shot_name in enumerate(files_shot):
        ffid = ".".join(shot_name.split(".")[:-1])
        sub_shots, info, sub_fbs = preprocess_shots_fbs(
            data_path=dataset_dir,
            shot_name=shot_name,
            fb_name = fb_files[i],
            split_nt=split_nt,
            overlap=overlap,
            time_window=time_window,
            fbt_file_header=fbt_file_header,
            fbt_time_column=fbt_time_column,
            scale=scale,
            grayscale=grayscale,
            dt=dt
            )
        
        data_info[[*info][0]] = info[[*info][0]]
        
        if sub_fbs[0] is not None:
            save_train_npy(
                shots = sub_shots,
                labels = np.int32(sub_fbs),
                file_name=f"{dir_to_save}/{ffid}"
                ) 
        else:
            save_test_npy(
                shots = sub_shots,
                file_name=f"{dir_to_save}/{ffid}"
                ) 
    df = info_dict_2_df(data_info)
    return df


def info_dict_2_df(data_info: dict):
    df = pd.DataFrame(data_info).T
    idx = df.index
    total_traces = list(df[0])
    number_of_subshots = list(df[1])
    
    df = pd.DataFrame({"shot_id": idx,
                   "total_traces": total_traces,
                   "number_of_subshots": number_of_subshots})
    return df 


def save_train_npy(shots: List[np.ndarray],
                    labels: List[np.ndarray],
                    file_name: np.ndarray):
    
    if len(shots) > 1:
        for i in range(len(shots)):
            data = np.array([shots[i], labels[i]], dtype=object)
            _save_final_npy(data=data, 
                        file_name=file_name, 
                        count=i)
    else:
        data = np.array([shots[0], labels[0]], dtype=object)
        _save_final_npy(data=data, 
                        file_name=file_name, 
                        count=None)
        
        
def save_test_npy(shots: List[np.ndarray],
                    file_name: np.ndarray):
    if len(shots) > 1:
        for i, sub in enumerate(shots):
            _save_final_npy(data=sub, 
                    file_name=file_name, 
                    count=i)
    else:
        _save_final_npy(data=shots[0], 
                    file_name=file_name, 
                    count=None)
            
    
def _save_final_npy(data: np.ndarray,
                    file_name: str,
                    count: int):
    """
    Save final files

    Parameters
    ----------
    data : np.ndarray
        _description_
    fimle_name : str
        _description_
    count : int
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if count is None:
        np.save(file_name, data)
    else:
        np.save(file_name+f"_{count}", data)


def get_args():
    path: str = os.path.abspath(os.path.join(__file__, "../../data_files"))
    
    parser = ArgumentParser()
    
    parser.add_argument(
        "--project_name", dest = "project_name",
        help = "Project name",
        type=str
    )
    
    parser.add_argument(
        "--operation", dest = "operation",
        help = "Type of dataset", 
        choices=["train", "test", "to_pick"],
        type=str, default="train"
    )
    
    parser.add_argument(
        "--data_dir", dest = "data_dir",
        help = "Raw data directory",
        default = path + "/raw_data"
    )
    
    parser.add_argument(
        "--save_dir", dest = "save_dir",
        help = "Preprocessed data directory",
        default = path + "/preprocessed"
    )
    
    parser.add_argument(
        "--time_window", dest = "time_window",
        help = "Time size for cropping in time direction",
        nargs="+", type=int,
        default = [0, 512]
    )
    
    parser.add_argument(
        "--split_nt", dest = "split_nt",
        help = "Preprocessed data directory",
        default = 17, type=int
    )
    
    parser.add_argument(
        "--fbt_time_column", dest = "fbt_time_column",
        help = "Column number in first break file",
        default = 4, type=int
    )
    
    parser.add_argument(
        "--fbt_file_header", dest = "fbt_file_header",
        help = "First break header",
        default = None, type=int
    )
    
    parser.add_argument(
        "--overlap", dest = "overlap",
        help = "Overlap for shot spliting",
        default = 0.15, type=float
    )
    
    args = parser.parse_args()
    
    return (args.data_dir, args.save_dir, args.project_name, args.operation, args.time_window,
            args.split_nt, args.overlap, args.fbt_time_column, args.fbt_file_header)
    
    
# if __name__ == "__main__":    
    # (path_to_load, path_to_save, project_name, 
    # operation, time_window, split_nt, overlap, 
    # fbt_time_column, fbt_file_header) = get_args()
    
    # if project_name is None:
    #     raise NameError("Please define the name of the project")
    
    # path_to_load: str = path_to_load + f"/{project_name}/{operation}"
    # path_to_save: str = path_to_save + f"/{project_name}/{operation}"
    
    # Path(path_to_save).mkdir(exist_ok=True, parents=True)

    # shot_to_npy(dataset_dir=path_to_load,
    #             dir_to_save=path_to_save,
    #             split_nt= split_nt,
    #             overlap = overlap,
    #             time_window=time_window,
    #             fbt_file_header=fbt_file_header,
    #             fbt_time_column=fbt_time_column,
    #             scale=True,
    #             grayscale=True, shot_ext=".npy")
    
    # Fast calling
        # python data_segy2npy.py --project_name amem --fbt_file_header 9

