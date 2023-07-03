import os 
from typing import List, Tuple
import pandas as pd 
import numpy as np 

import first_break_picking.data.data_utils as data_tools

def preprocess_shots_fbs(data_path: str,
                    shot_name: str,
                    fb_name: str,
                split_nt: int,
                overlap: float,
                time_window: List[int],
                fbt_file_header: int,
                fbt_time_column: int,
                scale: bool,
                grayscale: bool,
                dt: float
                ) -> Tuple[List[np.ndarray],
                          dict, 
                          List[np.ndarray]]:
    """
    Make function to load segy and first break files

    Parameters
    ----------
    shot_path : str
        Path of the shot
    dir_to_save : str
        Directory to save files
    split_nt : int
        Number of traces in splited shots
    overlap : float
        Overlap for spliting shots
    time_window : List[int]
        Time windowing if we don't want to consider the whole data
    fbt_file_header : int
        Number of header lines in fbt_file_header
    fbt_time_column : int
        Number of arrival time column in first break file
    overlap: float
        Overlap of subimages
    shot_ext: str
        Extension of shot files
    dt: float
        Temporal sampling rate
        
    Returns
    -------
    data_info : pd.DataFrame
        A dataframe containing shot_id, number of traces in each shot and number of subshots
    
    Raises
    ------
    RuntimeError
        If equivalent file doesn't exist for a first break file
    """
    data_info = {}
    filename = ".".join(shot_name.split(".")[:-1])
    
    if os.path.exists(f"{data_path}/{shot_name}"):
        sub_shots, n_traces, sub_fbs = _read_shot_fb(
            data_path=data_path,
                shot_name=shot_name, 
                fb_file = fb_name, 
                split_nt=split_nt,
                overlap=overlap,
                time_window = time_window,
                fbt_file_header=fbt_file_header,
                fbt_time_column=fbt_time_column,
                scale=scale,
                grayscale=grayscale,
                dt=dt)
    else:
            raise RuntimeError(f"The equivalent shot for first break {fb_name} doesn't exist")
    
    data_info[filename] = [n_traces, len(sub_shots)]

    return sub_shots, data_info, sub_fbs


def _read_first_break(filename: str,
                    header: int,
                    time_coloumn: int):
    """
    Read first break file

    Parameters
    ----------
    filename : str
        _description_
    header : int
        _description_
    time_coloumn : int
        _description_

    Returns
    -------
    _type_
        _description_
    """

    df = pd.read_csv(filename, sep=",", header=header-1)
    fb = df.iloc[:, time_coloumn]
    fb.replace(-999.031250, np.nan, inplace=True)
    fb= fb.ffill().bfill()

    return fb.to_numpy()

def _read_shot_fb(data_path: str,
                  shot_name: str,
                fb_file: str,
                split_nt,
                overlap,
                time_window: List[int],
                fbt_file_header: int,
                fbt_time_column: int,
                scale: bool,
                grayscale: bool,
                dt: float):
    """
    Read one shot and first break file

    Parameters
    ----------
    file : str
        _description_
    fb_file : str
        _description_
    n_traces : int
        _description_
    time_window : List[int]
        _description_
    fbt_file_header : int
        _description_
    fbt_time_column : int
        _description_
    shot_ext: str
        Extension of shot files. Default is ".sgy"
    dt: float
        Temporal sampling rate
    Returns
    -------
    _type_
        _description_
    """
    shot_ext = shot_name.split(".")[-1]
    shot_path = f"{data_path}/{shot_name}"
    
    if shot_ext in ["sgy", "segy"]:  
        raise RuntimeError("This package can't be used to read segy file. "
                           "Please convert all files to numpy, .npy")        
        # shot, dt, twt = _read_segy(shot_path, False)
    elif shot_ext == "npy":
        shot = np.load(shot_path)
    else:
        raise RuntimeError("Diles with extension of '.npy' and '.sgy' can be loaded"
                           f", but got .{shot_ext}")
        
    if time_window is not None:
        shot = shot[time_window[0]:time_window[1], :]
    if scale:
        shot = data_tools.data_normalize_and_limiting(data=shot)
    if grayscale:
        shot = data_tools.shot_to_gray_scale(shot)

    points = data_tools.starting_points(shot.shape[1], split_nt, overlap)
    sub_shots = data_tools.shot_spliting(
        shot=shot,
        points=points,
        split_nt=split_nt
        )
    sub_fbs = [None]
    
    if fb_file is not None:
        fb = _read_first_break(f"{data_path}/{fb_file}",
                        header=fbt_file_header,
                        time_coloumn=fbt_time_column)//dt
        
        sub_fbs = data_tools.fb_spliting(
            fb=fb,
            points=points,
            split_nt=split_nt
        )
        
    return sub_shots, shot.shape[1], sub_fbs
