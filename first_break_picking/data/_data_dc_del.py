from pathlib import Path
import os 
import numpy as np 
import pandas as pd
import logging 
from typing import List, Union
from scipy.ndimage import gaussian_filter

import first_break_picking.data.data_utils as data_tools

def preprocess_shots_dcs(dataset_dir: str,
            shot_name:str, 
            curve_name: str,
            grayscale: bool,
            smoothing_value):

    shot =np.load(f"{dataset_dir}/{shot_name}")
    
    if curve_name is not None:
        curve = pd.read_csv(f"{dataset_dir}/{curve_name}").to_numpy()
    else:
        curve = None
    
    if grayscale:
        shot = data_tools.shot_to_gray_scale(data=shot)
    shot = gaussian_filter(shot, smoothing_value)
    
    return shot, curve
   
   
def save_data_pairs(data_path_loading: str,
                    data_path_saving: str) -> None:
    """
    Save pairs of dispersion array and dispersion curve in one `npy` file.

    Parameters
    ----------
    data_path_loading : str
        Path to load dispersion arrays and curves (to separate files per shot)
        (e.g. 141.npy and 131_disp.txt)
    data_path_saving : str
        Path to save dispersion arrays and curves in one single `npy` file
    """
    
    if not Path(data_path_saving).exists():
        Path(data_path_saving).mkdir(parents=True)
    
    files = os.listdir(data_path_loading)
    files = data_tools.delete_ds_store(files=files)

    ffids = [file.split('.')[0] for file in files if file.endswith(".npy")]
    ffids = np.unique(ffids)

    for ffid in ffids:
        disp_curve = pd.read_csv(
            f"{data_path_loading}/{ffid}_disp.txt").to_numpy()
        
        disp_array = np.load(f"{data_path_loading}/{ffid}.npy")
        
        __save_pairs(shot=disp_array,
                       label=disp_curve,
                       file_name=f"{data_path_saving}/{ffid}.npy"
                        )
        
def __save_pairs(shot: np.ndarray,
                 label: pd.DataFrame,
                 file_name: str
                 ) -> None:

    data_sample = np.array([shot, label], dtype=object)
    np.save(file_name, data_sample)
    
    
def load_available_dispersion_curves(
    file_name: str) -> Union[pd.DataFrame,
                             list]:
    """
    Load the dispersion curves and return the them plus shot names

    Parameters
    ----------
    file_name : str
        _description_

    Returns
    -------
    Union[pd.DataFrame, list]
        _description_
    """
        
    curves = pd.read_json(file_name)
    return curves, list(curves.columns)


def check_dispersion_curve_file_name(picks_file_name: str) -> str:
    """
    Verifies the format of picks name

    Parameters
    ----------
    picks_file_name : str
        _description_

    Returns
    -------
    str
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    name_parts = picks_file_name.split(".")
    if picks_file_name.endswith(".json"):
        return picks_file_name
    elif len(name_parts) == 1:
        return f"{picks_file_name}.json"
    elif len(name_parts) > 1:
        suffix = name_parts[1]
        raise ValueError(f"Dispersion curve format should be '.json', but got .{suffix}")
    