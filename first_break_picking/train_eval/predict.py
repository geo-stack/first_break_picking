# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc.
#
# This file is part of the First-Break Picking package.
# Licensed under the terms of the MIT License.
#
# https://github.com/geo-stack/first_break_picking
# =============================================================================

import torch 
from pathlib import Path
import shutil
from typing import Any, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

import first_break_picking.data.data_utils as data_tools
from first_break_picking.data.dataset import get_predict_dataset
from first_break_picking.train_eval.unet import UNet
import first_break_picking.train_eval.parameter_tools as ptools
import first_break_picking.train_eval.ai_tools as tools
from first_break_picking.data.data_segy2npy import (load_one_general_file,
                                                    general_transform,
                                                    info_dict_2_df)

class Predictor:
    def __init__(self,
                 path_to_save: str, # location to save the result
                 upsampled_size_row: int,  # time, vel
                 upsampled_size_col: int,  # trace, freq
                 type_of_problem: str="fb",
                 x_axis: np.ndarray=None,
                 y_axis: np.ndarray=None,
                 original_dispersion_size: Tuple=(157, 490),
                 split_nt: int=0,
                 overlap: float=0,
                 dt: float=1.0,
                 checkpoint_path: str = None,
                 model_name: str = "unet_resnet",
                 smoothing_threshold: int = 20,
                 features: List[int] = [16, 32, 64, 128],
                 in_channels: int = 1,
                 out_channels: int = 2,
                 save_segmentation: bool = False
            ) -> None:
        """An object for predicting the first break picking

        Parameters
        ----------
        path_to_save : str
            location to save the result
        upsampled_size_col : int
            Number of columns in the upsampled shot
        upsampled_size_row : int
            Number of rows in the upsampled shot
        type_of_problem : str, optional
            _description_, by default "fb"
        x_axis : np.ndarray, optional
            _description_, by default None
        y_axis : np.ndarray, optional
            _description_, by default None
        original_dispersion_size : Tuple, optional
            _description_, by default (157, 490)
        split_nt : int, optional
            _description_, by default 0
        overlap : float, optional
            _description_, by default 0
        dt : float, optional
            _description_, by default 1.0
        checkpoint_path : str, optional
            _description_, by default None
        model_name : str, optional
            _description_, by default "unet_resnet"
        smoothing_threshold : int, optional
            _description_, by default 20
        features : List[int], optional
            _description_, by default [16, 32, 64, 128]
        in_channels : int, optional
            _description_, by default 1
        out_channels : int, optional
            _description_, by default 2
        save_segmentation : bool, optional
            _description_, by default False
        """
        
        self.path_to_save = path_to_save
        self.smoothing_threshold = smoothing_threshold
        self.upsampled_size_row = upsampled_size_row
        self.save_segmentation = save_segmentation
        
        self.split_nt = split_nt
        self.overlap = overlap
        
        self.type_of_problem = type_of_problem
        device = "cpu"  # For prediction we don't need cuda nor mps
        
        self.x_axis = x_axis
        self.y_axis = y_axis

        (self.model, self.device, self.upsampler) = \
            ptools.define_general_parmeters(
                upsampled_size_row=upsampled_size_row,
                upsampled_size_col=upsampled_size_col,
                model_name=model_name,
                in_channels=in_channels,
                out_channels=out_channels,
                features=features,
                encoder_weight="imagenet",
                device=device
                )
            
        (self.predict_validation, self.predict_test,
         self.case_specific_parameters, self.show_prediction,
         self.dt)= ptools.define_predict_parameters_one_shot(
             model=self.model,
             upsampled_size_row=upsampled_size_row,
             checkpoint_path=checkpoint_path,
             device=device,
             type_of_problem=type_of_problem,
             dt=dt,
             original_dispersion_size=original_dispersion_size
             )
         
        self.smoothing_value = smoothing_threshold
    
    def predict_ndarray_fb(self,
                           shot: np.ndarray) -> np.ndarray:
        
        # No need to change. It is important when we have more than one shot
        ffid: str = "12" 
        
        sub_shots, n_total_traces, _ = data_tools.fb_pre_process_data(
            shot, fb=None,
            split_nt=self.split_nt,
            overlap=self.overlap,
            time_window=[0, self.upsampled_size_row],
            scale=True,
            grayscale=True
            )
        
        n_subshots = len(sub_shots)
        data = torch.zeros((1, n_subshots, 1, 
                            self.upsampled_size_row, self.split_nt))
        #nsp:  data.shape=[1, 2, 1, 512, 32])
        
        _transformer = general_transform()
        for i, sub in enumerate(sub_shots):
            data[0, i, 0, ...] = _transformer(sub)
        #nsp:  data.shape=[1, 2, 1, 512, 32])
        
        data_info = info_dict_2_df({str(ffid): [n_total_traces, n_subshots]})
        data_info = data_info.set_index("shot_id")
        
        # nsp:  data.shape=[1, 2, 1, 512, 32]
        data, _ = self.upsampler(data.squeeze(0), data.squeeze(0))
        #nsp:  data.shape= [2, 512, 512])
        # data = data.unsqueeze(1)
        
        shot, predicted_pick, predicted_segment = self.predict_test(
                    batch=data, 
                    model=self.model,
                    split_nt=self.split_nt,
                    overlap=self.overlap,
                    shot_id=ffid,
                    smoothing_threshold=self.smoothing_threshold,
                    data_info=data_info,
                    case_specific_parameters=self.case_specific_parameters
                )
        if predicted_pick is None:
            return -999.03125 * np.ones(shot.shape[1])
        else:
            return np.array(predicted_pick * self.dt)
    
    def predict(self, path_data: str):
        """Predict FB

        Parameters
        ----------
        path_data : str
            Paht to the data
        """
        file_name = path_data.split("/")[-1]
        path_data = "/".join(path_data.split("/")[:-1])
        ffid = ".".join(file_name.split(".")[:-1])
        
        data, data_info = load_one_general_file(
                data_path=path_data,
                shot_name=file_name,
                label_name=None,
                split_nt=self.split_nt,
                overlap=self.overlap,
                time_window=[0, self.upsampled_size_row],
                fbt_file_header=None,
                fbt_time_column=0,
                scale=True,
                grayscale=True,
                dt=self.dt,
                type_of_problem=self.type_of_problem,
                smoothing_value=self.smoothing_value
                )
            
        shot, predicted_pick, predicted_segment = self.predict_test(
                    batch=data, 
                    model=self.model,
                    split_nt=self.split_nt,
                    overlap=self.overlap,
                    shot_id=ffid,
                    smoothing_threshold=self.smoothing_threshold,
                    upsampler=self.upsampler,
                    data_info=data_info,
                    case_specific_parameters=self.case_specific_parameters
                )
                
        self.show_prediction([shot],
                              [predicted_segment],
                              [predicted_pick],
                              true_masks = [None],
                              path_save_fb=self.path_to_save,
                              dt=self.dt,
                              ffids=[ffid],
                              save_segmentation=self.save_segmentation,
                              x_axis=self.x_axis,
                              y_axis=self.y_axis)

    
def predict(base_dir: str,
            path_to_save: str,
            upsampled_size_row: int,  # time, vel
            upsampled_size_col: int,  # trace, freq
            x_axis: np.ndarray=None,
            y_axis: np.ndarray=None,
            split_nt: int=0,
            overlap: float=0.0,
            original_dispersion_size: Tuple=(157, 490),
            dt: float=1,
            data_info: pd.DataFrame = None,
            checkpoint_path: str = None,
            model_name: str = "unet_resnet",
            smoothing_threshold: int = 50,
            features: List[int] = [16, 32, 64, 128],
            in_channels: int = 1,
            out_channels: int = 2,
            validation: str =  False,
            save_list: List[str] = None,
            save_segmentation: bool = False):
    """
    This function is to be called to predict the results.

    Parameters
    ----------
    base_dir : str
        Directory where datasets (*.npy) are saved
    split_nt : int
        Number of traces in splitted shot (for example 17)
    overlap : float
        Overlap betwwen each shot for spliting
    n_time_sampels : int
        Number of time samples in each shot
    width_enlarged_subshot : int
        Number of traces in upsampled shot (devisable by 16)
    dt : float
        Temporal sampling rate
    data_info : pd.DataFrame,
        A dataframe containing name of each shot (FFID) and its number of traces and subshots
    checkpoint_path : str, optional
        Path to the checkpoints. If not specified, it uses a pre-trained model
    model_name : str, optional
        Name of the network, by default "unet_resnet"
        It can be either 'unet' or 'unet_resnet'
    smoothing_threshold: int, optoional
        In each trace, if there is there is multiple segment in smoothing_threshold,
        model does picks the first occurance of the data segment as anomaly and moves 
        to the next occurance of data segment.
    features : List[int], optional
        List of number of channels for each conv layer, by default [16, 32, 64, 128]
    n_channels : int, optional
        Number of channels in the input shot, by default 1
    out_channels : int, optional
        Number of out-channels , by default 2
    validation : bool, optional
        Specify it is validation set (with label) or the dataset is without label, by default False
    save_list : List[str], optional
        List of FFIDs to save 
    save_segmentation : bool, optional
        Specify if user desires to save the segmentation, by default False        
    """
    device = "cpu"  # For prediction we don't need cuda nor mps
        
    (model, device, upsampler) = \
        ptools.define_general_parmeters(
            upsampled_size_row=upsampled_size_row,
            upsampled_size_col=upsampled_size_col,
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            encoder_weight="imagenet",
            device=device
            )
    
    (data_info, type_of_problem,
     predict_validation, predict_test,
     case_specific_parameters, show_prediction,
     dt)= ptools.define_predict_parameters(
        model=model,
        upsampled_size_row=upsampled_size_row,
        checkpoint_path=checkpoint_path,
        device=device,
        data_info=data_info,
        save_list=save_list,
        dt=dt,
        original_dispersion_size=original_dispersion_size
    )

    if validation:
        path_save_fb = f"{path_to_save}/picked_{type_of_problem}_train_data"
    else:
        path_save_fb = f"{path_to_save}/picked_{type_of_problem}_test_data"
    if os.path.exists(path_save_fb):
        raise RuntimeError(f"Selected path for saving data, {path_save_fb} already exists.")
        # shutil.rmtree(path_save_fb)
    Path(path_save_fb).mkdir(parents=True)
    
    Dataset = get_predict_dataset(type_of_problem=type_of_problem)
    
    test_dataset = Dataset(
            data_dir=base_dir,
            data_info=data_info, 
            with_label=validation,
            file_names=save_list)
    
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    n_data = test_dataset.n_data
    
    shots = []
    predicted_picks = []
    predicted_segments = []
    fbt_file_names = []
    
    loop = tqdm(loader)
    with torch.no_grad():
        if validation:
            true_masks = []
            for shot_number, (batch, true_mask, fbt_file_name) in enumerate(loop):
                fbt_file_name = fbt_file_name[0]
                
                shot1, predicted_pick, predicted_segment, true_mask1 = predict_validation(
                    batch=batch, 
                    model=model,
                    true_mask=true_mask,
                    split_nt=split_nt,
                    overlap=overlap,
                    shot_id=fbt_file_name,
                    smoothing_threshold=smoothing_threshold,
                    # upsampler=upsampler,
                    data_info=data_info,
                    case_specific_parameters=case_specific_parameters
                )
                
                shots.append(shot1)
                predicted_picks.append(predicted_pick)
                predicted_segments.append(predicted_segment)
                fbt_file_names.append(fbt_file_name)
                true_masks.append(true_mask1)

        else:
            for shot_number, (batch, fbt_file_name) in enumerate(loop):
                fbt_file_name = fbt_file_name[0]
                
                # nsp:  data.shape=[1, 3, 1, 512, 22]
                batch, _ = upsampler(batch.squeeze(0), batch.squeeze(0))
                #nsp:  data.shape= [3, 1, 512, 512])
                
                shot, predicted_pick, predicted_segment = predict_test(
                    batch=batch.unsqueeze(0), 
                    model=model,
                    split_nt=split_nt,
                    overlap=overlap,
                    shot_id=fbt_file_name,
                    smoothing_threshold=smoothing_threshold,
                    # upsampler=upsampler,
                    data_info=data_info,
                    case_specific_parameters=case_specific_parameters
                )
                
                shots.append(shot)
                predicted_picks.append(predicted_pick)
                predicted_segments.append(predicted_segment)
                fbt_file_names.append(fbt_file_name)
                
                # if shot_number == n_data - 1:
                #     break
            true_masks = [None]
        
        show_prediction(shots,
                              predicted_segments,
                              predicted_picks,
                              true_masks = true_masks,
                              path_save_fb=path_save_fb,
                              dt=dt,
                              ffids=fbt_file_names,
                              save_segmentation=save_segmentation,
                              x_axis=x_axis,
                              y_axis=y_axis)
        plt.close()
    

if __name__ == "__main__":
    from default_values import *
    from data.data_segy2npy import shot_to_npy
    import numpy as np
    import pandas as pd 
    from first_break_picking.tools import seed_everything
    
    seed_everything(seed=10)
    (data_dir, checkpoint, n_trace,
        split_nt, overlap, n_time_sampels, upsampled_size,
        smoothing_threshold, 
        n_subshots, device, model_name, phase) = get_pred_args()
    
    smoothing_threshold = 50
    dt = 0.00025

    checkpoint = None 
    project_name = "mtq"
    phase = "test"
    
    split_nt: int = 30
    n_subshots: int = 7 # Important for prediction
    overlap: float = 0.15
    n_time_sampels: int = 512
    upsampled_size = 256
    model_name: str = "unet_resnet"

    data_dir = f"/Users/amir/repos/fbp_edf/data_files/preprocessed/{project_name}/{phase}"

    data_info = pd.read_csv(data_dir + f"_data_info.txt", converters={'shot_id': str})
    
    predict(data_dir, 
        checkpoint_dir=checkpoint,
        split_nt=split_nt, 
        overlap=overlap, 
        n_time_sampels=n_time_sampels, 
        width_enlarged_subshot=upsampled_size,
        dt=dt,
        smoothing_threshold=smoothing_threshold,
        data_info=data_info, 
        model_name="unet_resnet",
        validation=True if phase == "train" else False,
        save_list=None
        )
    
    plt.show()





