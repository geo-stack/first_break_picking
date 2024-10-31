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
import torch 
from typing import Optional, Union, Tuple, List
import json
import logging
from matplotlib.widgets import Button
import matplotlib.pyplot as plt 

import first_break_picking.train_eval._predicting_tools as ptools
class Upsample:
    def __init__(self, size: Tuple[int, int]):
        
        self._upsampler = torch.nn.Upsample(size=size)
        self.size = self._upsampler.size
        
    def __call__(self, data, label):
        
        return (self._upsampler(data), 
                (self._upsampler(label)).long())
    
    
def normalize_metrics(metrics: dict) -> dict:
    """Normalize the loss in metrics dictionary

    Parameters
    ----------
    metrics : dict
        metrics dictionary

    Returns
    -------
    dict
        metrics dictionary
    """
    for key, value in metrics.items():
        if key.split("_")[1] == "loss":
            value = np.array(value)
            metrics[key] = value / value.max()
    return metrics


def save_checkpoint(
    model,
    file:str) -> None:
    """
    saves the checkpoints

    Parameters
    ----------
    model : 
        _description_
    file : str
        _description_
    """
    
    torch.save(model.state_dict(), file)
    print("== Checkpoint is saved! ==")
    
    
def load_checkpoint(model, 
                    file: str,
                    device: str) -> None:
    """
    Load checkpoint for selected model

    Parameters
    ----------
    model
        A DL network
    file : str
        Checkpoint's file
    device : str
        Name of device
    """
    try:
        state = torch.load(file, map_location=torch.device(device))
        model.load_state_dict(state) 
        print("=== Checkpoint is loaded! ===")
    except:
        raise RuntimeError("Please enter a valid checkpoint file.")
    

def setup_predictor(type_of_problem: str):
    predict_validation = ptools.fb_predict_validation
    predict_test = ptools.fb_predict_test
    show_prediction = ptools.fb_show_prediction
    return predict_validation, predict_test, show_prediction

