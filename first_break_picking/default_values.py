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
import os 
from argparse import ArgumentParser, Namespace
from typing import Optional, Union, List


LEARNING_RATE = 1e-4
DEVICE = (("cpu", "cuda")[torch.cuda.is_available()], 
        "mps")[torch.backends.mps.is_available()]
BATCH_SIZE: int = 15
NUM_EPOCHS: int = 10
SAVE_FREQ = NUM_EPOCHS
NUM_WORKER = 2
N_CLASSES: int = 2
N_CHANNELS: int = 1
OUT_CHANNELS: int = N_CLASSES
PIN_MEMORY = True
LOAD_MODEL: str = None 
VAL_PERCENTAGE: float = 0.1
UPSAMPLED_SIZE = 256
FEATURES: List[int] = [16, 32, 64, 128] 
N_TRACES: int = 48
N_SUBSHOTS: int = 4
OVERLAP: float = 0.15
N_TIME_SAMPLES: int = 512
SPLIT_NT: int = 17
BAND_SIZE: int = 0
STRIP_WEIGHT: float = 0.0
HEIGHT_MODEL: int = 1000
LOSS_FN: str = "ce"
MODEL = "unet_resnet"

DATA_DIR: str = os.path.abspath(os.path.join(__file__, "../../data_files/raw_data/synthetic/train"))
PATH_TO_SAVE: str = os.path.abspath(os.path.join(__file__, "../../"))


def get_train_args():
    
    parser = ArgumentParser()
    
    parser.add_argument(
        "--data_dir", dest="data_dir",
        help="Data directory",
        default=DATA_DIR
    )

    parser.add_argument(
        "--load_model", dest="load_model",
        help="Checkpoint directory to load",
        default=LOAD_MODEL
    )

    parser.add_argument(
       "--result_dir", dest="result_dir",
       help="Directory to save checkpoints, loss, and some examples",
       default=PATH_TO_SAVE
    )
    
    parser.add_argument(
        "--epochs", dest="epochs",
        help="Number of epochs for training",
        default=NUM_EPOCHS,
        type=int
    )
    
    parser.add_argument(
        "--batch_size", dest="batch_size",
        help="Size of batches",
        default=BATCH_SIZE,
        type=int
    )
    
    parser.add_argument(
        "--val_percentage", dest="val_percentage",
        help="Relative size of validation set in percentage [0, 1.0]",
        default=VAL_PERCENTAGE,
        type=float
    )
    
    parser.add_argument(
        "--learning_rate", dest="learning_rate",
        help="Learning rate",
        default=LEARNING_RATE,
        type=float
    )
    
    parser.add_argument(
        "--device", dest="device", 
        help="Name of device for training (cpu, cuda)",
        default=DEVICE,
        choices=["cpu", "cuda", "mps"]
    )
    
    parser.add_argument(
        "--n_traces", dest="n_trace",
        help="Number of traces in one shot gather",
        default=N_TRACES,
        type=int)
    
    parser.add_argument(
        "--band_size", dest="band_size",
        help="Height of the narrow layer in number of samples",
        default=BAND_SIZE,
        type=int)

    parser.add_argument(
        "--strip_weight", dest="strip_weight",
        help="Regularization weight for the strip around the first break",
        type=float, default=STRIP_WEIGHT) 
    
    parser.add_argument(
        "--upsampled_size", dest="upsampled_size",
        help="Size of upsampled shot",
        type=int, default=UPSAMPLED_SIZE) 
    
    parser.add_argument(
        "--height_model", dest="height_model",
        help="Number ot time sample in a shot",
        type=int, default=HEIGHT_MODEL)   

    parser.add_argument(
        "--loss_fn", dest="loss_fn",
        help="Type of loss function",
        type=str, default=LOSS_FN,
        choices=["ce", "dice", "binary"]
        ) 
    
    parser.add_argument(
        "--model_name", dest="model_name",
        help="Type of loss function",
        type=str, default=MODEL,
        choices=["unet", "unet_resnet", "unetpp_resnet"]
        ) 

    args = parser.parse_args()
    return check_args_train(args)


def check_args_train(args: Namespace):
    data_dir = args.data_dir
    path_to_save = args.result_dir
    epochs = args.epochs
    batch_size = args.batch_size
    val_percentage = args.val_percentage
    learning_rate = args.learning_rate
    device = args.device
    n_trace = args.n_trace
    band_size = args.band_size
    strip_weight = args.strip_weight
    upsampled_size = args.upsampled_size
    height_model = args.height_model
    loss_fn = args.loss_fn
    load_model = args.load_model
    model_name = args.model_name

    return (data_dir, n_trace, height_model,
            band_size, path_to_save, epochs,
            batch_size, val_percentage,
            learning_rate, device, strip_weight,
            upsampled_size, loss_fn, model_name, load_model)


def get_pred_args():
    
    parser = ArgumentParser()
    
    parser.add_argument(
        "--data_dir", dest="data_dir",
        help="Data directory",
        default=DATA_DIR
    )

    parser.add_argument(
        "--checkpoint", dest="checkpoint",
        help="Checkpoint directory to load",
        default=LOAD_MODEL
    )
    
    parser.add_argument(
        "--n_subshots", dest="n_subshots",
        help="Number of subshots",
        default=N_SUBSHOTS,
        type=int
    )
    
    parser.add_argument(
        "--device", dest="device", 
        help="Name of device for training (cpu, cuda)",
        default=DEVICE,
        choices=["cpu", "cuda", "mps"]
    )
    
    parser.add_argument(
        "--n_traces", dest="n_trace",
        help="Number of traces in one shot gather",
        default=N_TRACES,
        type=int)
    
    parser.add_argument(
        "--split_nt", dest="split_nt",
        help="Number of traces in each subshot",
        default=SPLIT_NT,
        type=int)
    
    parser.add_argument(
        "--overlap", dest="overlap",
        help="Overlap of each subshot",
        default=OVERLAP,
        type=float)
    

    parser.add_argument(
        "--n_time_sampels", dest="n_time_sampels",
        help="Number of time samples in each shot",
        type=int, default=N_TIME_SAMPLES) 
    
    parser.add_argument(
        "--upsampled_size", dest="upsampled_size",
        help="Size of upsampled shot",
        type=int, default=UPSAMPLED_SIZE) 
    
    parser.add_argument(
        "--model_name", dest="model_name",
        help="Type of loss function",
        type=str, default=MODEL,
        choices=["unet", "unet_resnet"]
        )
    
    parser.add_argument(
        "--phase", dest="phase",
        help="Validation or test (with label or without label)",
        type=str, default="validation",
        choices=["validation", "test"]
        ) 
    
    parser.add_argument(
        "--smoothing", dest="smoothing_threshold",
        help="Smoothing value for prediction",
        type=float, default=30.0
        )

    args = parser.parse_args()
    return check_args_predict(args)


def check_args_predict(args: Namespace):
    data_dir = args.data_dir
    checkpoint = args.checkpoint
    n_trace = args.n_trace
    split_nt = args.split_nt
    n_time_sampels = args.n_time_sampels
    upsampled_size = args.upsampled_size
    n_subshots = args.n_subshots
    device = args.device
    model_name = args.model_name
    phase = args.phase
    overlap = args.overlap
    smoothing_threshold = args.smoothing_threshold
    
    if phase == "validation":
        phase = True
    else:
        phase = False
        
    return (data_dir, checkpoint, n_trace,
            split_nt, overlap, n_time_sampels, upsampled_size,
            smoothing_threshold,
            n_subshots, device, model_name, phase)