import torch
from first_break_picking.data.dataset import get_loaders
from pathlib import Path
from typing import Any, List, Optional, Union
import os 
import pandas as pd
import matplotlib.pyplot as plt
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
package_path = os.path.abspath(os.path.join(__file__, "../"))
sys.path.append(package_path)
# import first_break_picking.tools as tools
import first_break_picking.train_eval.metrics as M
from first_break_picking.train_eval.metrics import check_accuracy
import first_break_picking.train_eval.parameter_tools as pt
import first_break_picking.train_eval.ai_tools as tools

def train(base_dir: str,
         batch_size: int, 
         val_percentage: float,
         epochs: int, 
         learning_rate: float, 
         device: torch.device,
         path_to_save: str, 
         save_frequency:int,
         upsampled_size_row: int,
         upsampled_size_col: int,
         type_of_problem: str="fb",
         loss_fn_name: str = "ce",
         model_name: str = "unet_resnet",
         checkpoint_path: Optional[str]=None,
         features: List = [16, 32, 64, 128],
         in_channels: int = 1, 
         out_channels: int = 2,
         encoder_weight: str = "imagenet",
         step_size_milestone: int = None,
         show: bool = False) -> None:
    """
    This function is the main function to be calld for training 

    Parameters
    ----------
    base_dir : str
        Directory of data
    height_model : int
        Number of time samples
    batch_size : int
        Batch size
    val_percentage : float
        Fraction of validation
    epochs : int
        Number of epochs
    learning_rate : float
        Learning rate
    upsampled_size : int
        Size of each subshot during training
    device : torch.device
        Device
    path_to_save : str
        Path to save the checkpoints
    save_frequency : int
        Frequency of saving checkpoints
    band_size : int, optional
        Size of the band if we consider a band on the first break, by default 0
    strip_weight : float, optional
        Weight of loss for band if we consider a band on the first break, by default 0.0
    loss_fn_name : str, optional
        Name of loss function, by default "ce"
    model_name : str, optional
        Name of desired network, by default "unet_resnet"
    checkpoint_path : Optional[str], optional
        Checkpointt address for loading, by default None
    features : List, optional
        Number of channels in each conv layer, by default [16, 32, 64, 128]
    n_channels : int, optional
        Number of input channels in iput shot, by default 1
    n_classes : int, optional
        Number of output channels, by default 2
    encoder_weight : str, by default imagenet
        Name of the weigths for initializing the network
    step_size_milestone : int, None
        Step size will be divided by 10 at `every step_size_milestone`, by default None which leads to a constant step size
    show : bool, optional
        If you need to show some sampels after training, by default False

    """
            
    # Setup logging
    metrics = pt.setup_report(epochs=epochs,
                   batch_size=batch_size,
                   learning_rate=learning_rate,
                   upsampled_size_row=upsampled_size_row,
                   val_percentage=val_percentage,
                   loss_fn_name=loss_fn_name,
                   upsampled_size_col=upsampled_size_col,
                   model_name=model_name,
                   step_size_milestone=step_size_milestone,
                   type_of_problem=type_of_problem,
                   log_name=f"{path_to_save}/job",
                   )
    
    (model, device, upsampler) = \
        pt.define_general_parmeters(
            upsampled_size_row=upsampled_size_row,
            upsampled_size_col=upsampled_size_col,
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            encoder_weight=encoder_weight,
            device=device
            )
    
    (step_size_milestone, 
     loss_fn , base_dir) = pt.define_train_parmeters(
         base_dir=base_dir,
         checkpoint_path=checkpoint_path,
         epochs=epochs, 
         loss_fn_name=loss_fn_name,
         model=model,
         device=device,
         step_size_milestone=step_size_milestone
    )
        
    path_to_save_checkpoints = path_to_save + "/checkpoints"
    Path(path_to_save_checkpoints).mkdir(parents=True, exist_ok=True)
    
    train_dl, val_dl = get_loaders(
        data_dir=base_dir,
        val_fraction=val_percentage,
        batch_size=batch_size, 
        problem=type_of_problem)
    
    optimizer, scaler, scheduler = pt.setup_optimizer(
        model=model, 
        learning_rate=learning_rate,
        step_size_milestone=step_size_milestone
    )
    
    train_plot = pt.TrainFigure(
        n_segments=out_channels,
        plot=show,
        figsize=(12, 5),
        type_of_problem=type_of_problem)
    
    for epoch in range(1, epochs+1):
        loss = _train(
            train_dl, model,
              optimizer=optimizer,
              loss_fn=loss_fn,
              scaler=scaler,
              device=device,
              n_classes=out_channels,
              upsampler=upsampler
              )
        
        metrics["train_loss"].append(loss)
        metrics["learning_rate"].append(optimizer.param_groups[0]["lr"])

        # save model
        if (epoch % save_frequency == 0) or (epoch==epochs):
            tools.save_checkpoint(model, path_to_save_checkpoints+f"/chp_{type_of_problem}_{epoch}.tar")
        
        # check accuracy
        (pixel_accuracy,
         dice_accuracy,
         loss_val) = check_accuracy(
            loader=val_dl, model=model,
            loss_fn=loss_fn,
            device=device,
            n_classes=out_channels,
            upsampler=upsampler
            )

        scheduler.step()
        
        metrics["valid_loss"].append(loss_val)
        metrics["dice_accuracy"].append(dice_accuracy)
        metrics["pixel_accuracy"].append(pixel_accuracy)
        
        model.eval()
        train_plot.plot(model=model,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        train_loss=metrics["train_loss"],
                        validation_loss = metrics["valid_loss"],
                        validation_accuracy=metrics["pixel_accuracy"],
                        device=device,
                        epoch=epoch,
                        upsampler=upsampler
                        )
        model.train()
        
        print(f"Epoch {epoch}:",
              f"\ntrain loss: {metrics['train_loss'][-1]}",
              f"\nvalidation loss: {metrics['valid_loss'][-1]}\n"
              )
        logging.info(f"Epoch {epoch} -> Training loss: {loss}, validation loss: {loss_val}, lr: {optimizer.param_groups[0]['lr']}")

    df = pd.DataFrame(metrics)
    df.to_csv(path_to_save_checkpoints + f"/metrics_{type_of_problem}.csv", index=False)
            
def _train(
    loader: DataLoader,
    model, 
    optimizer: torch.optim.Optimizer, 
    loss_fn,  
    scaler,
    device: torch.device,
    n_classes: int,
    upsampler: tools.Upsample
):
    n_loader = len(loader)
    loop = tqdm(loader)
    loss_epoch = 0.0
    model.train()
    for idx, (img, target, band_mask) in enumerate(loop):
        # img = upsampler(img)
        img, target = upsampler(img, target)
        img = img.to(device=device)
        target = target.to(device=device)
        # target = target.long()
        # band_mask = band_mask.to(device=device)

        with torch.cuda.amp.autocast():
            prediction = model(img)
            if n_classes == 1:
                loss = loss_fn(prediction, target.to(dtype=torch.float))
            else:
                loss = loss_fn(prediction, target.squeeze(1))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        loss_epoch += loss.item()
    return loss_epoch/n_loader


if __name__ == "__main__":
    from default_values import *
    SHOW = False
    
    tools.seed_everything(seed=10)
    (data_dir, n_trace, height_model, band_size,
    path_to_save, num_epochs, 
    batch_size, val_percentage,
    learning_rate, device,
    strip_weight, upsampled_size, loss_fn_name,
    model_name, load_model) = get_train_args()
    # data_dir = ["/Users/amir/repos/first_break_picking/data_files/preprocessed/amem/train",
    #             "/Users/amir/repos/first_break_picking/data_files/preprocessed/amem/test"]
    
    train(data_dir,
         n_trace = n_trace, 
         height_model = height_model, 
         batch_size = batch_size,
         val_percentage = val_percentage,
         epochs = num_epochs, 
         learning_rate = learning_rate,
         upsampled_size=upsampled_size, 
         device = device, 
         path_to_save = path_to_save, 
         save_frequency = SAVE_FREQ,
         loss_fn_name = loss_fn_name,
         model_name=model_name,
         checkpoint_path = load_model,
         features = FEATURES, 
         in_channels = N_CHANNELS, 
         out_channels = N_CLASSES,
         encoder_weight = None,
         step_size_milestone=None,
         show = SHOW)