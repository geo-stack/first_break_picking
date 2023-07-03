import os 
import logging 
import torch 
import segmentation_models_pytorch as smp
from typing import List, Tuple
from torch.optim import Adam 
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd 

from first_break_picking.train_eval.unet import UNet
import first_break_picking.train_eval.metrics as M
import first_break_picking.train_eval.ai_tools as tools

def setup_report(epochs: int,
                batch_size: int,
                learning_rate: int,
                upsampled_size_row: int,
                val_percentage: float,
                loss_fn_name: str,
                upsampled_size_col: int,
                model_name: str,
                step_size_milestone: str,
                type_of_problem: str,
                log_name: str)->None:
    # try:
    #     os.remove("job.log")
    # except:
    #     pass
    logging.basicConfig(filename=f"{log_name}.log", 
                        level=logging.INFO,
                        filemode='w',
                        format='%(levelname)s: %(message)s')
    if model_name is not None and not isinstance(model_name, str):
        model_name = "custom model"
        
    logging.info(f'''Start training:
                 Type of problem: {type_of_problem}
                 Epochs: {epochs}
                 Batch size: {batch_size}
                 Initial learning rate: {learning_rate},
                 Upsampled size (row): {upsampled_size_row}
                 Upsampled size (col); {upsampled_size_col}
                 val_percentage: {val_percentage}
                 Loss function: {loss_fn_name}
                 Model name: {model_name}
                 Step size milestone: {step_size_milestone}
                 ''')
    
    metrics = {
        "train_loss": [],
        "valid_loss": [],
        "dice_accuracy": [],
        "pixel_accuracy": [],
        "learning_rate": []
    }
    
    return metrics
    

def define_general_parmeters(upsampled_size_row: int,
                            upsampled_size_col: int,
                            model_name: str,
                             in_channels: int,
                             out_channels: int,
                             features: List[int],
                             encoder_weight: str,
                             device: str):
    
    _check_general_parmeters(
        encoder_weight,
        device=device,
        upsampled_size_row=upsampled_size_row,
        upsampled_size_col=upsampled_size_col
        )
    
    model = _define_model(model_name=model_name,
                          in_channels=in_channels,
                          out_channels=out_channels,
                          features=features,
                          encoder_weights=encoder_weight)
    upsampler = tools.Upsample(
        size=(upsampled_size_row, upsampled_size_col)
        )
    return model, device, upsampler


def define_predict_parameters_one_shot(model, 
                              checkpoint_path: str,
                              upsampled_size_row: int,
                              device: str,
                              type_of_problem: str,
                              dt: float,
                              original_dispersion_size: Tuple):
    
    if checkpoint_path is None:
        checkpoint_path = os.path.abspath(os.path.join(__file__, "../../saved_checkpoints/fb_20.tar"))
    
    tools.load_checkpoint(
        model=model, file=checkpoint_path,
        device=device)
    
    (predict_validation, predict_test, 
     show_prediction) = tools.setup_predictor(type_of_problem)
    
    case_specific_parameters = {
            "n_original_time_sampels": upsampled_size_row
        }
    dt = dt
        
    return (predict_validation, predict_test,
            case_specific_parameters, show_prediction, dt)
    

def define_predict_parameters(model, 
                              checkpoint_path: str,
                              device: str,
                              data_info: pd.DataFrame,
                              save_list: bool,
                              upsampled_size_row: int,
                              original_dispersion_size: Tuple[int],
                              dt:float):
    
    type_of_problem = "fb"
    data_info = data_info.set_index("shot_id")
    if save_list is not None:
        data_info =  data_info[data_info.index.isin(save_list)]
    dt = dt
    
    (predict_validation, predict_test,
     case_specific_parameters, show_prediction, dt) = define_predict_parameters_one_shot(
                model=model, 
                upsampled_size_row=upsampled_size_row,
                checkpoint_path=checkpoint_path,
                device=device,
                type_of_problem=type_of_problem,
                original_dispersion_size=original_dispersion_size, 
                dt=dt
            )
    
    return (data_info, type_of_problem,
            predict_validation, predict_test,
            case_specific_parameters, show_prediction,
            dt)


def define_train_parmeters(base_dir: List["str"],
                                   checkpoint_path:str,
                                   epochs: int,
                                   loss_fn_name: str,
                                   model, 
                                   device: str,
                                   step_size_milestone: int=None,
                                ):
    if checkpoint_path is not None:
        tools.load_checkpoint(model, checkpoint_path, device=device)
        
    if not isinstance(base_dir, list):
        base_dir = [base_dir]
        
    if step_size_milestone is None:
        step_size_milestone = epochs
        
    loss_fn = _set_loss(loss_fn_name=loss_fn_name)
    model = model.to(device)
    
    return (step_size_milestone, 
            loss_fn, base_dir)


def _set_loss(loss_fn_name: str):
    if loss_fn_name == "ce":
        loss_fn = torch.nn.CrossEntropyLoss(
            # weight = weight
            )
    elif loss_fn_name == "dice":
        loss_fn = M.DiceLoss()
        
    elif isinstance(loss_fn_name, str):
        raise NotImplementedError(
            "loss_fn must be either ce (cross entropy) or dice "
            f"while you entered {loss_fn_name}"
            )
    else:
        loss_fn = loss_fn_name
        logging.warning("Custom cross entropy is chosen. "
                        "Be aware that data should comply "
                        "with input of CrossEntropyLoss")
        
    return loss_fn

def _check_general_parmeters(encoder_weight: str,
                             device: str,
                             upsampled_size_row: int,
                             upsampled_size_col: int):
        
    if encoder_weight not in [None, "imagenet"]:
        raise RuntimeError(f"encoder_weights should be either None or imagenet, "\
            f"but got {encoder_weight}")
    
    _check_device(device=device)
    _check_upsample(upsampled_size_row=upsampled_size_row)
    _check_upsample(upsampled_size_col=upsampled_size_col)
    
def _check_upsample(**kwrgs: dict):
    key = [*kwrgs][0]
    value = kwrgs[key]
    if not (isinstance(value, int) and 
            value % 16 ==0):
        raise AssertionError(f"{key} should be divisible by 16, but got {value} as {type(value).__name__}.")
        
         
def _check_device(device: str):
    def __device_error(device: str):
        raise RuntimeError("Device should be either 'cpu', 'cuda' or 'mps', "\
                f"but {device} is entered.")
        
    if isinstance(device, str):
        if device not in ["cpu", "mps", "mps"]:
            __device_error(device)
            
    elif isinstance(device, torch.device):
        pass
    else:
        __device_error(device)


def _define_model(model_name: str,
                  in_channels: int,
                  out_channels: int,
                  features: List[int],
                  encoder_weights: str
                  ):
    
    if model_name == "unet":
        model = UNet(in_channels = in_channels,
                    classes = out_channels, 
                    features=features,
                    bilinear=False) 
               
    elif model_name == "unet_resnet": 
        model = smp.Unet(
            encoder_name="resnet34",        
            encoder_weights=encoder_weights,     
            in_channels=in_channels,              
            classes=out_channels,                    
            decoder_channels =features[::-1],
            encoder_depth=len(features)
        )
        
    elif model_name == "unetpp_resnet":
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",       
            encoder_weights=encoder_weights,     
            in_channels=in_channels,                  
            classes=out_channels,                  
            decoder_channels =features[::-1],
            encoder_depth=len(features)
        )
        
    return model

def setup_optimizer(
    model,
    learning_rate: float,
    step_size_milestone: int):
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
            step_size=step_size_milestone, gamma=0.1, verbose=False)
    
    return optimizer, scaler, scheduler


class TrainFigure:
    def __init__(self, 
                 plot: bool, # to keep the training module clean, I check it here if user asks to plot
                 n_segments: int,
                 type_of_problem: str,
                 **fig_kw) -> None:
        
        cmap = "gray"
        cmap_mask = "coolwarm"
        
        self.legend = None
        if plot:
            cmap = plt.get_cmap(cmap, n_segments)
            cmap_mask = plt.get_cmap(cmap_mask, n_segments)
            self.fig, self.ax = plt.subplots(2, 3, **fig_kw)
            
            self._im_valid = self.ax[0, 0].imshow(np.empty((2, 2)), 
                        aspect="auto", cmap=cmap_mask,
                        vmin=0, vmax=n_segments)
            
            self._im_valid_data = self.ax[1, 0].imshow(np.empty((2, 2)), 
                        aspect="auto", cmap="jet")
            
            self._im_valid_mask = self.ax[1, 0].imshow(np.empty((2, 2)), 
                        aspect="auto", cmap=cmap_mask,
                        vmin=0, vmax=n_segments, alpha=0.7)
            
            self.ax[0, 0].set_title("Validation")
            
            self._im_train = self.ax[0, 1].imshow(np.empty((2, 2)), 
                        aspect="auto", cmap=cmap_mask,
                        vmin=0, vmax=n_segments)
            self.ax[0, 1].set_title("Train")
            
            self._im_train_data = self.ax[1, 1].imshow(np.empty((2, 2)), 
                        aspect="auto", cmap="jet")
            self._im_train_mask = self.ax[1, 1].imshow(np.empty((2, 2)), 
                        aspect="auto", cmap=cmap_mask,
                        vmin=0, vmax=n_segments, alpha=0.7)

            self.ax[0, 2].grid()
            self.ax[0, 2].set_title("Loss")
            
            self.ax[1, 2].grid()
            self.ax[1, 2].set_title("Accuracy")
            self.__turn_off_axis()

            self.plot = self.__plot_results
            self.fig.subplots_adjust(hspace=0.23)
        else:
            # for avoiding if, this method exis
            self.__plot = self.__plot_skip

    def plot(self, model, 
             train_dl: DataLoader,
             val_dl: DataLoader,
             train_loss: List,
             validation_loss: List,
             validation_accuracy: List,
             device: str,
             epoch: int):
        
        self.__plot(model,
                    train_dl,
                        val_dl,
                        train_loss,
                        validation_loss,
                        validation_accuracy,
                        device,
                        epoch)
        
    def __plot_skip(self, 
                    model, 
             train_dl: DataLoader,
             val_dl: DataLoader,
             train_loss: List,
             validation_loss: List,
             validation_accuracy: List,
             device: str,
             epoch: int,
             upsampler):
        pass
        
    def __plot_results(self, model, 
             train_dl: DataLoader,
             val_dl: DataLoader,
             train_loss: List,
             validation_loss: List,
             validation_accuracy: List,
             device: str,
             epoch: int, 
             upsampler):
        
        with torch.no_grad():
            example_shot_val, _, __ = next(iter(val_dl))
            example_shot_train, _, __ = next(iter(train_dl))
            
            # example_shot_val = example_shot_val.to(device=device)
            example_shot_val, __ = upsampler(example_shot_val, _)
            example_val = model(example_shot_val.to(device=device))
            
            example_val = torch.argmax(torch.softmax(example_val, dim=1), dim=1).cpu()
            
            # example_shot_train = example_shot_train
            example_shot_train, __ = upsampler(example_shot_train, _)
            example_train = model(example_shot_train.to(device=device))
            
            example_train = torch.argmax(torch.softmax(example_train, dim=1), dim=1).cpu()
            
            self._im_valid.set_data(example_val[0, ...])
            self._im_train.set_data(example_train[0, ...])
            
            self._im_train_data.set_data(example_shot_train[0, 0, ...])
            self._im_train_data.set_clim(example_shot_train[0, 0, ...].min(), 
                                                example_shot_train[0, 0, ...].max())

            self._im_valid_data.set_data(example_shot_val[0, 0, ...])
            self._im_valid_data.set_clim(example_shot_val[0, 0, ...].min(), 
                                               example_shot_train[0, 0, ...].max())
            
            self._im_valid_mask.set_data(example_val[0, ...])
            self._im_valid_mask.set_clim(example_val[0, ...].min(), 
                                        example_val[0, ...].max())
            
            self._im_train_mask.set_data(example_train[0, ...])
            self._im_train_mask.set_clim(example_train[0, ...].min(), 
                                        example_train[0, ...].max())
            
            self.fig.suptitle(f"Epoch: {epoch}, train: {train_loss[-1]:2.4}, val: {validation_loss[-1]:2.4}")
                        
            loss_line = self.ax[0, 2].plot(train_loss, 
                                           marker="o", 
                                           color="b",
                                           label="Train")
            loss_line = self.ax[0, 2].plot(validation_loss, 
                                           marker="*", 
                                           color="r",
                                           label="Test")
            
            self.ax[1, 2].plot(validation_accuracy, marker="*", color="r")
            
            self.__set_legend(example_train[0, ...])
            
            plt.pause(.2)
    
    def __set_legend_inactive(self, a) :
        """ 
        This is used just to avoid using if when the legend is added
        """
        pass
    
    def __set_legend(self, data):
        ny, nx = data.shape
        
        self.legend = self.ax[0, 2].legend()
        self.__set_extent_axis(nx, ny)
        # Once the legend is created, this function is replaced by an inactive version
        self.__set_legend = self.__set_legend_inactive
        
    def  __set_extent_axis(self, nx, ny):
        self._im_valid.set_extent([0, nx, 0, ny])
        self._im_train.set_extent([0, nx, 0, ny])
        self._im_valid_data.set_extent([0, nx, 0, ny])
        self._im_train_data.set_extent([0, nx, 0, ny])
        self._im_valid_mask.set_extent([0, nx, 0, ny])
        self._im_train_mask.set_extent([0, nx, 0, ny])
        
    def  __turn_off_axis(self):
        self.ax[0, 0].set_xticklabels('')
        self.ax[0, 1].set_xticklabels('')
        self.ax[0, 1].set_yticklabels('')
        self.ax[0, 2].set_xticklabels('')
        self.ax[1, 1].set_yticklabels('')
        self.ax[1, 2].set_label("Epochs")
