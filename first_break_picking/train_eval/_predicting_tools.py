from typing import Optional, Union, Tuple, List
import matplotlib.pyplot as plt 
import pandas as pd
import torch 
import numpy as np
import json
import logging
from matplotlib.widgets import Button
import matplotlib.pyplot as plt 

from first_break_picking.data import data_utils as data_tools


def get_downsample_points(n_time_sampels: int,
                          split_nt: int,
                          n_trace: int, 
                          overlap: float) -> Tuple[torch.nn.Upsample, list]:
    downsample = torch.nn.Upsample(size=(n_time_sampels, split_nt))
    points = data_tools.starting_points(n_traces=n_trace, split_nt=split_nt, overlap=overlap)
    points.append(n_trace)
    
    return downsample, points


def predict(model,
            data: torch.Tensor, 
            binary: bool):
    """
    Predict the segments

    Parameters
    ----------
    model : UNet
        _description_
    data : torch.Tensor
        _description_

    Returns
    -------
    _type_
        segmented result
    """
    predicted_logit = model(data)

    if binary:
        prob = torch.sigmoid(predicted_logit.squeeze(1))
        # predicted = predicted[predicted>0.5]
        predicted = torch.argmax(torch.diff(prob, dim=1), dim=1)

    else:
        prob = torch.softmax(predicted_logit, dim=1)
        predicted = torch.argmax(prob, dim=1)
    return predicted, prob


def fb_predict_validation(
            batch: torch.Tensor,
            model,
            upsampler,
            split_nt: int,
            overlap: float,
            shot_id: str,
            smoothing_threshold: int,
            data_info: pd.DataFrame,
            true_mask: torch.Tensor,
            # case_specific_parameters: dict,
            case_specific_parameters
            ):
    n_original_time_sampels = case_specific_parameters["n_original_time_sampels"]
    
    n_trace = data_info.loc[shot_id][0]
    #TODO: This function should be mixed with predict_test
    device = "cpu"
    model = model.to(device=device)
    
    downsample, points = get_downsample_points(
            n_time_sampels = n_original_time_sampels,
            split_nt = split_nt,
            n_trace = n_trace,
            overlap = overlap
            )
    
    batch = batch.squeeze(0).to(device=device)
    n_subshots = batch.shape[0]
    
    batch, true_mask = upsampler(
        batch,
        true_mask.squeeze(0))
        
    predicted, prob = predict(model=model, data=batch,
                    binary=False)
    predicted = predicted.unsqueeze(1).to(device=device, dtype=float) # int doesn't work
    
    true_mask = downsample(true_mask.to(torch.float32)).squeeze(1)
    predicted = downsample(predicted).to(dtype=torch.int).squeeze(1)
    batch = downsample(batch).squeeze(1)
                
    predicted1 = torch.zeros((n_original_time_sampels, n_trace))
    true_mask1 = torch.zeros((n_original_time_sampels, n_trace))
    shot1 = torch.zeros((n_original_time_sampels, n_trace))
    for i in range(n_subshots):
                    
        predicted1[:, points[i]:points[i+1]] = predicted[i,:, :points[i+1] - points[i]]
        true_mask1[:, points[i]:points[i+1]] = true_mask[i,:, :points[i+1] - points[i]]
        shot1[:, points[i]:points[i+1]] = batch[i,:, :points[i+1] - points[i]]

    predicted_pick = _fb_smooth_result(
        predicted=predicted1, 
        n_trace=n_trace, 
        shot_id=shot_id,
        threshold=smoothing_threshold)
    
    return shot1, predicted_pick, predicted1, true_mask1
    
    
def fb_predict_test(
    batch: torch.Tensor,
    model,
    upsampler,
    split_nt: int,
    overlap: float,
    shot_id: str,
    smoothing_threshold: int,
    data_info: pd.DataFrame,
    case_specific_parameters: dict
            ):
    n_original_time_sampels = case_specific_parameters["n_original_time_sampels"]
    
    n_trace = data_info.loc[shot_id][0]
    device = "cpu"
    model = model.to(device=device)
    
    downsample, points = get_downsample_points(
        n_time_sampels = n_original_time_sampels,
        split_nt = split_nt,
        n_trace = n_trace,
        overlap = overlap
        )
    
    batch = batch.squeeze(0).to(device=device)
    n_subshots = batch.shape[0]
    
    batch, _ = upsampler(
        batch, batch)
    
    predicted, prob = predict( # Check why I use prob here but not for validation
        model=model, data=batch,
        binary=False)
                    
    prob1 = downsample(prob)
    predicted = torch.argmax(prob1, dim=1)
    batch = downsample(batch).squeeze(1)
                
    predicted1 = torch.zeros((n_original_time_sampels, n_trace))
    shot1 = torch.zeros((n_original_time_sampels, n_trace))

    for i in range(n_subshots):
        predicted1[:, points[i]:points[i+1]] = predicted[i,:, :points[i+1] - points[i]]
        shot1[:, points[i]:points[i+1]] = batch[i,:, :points[i+1] - points[i]]
        
    predicted_pick = _fb_smooth_result(
        predicted=predicted1, 
        n_trace=n_trace, 
        shot_id=shot_id,
        threshold=smoothing_threshold)
    
    return shot1, predicted_pick, predicted1


def _fb_smooth_result(
    predicted: torch.Tensor,
    n_trace: int,
    shot_id: str = None,
    threshold: int =50):
    """
    This function is used to not pick the sample with value of one in each trace
    as a first sttival. Using higher threshold cases more accurate result. However, 
    very larger thereshold can cause error.

    Parameters
    ----------
    predicted : torch.Tensor
        _description_
    n_trace : int
        _description_
    shot_id : str
        Name of shot (FFID)
    threshold : int, optional
        _description_, by default 50

    Returns
    -------
    pick : torch.Tensor[torch.int]
        Sample number of the picked first-arrival time 
    """
    
    pick = torch.zeros(n_trace, dtype=int)
    try:
        for i in range(n_trace):
            count = 0
            ones = torch.argwhere(predicted[:, i]==1).view(-1)
            while True:
                a = sum(predicted[ones[count]:ones[count]+threshold, i])
                b = sum(predicted[ones[count+1]:ones[count+1]+threshold, i])
                if a == b:
                    pick[i] = ones[count].item()
                    break
                else:
                    count += 1
                    
        return pick
    except:
        if shot_id:
            print(f"Not possible to pick an appropriate time for shot {shot_id}. "
                "You can use a smaller smoothing_threshold.")
        else:
            pass
        return None
    
def __setup_ax():
    fig, ax = plt.subplots(1,1)
    fig.subplots_adjust(bottom=0.2)
        
            # [left, bottom, width, height]
    axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    # TODO: Add exit button
    button_save = Button(axnext, 'Save',
                            color='lightgoldenrodyellow',
                            hovercolor='0.975')
    button_skip = Button(axprev, 'Skip',
                            color='lightgoldenrodyellow',
                            hovercolor='0.975')
    
    return fig, ax, button_save, button_skip

    
def fb_show_prediction(shots: List[torch.Tensor],
                    predicted_segments: List[torch.Tensor],
                    predicted_picks: List[torch.Tensor],
                    ffids: List[str],
                    path_save_fb: str,
                    dt: float,
                    true_masks: List[torch.Tensor]= [None],
                    save_segmentation: bool= False,
                    x_axis: np.ndarray=None,
                    y_axis: np.ndarray=None,) -> None:
    """
    A function to visulaize the result

    Parameters
    ----------
    shot : torch.Tensor
        A shot gather
    predicted_segment : torch.Tensor
        _description_
    predicted_pick : torch.Tensor
        _description_
    n : int
        _description_
    n_data : int
        _description_
    true_mask : torch.Tensor, optional
        _description_, by default None
    save_segmentation: bool, optional
        Specify if user desires to save the segmentation, by default False

    """
    n_shots = len(shots)
    for i in range(n_shots):
        (fig, ax, button_save, button_skip) = __setup_ax()
        
        m,n = shots[i].shape
        ax.imshow(shots[i], aspect="auto", cmap="gray",
                #   extent=[0, n, m*dt, 0]
                  )
        ax.imshow(predicted_segments[i], aspect="auto", 
                  cmap="coolwarm", alpha=0.1,
                #   extent=[0, n, m*dt, 0]
                  )
        try:
            ax.plot(predicted_picks[i], label="Predicted")
            if true_masks[0] is not None:
                ax.plot(torch.argmax(true_masks[i], dim=0), "--", label="True")
                ax.legend()
        except:
            logging.warning(f"Can't pick the first breaks for shot {ffids[i]}")
        
        ax.set_title(f"FFID: {ffids[i]}")
        ax.set_xlabel("Trace Number")
        ax.set_ylabel("Time Step")
            
        if not save_segmentation:
            predicted_segments[i] = None
            
        a = plt.waitforbuttonpress()
        if a is False:
            button_save.on_clicked(
                __save_fb(path_save_fb,
                    fbt_file_name=ffids[i],
                    n_trace=shots[i].shape[1],
                    predicted_segments=predicted_segments[i],
                    predicted_pick=predicted_picks[i],
                    dt=dt, comment=f"{i+1}/{n_shots}")
            )
        
            button_skip.on_clicked(
                skip(fbt_file_name=ffids[i], comment=f"{i+1}/{n_shots}")
                )

            plt.pause(0.5)
        #TODO : optimize here
        fig.clear()
        plt.close()
        

def skip(fbt_file_name: str,
         comment: str) :   
    def skipped(event):
        print(f"{comment}: Shot {fbt_file_name} is skipped")
    return skipped 


def __save_fb(path_save_fb: str,
            fbt_file_name: str,
             n_trace: int,
             predicted_segments: Union[torch.Tensor, bool],
             predicted_pick: torch.Tensor,
             dt: float,
             comment: str
             ) -> None:
    """
    This function saves first-break files as a json file

    Parameters
    ----------
    path_save_fb : str
        Path to save the first-break files
    fbt_file_name : str
        File name
    n_trace : int
        Number of traces
    predicted_segments : Union[torch.Tensor, bool]
        Predicted segmentation array
    predicted_pick : torch.Tensor
        First break whose dtype is torch.int
    dt : float
        Temporal sampling rate
    comment : str
        Comment on the total and processed shot
    """
    def clicked(event):
        save_fb(path_save_fb=path_save_fb,
            fbt_file_name=fbt_file_name,
             n_trace=n_trace,
             predicted_segments=predicted_segments,
             predicted_pick=predicted_pick,
             dt=dt,
             comment=comment)
        
    return clicked


def save_fb(path_save_fb: str,
            fbt_file_name: str,
             n_trace: int,
             predicted_pick: torch.Tensor,
             dt: float,
             comment: str,
             predicted_segments: torch.Tensor = None
             ) -> None:
    """
    This function saves first-break files as a json file

    Parameters
    ----------
    path_save_fb : str
        Path to save the first-break files
    fbt_file_name : str
        File name
    n_trace : int
        Number of traces
    predicted_pick : torch.Tensor
        First break whose dtype is torch.int
    dt : float
        Temporal sampling rate
    comment : str
        Comments to print (number of shot / number of all shots)
    predicted_segments : torch.Tensor, optional
        The predicted segentation to be saved, by default None
    """
    result = {
        fbt_file_name: {
            "xs": list(range(n_trace)),
            "ys": list(predicted_pick.numpy().astype(int) * dt)
            }
        }
    
    # Serializing json
    json_object = json.dumps(result, indent=4)
    
    # Save json file
    with open(path_save_fb + f"/{fbt_file_name}.json", "w") as outfile:
        outfile.write(json_object)
    print(f"{comment}: Shot {fbt_file_name} is saved")
    if predicted_segments is not None:
        np.save(path_save_fb + f"/{fbt_file_name}_seg.npy", predicted_segments)