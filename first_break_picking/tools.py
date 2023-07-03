import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from typing import Optional, Union, Tuple, List
from matplotlib.patches import Polygon
import torch 
import random
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from first_break_picking.data.data_utils import get_test_data
from first_break_picking.data import data_utils as data_tools
import json
import logging
from matplotlib.widgets import Button

 
def seed_everything(seed: int) -> None:
    """
    It set a seed for all  packages to guarantee the reproducibility of
    results. 

    Parameters
    ----------
    seed : int
        An integer value for seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
  
def save_predictions_as_image(
    loader: DataLoader,
    model,
    folder: str,
    device: torch.device,
    n_classes: int
    ) -> None:
    
    model.eval()
    for i, (img, mask) in enumerate(loader):
        img = img.unsqueeze(1).to(device=device)
        
        out = model(img)
        with torch.no_grad():
            if n_classes == 2:
                preds = torch.sigmoid(out)
                preds = (preds > 0.5).float()
            else:
                preds = torch.softmax(out, dim=1)
                preds = torch.argmax(preds, dim=1).unsqueeze(1).float()
            
        torchvision.utils.save_image(
            preds, fp=f"{folder}/pred_{i}.png"
        )
            
        torchvision.utils.save_image(
            mask, fp=f"{folder}/true_{i}.png"
        )
    
    
      
def plotseis(ax,
             data: np.ndarray,
             picking: Optional[np.ndarray] = None,
             add_picking: Optional[np.ndarray] = None,
             normalizing: Union[str, int] = 'entire',
             clip: float = 0.9,
             ampl: float = 1.0,
             patch: bool = True,
             colorseis: bool = False,
             wiggle: bool = True,
             background: Optional[np.ndarray] = None,
             colorbar: bool = False,
             dt: float = 1.0,
             show: bool = True) -> matplotlib.figure.Figure:
    """
    plotseis for plotting seismic data
    
    This function belongs to Aleksei Tarasov
    https://github.com/DaloroAT

    _extended_summary_

    Parameters
    ----------
    data : np.ndarray
        _description_
    picking : Optional[np.ndarray], optional
        _description_, by default None
    add_picking : Optional[np.ndarray], optional
        _description_, by default None
    normalizing : Union[str, int], optional
        _description_, by default 'entire'
    clip : float, optional
        _description_, by default 0.9
    ampl : float, optional
        _description_, by default 1.0
    patch : bool, optional
        _description_, by default True
    colorseis : bool, optional
        _description_, by default False
    wiggle : bool, optional
        _description_, by default True
    background : Optional[np.ndarray], optional
        _description_, by default None
    colorbar : bool, optional
        _description_, by default False
    dt : float, optional
        _description_, by default 1.0
    show : bool, optional
        _description_, by default True

    Returns
    -------
    matplotlib.figure.Figure
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    num_time, num_trace = np.shape(data)

    if normalizing == 'indiv':
        norm_factor = np.max(np.abs(data), axis=0)
        norm_factor[np.abs(norm_factor) < 1e-9 * np.max(np.abs(norm_factor))] = 1
    elif normalizing == 'entire':
        norm_factor = np.tile(np.max(np.abs(data)), (1, num_trace))
    elif np.size(normalizing) == 1 and not None:
        norm_factor = np.tile(normalizing, (1, num_trace))
    elif np.size(normalizing) == num_trace:
        norm_factor = np.reshape(normalizing, (1, num_trace))
    else:
        raise ValueError('Wrong value of "normalizing"')

    data = data / norm_factor * ampl

    mask_overflow = np.abs(data) > clip
    data[mask_overflow] = np.sign(data[mask_overflow]) * clip

    data_time = np.tile((np.arange(num_time) + 1)[:, np.newaxis], (1, num_trace)) * dt

    # fig, ax = plt.subplots()

    plt.xlim((0, num_trace + 1))
    plt.ylim((0, num_time * dt))
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    if wiggle:
        data_to_wiggle = data + (np.arange(num_trace) + 1)[np.newaxis, :]

        ax.plot(data_to_wiggle, data_time,
                color=(0, 0, 0))

    if colorseis:
        if not (wiggle or patch):
            ax.imshow(data,
                      aspect='auto',
                      interpolation='bilinear',
                      alpha=1,
                      extent=(1, num_trace, (num_time - 0.5) * dt, -0.5 * dt),
                      cmap='gray')
        else:
            ax.imshow(data,
                      aspect='auto',
                      interpolation='bilinear',
                      alpha=1,
                      extent=(-0.5, num_trace + 2 - 0.5, (num_time - 0.5) * dt, -0.5 * dt),
                      cmap='gray')

    if patch:
        data_to_patch = data
        data_to_patch[data_to_patch < 0] = 0

        for k_trace in range(num_trace):
            patch_data = ((data_to_patch[:, k_trace] + k_trace + 1)[:, np.newaxis],
                          data_time[:, k_trace][:, np.newaxis])
            patch_data = np.hstack(patch_data)

            head = np.array((k_trace + 1, 0))[np.newaxis, :]
            tail = np.array((k_trace + 1, num_time * dt))[np.newaxis, :]
            patch_data = np.vstack((head, patch_data, tail))

            polygon = Polygon(patch_data,
                              closed=True,
                              facecolor='black',
                              edgecolor=None)
            ax.add_patch(polygon)

    if picking is not None:
        ax.plot(np.arange(num_trace) + 1, picking * dt,
                linewidth=2,
                color='blue', linestyle="-")

    if add_picking is not None:
        ax.plot(np.arange(num_trace) + 1, add_picking * dt,
                linewidth=2,
                color='green', linestyle="--")

    if background is not None:
        bg = ax.imshow(background,
                       aspect='auto',
                       extent=(0.5, num_trace + 1 - 0.5, (num_time - 0.5) * dt, -0.5 * dt),
                       cmap='YlOrRd')

        if colorbar:
            plt.colorbar(mappable=bg)

    if show:
        plt.show()
    return ax
    

def plot_picks(shot, predicted, truth=None,
               ax=None,
               band_size: int=0
               ):
    if ax is None:
        fig, ax = plt.subplots()

    n_t, n_traces = shot.shape

    if band_size > 0:
        pad = torch.where(predicted == 2, 1, 0)
        t = torch.argmax(pad, dim=0)
        pad_flipped = torch.flipud(pad)
        t2 = n_t - torch.argmax(pad_flipped, dim=0)
        ax = plotseis(ax, shot, picking=t,
                     add_picking=t2,
                     background=truth,
                      show=False)
    else:
        t = predicted.max(dim=0)[1]
        t2 = torch.argmax(truth, dim=0)

        ax = plotseis(ax, shot,
                    picking=t, add_picking=t2, background=truth, 
                             show=False)


def load_test_val(path: str,
                n_traces: int,
                height_model: int,
                test: bool= False):
    """
    laod files in test/validatioon folder and creats a batch 

    Parameters
    ----------
    path : str
        _description_
    n_traces : int
        _description_
    height_model : int
        _description_
    test : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    files = os.listdir(path)
    try:
        files.remove(".DS_Store")
    except:
        pass
    data = torch.empty((len(files), 1,
                        height_model, n_traces))
    
    for i, file in enumerate(files):
        npy = np.load(file=path+"/"+file, allow_pickle=True)
        if test: # file has just shot
            data[i, 0, ...] = get_test_data(npy.astype(np.float32))
        else:  # File has both shot and picks
            data[i, 0, ...] = get_test_data(npy[0].astype(np.float32))

    return data


def read_npy_list(file: str, 
                n_traces: int):
       
    npy = np.load(file=file)
    n_sources = npy.shape[1] // n_traces 

    data = [npy[:, :n_traces]]
    for i in range(1, n_sources):
        data.append(npy[:, i*n_traces:(i+1)*n_traces])
    return data


def _sta_lta(shot: np.ndarray,
         window_size: int,
        n_trace: int,
        n_samples: int,
        betta: float
        ):
    ratio = np.zeros((n_samples, n_trace))
    for trace_num in range(n_trace):
        i = 0
        temp = np.zeros(n_samples)
        while i + window_size < n_samples:
            
            s = np.sum(shot[i:i + window_size, trace_num] ** 2)
            l = np.sum(shot[: i + window_size, trace_num] ** 2)
           
            temp[i] = np.divide(s, l + betta, where=l!=np.nan)
            i += 1
        non_zero = temp[temp!=0]
        numbers = np.isnan(non_zero)
        
        if len(non_zero) ==sum(numbers):
            temp[:] = np.nan
            
        ratio[:, trace_num] = temp
    return ratio


def sta_lta(shot: np.ndarray,
            window_size:int,
            betta: float = 0.01,
            threshold: float = 0.1) -> np.ndarray:
    """
    This function performs Short Time Average over Long Time Average (STA/LTA) method

    Parameters
    ----------
    shot : np.ndarray
        A seismic shot
    window_size : int
        Length of window
    betta : float
        A stabilization constant that helps reduce the rapid fluctuations of ratio
    threshold: float
        A threshold for detecting the first break
        
    Returns
    -------
    np.ndarray
        masked version of picks
        
    Reference:
    ----------
    Sabbione J. and D. Velis, 2010, 
    Automatic first-breaks picking: New strategies and algorithms, 
    Geophysics, doi: 10.1190/1.3463703
    """
    n_trace = shot.shape[1]
    n_samples = shot.shape[0]
    
    ratio = _sta_lta(shot=shot,
                     window_size=window_size,
                     n_trace=n_trace,
                     n_samples=n_samples,
                     betta=betta)
    
    diff = np.diff(ratio, axis=0)
    diff[np.isnan(diff)] = 0
    # diff = data_normalize_and_limiting(diff)
    pick = (diff >= threshold).argmax(axis=0)
    
    ind = np.where(np.sum(diff, axis=0) ==0)[0]
    skept_pick = pick[ind]

    for i in range(len(skept_pick)):
        if skept_pick[i] == 0:
            pick[ind[i]] = -999
    masked = np.ma.masked_equal(pick, -999)
    return masked #, diff