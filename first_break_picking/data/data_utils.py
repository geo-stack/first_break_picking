import random
from math import ceil
from typing import Tuple, List, Any, Union, Optional
from torch.utils.data import Dataset
import numpy as np
import matplotlib
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import os
import torchvision.transforms as t
from scipy.interpolate import interp1d
import logging
import pandas as pd

def data_normalize_and_limiting(data: np.ndarray) -> np.ndarray:
    """
    Trace based scale a shot

    Parameters
    ----------
    data : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    norma = np.max(np.abs(data), axis=0)
    norma[np.abs(norma) < 1e-9 * np.max(np.abs(norma))] = 1
    data = data / norma * 2
    data[data < -1] = -1
    data[data > 1] = 1
    return data


def test_transforms():
    return t.Compose([
                data_normalize_and_limiting,
                t.ToTensor()
                ])
    
    
def get_test_data(data: np.ndarray):
    transforms = test_transforms()
    return transforms(data).float()


def starting_points(n_traces: int,
                    split_nt: int,
                    overlap: float)-> List[int]:
    points = [0]
    stride = int(split_nt * (1 - overlap))
    count = 1
    if split_nt >= n_traces:
        return points
    
    while True:
        pt = count * stride
        if pt + split_nt >= n_traces:
            if pt == n_traces:
                break
            points.append(n_traces - split_nt)
            break
        else:
            points.append(pt)
        
        count += 1

    return points


def shot_spliting(shot: np.ndarray,
                  points: list,
                split_nt: int,
                )-> List[np.ndarray]:
    """
    Split a shot to subshots

    Parameters
    ----------
    shot : Union[np.ndarray, List]
        An ndarray if it is test data
        A list if it is train data
    split_nt : int
        Number of traces in splitted version
    overlap : float
        Overlap between subshots

    Returns
    -------
    List of subshots
        
    """
    data_shot = [shot[:, t:t+split_nt] for t in points]

    return data_shot#, data_fb


def fb_spliting(fb: np.ndarray,
                  points: list,
                split_nt: int,
                )-> List[np.ndarray]:
    """
    Split a shot to subshots

    Parameters
    ----------
    shot : Union[np.ndarray, List]
        An ndarray if it is test data
        A list if it is train data
    split_nt : int
        Number of traces in splitted version
    overlap : float
        Overlap between subshots

    Returns
    -------
    List of subshots
        
    """
    data_fb = [fb[t:t+split_nt] for t in points]
    return data_fb


def _exclude_hidden(files: List[str]):
    files = [file for file in files if not file.startswith(".")]
    return files


def read_files_name(path: str,
                extension: Optional[str]= None) -> List[str]:
    """
    Read all files in folder

    Parameters
    ----------
    path : str
        _description_
    """
    files: List[str] = os.listdir(path)

    if extension:
        new_files = [file for file in files if file.endswith(f".{extension}")]
        return new_files
    
    else:
        files = _exclude_hidden(files)
        return files


def read_npy(path: str):
    files = read_files_name(path=path, extension=None)
    return files


def _num_sub_images(files: List[str]) -> int:        
    n_subimage: int = len(np.unique([file.split("_")[1].split(".npy")[0] for file in files]))
    # shot_name: List[str] = [file.split("_")[0] for file in files]
    return n_subimage#, shot_name


def load_one_shot(dir, file_names: List[str]):
    shot = [np.load(dir + f"/{file}", allow_pickle=True) for file in file_names]
    return shot 


def group_shots(dir: str,
                files: List[str]):
    n_subimage: int = _num_sub_images(files)
    n_shots: int = len(files) // n_subimage

    files.sort()
    print(files)
    shots= []
    
    for i in range(0, n_shots, n_subimage):
        shot = load_one_shot(dir, files[i:i+n_subimage])
        
    return shot


def show_batch(batch_size: int, 
               img: torch.Tensor, 
               mask: torch.Tensor,
               suptitle: str = None):
    """
    Function to show a batch

    Parameters
    ----------
    batch_size : int
        Number of shots to be shown
    img : torch.Tensor
        Seismic shot
    mask : torch.Tensor
        Mask
    suptitle : str, optional
        Super title for plot, by default None
    """
    fig, axs = plt.subplots(batch_size, 2, figsize=(4, 2*batch_size))
    for i, ax in enumerate(axs.flatten()):
        if not (i+1) % 2 == 0:
            ax.imshow(img[i//2, 0, ...], cmap="gray", aspect="auto")
        else:
            ax.imshow(mask[i//2, 0, ...], cmap="coolwarm", aspect="auto")
            ax.set_yticks([])
        
        if i < 2 * (batch_size - 1):
             ax.set_xticks([])
        if i == 0:
            ax.set_title("Seismic Gather")
        elif i == 1:
            ax.set_title("Segemented Gather")
    fig.suptitle(suptitle)


'''
The oter functions are mostly coded by Aleksei Tarasov
    https://github.com/DaloroAT

'''
def split_dataset(path: Path, fracs: Tuple[int, int, int]) -> Tuple[List[Path], List[Path], List[Path]]:
    filenames = list(path.glob('*.npy'))
    train_num = ceil(len(filenames) * fracs[0])
    valid_num = ceil(len(filenames) * fracs[1])
    test_num = ceil(len(filenames) * fracs[2])

    if train_num + valid_num + test_num > len(filenames):
        raise ValueError('Invalid fracs')

    if 0 in [train_num, valid_num, test_num]:
        raise ValueError('Insufficient size of dataset for split with such fractions')

    shuffled_names = random.choices(filenames, k=len(filenames))

    train_set, valid_set, test_set = shuffled_names[:train_num], \
                                     shuffled_names[train_num:train_num + valid_num], \
                                     shuffled_names[train_num + valid_num:]
    return train_set, valid_set, test_set


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AvgMoving:
    n: int
    abg: float

    def __init__(self):
        self.n = 0
        self.avg = 0

    def add(self, val: float) -> None:
        self.n += 1
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg


class AvgMovingVector:
    n: np.ndarray
    avg: np.ndarray

    def __init__(self, num_elem: int):
        self.n = np.zeros(num_elem)
        self.avg = np.zeros(num_elem)

    def add(self, val: np.ndarray, idx: np.ndarray) -> None:
        if val.ndim == 1 and idx.ndim == 1 and np.shape(val) == np.shape(idx):
            self._add_one_vec(val, idx)
        if val.ndim == 2 and idx.ndim == 2 and np.shape(val) == np.shape(idx):
            for batch in range(np.shape(val)[0]):
                self._add_one_vec(val[batch, :], idx[batch, :])

    def _add_one_vec(self, val: np.ndarray, idx: np.ndarray) -> None:
        self.n[idx] += 1
        self.avg[idx] = val / self.n[idx] + (self.n[idx] - 1) / self.n[idx] * self.avg[idx]


class Stopper:
    max_wrongs: int
    n_obs_wrongs: int
    delta: float
    best_value: float

    def __init__(self, max_wrongs: int, delta: float):
        assert max_wrongs > 1 and delta > 0
        self.max_wrongs = max_wrongs
        self.n_obs_wrongs = 0
        self.delta = delta
        self.best_value = 0

    def update(self, new_value: float) -> None:
        if new_value - self.best_value < self.delta or new_value < self.best_value:
            self.n_obs_wrongs += 1
        else:
            self.n_obs_wrongs = 0
            self.best_value = new_value

    def is_need_stop(self) -> bool:
        return self.n_obs_wrongs >= self.max_wrongs


def sinc_interp(x: np.ndarray, t_prev: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    shape_x = np.shape(x)
    period = t_prev[1] - t_prev[0]

    if len(shape_x) == 1:
        t_prev = np.reshape(t_prev, (1, np.size(t_prev)))
        t_new = np.reshape(t_new, (1, np.size(t_new)))
        time_matrix = np.tile(t_new, (len(t_prev), 1)) - np.tile(t_prev.transpose(), (1, len(t_new)))
        return np.dot(x, np.sinc(time_matrix / period))
    elif shape_x[0] == 1 and shape_x[1] > 1:
        t_prev = np.reshape(t_prev, (1, np.size(t_prev)))
        t_new = np.reshape(t_new, (1, np.size(t_new)))
        time_matrix = np.tile(t_new, (len(t_prev), 1)) - np.tile(t_prev.transpose(), (1, len(t_new)))
        return np.dot(x, np.sinc(time_matrix / period))
    elif shape_x[0] > 1 and shape_x[1] == 1:
        t_prev = np.reshape(t_prev, (np.size(t_prev), 1))
        t_new = np.reshape(t_new, (np.size(t_new), 1))
        time_matrix = np.tile(t_new, (1, len(t_prev))) - np.tile(t_prev.transpose(), (len(t_new), 1))
        return np.dot(np.sinc(time_matrix / period), x)


def sinc_interp_factor(x: np.ndarray, factor: int) -> np.ndarray:
    num_elem = np.max(np.shape(x))
    t_prev = np.linspace(0, num_elem - 1, num_elem)
    t_new = np.linspace(0, num_elem - 1, (num_elem - 1) * factor + 1)
    return sinc_interp(x, t_prev, t_new)


def plotseis(data: np.ndarray,
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

    fig, ax = plt.subplots()

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
                linewidth=1,
                color='blue')

    if add_picking is not None:
        ax.plot(np.arange(num_trace) + 1, add_picking * dt,
                linewidth=1,
                color='green')

    if background is not None:
        bg = ax.imshow(background,
                       aspect='auto',
                       extent=(0.5, num_trace + 1 - 0.5, (num_time - 0.5) * dt, -0.5 * dt),
                       cmap='YlOrRd')

        if colorbar:
            plt.colorbar(mappable=bg)

    if show:
        plt.show()
    return fig


def plotseis_batch(data_batch: np.ndarray,
                   picking_batch: Optional[np.ndarray] = None,
                   add_picking_batch: Optional[np.ndarray] = None,
                   normalizing: Union[str, int] = 'entire',
                   clip: float = 0.9,
                   ampl: float = 1.0,
                   patch: bool = True,
                   colorseis: bool = False,
                   wiggle: bool = True,
                   background_batch: Optional[np.ndarray] = None,
                   colorbar: bool = False,
                   dt: float = 1,
                   show: float = True) -> matplotlib.figure.Figure:

    *num_batch, num_time, num_trace = np.shape(data_batch)
    assert len(num_batch) == 1

    num_batch = num_batch[0]

    fig = plt.figure()

    if num_batch == 1:
        num_col = 1
        num_row = 1

    else:
        num_col = np.floor(np.sqrt(num_batch))
        num_col = int((num_col if num_col == np.sqrt(num_batch) else num_col + 1))
        num_row = np.array(num_batch // num_col)
        num_row = int((num_row if num_batch / num_col == num_row else num_row + 1))

    gs = fig.add_gridspec(num_row, num_col)

    for batch in range(num_batch):
        idx = np.unravel_index(batch, (num_row, num_col), order='C')
        ax = fig.add_subplot(gs[idx])
        data = data_batch[batch, :]

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

        plt.xlim((0, num_trace + 1))
        plt.ylim((0, num_time * dt))
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        if wiggle:
            data_to_wiggle = data + (np.arange(num_trace) + 1)[np.newaxis, :]

            ax.plot(data_to_wiggle, data_time,
                    color=(0, 0, 0))

        if colorseis:
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

        if picking_batch is not None:
            picking = picking_batch[batch, :]
            if picking is not None:
                ax.plot(np.arange(num_trace) + 1, picking * dt,
                        linewidth=1,
                        color='blue')

        if add_picking_batch is not None:
            add_picking = add_picking_batch[batch, :]
            if add_picking is not None:
                ax.plot(np.arange(num_trace) + 1, add_picking * dt,
                        linewidth=1,
                        color='green')

        if background_batch is not None:
            background = background_batch[batch, :]
            if background is not None:
                bg = ax.imshow(background,
                               aspect='auto',
                               extent=(0.5, num_trace + 1 - 0.5, (num_time - 0.5) * dt, -0.5 * dt),
                               cmap='Wistia')

                if colorbar:
                    plt.colorbar(mappable=bg)
    if show:
        plt.show()
    return fig


def delete_ds_store(files: List[str])->List[str]:
    """
    It deletes '.DS_Store in a list

    Parameters
    ----------
    files : List[str]
        A list of files in a folder

    Returns
    -------
    List[str]
        The same list as inoput without '.DS_Store'
    """
    try:
        files.remove(".DS_Store")
    except:
        pass
    return files


def make_mask(dispersion_array: np.ndarray,
              dispersion_curve: np.ndarray) -> np.ndarray:
    """
    Create mask for a dispersion array and dispersion curve

    Parameters
    ----------
    dispersion_array : np.ndarray
        _description_
    dispersion_curve : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    mask = np.ones_like(dispersion_array)
    f = interp1d(dispersion_curve[:,1], dispersion_curve[:,0])
    freqs = np.arange(dispersion_curve[:,1].min(), dispersion_curve[:,1].max(), dtype=np.int32)
    vels = f(freqs)
    vels = vels.astype(np.int32)
    
    for i, freq in enumerate(freqs):
        mask[freq, :vels[i]] = 0
    return mask


def shot_to_gray_scale(data: np.ndarray) -> np.ndarray:
    """
    Transform a shot to grayscale image in [0, 255]

    Parameters
    ----------
    data : np.ndarray
        One seismic shot

    Returns
    -------
    np.ndarray
        The shot in gray scale
    """
    nom = data - np.mean(data, axis=0)
    denom = 2 * (np.max(data, axis=0) - np.min(data, axis=0))
    data = 255 * (nom/ denom + 0.5)
    data[np.isnan(data)] = 0
    return data


def save_dispersion_array_curve(data_path_loading: str,
                                picks_file_path: str,
                                data_path_saving: str,
                                dv: float,
                                df: float,
                                vmin: float, 
                                vmax: float,
                                fmin: float, 
                                fmax: float,
                                save_pair:bool=False) -> None:
    """
    Save one array for dispersion array and one array for dispersion curve

    Parameters
    ----------
    data_path_loading : str
        Path to load dispersion arrays (multiple files)
    picks_file_path : str
        Path to load json dispersion curves (one file)
    data_path_saving : str
        Path to save dispersion arrays  and curves (one file for each ffid)
    dv : float
        sampling rate of velocity for creating the dispersin curve
    save_pair : bool, optional
        Users can select if they want to save dispersion array and 
        curve of one shot in one folder (`save_pair=True`) or separately 
        (`save_pair=False`), default is save_pair=True`.
    """
    
    # Create the directory to save the files
    if not Path(data_path_saving).exists():
        Path(data_path_saving).mkdir(parents=True)
    
    # Seperate to check the name of file
    picks_file_path = picks_file_path.split("/")
    picks_file_path[-1] = check_dispersion_curve_file_name(picks_file_path[-1])
    
    # Stick the files together
    picks_file_path = "/".join(picks_file_path)
    
    # Load all available curves as pd.DataFrame
    curves, ffids = load_available_dispersion_curves(picks_file_path)
    disp_files = os.listdir(data_path_loading)
    disp_files = delete_ds_store(disp_files)
    
    for ffid in ffids:
        disp_curve = load_dispersion_curve(curves=curves,
                                            dv=dv, df=df, ffid=ffid,
                                            vmin=vmin, vmax=vmax, 
                                            fmin=fmin, fmax=fmax)
        
        
        try: # in case we don't have dispersion array of a shot in a folder while its pick is in json file
            disp_array = np.load(f"{data_path_loading}/{ffid}.npy")
            
            if save_pair:
                disp_curve = disp_curve.to_numpy()
                
                __save_pairs(disp_array=disp_array,
                     disp_curve=disp_curve,
                     path=f"{data_path_saving}/{ffid}.npy")
                
            else:
                disp_curve.to_csv(f"{data_path_saving}/{ffid}_disp.txt",
                            index=False)
                np.save(f"{data_path_saving}/{ffid}.npy", disp_array)
            
        except:
            logging.warn(f" Dispersion array for FFID: {ffid} does not exist.")
               
                        
def __save_pairs(disp_array: np.ndarray,
                 disp_curve: pd.DataFrame,
                 path: str
                 ) -> None:
    data_sample = np.array([disp_array, disp_curve], dtype=object)
    np.save(path, data_sample)
             
     
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

    
def load_dispersion_curve(curves: pd.DataFrame, 
                          dv: float, 
                          df: float,
                          ffid: int,
                          vmin: float, vmax:float,
                          fmin: float, fmax: float):
    if not isinstance(ffid, int):
        raise RuntimeError (f"FFID should be integer, but got {type(ffid)}.")
    curve = curves[ffid]
    
    curve = np.vstack((curve["xs"], curve["ys"])).T # - [0, 2] 
    # I don't understand how seismate generate the curve but it works with [0, 2] here
    
    curve = pd.DataFrame(curve, 
                         columns=['velocity', "frequency"])
    
    # In case, dispersion is done using different parameters (frequency, velocity)
    # we might have some values beyond the dispersion curve
    curve = curve[(curve["velocity"] <= vmax) & (curve["velocity"] >= vmin)]
    curve = curve[(curve["frequency"] <= fmax) & (curve["frequency"] >= fmin)]
    
    return np.floor(curve) #  / [dv, df])
