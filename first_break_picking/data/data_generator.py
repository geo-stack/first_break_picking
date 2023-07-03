import numpy as np
from pathlib import Path
import argparse
from typing import Union
from argparse import ArgumentParser
from first_break_picking.data.seis_model import SeisModel
import os 
import shutil
from first_break_picking.utils import seed_everything
    
def generate(data_dir: str,
        num_data: int = 10,
        interp_signal: Union[int, int]=(4, 8),
        min_thickness: int=30,
        max_thickness: int=80,
        smoothness: int=5,
        height: int=1000,
        width: int=48,
        filename_signal: str ='',
        seed: int=1):
    """
    This function generates synthetic data

    Parameters
    ----------
    data_dir : str
        Directory to store models
    num_data : int, optional
        Number of generated model, by default 10
    interp_signal : Union, optional
        Two numbers with minimum and maximum factor of signal pulse interpolation, by default (4, 8)
    min_thickness : int, optional
        Minimum distance between layers in samples, by default 30
    max_thickness : int, optional
        Maximum distance between layers in samples, by default 80
    smoothness : int, optional
        Smoothness of the variability of the travel time curves of the waves, by default 5
    height : int, optional
        Height of model in samples, by default 1000
    width : int, optional
        Width of model in samples, by default 48
    filename_signal : str, optional
        Select a file with signal with specific name (without extension).'
                             'If nothing is specified, then a random file is selected with each generation, by default ''
    seed : int, optional
        Fix seeds for reproducibility, by default 1
    """
    
    seed_everything(seed)
    model_generator = SeisModel(interp_signal=interp_signal,
                                min_thickness=min_thickness,
                                max_thickness=max_thickness,
                                smoothness=smoothness,
                                height=height,
                                width=width,
                                filename_signal=filename_signal)

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    Path(data_dir).mkdir(exist_ok=False, parents=True)

    for k in range(num_data):
        model_and_picking = model_generator.get_random_model()
        model_and_picking = np.array(model_and_picking, dtype="object")
        np.save(data_dir + f'/model_{k}.npy', model_and_picking)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', dest='data_dir', type=Path,
                        default=os.path.abspath(os.path.join(__file__, "../../data_files/preprocessed/syn_data/train")),
                        help='Directory to store models')
    parser.add_argument('--num_data', dest='num_data', type=int,
                        default=10,
                        help='Number of generated model')

    parser.add_argument('--smoothness', dest='smoothness', type=int, default=5,
                        help='Smoothness of the variability of the travel time curves of the waves')
    parser.add_argument('--height', dest='height', type=int, default=1000,
                        help='Height of model in samples')
    parser.add_argument('--width', dest='width', type=int, default=24,
                        help='Width of model in samples')
    parser.add_argument('--min_thickness', dest='min_thickness', type=int, default=30,
                        help='Minimum distance between layers in samples')
    parser.add_argument('--max_thickness', dest='max_thickness', type=int, default=80,
                        help='Maximum distance between layers in samples')
    parser.add_argument('--interp_signal', dest='interp_signal', nargs=2, type=int, default=(4, 8),
                        help='Two numbers with minimum and maximum factor of signal pulse interpolation')
    parser.add_argument('--file_signal', dest='filename_signal', type=str, default='',
                        help='Select a file with signal with specific name (without extension).'
                             'If nothing is specified, then a random file is selected with each generation')
    parser.add_argument('--random_seed', dest='seed', type=int, default=1,
                        help='Fix seeds for reproducibility')
    return parser


if __name__ == '__main__':
    arg_parser = get_parser()
    args = arg_parser.parse_args()
    
    generate(args.data_dir,
         args.num_data,
        args.interp_signal,
        args.min_thickness,
        args.max_thickness,
        args.smoothness,
        args.height,
        args.width,
        args.filename_signal,
        args.seed
         )

