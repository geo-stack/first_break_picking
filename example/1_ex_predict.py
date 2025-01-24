#%% ========== Loading required packages 
import os.path as osp
from pathlib import Path
import shutil

# import sys
# sys.path.append(osp.abspath(osp.join(__file__, "../../")))
from first_break_picking.data import save_shots_fb
from first_break_picking import predict


from config import  (split_nt,
                     overlap,
                     dt,
                     n_time_sampels,
                     upsampled_size,
                     num_epcohs,
                     )
#%% ============ Define paths ===============
project_path = osp.abspath(osp.join(__file__, "../"))
data_base_train = project_path + "/data/"

# Define all data files for picking
raw_data_path = data_base_train + "/raw/dataset1/test"

preprocessed_path = f"{data_base_train}/preprocessed/dataset1/test"

# Path(path_to_save).mkdir(exist_ok=True, parents=True)
Path(preprocessed_path).mkdir(exist_ok=True, parents=True)
#%% =================== Prepare data =============
data_info = save_shots_fb(
        dataset_dir=raw_data_path,
        dir_to_save=preprocessed_path,
        split_nt= split_nt,
        overlap = overlap,
        time_window=[0, n_time_sampels],
        fbt_file_header=1,
        fbt_time_column=0,
        scale=True,
        grayscale=True,
        dt=dt,
        shot_ext=".npy",
        fb_ext=None
        )
    
# %% ======== Show results ===========
checkpoint_path = project_path + f"/checkpoints/chp_fb_{num_epcohs}.tar"
print(f"checkpoint is loaded: {checkpoint_path}")

predict(base_dir=preprocessed_path, 
        path_to_save=data_base_train,
        checkpoint_path=checkpoint_path,
        split_nt=split_nt, 
        overlap=overlap, 
        upsampled_size_row=n_time_sampels, 
        upsampled_size_col=upsampled_size,
        dt=dt, 
        smoothing_threshold=50,
        data_info=data_info, 
        model_name="unet_resnet",
        validation=False,
        save_list=None,
        out_channels=2,
        save_segmentation=False
        )

shutil.rmtree(f"{data_base_train}/preprocessed")
