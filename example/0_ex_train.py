#%% ========== Loading required packages 
import pandas as pd
import sys
import os.path as osp
import matplotlib.pyplot as plt
from time import time
from pathlib import Path
import shutil
from matplotlib.ticker import MaxNLocator

sys.path.append(osp.abspath(osp.join(__file__, "../../")))
from first_break_picking import train
from first_break_picking.tools import seed_everything
from first_break_picking.data import save_shots_fb

from config import  (split_nt,
                     overlap,
                     dt,
                     n_time_sampels,
                     upsampled_size,
                     batch_size,
                     val_percentage,
                     num_epcohs,
                     step_size_milestone
                     )

seed_everything(10)

#%% ============ Define paths ===============
project_path = osp.abspath(osp.join(__file__, "../"))
data_base_train = project_path + "/data/"

preprocessed_path = [
    f"{data_base_train}/preprocessed/dataset1/train",
    # f"{data_base_train}/preprocessed/dataset2/train"
]
# Define all training data set
train_data_path = [
    data_base_train + "/raw/dataset1/train",
    # data_base + "/raw/dataset2/train"
    # etc
]

#%% =================== Prepare data =============

for i, subfolder in enumerate(preprocessed_path):
    Path(subfolder).mkdir(exist_ok=True, parents=True)

    data_info = save_shots_fb(
        dataset_dir=train_data_path[i],
        dir_to_save=subfolder,
        split_nt= split_nt,
        overlap = overlap,
        time_window=[0, n_time_sampels],
        fbt_file_header=1,
        fbt_time_column=0,
        scale=True,
        grayscale=True,
        dt=dt,
        shot_ext=".npy",
        fb_ext=".txt"
        )
#%% =================== Training =================
tic = time()
train(preprocessed_path, 
    upsampled_size_row=n_time_sampels,
    upsampled_size_col=upsampled_size,
    batch_size=batch_size, 
    val_percentage=val_percentage,
    epochs=num_epcohs, 
    learning_rate=1e-4, 
    type_of_problem="fb",
    device="mps",
    checkpoint_path=None,
    path_to_save= project_path,
    save_frequency=num_epcohs,
    show=True,
    step_size_milestone=step_size_milestone
    )
toc = time()
print(f"It took {toc - tic} s for {num_epcohs} epochs")

# %% ============= plot metrics =================
metrics_path = f"{project_path}/checkpoints/metrics_fb.csv"
metrics = pd.read_csv(metrics_path)

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(121)
ax.plot(metrics.train_loss, label="Training set", 
    color="k")
ax.plot(metrics.valid_loss, linestyle="--",
     color="k", label="Validation set", linewidth=3
     )
ax.grid()
ax.set_ylabel("Loss")
ax.set_xlabel("Epoch")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()

ax = fig.add_subplot(122)
ax.plot(metrics.pixel_accuracy, label="Pixel",
    color="k")
ax.set_ylabel("Validation Accuracy")
ax.set_xlabel("Epoch")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid()

shutil.rmtree(f"{data_base_train}/preprocessed")
plt.show()
