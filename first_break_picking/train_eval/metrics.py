import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Union
from torch import Tensor
import matplotlib.pyplot as plt

from first_break_picking.train_eval.unet import UNet
import first_break_picking.train_eval.ai_tools as tools



class BDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = F.sigmoid(inputs)       
        
        dice = dice_calculation(inputs, targets)

        return 1 - dice


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        n_classes = inputs.shape[1]
        
        inputs = torch.softmax(inputs, dim=1)
        targets = F.one_hot(targets, n_classes).permute(0, 3, 1, 2)

        dice = dice_calculation(inputs, targets)

        return 1 - dice


def check_accuracy(loader: DataLoader,
                   model: UNet,
                   device: torch.device,
                   loss_fn,
                   n_classes: int,
                   upsampler: tools.Upsample) -> Union[float,
                                     float, 
                                     float]:
                       
    n_loader = len(loader)
    
    num_corrects = 0
    num_pixels = 0
    dice_score = 0.0
    loss = 0.0
        
    model.eval()
    with torch.no_grad():
        for img, mask, band_mask in loader:
            img, mask = upsampler(img, mask)
            img = img.to(device=device)
            mask = mask.to(device=device)
            
            out = model(img)
            if n_classes == 1:
                # mask.unsqueeze_(1)
                loss += loss_fn(out, mask.to(torch.float32))
                preds = torch.sigmoid(out)
                preds = (preds > 0.5).float()
                num_corrects += torch.eq(preds, mask).sum()

                dice_score += dice_calculation(preds, mask)

            else:
                # mask = mask.to(dtype=torch.long)
                loss += loss_fn(out, mask.squeeze_(1))
                mask = F.one_hot(mask, n_classes).permute(0, 3, 1, 2)
                preds = F.one_hot(out.argmax(dim=1), n_classes).permute(0, 3, 1, 2)
                
                num_corrects += torch.eq(preds, mask).sum()

                dice_score += dice_calculation(preds, mask)

            num_pixels += torch.numel(preds)

            
        
        print(
            f"Got {num_corrects}/{num_pixels} with accuracy {num_corrects/num_pixels*100:.2f}"
            f"\nDice score: {dice_score/len(loader)}"
        )
    return ((num_corrects/num_pixels).item(), 
            (dice_score/n_loader).item(),
            (loss/n_loader).item())


def dice_calculation(outputs: Tensor,
                    label: Tensor):

    intersection = (outputs * label).sum()
    
    union_intersection = (outputs + label).sum()
    return 2 * intersection / (union_intersection  + 1e-8)

# def biou_computation(prediction: Tensor,
#                     labels: Tensor) -> Tensor:
#     max_outputs = torch.mean(torch.max(prediction, dim=0)[0])
#     prediction = torch.ge(prediction, 0.5 * max_outputs)

