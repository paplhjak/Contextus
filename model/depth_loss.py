import torchvision
import torch
import numpy as np
import math

EPS = 1e-6

def masked_mse_loss(output, target):
    valid_mask = (target > 0).detach()
    diff = target - output
    diff = diff[valid_mask]
    loss = (diff ** 2).mean()
    return loss


def masked_l1_loss(output, target):
    valid_mask = (target > 0).detach()
    diff = target - output
    diff = diff[valid_mask]
    loss = diff.abs().mean()
    return loss

def masked_silog_loss(output, target):
    output = torch.nn.functional.relu(output)
    valid_mask = (target > 0).detach()
    num = torch.sum(valid_mask, dim=(1,2,3), keepdim=True)
    di = torch.log(output+EPS) - torch.log(target+EPS)
    # mask out invalid values
    di[target<=0]=0
    D1 = torch.sum(torch.pow(di,2), dim=(1,2,3), keepdim=True)
    D2 = torch.pow(torch.sum(di, dim=(1,2,3), keepdim=True),2)
    D = 1/num.float() * D1 - 1/torch.pow(num.float(),2) * D2
    return D.mean()
    
def masked_irmse_loss(predictions, target):
    valid_mask = (target > 0).detach()
    num = valid_mask.sum().item()
    predictions = predictions[valid_mask]
    target = target[valid_mask]
    inv_predictions = 1 / predictions
    inv_target = 1 / target
    abs_inv_diff = (inv_predictions - inv_target).abs()
    return torch.sqrt((torch.pow(abs_inv_diff, 2)).mean())