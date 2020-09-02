import torch
import math
import numpy as np

EPS = 1e-7


def abs_diff(predictions, target):
    valid_mask = (target > 0)
    num = valid_mask.sum().item()
    predictions = predictions[valid_mask]
    target = target[valid_mask]
    return torch.sum((predictions - target).abs())


def abs_log_diff(predictions, target):
    valid_mask = (target > 0)
    num = valid_mask.sum().item()
    predictions = predictions[valid_mask]
    target = target[valid_mask]
    return torch.sum((torch.log(predictions + EPS) - torch.log(target + EPS)).abs())


def rmse(predictions, target):
    valid_mask = (target > 0)
    num = valid_mask.sum().item()
    predictions = predictions[valid_mask]
    target = target[valid_mask]
    abs_diff = (predictions - target).abs()
    return math.sqrt((torch.pow(abs_diff, 2)).mean())


def mae(predictions, target):
    valid_mask = (target > 0)
    num = valid_mask.sum().item()
    predictions = predictions[valid_mask]
    target = target[valid_mask]
    abs_diff = (predictions - target).abs()
    return abs_diff.mean()


def log_rmse(predictions, target):
    valid_mask = (target > 0)
    num = valid_mask.sum().item()
    predictions = predictions[valid_mask]
    target = target[valid_mask]
    abs_log_diff = (torch.log(predictions + EPS) - torch.log(target + EPS)).abs()
    return math.sqrt((torch.pow(abs_log_diff, 2)).mean())


def log_mae(predictions, target):
    valid_mask = (target > 0)
    num = valid_mask.sum().item()
    predictions = predictions[valid_mask]
    target = target[valid_mask]
    abs_log_diff = (torch.log(predictions + EPS) - torch.log(target + EPS)).abs()
    return abs_log_diff.mean()


def absrel(predictions, target):
    valid_mask = (target > 0)
    num = valid_mask.sum().item()
    predictions = predictions[valid_mask]
    target = target[valid_mask]
    abs_diff = (predictions - target).abs()
    return (abs_diff / target).mean()


def sqrel(predictions, target):
    valid_mask = (target > 0)
    num = valid_mask.sum().item()
    predictions = predictions[valid_mask]
    target = target[valid_mask]
    abs_diff = (predictions - target).abs()
    return (torch.pow(abs_diff, 2) / (torch.pow(target, 2))).mean()


def irmse(predictions, target):
    valid_mask = (target > 0)
    num = valid_mask.sum().item()
    predictions = predictions[valid_mask]
    target = target[valid_mask]
    inv_predictions = 1 / predictions
    inv_target = 1 / target
    abs_inv_diff = (inv_predictions - inv_target).abs()
    return math.sqrt((torch.pow(abs_inv_diff, 2)).mean())


def imae(predictions, target):
    valid_mask = (target > 0)
    num = valid_mask.sum().item()
    predictions = predictions[valid_mask]
    target = target[valid_mask]
    inv_predictions = 1 / predictions
    inv_target = 1 / target
    abs_inv_diff = (inv_predictions - inv_target).abs()
    return abs_inv_diff.mean()
    
def SILog(predictions, target):
    valid_mask = (target > 0)
    num = valid_mask.sum().item()
    predictions = predictions[valid_mask]
    target = target[valid_mask]
    error = torch.log(predictions) - torch.log(target)
    silog = torch.sqrt(torch.mean(torch.pow(error,2)) - torch.pow(torch.mean(error), 2))
    return silog