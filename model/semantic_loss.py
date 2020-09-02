import torch.nn.functional as F
import torch
import numpy as np

class_num = 19.


def nll_loss(output, target):
    return F.nll_loss(F.log_softmax(output, 1), target)


def cross_entropy_loss(output, target):
    #print(torch.argmax(output, 0).size())
    out = F.cross_entropy(output, target, reduction='mean', ignore_index=255)
    return out


def focal_loss(output, target, gamma=1):
    tweight = torch.from_numpy(np.load("/mnt/datagrid/personal/paplhjak/PapLab/weights.npy")).to(output.device).float()
    prediction_prob = F.softmax(output, dim=1)

    # cross entropy part
    result = F.cross_entropy(output, target, reduction='none', weight=tweight, ignore_index=255)[:, None, ...]

    # new part
    loss_weight = (1 - prediction_prob.gather(1, target[:, None, ...])) ** gamma

    # final form
    loss = loss_weight * result

    return torch.mean(loss)


def focal_loss_no_weights(output, target, gamma=1): 
    prediction_prob = F.softmax(output, dim=1)

    # cross entropy part
    #target[target==255] = 1
    result = F.cross_entropy(output, target, reduction='none', ignore_index=255)[:, None, ...]
    #print(target.unique())
    #print(result.unique())
    # new part
    loss_weight = (1 - prediction_prob.gather(1, target[:, None, ...])) ** gamma
    loss_weight[target[:, None, ...]==255] = 0
    # print(loss_weight)
    # final form
    loss = loss_weight * result

    return torch.mean(loss)