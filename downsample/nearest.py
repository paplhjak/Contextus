import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    # Defining the forward pass
    def forward(self, x):
        down_1 = torch.nn.functional.interpolate(input=x, scale_factor=(0.5) ** (1), mode='nearest')
        down_2 = torch.nn.functional.interpolate(input=x, scale_factor=(0.5) ** (2), mode='nearest')
        down_3 = torch.nn.functional.interpolate(input=x, scale_factor=(0.5) ** (3), mode='nearest')
        down_4 = torch.nn.functional.interpolate(input=x, scale_factor=(0.5) ** (4), mode='nearest')
        down_5 = torch.nn.functional.interpolate(input=x, scale_factor=(0.5) ** (5), mode='nearest')

        return x, down_1, down_2, down_3, down_4, down_5
