import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    # Defining the forward pass
    def forward(self, x):
        kernel = torch.zeros(3,1,3,3)#.cuda()
        kernel[0,0,1,1]=1.
        kernel[1,0,1,1]=1.
        kernel[2,0,1,1]=1.
        down_1 = torch.nn.functional.conv2d(x, kernel, bias=None, stride=2, padding=1, dilation=1, groups=3)
        down_2 = torch.nn.functional.conv2d(down_1, kernel, bias=None, stride=2, padding=1, dilation=1, groups=3)
        down_3 = torch.nn.functional.conv2d(down_2, kernel, bias=None, stride=2, padding=1, dilation=1, groups=3)
        down_4 = torch.nn.functional.conv2d(down_3, kernel, bias=None, stride=2, padding=1, dilation=1, groups=3)
        down_5 = torch.nn.functional.conv2d(down_4, kernel, bias=None, stride=2, padding=1, dilation=1, groups=3)
        return down_1, down_2, down_3, down_4, down_5