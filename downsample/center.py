import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, channels):
        self.channels = channels

        self.kernel = torch.zeros(self.channels, 1, 3, 3).cuda()
        for i in range(self.channels):
            self.kernel[i, 0, 1, 1] = 1.

        super(Net, self).__init__()

    # Defining the forward pass
    def forward(self, x):
        down_1 = torch.nn.functional.conv2d(x, self.kernel, bias=None, stride=2, padding=1, dilation=1,
                                            groups=self.channels)
        down_2 = torch.nn.functional.conv2d(down_1, self.kernel, bias=None, stride=2, padding=1, dilation=1,
                                            groups=self.channels)
        down_3 = torch.nn.functional.conv2d(down_2, self.kernel, bias=None, stride=2, padding=1, dilation=1,
                                            groups=self.channels)
        down_4 = torch.nn.functional.conv2d(down_3, self.kernel, bias=None, stride=2, padding=1, dilation=1,
                                            groups=self.channels)
        down_5 = torch.nn.functional.conv2d(down_4, self.kernel, bias=None, stride=2, padding=1, dilation=1,
                                            groups=self.channels)
        return x, down_1, down_2, down_3, down_4, down_5
