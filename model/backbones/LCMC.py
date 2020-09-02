from __future__ import division
import torch
from torch import nn
from model.backbones.functional import masked_conv2d, MaskedConvScaling


class LCMC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 scaling=MaskedConvScaling.unit):
        super(LCMC, self).__init__()

        assert in_channels % groups == 0, "in_channels of Conv2d must be divisible by groups."
        assert out_channels % groups == 0, "out_channels of Conv2d must be divisible by groups."

        self.groups = groups
        self.scaling = scaling

        self.in_channels = in_channels
        self.out_channels = out_channels

        # kernel_size
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "expected kernel_size to contain 2 elements"
            assert kernel_size[0] == kernel_size[1], "expected kernel_size to be of size N x N"
            self.kernel_size = kernel_size[0]
        else:
            assert isinstance(kernel_size, int), "expected kernel_size to be of type int"
            self.kernel_size = kernel_size

        # stride
        if isinstance(stride, tuple):
            assert len(stride) == 2, "expected stride to contain 2 elements"
            assert stride[0] == stride[1], "expected stride to be of size N x N"
            self.stride = stride[0]
        else:
            assert isinstance(stride, int), "expected stride to be of type int"
            self.stride = stride

        # padding
        if isinstance(padding, tuple):
            assert len(padding) == 2, "expected padding to contain 2 elements"
            assert padding[0] == padding[1], "expected padding to be of size N x N"
            self.padding = padding[0]
        else:
            assert isinstance(padding, int), "expected padding to be of type int"
            self.padding = padding

        # dilation
        if isinstance(dilation, tuple):
            assert len(dilation) == 2, "expected dilation to contain 2 elements"
            assert dilation[0] == dilation[1], "expected dilation to be of size N x N"
            self.dilation = dilation[0]
        else:
            assert isinstance(dilation, int), "expected dilation to be of type int"
            self.dilation = dilation

        # bias
        if bias is True:
            self.bias = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))
            self.register_parameter('bias', self.bias)
        else:
            self.bias = None

        # kernel weights
        self.weights = nn.Parameter(
            torch.zeros(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size),
            requires_grad=True)
        self.register_parameter('weight', self.weights)

        # xavier uniform initialization of weights and biases
        self.init_weights()

    def init_weights(self):
        if self.bias is not None:
            torch.nn.init.xavier_uniform_(self.bias)
        torch.nn.init.xavier_uniform_(self.weights)

    def init_weights_ones(self):
        if self.bias is not None:
            torch.nn.init.ones_(self.bias)
        torch.nn.init.ones_(self.weights)

    def forward(self, input, unfolded_mask):
        output = masked_conv2d(input, self.weights, unfolded_mask, padding=self.padding, groups=self.groups,
                               dilation=self.dilation,
                               stride=self.stride, bias=self.bias, scaling=self.scaling)

        return output
