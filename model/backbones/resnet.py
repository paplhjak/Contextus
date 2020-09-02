import torch
import torch.nn as nn
from enum import Enum

from model.backbones.functional import MaskedConvScaling
from model.backbones.ContextWeights import LaplacianContextWeights
from model.backbones.ContextWeights import GaussianContextWeights
from model.backbones.ContextWeights import InvL2ContextWeights
from model.backbones.multi_input_sequential import miSequential
from downsample.bilinear import Net as bilinear_downsample

from model.backbones.LCMC import LCMC
from pacnet.pac import PacConv2d
# from torchvision.models.utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ConvType(Enum):
    """Types of convolution implementation."""
    lcmc = 'LCMC'
    pac = 'PAC'
    standard = 'standard'


class ContextType(Enum):
    """Types of context."""
    laplacian = 'laplacian'
    gaussian = 'gaussian'
    dist = 'invdistance'


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def LCMCconv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, scaling=MaskedConvScaling.unit):
    """3x3 LCMC convolution with padding"""
    return LCMC(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups,
                bias=False,
                scaling=scaling)


def PACconv3x3(in_planes, out_planes, stride=1, dilation=1, normalize=True):
    return PacConv2d(in_planes, out_planes, 3, stride=stride, padding=dilation, bias=False, dilation=dilation,
                     normalize_kernel=normalize)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, convolution=ConvType.lcmc.value, scaling=MaskedConvScaling.unit):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.convolution = convolution

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # Choose from convolution implementations
        if self.convolution == ConvType.standard.value:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        elif self.convolution == ConvType.lcmc.value:
            self.conv2 = LCMCconv3x3(width, width, stride, groups, dilation, scaling)
        elif self.convolution == ConvType.pac.value:
            self.conv2 = PACconv3x3(width, width, stride, dilation, normalize=(scaling != MaskedConvScaling.none))

        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, unfolded_context):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # Choose from convolution implementations
        if self.convolution == ConvType.standard.value:
            out = self.conv2(out)
        elif self.convolution == ConvType.lcmc.value:
            out = self.conv2(out, unfolded_context)
        elif self.convolution == ConvType.pac.value:
            out = self.conv2(out, unfolded_context)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, unfolded_context

class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=[False, True, True],
                 norm_layer=None, convolution_type=ConvType.lcmc.value, scaling=MaskedConvScaling.unit, in_channels=1,
                 context_type=ContextType.laplacian, sigma=200.0,
                 learnable_sigma=True):

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.convolution_type = convolution_type
        self.convolution_scaling = scaling
        self.context_type = context_type

        """
        conv 1
        """
        if self.convolution_type == ConvType.standard.value:
            self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif self.convolution_type == ConvType.lcmc.value:
            self.conv1 = LCMC(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                              bias=False, scaling=self.convolution_scaling)
        elif self.convolution_type == ConvType.pac.value:
            self.conv1 = PacConv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False, normalize_kernel=(scaling != MaskedConvScaling.none))
        if self.context_type == ContextType.laplacian.value:
            self.conv1_context_fetcher = LaplacianContextWeights(kernel_size=7, stride=2, padding=3, dilation=1,
                                                                 initial_sigma=sigma,
                                                                 learnable_sigma=learnable_sigma)
        elif self.context_type == ContextType.gaussian.value:
            self.conv1_context_fetcher = GaussianContextWeights(kernel_size=7, stride=2, padding=3, dilation=1,
                                                                initial_sigma=sigma,
                                                                learnable_sigma=learnable_sigma)
        elif self.context_type == ContextType.dist.value:
            self.conv1_context_fetcher = InvL2ContextWeights(kernel_size=7, stride=2, padding=3, dilation=1,
                                                             initial_a=sigma, learnable_a=learnable_sigma)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        """
        layer 1
        """
        self.pre_layer1_context_fetcher = self._make_pre_layer_context_fetcher(sigma=sigma,
                                                                               learnable_sigma=learnable_sigma)
        self.pre_layer1 = self._make_pre_layer(block, 64, convolution=self.convolution_type,
                                               scaling=self.convolution_scaling)
        self.layer1 = self._make_layer(block, 64, layers[0], convolution=self.convolution_type,
                                       scaling=self.convolution_scaling)

        """
        layer 2
        """
        self.pre_layer2_context_fetcher = self._make_pre_layer_context_fetcher(stride=2,
                                                                               dilate=replace_stride_with_dilation[0],
                                                                               sigma=sigma,
                                                                               learnable_sigma=learnable_sigma)
        self.layer2_context_fetcher = self._make_pre_layer_context_fetcher(sigma=sigma,
                                                                           learnable_sigma=learnable_sigma)
        self.pre_layer2 = self._make_pre_layer(block, 128, stride=2,
                                               dilate=replace_stride_with_dilation[0],
                                               convolution=self.convolution_type, scaling=self.convolution_scaling)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], convolution=self.convolution_type,
                                       scaling=self.convolution_scaling)

        """
        layer 3
        """
        self.pre_layer3_context_fetcher = self._make_pre_layer_context_fetcher(stride=2,
                                                                               dilate=replace_stride_with_dilation[1],
                                                                               sigma=sigma,
                                                                               learnable_sigma=learnable_sigma)
        self.layer3_context_fetcher = self._make_pre_layer_context_fetcher(sigma=sigma,
                                                                           learnable_sigma=learnable_sigma)
        self.pre_layer3 = self._make_pre_layer(block, 256, stride=2,
                                               dilate=replace_stride_with_dilation[1],
                                               convolution=self.convolution_type,
                                               scaling=self.convolution_scaling)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], convolution=self.convolution_type,
                                       scaling=self.convolution_scaling)

        """
        layer 4
        """
        self.pre_layer4_context_fetcher = self._make_pre_layer_context_fetcher(stride=2,
                                                                               dilate=replace_stride_with_dilation[2],
                                                                               sigma=sigma,
                                                                               learnable_sigma=learnable_sigma)
        self.layer4_context_fetcher = self._make_pre_layer_context_fetcher(sigma=sigma,
                                                                           learnable_sigma=learnable_sigma)
        self.pre_layer4 = self._make_pre_layer(block, 512, stride=2,
                                               dilate=replace_stride_with_dilation[2],
                                               convolution=self.convolution_type,
                                               scaling=self.convolution_scaling)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], convolution=self.convolution_type,
                                       scaling=self.convolution_scaling)

        """
        context scales 2**x
        """
        self.scales = [0, 2, (2 + ([not i for i in replace_stride_with_dilation])[0]),
                       (2 + sum(([not i for i in replace_stride_with_dilation])[0:2])),
                       (2 + sum(([not i for i in replace_stride_with_dilation])[0:3]))]

        """
        context downsampling network
        """
        self.context_downsampler = bilinear_downsample()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_pre_layer(self, block, planes, stride=1, dilate=False, convolution=ConvType.lcmc.value,
                        scaling=MaskedConvScaling.unit):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = miSequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        return block(self.inplanes, planes, stride, downsample, self.groups,
                     self.base_width, previous_dilation, norm_layer, convolution, scaling=scaling)

    def _make_pre_layer_context_fetcher(self, stride=1, dilate=False, sigma=200.0, learnable_sigma=True):
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if self.context_type == ContextType.laplacian.value:
            return LaplacianContextWeights(kernel_size=3, stride=stride, padding=previous_dilation,
                                           dilation=previous_dilation,
                                           initial_sigma=sigma, learnable_sigma=learnable_sigma)
        elif self.context_type == ContextType.gaussian.value:
            return GaussianContextWeights(kernel_size=3, stride=stride, padding=previous_dilation,
                                          dilation=previous_dilation,
                                          initial_sigma=sigma, learnable_sigma=learnable_sigma)
        elif self.context_type == ContextType.dist.value:
            return InvL2ContextWeights(kernel_size=3, stride=stride, padding=previous_dilation,
                                       dilation=previous_dilation,
                                       initial_a=sigma, learnable_a=learnable_sigma)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, convolution=ConvType.lcmc.value,
                    scaling=MaskedConvScaling.unit):
        norm_layer = self._norm_layer

        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
        #                    self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, convolution=convolution, scaling=scaling))

        return miSequential(*layers)

    def _forward_impl(self, x, context_source=None):
        guided = self.convolution_type != ConvType.standard.value
        if guided:
            contexts = self.context_downsampler(context_source)

        # See note [TorchScript super()]
        # print('conv1')
              
        if self.convolution_type == ConvType.lcmc.value:
            unfolded_context = self.conv1_context_fetcher(contexts[self.scales[0]])
            x = self.conv1(x, unfolded_context)
        elif self.convolution_type == ConvType.pac.value:
            x = self.conv1(x, contexts[self.scales[0]])
        else:
            x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)

        # print('maxpool')
        x = self.maxpool(x)

        # print('layer1')

        if self.convolution_type == ConvType.lcmc.value:
            unfolded_context = self.pre_layer1_context_fetcher(contexts[self.scales[1]])
            x, _ = self.pre_layer1(x, unfolded_context)
            x, _ = self.layer1(x, unfolded_context)
        elif self.convolution_type == ConvType.pac.value:
            x, _ = self.pre_layer1(x, contexts[self.scales[1]])
            x, _ = self.layer1(x, contexts[self.scales[1]])
        else:
            x, _ = self.pre_layer1(x, None)
            x, _ = self.layer1(x, None)

        # print('layer2')

        if self.convolution_type == ConvType.lcmc.value:
            unfolded_context = self.pre_layer2_context_fetcher(contexts[self.scales[1]])
            x, _ = self.pre_layer2(x, unfolded_context)
            unfolded_context = self.layer2_context_fetcher(contexts[self.scales[2]])
            x, _ = self.layer2(x, unfolded_context)
        elif self.convolution_type == ConvType.pac.value:
            x, _ = self.pre_layer2(x, contexts[self.scales[1]])
            x, _ = self.layer2(x, contexts[self.scales[2]])
        else:
            x, _ = self.pre_layer2(x, None)
            x, _ = self.layer2(x, None)

        # print('layer3')

        if self.convolution_type == ConvType.lcmc.value:
            unfolded_context = self.pre_layer3_context_fetcher(contexts[self.scales[2]])
            x, _ = self.pre_layer3(x, unfolded_context)
            unfolded_context = self.layer3_context_fetcher(contexts[self.scales[3]])
            x, _ = self.layer3(x, unfolded_context)
        elif self.convolution_type == ConvType.pac.value:
            x, _ = self.pre_layer3(x, contexts[self.scales[2]])
            x, _ = self.layer3(x, contexts[self.scales[3]])
        else:
            x, _ = self.pre_layer3(x, None)
            x, _ = self.layer3(x, None)

        # print('layer4')

        if self.convolution_type == ConvType.lcmc.value:
            unfolded_context = self.pre_layer4_context_fetcher(contexts[self.scales[3]])
            x, _ = self.pre_layer4(x, unfolded_context)
            unfolded_context = self.layer4_context_fetcher(contexts[self.scales[4]])
            x, _ = self.layer4(x, unfolded_context)
        elif self.convolution_type == ConvType.pac.value:
            x, _ = self.pre_layer4(x, contexts[self.scales[3]])
            x, _ = self.layer4(x, contexts[self.scales[4]])
        else:
            x, _ = self.pre_layer4(x, None)
            x, _ = self.layer4(x, None)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x, context):
        return self._forward_impl(x, context)


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)

    return model


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3],
                   **kwargs)


def resnet101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3],
                   **kwargs)


def resnet152(**kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3],
                   **kwargs)


if __name__ == '__main__':
    model = resnet50().cuda()
    inp = torch.ones(1, 1, 1024, 128, requires_grad=True).cuda()
    con = torch.ones(1, 3, 1024, 128, requires_grad=True).cuda()

    a = model(inp, con).norm().backward()
