from base import BaseModel
import torch.nn.functional as F


def load_backbone(backbone, convolution, context, in_channels):
    if "resnet" in backbone["type"]:
        import model.backbones.resnet
        if "resnet50" == backbone["type"]:
            return model.backbones.resnet.resnet50(replace_stride_with_dilation=backbone["dilation"], convolution_type=convolution["type"],
                            scaling=convolution["scaling"], context_type=context["type"],
                            in_channels=in_channels, sigma=context["sigma"],
                            learnable_sigma=context["learnable_sigma"])
        if "resnet101" == backbone["type"]:
            return model.backbones.resnet.resnet101(replace_stride_with_dilation=backbone["dilation"], convolution_type=convolution["type"],
                             scaling=convolution["scaling"], context_type=context["type"],
                             in_channels=in_channels, sigma=context["sigma"],
                             learnable_sigma=context["learnable_sigma"])
        if "resnet152" == backbone["type"]:
            return model.backbones.resnet.resnet152(replace_stride_with_dilation=backbone["dilation"], convolution_type=convolution["type"],
                             scaling=convolution["scaling"], context_type=context["type"],
                             in_channels=in_channels, sigma=context["sigma"],
                             learnable_sigma=context["learnable_sigma"])


def load_classifier(classifier):
    if "fcn" in classifier["type"]:
        import model.classifier_heads.fcn
        return model.classifier_heads.fcn.FCNHead(2048, 1)


class DepthCompletion(BaseModel):
    def __init__(self, in_channels, context, convolution, backbone, classifier):
        super(DepthCompletion, self).__init__()
        """
        :param in_channels: int:
            number of input channels; 1 for depth input; 4 for cat(depth,rgb)
        :param context: dict:
            'type' - 'gaussian', 'laplacian'
            'sigma', - float
            'learnable sigma' - True, False
        :param convolution: dict:
            'type' - 'LCMC', 'PAC', "standard"
            'scaling' - 'none', 'unit', 'inputs'
        :param backbone: dict:
            'type' - 'resnet50', 'resnet101', 'resnet152'
            'dilation' - [bool bool bool]
        :param classifier: dict:
            'type' - 'fcn'
        """
        self.backbone = load_backbone(backbone, convolution, context, in_channels)
        self.classifier = load_classifier(classifier)

    def forward(self, x, context_source):
        input_shape = x.shape[-2:]
        features = self.backbone(x, context_source)
        depth = self.classifier(features)
        depth_ipol = F.interpolate(depth, size=input_shape, mode='bilinear', align_corners=False)
        return depth_ipol

    def load(self):
        return self
