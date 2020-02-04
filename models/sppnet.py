import torch
from torch import nn
from torchvision import models
import math


class ResNet(nn.Module):
    def __init__(self, num_class=2, pretrained=True):
        super(ResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.num_class = num_class
        self.fc = nn.Linear(512 * 4, num_class)

    def conv_base(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        layer1 = self.resnet.layer1(x)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)
        return layer1, layer2, layer3, layer4

    def forward(self, x):
        layer1, layer2, layer3, layer4 = self.conv_base(x)
        x = self.resnet.avgpool(layer4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SPPNet(nn.Module):
    def __init__(self, num_class=2, pool_size=(1, 2, 6), pretrained=True):
        # Only resnet is supported in this version
        super(SPPNet, self).__init__()
        self.resnet = ResNet(num_class, pretrained)
        self.c = 2048

        self.spp = SpatialPyramidPool2D(out_side=pool_size)
        num_features = self.c * (
            pool_size[0] ** 2 + pool_size[1] ** 2 + pool_size[2] ** 2
        )
        self.classifier = nn.Linear(num_features, num_class)

    def forward(self, x):
        _, _, _, x = self.resnet.conv_base(x)
        x = self.spp(x)
        x = self.classifier(x)
        return x

    def get_mean_std(self):
        return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def get_image_size(self):
        return (224, 224)


class SpatialPyramidPool2D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(SpatialPyramidPool2D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        # batch_size, c, h, w = x.size()
        out = None
        for n in self.out_side:
            w_r, h_r = map(
                lambda s: math.ceil(s / n), x.size()[2:]
            )  # Receptive Field Size
            s_w, s_h = map(lambda s: math.floor(s / n), x.size()[2:])  # Stride
            max_pool = nn.MaxPool2d(kernel_size=(w_r, h_r), stride=(s_w, s_h))
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], -1)
            else:
                out = torch.cat((out, y.view(y.size()[0], -1)), 1)
        return out


def get_model(num_classes=2, pretrained=True):
    model = SPPNet(num_class=num_classes, pretrained=pretrained)
    if pretrained:
        for parameter in model.named_parameters():
            if parameter[0].startswith('resnet.resnet'):
                parameter[1].requires_grad = False
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters is {pytorch_total_params}")
    return model
