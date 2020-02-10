import torchvision.models as models
import torch.nn as nn


class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(
            block=models.resnet.Bottleneck,
            layers=[3, 4, 6, 3],
            groups=32,
            width_per_group=4,
        )
        # Override the existing FC layer with a new one.
        self.fc = nn.Linear(2048, 2)

    def get_mean_std(self):
        return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def get_image_size(self):
        return (224, 224)


def get_model(num_classes=2):
    model = MyResNeXt(True)
    for parameter in model.named_parameters():
        parameter[1].requires_grad = True
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters is {pytorch_total_params}")
    return model
