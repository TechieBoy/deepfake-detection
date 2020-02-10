import pretrainedmodels
import torch.nn as nn


class PretrainedModel:
    def __init__(self, model_name, num_classes, dataset):
        self.model = pretrainedmodels.__dict__[model_name](
            num_classes=1000, pretrained=dataset
        )
        dim_feats = self.model.last_linear.in_features
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.last_linear = nn.Linear(dim_feats, num_classes)
        self.model.get_image_size = self.get_image_size
        self.model.get_mean_std = self.get_mean_std

    def forward(self, input):
        return self.model(input)

    def get_mean_std(self):
        return (self.model.mean, self.model.std)

    def get_image_size(self):
        return tuple(self.model.input_size[1:])

    def get_model_colorspace(self):
        return self.model.input_space


def get_model(model_name, num_classes=2, dataset="imagenet"):
    model = PretrainedModel(model_name, num_classes, dataset).model
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Number of trainable parameters is {pytorch_total_params}")
    return model


if __name__ == "__main__":
    model = get_model("xception")
    print(model.get_image_size())
    print(model.get_mean_std())
