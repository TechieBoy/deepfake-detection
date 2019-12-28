import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import torch
import torch.nn as nn
import cvlib as cv
import cv2
import os
from torchvision import models, transforms
from glob import glob
import csv


class MesoInception4(nn.Module):
    def __init__(self, num_classes=2):
        super(MesoInception4, self).__init__()
        self.num_classes = num_classes
        # InceptionLayer1
        self.Incption1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
        self.Incption1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption1_bn = nn.BatchNorm2d(11)

        # InceptionLayer2
        self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption2_bn = nn.BatchNorm2d(12)

        # Normal Layer
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    # InceptionLayer
    def InceptionLayer1(self, input):
        x1 = self.Incption1_conv1(input)
        x2 = self.Incption1_conv2_1(input)
        x2 = self.Incption1_conv2_2(x2)
        x3 = self.Incption1_conv3_1(input)
        x3 = self.Incption1_conv3_2(x3)
        x4 = self.Incption1_conv4_1(input)
        x4 = self.Incption1_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption1_bn(y)
        y = self.maxpooling1(y)

        return y

    def InceptionLayer2(self, input):
        x1 = self.Incption2_conv1(input)
        x2 = self.Incption2_conv2_1(input)
        x2 = self.Incption2_conv2_2(x2)
        x3 = self.Incption2_conv3_1(input)
        x3 = self.Incption2_conv3_2(x3)
        x4 = self.Incption2_conv4_1(input)
        x4 = self.Incption2_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption2_bn(y)
        y = self.maxpooling1(y)

        return y

    def forward(self, input):
        x = self.InceptionLayer1(input)  # (Batch, 11, 128, 128)
        x = self.InceptionLayer2(x)  # (Batch, 12, 64, 64)

        x = self.conv1(x)  # (Batch, 16, 64 ,64)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (Batch, 16, 32, 32)

        x = self.conv2(x)  # (Batch, 16, 32, 32)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling2(x)  # (Batch, 16, 8, 8)

        x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
        x = self.dropout(x)
        x = self.fc1(x)  # (Batch, 16)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def get_model():
    model = MesoInception4(2)
    return model


def find_max_face(input_image):
    """
    Finds face in input_image with maximum confidence and returns it
    Adds padding of 15px around face
    """
    detection = cv.detect_face(input_image)
    if detection is not None:
        faces, confidences = detection

        if confidences:
            max_conf = max(confidences)
            face = faces[confidences.index(max_conf)]

            (startX, startY) = face[0], face[1]
            (endX, endY) = face[2], face[3]
            height, width, _ = input_image.shape

            y_top = max(startY - 15, 0)
            x_top = max(startX - 15, 0)
            y_bot = min(endY + 15, height)
            x_bot = min(endX + 15, width)

            fimg = input_image[y_top:y_bot, x_top:x_bot]
            return fimg
    return None


def get_frame(video_path, per_n):
    capture = cv2.VideoCapture(video_path)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(0, num_frames):
        ret = capture.grab()
        if i % per_n == 0:
            ret, frame = capture.retrieve()
            # Get face from video
            a = find_max_face(frame)
            if a is not None and len(a) > 0:
                capture.release()
                return a
    capture.release()
    return None


def mesonet_eval(model, video_path):

    # Load video
    a = None
    per_n = 40
    sol = 0
    while a is None and per_n > 0:
        per_n = per_n // 2
        if per_n == 0:
            break
        a = get_frame(video_path, per_n)
    if a is None:
        # Could not find face return 0.5
        sol = 0.5
    else:
        d = data_transform(a)
        d = d.unsqueeze(0)
        outputs = model(d)

        _, preds = torch.max(outputs, 1)
        outputs = outputs[0].detach().numpy()
        from scipy.special import softmax

        # eval_dict = {
        #     'fake' : 0,
        #     'real' : 1
        # }
        sol = softmax(outputs)[0]
    return sol



if __name__ == "__main__":
    model = get_model()
    model.to(torch.device("cpu"))
    model.load_state_dict(torch.load("saved_models/meso.pt"))
    # Eval mode
    model.eval()

    fd = {}

    files = glob("../kaggle/test_videos/*")
    for fil in files:
        video_name = fil.split("/")[-1]
        if video_name != "metadata.json":
            print(video_name)
            sol = mesonet_eval(model, fil)
            print(sol)
            fd[video_name] = sol
    with open("submission.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "label"])
        for key, value in fd.items():
            writer.writerow([key, value])
    # resnet_eval('/home/teh_devs/deepfake/raw/dfdc_train_part_1/hkgldamgcb.mp4')
