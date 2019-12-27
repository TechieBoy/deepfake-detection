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

data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_model():
    model_v1 = models.resnet101(pretrained=False)
    for param in model_v1.parameters():
        param.requires_grad = True

    num_last_layer = model_v1.fc.in_features
    model_v1.fc = nn.Linear(num_last_layer, 2)

    model_v1 = model_v1.to(device)
    return model_v1


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


def resnet_eval(model, video_path):

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


if __name__ == '__main__':
    device = torch.device("cuda:0")
    # Load model
    model = get_model()
    model.to(torch.device('cpu'))
    model.load_state_dict(torch.load('saved_models/resnet101.pt'))
    # Eval mode
    model.eval()

    fd = {}

    files = glob('../kaggle/test_videos/*')
    for fil in files:
        video_name = fil.split('/')[-1]
        if video_name != 'metadata.json':
            sol = resnet_eval(model, fil)
            fd[video_name] = sol
    with open('submission.csv', 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['filename', 'label'])
        for key, value in fd.items():
            writer.writerow([key, value])
    # resnet_eval('/home/teh_devs/deepfake/raw/dfdc_train_part_1/hkgldamgcb.mp4')

