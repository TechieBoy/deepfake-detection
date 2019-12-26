import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import torch
import torch.nn as nn
import cvlib as cv
import cv2
import os
from torchvision import models, transforms
from glob import glob
import multiprocessing as mp

data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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

            return input_image[y_top:y_bot, x_top:x_bot]
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
            if a is not None:
                capture.release()
                return a
    return None



def resnet_eval(model, video_path, video_name, q):

    # Load video
    a = None
    per_n = 40
    sol = 0
    while a is None and per_n > 0:
        per_n = per_n // 2
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
    result = video_name + "," + str(sol)
    q.put(result)
    return result


def listener(q):
    '''listens for messages on the q, writes to file. '''
    with open('submission.csv', 'w') as f:
        while True:
            m = q.get()
            if m == 'kill':
                print("Done")
                break
            f.write(str(m) + os.linesep)
            f.flush()


if __name__ == '__main__':
    with open('submission.csv', 'w') as f:
        f.write('filename,label' + os.linesep)
    
    device = torch.device("cuda:0")
    # Load model
    model = get_model()
    model.load_state_dict(torch.load('saved_models/resnet101.pt'))
    # Eval mode
    model.eval()

    files = glob('files/*')
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count(0 + 2))
    watcher = pool.apply_async(listener, (q,))
    jobs = []
    for fil in files:
        video_name = fil.split('/')[-1]
        job = pool.apply_async(resnet_eval, (model, fil, video_name, q))
        jobs.append(job)
    
    for job in jobs:
        job.get()
    
    q.put('kill')
    pool.close()
    pool.join()

    resnet_eval('/home/teh_devs/deepfake/raw/dfdc_train_part_1/hkgldamgcb.mp4')

