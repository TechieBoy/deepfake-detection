import torch
import warnings
warnings.simplefilter("ignore", UserWarning)
from audio import read_as_melspectrogram
from audio_cnn import get_model
import os
from audio_hp import audio_config
from torch.nn.functional import softmax
from scipy.stats.mstats import gmean
import csv


def eval_model(device, class_mapping, model, specs, data_transform):
    """Return geometric mean of frame detections"""

    fake_class = class_mapping["fake"]
    specs = [data_transform(f) for f in specs if f is not None]
    if not specs:
        return None
    specs = torch.stack(specs)
    d = specs.to(device)
    outputs = model(d)

    outputs = softmax(outputs, dim=1)
    probs = outputs[:, fake_class].cpu().detach().numpy()
    return gmean(probs)


def data_transform_audio(spec):
    spec -= audio_config.mean
    spec /= audio_config.std
    spec = torch.from_numpy(spec).unsqueeze(0)
    return spec


if __name__ == "__main__":
    # dlib.DLIB_USE_CUDA
    class_mapping = {"fake": 0, "real": 1}
    model = get_model(2, 1)
    device = torch.device("cuda:0")
    model.to(device)
    model.load_state_dict(torch.load("/home/teh_devs/deepfake/deepfake-detection/saved_models/audio_base_cnn_step_lr_test_convergence22Jan12:41PM/audio_base_cnn_step_lr_test_convergence_17.pt"))
    # Eval mode
    model.eval()

    fd = {}

    videos = os.listdir("../kaggle/test_videos/")
    for video_name in videos:
        try:
            video_path = os.path.join('../kaggle/test_videos/', video_name)
            spec = read_as_melspectrogram(video_path)
            sol = eval_model(device, class_mapping, model, [spec], data_transform_audio)
        except Exception as e:
            print(e)
            sol = 0.5
        print(sol)
        fd[video_name] = sol
    with open("submission.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "label"])
        for key, value in fd.items():
            writer.writerow([key, value])
