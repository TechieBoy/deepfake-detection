from constants import ConstDict


class AudioConfig:
    sampling_rate = 44100
    hop_length = 512  # 64 - 512 larger gets larger images for gpu
    fmin = 0  # Try 0 to 20
    fmax = 16000  # Try 8000 for speech
    n_mels = 512  # 64 - 128, upscaling image is more effective
    n_fft = n_mels * 20  # Change if you see black horizontal lines
    top_db = 60  # 80 to 120 seems good (brightest pixel, everything above this clips)
    tot_seconds = 6
    time_stretch = 1.0
    pitch_shift = 0.0
    # For above values
    img_w = 512
    img_h = 517
    mean = -19.88055631
    std = 14.3063872


audio_config = ConstDict(
    **{key: val for key, val in vars(AudioConfig).items() if not key.startswith("__")}
)
