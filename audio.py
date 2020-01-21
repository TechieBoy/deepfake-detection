import numpy as np
from audio_hp import audio_config
import librosa
import librosa.display


def read_audio(file_path):
    tot_samples = int(audio_config.tot_seconds * audio_config.sampling_rate)
    try:
        y, sr = librosa.load(file_path, sr=audio_config.sampling_rate)
        trim_y, trim_idx = librosa.effects.trim(y, top_db=audio_config.top_db)  # trim

        if len(trim_y) != tot_samples:
            center = (trim_idx[1] - trim_idx[0]) // 2
            left_idx = max(0, center - tot_samples // 2)
            right_idx = min(len(y), center + tot_samples // 2)
            trim_y = y[left_idx:right_idx]

            if len(trim_y) != tot_samples:
                print(f"Padding, {file_path}")
                padding = tot_samples - len(trim_y)
                offset = padding // 2
                trim_y = np.pad(trim_y, (offset, padding - offset), "constant")
        return trim_y
    except BaseException as e:
        print(f"Exception while reading file {e}, {file_path}")
        return np.zeros(tot_samples, dtype=np.float32)


def audio_to_melspectrogram(audio):
    spectrogram = librosa.feature.melspectrogram(
        audio,
        sr=audio_config.sampling_rate,
        n_mels=audio_config.n_mels,
        hop_length=audio_config.hop_length,
        n_fft=audio_config.n_fft,
        fmin=audio_config.fmin,
        fmax=audio_config.fmax,
    )
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def show_melspectrogram(mels, title="Log-frequency power spectrogram"):
    import matplotlib.pyplot as plt

    librosa.display.specshow(
        mels,
        x_axis="time",
        y_axis="mel",
        sr=audio_config.sampling_rate,
        hop_length=audio_config.hop_length,
        fmin=audio_config.fmin,
        fmax=audio_config.fmax,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.show()


def read_as_melspectrogram(
    file_path,
    time_stretch=audio_config.time_stretch,
    pitch_shift=audio_config.pitch_shift,
    debug_display=False,
):
    x = read_audio(file_path)
    if time_stretch != 1.0:
        x = librosa.effects.time_stretch(x, time_stretch)

    if pitch_shift != 0.0:
        librosa.effects.pitch_shift(x, audio_config.sampling_rate, n_steps=pitch_shift)

    mels = audio_to_melspectrogram(x)
    if debug_display:
        import IPython

        IPython.display.display(
            IPython.display.Audio(x, rate=audio_config.sampling_rate)
        )
        show_melspectrogram(mels)
    return mels
