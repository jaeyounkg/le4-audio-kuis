import librosa
import numpy as np


class AudioAnalyzer:
    SR = 16000
    frame_size = 4096

    def __init__(self, audio_path):
        self.wave, _ = librosa.load(audio_path, sr=self.SR)
        self.spectrogram = self._get_spectrogram(self.wave)

    def _get_spectrogram(self, x):
        hamming_window = np.hamming(self.frame_size)  # フレームサイズに合わせてハミング窓を作成
        shift_size = self.SR / 100  # 0.01 秒 (10 msec)
        spectrogram = []
        for i in range(0, len(x) - self.frame_size, int(shift_size)):
            x_frame = x[i : i + self.frame_size]
            x_fft = np.log(np.abs(np.fft.rfft(x_frame * hamming_window)))
            spectrogram.append(x_fft)
        return np.array(spectrogram)


def get_spectrogram(wave, sr, frame_size):
    hamming_window = np.hamming(frame_size)  # フレームサイズに合わせてハミング窓を作成
    shift_size = sr / 100  # 0.01 秒 (10 msec)
    spectrogram = []
    for i in range(0, len(wave) - frame_size, int(shift_size)):
        x_frame = wave[i : i + frame_size]
        x_fft = np.log(np.abs(np.fft.rfft(x_frame * hamming_window)))
        spectrogram.append(x_fft)
    return np.array(spectrogram)


def get_f0(wave):
    corr = np.correlate(wave, wave, "full")
    corr = corr[len(corr) // 2 :]

    def is_peak(corr, i):
        return 0 < i < len(corr) - 1 and corr[i - 1] < corr[i] < corr[i + 1]

    peakindices = [i for i in range(len(corr)) if is_peak(corr, i)]
    peakindices = [i for i in peakindices if i != 0]
    return (
        0 if len(peakindices) == 0 else max(peakindices, key=lambda index: corr[index])
    )
