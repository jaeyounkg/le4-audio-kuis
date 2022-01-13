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
