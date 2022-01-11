#
# 計算機科学実験及演習 4「音響信号処理」
# サンプルソースコード
#
# 正弦波を生成し，音声ファイルとして出力する
#

import math
import sys

import numpy as np
import scipy.io.wavfile


# 正弦波を生成する関数
# sampling_rate ... サンプリングレート
# frequency ... 生成する正弦波の周波数
# duration ... 生成する正弦波の時間的長さ
def generate_sinusoid(sampling_rate, frequency, duration):
    sampling_interval = 1.0 / sampling_rate
    t = np.arange(sampling_rate * duration) * sampling_interval
    waveform = np.sin(2.0 * math.pi * frequency * t)
    return waveform


SR = 16000.0
f = 440.0
duration = 2.0

w1 = generate_sinusoid(SR, 440, duration)
w2 = generate_sinusoid(SR, 880, duration)
waveform = w1 + w2

# waveform = waveform * 0.9
# waveform = (waveform * 32768.0).astype("int16")

# 音声ファイルとして出力する
filename = "sinuoid_test.wav"
scipy.io.wavfile.write(filename, int(SR), waveform)
