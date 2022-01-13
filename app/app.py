from __future__ import annotations

import io
from random import random

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.widget import Widget


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


class AudioView(BoxLayout):
    audio = ObjectProperty(None)
    ax: plt.Axes
    fig: matplotlib.figure.Figure

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(audio=self.init)

    def init(self, *args):
        pass

    def update_fig(self):
        # self.ax.relim()
        # self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class SpectrogramView(AudioView):
    audio = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(audio=self.init)

    def init(self, *args):
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(
            np.flipud(self.audio.spectrogram.T),
            extent=[0, len(self.audio.wave), 0, self.audio.SR / 2],
            aspect="auto",
            interpolation="nearest",
        )
        (self.line,) = self.ax.plot([0, 0], [0, self.audio.SR / 2], color="white")
        widget = FigureCanvasKivyAgg(self.fig)
        self.add_widget(widget)

    def update_view(self, value: int, *args, **kwargs):
        x = value * len(self.audio.wave) / self.audio.spectrogram.shape[0]
        self.line.set_data(
            [x, x],
            [0, self.audio.SR / 2],
        )
        self.update_fig()


class SpectrumView(AudioView):
    audio = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(audio=self.init)

    def init(self, *args):
        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot(
            list(range(self.audio.spectrogram.shape[1])),
            self.audio.spectrogram[0],
        )
        self.ax.set_ylim(self.audio.spectrogram.min(), self.audio.spectrogram.max())
        widget = FigureCanvasKivyAgg(self.fig)
        self.add_widget(widget)

    def update_view(self, value: int, *args, **kwargs):
        self.line.set_data(
            list(range(self.audio.spectrogram.shape[1])),
            self.audio.spectrogram[value, :],
        )
        self.update_fig()


class MainWidget(BoxLayout):
    spectrogram = ObjectProperty(None)
    slider = ObjectProperty(None)
    spectrum = ObjectProperty(None)
    audio = ObjectProperty(AudioAnalyzer("data/aiueo.wav"))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.spectrogram = SpectrogramView(self.audio)
        # self.slider = Slider(
        #     size_hint=(1, 0.075), min=0, max=self.audio.spectrogram.shape[0] - 1
        # )
        # self.spectrum = SpectrumView(self.audio)

        self.slider.bind(value=self.update_view)

        # self.add_widget(self.spectrogram)
        # self.add_widget(self.slider)
        # self.add_widget(self.spectrum)

    def update_view(self, *args, **kwargs):
        self.spectrogram.update_view(int(self.slider.value))
        self.spectrum.update_view(int(self.slider.value))


class TebuAudioApp(App):
    def build(self):
        self.root = MainWidget()
        return self.root


if __name__ == "__main__":
    TebuAudioApp().run()
