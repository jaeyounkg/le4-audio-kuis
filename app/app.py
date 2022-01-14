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

from analyze import AudioAnalyzer, get_f0


class AudioView(BoxLayout):
    """
    Parent class for all BoxLayout classes defined for showing some kind of plot for given waveform. Defines some common patterns for showing plot.
    """

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


class WaveView(AudioView):
    """Shows raw waveform"""

    audio = ObjectProperty(None)

    def init(self, *args):
        """Initialize waveform plot and 2 lines showing the selected range for analysis"""
        wave = self.audio.wave
        self.fig, self.ax = plt.subplots()
        (self.plot,) = self.ax.plot(np.arange(wave.shape[0]), wave)
        (self.line1,) = self.ax.plot([0, 0], [min(wave), max(wave)], color="black")
        (self.line2,) = self.ax.plot([0, 0], [min(wave), max(wave)], color="black")
        self.ax.set_title(f"Waveform\nF0: {get_f0(self.audio.wave)}")
        self.ax.set_xlabel("Sample")
        widget = FigureCanvasKivyAgg(self.fig)
        self.add_widget(widget)

    def update_view(self, s: int, t: int, *args, **kwargs):
        """Update the 2 lines showing selected range"""
        wave = self.audio.wave
        self.line1.set_data([s, s], [min(wave), max(wave)])
        self.line2.set_data([t, t], [min(wave), max(wave)])
        self.update_fig()


class SpectrogramView(AudioView):
    """Shows spectrogram"""

    audio = ObjectProperty(None)

    def init(self, *args):
        """Initialize spcectrogram plot and a line showing selected sample to show spectrum from"""
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(
            np.flipud(self.audio.spectrogram.T),
            extent=[0, len(self.audio.wave), 0, self.audio.SR / 2],
            aspect="auto",
            interpolation="nearest",
        )
        (self.line,) = self.ax.plot([0, 0], [0, self.audio.SR / 2], color="white")
        self.ax.set_title("Spectrogram")
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("frequency [Hz]")
        widget = FigureCanvasKivyAgg(self.fig)
        self.add_widget(widget)

    def update_view(self, s: int, t: int, value: int, *args, **kwargs):
        """Update spectrogram view to the given [s, t] range, and the line showing selected sample"""
        self.ax.set_xlim(s, t)

        x = value * len(self.audio.wave) / self.audio.spectrogram.shape[0]
        self.line.set_data(
            [x, x],
            [0, self.audio.SR / 2],
        )
        self.update_fig()


class SpectrumView(AudioView):
    """Shows spectrum in selected sample time"""

    audio = ObjectProperty(None)

    def xs(self):
        """Get x values of plot"""
        t = self.audio.spectrogram.shape[1]
        return np.arange(t) * (self.audio.SR / 2) / t

    def init(self, *args):
        """Initialize spectrum plot and a black line showing where the value is 0"""
        self.fig, self.ax = plt.subplots()
        (self.plot,) = self.ax.plot(self.xs(), self.audio.spectrogram[0])
        (self.line,) = self.ax.plot([0, self.audio.SR / 2], [0, 0], color="black")
        self.ax.set_xlim(0, self.audio.SR / 2)
        self.ax.set_ylim(self.audio.spectrogram.min(), self.audio.spectrogram.max())
        self.ax.set_title("Spectrum")
        self.ax.set_xlabel("Frequency [Hz]")
        widget = FigureCanvasKivyAgg(self.fig)
        self.add_widget(widget)

    def update_view(self, sample_t: int, max_freq: int, *args, **kwargs):
        """Update plot for updated sample & max_freq values"""
        self.plot.set_data(self.xs(), self.audio.spectrogram[sample_t, :])
        self.line.set_data([0, max_freq], [0, 0])
        self.ax.set_xlim(0, max_freq)
        self.update_fig()


class MainWidget(BoxLayout):
    wave = ObjectProperty(None)
    spectrogram = ObjectProperty(None)
    spectrum = ObjectProperty(None)

    wave_slider = ObjectProperty(None)
    wave_frame_slider = ObjectProperty(None)
    slider = ObjectProperty(None)
    freq_slider = ObjectProperty(None)

    audio = ObjectProperty(AudioAnalyzer("data/aiueo.wav"))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wave_slider.bind(value=self.wave_slider_update_view)
        self.wave_frame_slider.bind(value=self.wave_slider_update_view)
        self.slider.bind(value=self.slider_update_view)
        self.freq_slider.bind(value=self.freq_slider_update_view)

    def s(self):
        return int(np.clip(self.wave_slider.value, 0, len(self.audio.wave) - 1))

    def t(self):
        t = self.s() + int(self.wave_frame_slider.value)
        return int(np.clip(t, 0, len(self.audio.wave) - 1))

    def wave_slider_update_view(self, *args, **kwargs):
        """Callback for both wave_slider and wave_frame_slider"""
        scaler = (
            lambda x: x
            / (len(self.audio.wave) - self.audio.frame_size)
            * self.audio.spectrogram.shape[0]
        )
        self.slider.min = scaler(self.s())
        self.slider.max = scaler(self.t())
        self.wave.update_view(self.s(), self.t())
        self.spectrogram.update_view(self.s(), self.t(), int(self.slider.value))
        self.spectrum.update_view(int(self.slider.value), int(self.freq_slider.value))

    def slider_update_view(self, *args, **kwargs):
        self.spectrogram.update_view(self.s(), self.t(), int(self.slider.value))
        self.spectrum.update_view(int(self.slider.value), int(self.freq_slider.value))

    def freq_slider_update_view(self, *args, **kwargs):
        self.spectrum.update_view(int(self.slider.value), int(self.freq_slider.value))


class TebuAudioApp(App):
    def build(self):
        self.root = MainWidget()
        return self.root


if __name__ == "__main__":
    TebuAudioApp().run()
