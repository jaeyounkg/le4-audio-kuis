from __future__ import annotations

import io
import sys
import threading
import wave
from datetime import datetime
from functools import partial
from random import random

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.core.image import Image as CoreImage
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import ListProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.widget import Widget

from analyze import AudioAnalyzer, get_f0, get_spectrogram


class AudioView(BoxLayout):
    """
    Parent class for all BoxLayout classes defined for showing some kind of plot for given waveform. Defines some common patterns for showing plot.
    """

    wave = ObjectProperty(None)
    ax: plt.Axes
    fig: matplotlib.figure.Figure

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(wave=self.init)

    def init(self, *args):
        pass

    def update_fig(self):
        # self.ax.relim()
        # self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class WaveView(AudioView):
    """Shows raw waveform"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init()

    def init(self, *args):
        """Initialize waveform plot and 2 lines showing the selected range for analysis"""
        self.fig, self.ax = plt.subplots()
        wave = np.zeros(100)
        (self.plot,) = self.ax.plot(np.arange(wave.shape[0]), wave)
        self.ax.set_title(f"Waveform")
        self.ax.set_xlabel("Sample")
        widget = FigureCanvasKivyAgg(self.fig)
        self.add_widget(widget)

    def update_view(self, wave, *args, **kwargs):
        """Update the 2 lines showing selected range"""
        self.plot.set_data(np.arange(len(wave)), wave)
        self.ax.set_xlim(0, len(wave))
        self.ax.set_ylim(-1, 1)
        # self.ax.set_ylim(0, 1000)
        self.update_fig()


class MainWidget(BoxLayout):
    wave_view = ObjectProperty(None)
    frames = ListProperty(list())

    CHANNELS = 2
    FORMAT = pyaudio.paInt16
    FR = 44100  # Frame Rate = Sample Rate * Channels
    CHUNKS = 1024

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=self.FORMAT,
            channels=2,
            rate=self.FR,
            input=True,
            frames_per_buffer=self.CHUNKS,
        )

        self.frames = list()
        self.recorded = list()
        self.f0s = list()
        self.dbs = list()
        thread = threading.Thread(target=self.record, args=(self.frames,), daemon=True)
        thread.start()
        self.tick = 0
        Clock.schedule_interval(self.handle_recorded, 1 / 60)

        # self.bind(frames=self.handle_recorded)

    def record(self, frames):
        st = datetime.now()
        while True:
            frame = self.stream.read(self.CHUNKS)
            print(len(frames), datetime.now() - st)
            frames.append(frame)
            st = datetime.now()

    def handle_recorded(self, *args):
        print(f"handle: {self.tick}, {len(self.frames)}")
        for i in range(self.tick, len(self.frames)):
            frame = self.frames[i]

            wf = wave.open(f"tmp/recorded{str(i)}.wav", "wb")
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.FORMAT))
            wf.setframerate(self.FR)
            wf.writeframes(frame)
            wf.close()

            x, _ = librosa.load(f"tmp/recorded{str(i)}.wav")
            self.recorded.extend(x)
            if len(self.recorded) > 20000:
                self.recorded = self.recorded[-10000:]
            if len(self.recorded) >= 10000:
                self.wave_view.update_view(self.recorded[-10000:])

        self.tick = len(self.frames)

    def old_record(self, *args):
        # current_recorded = list(self.stream.read(self.chunks))
        current_recorded = [x - 127 for x in list(self.stream.read(self.chunks))]
        print(min(current_recorded), max(current_recorded))
        self.recorded.extend(current_recorded)
        if len(self.recorded) > 20000:
            self.recorded = self.recorded[-10000:]

        N = self.chunks * 10
        if len(self.recorded) > N:
            # self.f0s.append(get_f0(self.recorded[-self.chunks :]))
            # if len(self.f0s) > 60:
            #     self.f0s = self.f0s[-60:]

            self.dbs.append(np.sqrt(np.sum(np.power(self.recorded[-N:], 2)) / N))
            if len(self.dbs) > 60:
                self.dbs = self.dbs[-60:]

            self.wave_view.update_view(self.dbs)


class KaraokeApp(App):
    def build(self):
        self.root = MainWidget()
        return self.root


if __name__ == "__main__":
    KaraokeApp().run()
