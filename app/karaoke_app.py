from __future__ import annotations

import io
import logging
import sys
import threading
import wave
from datetime import datetime
from functools import partial
from pathlib import Path
from random import random

import cv2
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
from playsound import playsound

from analyze import AudioAnalyzer, get_f0, get_spectrogram, hz2nn

logger = logging.getLogger(__file__)


class AudioView(BoxLayout):
    """
    Parent class for all BoxLayout classes defined for showing some kind of plot for given waveform. Defines some common patterns for showing plot.
    """

    ax: plt.Axes
    fig: matplotlib.figure.Figure

    def update_fig(self):
        # self.ax.relim()
        # self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class WaveView(AudioView):
    """Shows raw waveform"""

    def init(self, title, *args):
        self.fig, self.ax = plt.subplots()
        wave = np.zeros(100)
        (self.plot,) = self.ax.plot(np.arange(wave.shape[0]), wave)
        self.ax.set_title(title)
        self.ax.set_xlabel("Sample")
        widget = FigureCanvasKivyAgg(self.fig)
        self.add_widget(widget)

    def update_view(self, wave, ymin=None, ymax=None, *args, **kwargs):
        if len(wave) == 0:
            return
        if ymin is None:
            ymin = min(wave)
            ymin = 0 if np.isnan(ymin) or np.isinf(ymin) else ymin
        if ymax is None:
            ymax = max(wave)
            ymax = 0 if np.isnan(ymax) or np.isinf(ymax) else ymax
        self.plot.set_data(np.arange(len(wave)), wave)
        self.ax.set_xlim(0, len(wave))
        self.ax.set_ylim(ymin, ymax)
        self.update_fig()


class SpectrogramView(AudioView):
    def init(self, spectrogram, sr, *args):
        spec = spectrogram[:, : spectrogram.shape[1] // 5]
        h, w = spec.T.shape
        img = cv2.resize(spec.T, (w // 3, h // 2))
        print(img.shape)
        self.max_hz = sr / 2 // 5

        print(f"max_hz: {self.max_hz}")

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(
            np.flipud(img),
            extent=[0, spectrogram.shape[0] / 100, 0, self.max_hz],
            aspect="auto",
            interpolation="nearest",
        )
        self.ax.set_title("Spectrogram")
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("frequency [Hz]")
        # self.ax.set_ylim(0, 1000)
        # self.ax.set_xlim(0, 10)
        widget = FigureCanvasKivyAgg(self.fig)
        self.add_widget(widget)

    def update_view(self, sec):
        self.ax.set_xlim(sec - 5, sec)
        self.update_fig()


class MainWidget(BoxLayout):
    spectrogram_view = ObjectProperty(None)
    db_view = ObjectProperty(None)
    f0_view = ObjectProperty(None)
    frames = ListProperty(list())

    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    FR = 16000  # Frame Rate = Sample Rate * Channels
    SR = FR // CHANNELS
    CHUNKS = 1024
    SHOW_SAMPLES = 12000

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

        audio_path = "data/not-anyone-else-mono.mp3"

        self.music, _ = librosa.load(audio_path, sr=self.SR)
        play_thread = threading.Thread(
            target=self.play_music, args=(audio_path,), daemon=True
        )
        play_thread.start()

        print(len(self.music))
        spectrogram = get_spectrogram(self.music, self.SR, 4096)
        self.spectrogram_view.init(spectrogram, self.SR)

        self.db_view.init("Decibel")
        self.f0_view.init("F0")

        self.frames = list()
        self.recorded = list()
        self.f0s = list()
        self.nns = list()
        self.dbs = list()
        record_thread = threading.Thread(
            target=self.record, args=(self.frames,), daemon=True
        )
        record_thread.start()
        self.tick = 0
        Clock.schedule_interval(self.handle_recorded, 1 / 60)

    def play_music(self, music_path):
        playsound(music_path)

    def record(self, frames):
        st = datetime.now()
        while True:
            frame = self.stream.read(self.CHUNKS)
            logger.debug(len(frames), datetime.now() - st)
            frames.append(frame)
            st = datetime.now()

    def handle_recorded(self, *args):
        # logger.debug(f"handle: {self.tick}, {len(self.frames)}")
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
            if len(self.recorded) >= self.SHOW_SAMPLES * 2:
                self.recorded = self.recorded[-self.SHOW_SAMPLES :]

            N = self.CHUNKS * 1
            if len(self.recorded) > N:
                f0 = get_f0(self.recorded[-self.CHUNKS :], self.SR)
                self.f0s.append(hz2nn(f0) if f0 > 0 else 0)
                if len(self.f0s) >= 60:
                    self.f0s = self.f0s[-60:]

                self.dbs.append(
                    np.log(np.sqrt(np.sum(np.power(self.recorded[-N:], 2)) / N))
                )
                if len(self.dbs) > 60:
                    self.dbs = self.dbs[-60:]

            DB_THRESHOLD = -7.6

            self.db_view.update_view(self.dbs, -9, -1)
            self.f0_view.update_view(
                np.array(self.f0s) * (np.array(self.dbs) > DB_THRESHOLD), 20, 80
            )

        self.tick = len(self.frames)

        sec = self.tick / (self.FR / self.CHUNKS)
        self.spectrogram_view.update_view(sec)


class KaraokeApp(App):
    def build(self):
        self.root = MainWidget()
        return self.root


if __name__ == "__main__":
    if not Path("tmp").exists():
        Path("tmp").mkdir()
    KaraokeApp().run()
