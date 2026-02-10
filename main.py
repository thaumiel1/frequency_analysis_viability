import logging
import threading
import time
from collections import deque

import numpy as np
import scipy
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger("__name__")


def main():
    logging.basicConfig(filename="freq_anal_log.log", level=logging.INFO)
    logger.info("Program started.")
    file = WavFile()
    file.read_wav()
    file.stereo_to_mono()


def get_band_data(data, low, high, fs, order=5):
    # Create a bandpass filter in SOS format
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    # Apply zero-phase filtering to prevent time-shifting
    return sosfiltfilt(sos, data)


class WavFile:
    def __init__(self) -> None:
        logger.info("Initialising audio holding...")
        self.data = []
        self.fs = -1

    def read_wav(self, filename="test.wav"):
        logger.info("Reading in .wav...")
        (self.fs, self.data) = scipy.io.wavfile.read(f"./assets/{filename}")

    def stereo_to_mono(self):
        logger.info("Converting .wav to mono signal...")
        self.data = self.data.mean(axis=1).astype(self.data.dtype)

    def split_signals(self):
        bands = {"bass": (20, 250), "mid": (250, 4000), "treble": (4000, 20000)}

        split_signals = {
            name: get_band_data(self.data, low, high, self.fs)
            for name, (low, high) in bands.items()
        }
        print(split_signals)


if __name__ == "__main__":
    main()
