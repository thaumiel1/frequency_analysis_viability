import logging
import threading
import time
import wave
from collections import deque

import numpy as np
import pyaudio

logger = logging.getLogger("__name__")


def main():
    logging.basicConfig(filename="freq_anal_log.log", level=logging.INFO)
    logger.info("Program started.")


if __name__ == "__main__":
    main()
