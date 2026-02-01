import logging
import threading
import time
import wave
from collections import deque

import numpy as np
import pyaudio

logger = logging.getLogger("__name__")


class audioBuffer:
    def __init__(
        self, seconds=30, sample_rate=44100, chunk=1024, device_index=None
    ) -> None:
        # Make attributes.
        # Rate at which you want samples taken, commonly 44.1KHz.
        self.sample_rate = sample_rate
        # Chunk size for PulseAudio context.
        self.chunk = chunk
        # Device index to read audio samples from.
        self.device_index = device_index
        # Deque with a max length of the total number of samples divided into chunks, each element being a chunk.
        self.deque = deque(maxlen=int((sample_rate * seconds) / chunk))
        # Attribute for stream.
        self.stream = None
        # If attribute is true the loop of collecting samples is running.
        self.running = False
        # Handle for connecting to pyaudio.
        self.p = pyaudio.PyAudio()

    def callback(self, in_data, frame_count, time_info, status):
        # Standard approach for callback, only unique aspect here is the int16 which is how I'll store the audio data to limit memory space usage.
        data = np.frombuffer(in_data, dtype=np.int16)
        self.deque.append(data)
        return (None, pyaudio.paContinue)

    def start(self):
        if self.stream is not None:
            logger.error("Attempted to start stream but stream was not empty.")
            return

        logger.info("Initialising stream.")
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk,
            stream_callback=self.callback,
        )
        self.running = True
        self.stream.start_stream()
        logger.info("Stream successfully started, buffering last 30 seconds.")

    def stop(self):
        self.running = False
        # If statement for if the stream exists.
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        # Kill the pyaudio connection to the audio driver.
        self.p.terminate()

    # A function that gets an iterator at a point in time of the deque containing the audio samples.
    def get_iterator(self):
        snapshot = list(self.deque)
        for chunk_bytes in snapshot:
            # Change the chunks to int16 samples
            samples = np.frombuffer(chunk_bytes, dtype=np.int16)
            for sample in samples:
                yield sample

    def save_to_wav(self, filename="out.wav"):
        if not self.deque:
            logger.warning("Buffer is empty. Nothing to save.")
            return

        snapshot = list(self.deque)
        # Combine the snapshot into bytes.
        data = b"".join(snapshot)

        try:
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)  # Mono signal
                wf.setsampwidth(
                    self.p.get_sample_size(pyaudio.paInt16)
                )  # 2 bytes per sample
                wf.setframerate(self.sample_rate)
                wf.writeframes(data)
            logger.info(
                f"Successfully saved {len(data) / self.sample_rate / 2:.2f} seconds to '{filename}'"
            )
        except Exception as e:
            logger.error(f"Error saving file: {e}")


def main():
    logging.basicConfig(filename="freq_anal_log.log", level=logging.INFO)
    logger.info("Program started.")
    # Demo area
    logger.info("Starting test. Expect two wav files to appear in project directory.")
    buf = audioBuffer(seconds=30)
    try:
        buf.start()

        logger.info("Waiting 10 seconds to fill some buffer...")
        time.sleep(10)

        # Save whatever we have so far (approx 10 seconds)
        logger.info("Capturing approximately 10 seconds into 'test_capture.wav'...")
        buf.save_to_wav("test_capture.wav")

        logger.info("Waiting 25 seconds to have a total time larger than 30 seconds.")
        time.sleep(25)

        # Save again. Since the buffer is circular, this will contain
        # only the *last* 30 seconds, not the full 35 seconds.
        logger.info(
            "Capturing full buffer of the last 30 seconds into 'last_30_seconds.wav'..."
        )
        buf.save_to_wav("last_30_seconds.wav")
    except KeyboardInterrupt:
        pass
    finally:
        buf.stop()
    logger.info("Program ending...")


if __name__ == "__main__":
    main()
