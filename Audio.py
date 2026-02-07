import sounddevice as sd
import numpy as np
import threading
import queue
import time

#======= Iniitialize parameters ========

duration = 0.5  # seconds
sample_rate = 16000  # Sample 
sample_size = int(sample_rate * duration)  # bits per sample

sd.default.samplerate = sample_rate
sd.default.channels = 1


#============ Audio Loop ================
print("Recording audio. Ctrl+C to stop.")

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata  # Echo the input to output)       

with sd.Stream(callback=audio_callback, dtype='float32'):
    try:
        while True:
            time.sleep(0.1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Audio recording stopped.")




