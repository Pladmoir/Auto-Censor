import sounddevice as sd
import numpy as np
import threading
import queue
import time
from collections import deque

#======= Iniitialize parameters ========

buffer_seconds = 0.5  
sample_rate = 16000  # Sample 
sample_size = int(sample_rate * buffer_seconds)  # bits per sample
buffer_size = int(sample_rate * buffer_seconds)  # bytes per buffer (16-bit audio)

audio_buffer = deque(maxlen=buffer_size)  # Circular buffer to hold audio data

sd.default.samplerate = sample_rate
sd.default.channels = 1


#============ Audio Loop ================
print("Recording audio. Ctrl+C to stop.")

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    audio_buffer.extend(indata[:, 0])  # Add incoming audio to the buffer   
    if len(audio_buffer) >= buffer_size:
        outdata[:] = np.zeros() 
    else:
        chunk = np.array(
            [audio_buffer.popleft() for _ in range(frames)], dtype='float32') 
        outdata[:, 0] = chunk
         

with sd.Stream(callback=audio_callback, dtype='float32'):
    try:
        while True:
            time.sleep(0.1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Audio recording stopped.")




