import sounddevice as sd
import numpy as np
import time

#======= Iniitialize parameters ========

duration = 1  # seconds
sample_rate = 16000  # Sample rate
sd.default.samplerate = sample_rate
sd.default.channels = 2  

#============ Audio Loop ================

try:
    while True:
        print("Recording audio...")
        myrecording = sd.rec(int(duration * sample_rate))
        sd.wait()  # Wait until recording is finished
        sd.play(myrecording, sample_rate)
        sd.wait()  # Wait until playback is finished
except KeyboardInterrupt:
    print("Audio recording stopped.")



