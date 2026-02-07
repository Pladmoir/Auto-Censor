import sounddevice as sd
import numpy as np
import time

#=======Iniitialize parameters========

duration = 1  # seconds
sample_rate = 16000  # Sample rate
sd.default.samplerate = sample_rate
sd.default.channels = 2  

#=======Record audio========

print("Recording audio...")
myrecording = sd.rec(int(duration * sample_rate))

sd.wait()  # Wait until recording is finished

#=======Playback audio========

print("Recording finished.")
print("Playing back the recording...")
sd.play(myrecording, sample_rate)
sd.wait()  # Wait until playback is finished
