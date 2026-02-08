import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import time
from collections import deque
import openai
import os
import tempfile
from scipy.io.wavfile import write


#======= Iniitialize parameters ========

buffer_seconds = 0.5  
chunk_seconds = 2
sample_rate = 16000  # Sample


sample_size = int(sample_rate * chunk_seconds)  # bits per sample

buffer_size = int(sample_rate * buffer_seconds)  # bytes per buffer (16-bit audio)

audio_buffer = deque(maxlen=buffer_size)  # Circular buffer to hold audio data

sd.default.samplerate = sample_rate
sd.default.channels = 1

beep_freq = 1000  # Frequency of the beep sound in Hz
beep_amplitude = 0.3
VAD_threshold = 0.01  # Threshold for voice activity detection (VAD)

openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key as an environment variable

speech_queue = queue.Queue()  # Queue to hold audio chunks for processing
play_queue = queue.Queue()  # Queue to hold audio chunks for playback

speech_buffer = []  # Buffer to hold audio chunks for transcription
speech_sample_target = int(sample_rate * 1.5)  # Target number of samples for transcription (1.5 seconds)

profanity = ["fuck", "shit", "bitch", "asshole", "damn", "cunt", "motherfucker", "cocksucker", "ass"]

#======Audio Playing Function======

def record_chunk():
    while True:
        audio = play_queue.get()  # Get audio chunk from the play queue
        sd.play(audio, sample_rate)  
        sd.wait() 

threading.Thread(target=record_chunk, daemon=True).start()  
#======Open AI Transcription Function======

def transcribe_audio():
    while True:
        audio_chunk = speech_queue.get()  # Get audio chunk from the queue
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_chunk, sample_rate)  
            filename = f.name

            try:
                with open(filename, "rb") as audio_file:
                    transcript = openai.Audio.transcribe(file=audio_file, model="whisper-1") 
                    text = transcript.text
            except Exception as e:
                print("Error during transcription: ", e)
                text = ""
            
            
            os.remove(filename)  
            print("Transcribed: ", transcript.text)  

            if any(word in text.lower() for word in profanity):
                print("Profanity detected.")
                t = np.linspace(0, chunk_seconds, sample_size, endpoint=False)
                beep = (beep_amplitude * np.sin(2 * np.pi * beep_freq * t)).astype('float32')
                play_queue.put(beep) 
            else:
                play_queue.put(audio_chunk)

threading.Thread(target=transcribe_audio, daemon=True).start() 

#============ Audio Loop ================

print("Recording audio. Ctrl+C to stop.")


def audio_callback(indata, outdata, frames, time_info, status):
    if status:
        print(status)
    audio_buffer.extend(indata[:, 0])  # Add incoming audio to the buffer   
    if len(audio_buffer) >= buffer_size:
        outdata[:] = np.zeros() 
    else:
        chunk = np.array(
            [audio_buffer.popleft() for _ in range(frames)], dtype='float32')     
         
        energy = np.sqrt(np.mean(chunk ** 2))  # Calculate energy of the audio chunk

        if VAD_threshold < energy: 
            speech_buffer.extend(chunk)  # Add chunk to speech buffer if voice is detected
            
            if len(speech_buffer) >= speech_sample_target:
                # Transcribe the speech buffer
                speech_queue.put(np.array(speech_buffer, dtype='float32'))
                speech_buffer.clear()  # Clear the buffer after transcription
        else:
            speech_buffer.clear()  # Clear the buffer if no voice is detected
         

with sd.Stream(callback=audio_callback, dtype='float32'):
    try:
        while True:
           threading.Event().wait()  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Audio recording stopped.")





