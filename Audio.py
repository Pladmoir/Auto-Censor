import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import re
from openai import OpenAI

# ================= CONFIG =================
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 1  # seconds per chunk for transcription
BEEP_FREQ = 1000
BEEP_AMPLITUDE = 0.3
PROFANITY = {
    "fuck", 
    "shit", 
    "bitch", 
    "asshole",
    "damn", 
    "cunt", 
    "motherfucker", 
    "cocksucker", 
    "ass",
    "arsehole",
    "asshat",
    "bastard",
    "bollocks",
    "blowjob",
    "bullshit",
    "ching chong",
    "cracker",
    "dick",
    "dickhead",
    "cock",
    "faggot",
    "fucker",
    "fucking",
    "nigger",
    "pussy",
    "nigga",
    "pajeet",
    "paki",
    "prick",
    'retard',
    "slut",
    "shitter",
    "whore"
}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================= RECORD AUDIO =================
print(" Recording… Press Ctrl+C to stop.")
audio_data = []

try:
    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_data.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=callback,
        dtype="float32"
    ):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n Recording stopped.")

# Flatten to 1D numpy array
audio_array = np.concatenate(audio_data, axis=0)[:, 0]
audio_censored = np.copy(audio_array)

# ================= SPLIT INTO CHUNKS =================
chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
chunks = [
    (i, audio_array[i:i + chunk_samples])
    for i in range(0, len(audio_array), chunk_samples)
]

print(f" Split into {len(chunks)} chunks of ~{CHUNK_DURATION}s each")

# ================= TRANSCRIBE & CENSOR =================
for idx, chunk in chunks:
    # Save chunk to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, chunk, SAMPLE_RATE)
        filename = f.name

    # Send to Whisper
    try:
        with open(filename, "rb") as af:
            result = client.audio.transcriptions.create(
                file=af,
                model="whisper-1",
                response_format="verbose_json"
            )
        segments = getattr(result, "segments", [])
    except Exception as e:
        print("Transcription error:", e)
        segments = []
    finally:
        try:
            os.remove(filename)
        except:
            pass

    # Censor each segment in this chunk
    for seg in segments:
        text = seg.text.lower()
        start_sample = max(0, int(seg.start * SAMPLE_RATE)) + idx
        end_sample = min(len(audio_array), int(seg.end * SAMPLE_RATE)) + idx

        if any(word in PROFANITY for word in re.findall(r"\b\w+\b", text)):
            t = np.linspace(0, (end_sample - start_sample) / SAMPLE_RATE,
                            end_sample - start_sample, endpoint=False)
            beep = BEEP_AMPLITUDE * np.sin(2 * np.pi * BEEP_FREQ * t)
            audio_censored[start_sample:end_sample] = beep
            print(f" Censoring segment: '{text.strip()}'")

# ================= PLAYBACK =================
print(" Playing censored audio…")
sd.play(audio_censored, SAMPLE_RATE)
sd.wait()

print(" Done.")
