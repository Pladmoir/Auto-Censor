import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import time
from openai import OpenAI

# ================= CONFIG =================
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 2  # seconds per chunk for transcription
BEEP_FREQ = 1000
BEEP_AMPLITUDE = 0.3

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================= RECORD AUDIO =================
print(" Recording… Press Ctrl+C to stop.")
audio_data = []

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_data.append(indata.copy())

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback,
        dtype="float32"
    ):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n Recording stopped.")

# Flatten audio into 1D array
audio_array = np.concatenate(audio_data, axis=0)[:, 0]
audio_censored = np.copy(audio_array)

# ================= SPLIT INTO CHUNKS =================
chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
chunks = [
    (i, audio_array[i:i + chunk_samples])
    for i in range(0, len(audio_array), chunk_samples)
]
print(f" Split into {len(chunks)} chunks of ~{CHUNK_DURATION}s each")

# ================= OPENAI CONTEXT CLASSIFIER =================
def should_censor(text: str) -> bool:
    """
    Uses GPT to classify if the text should be censored.
    Returns True if GPT thinks it should be censored.
    """
    prompt = (
        "Decide if the following text contains sensitive, offensive, "
        "or inappropriate content that should be censored in audio playback. "
        "Answer YES or NO only.\n\n"
        f"Text: \"{text.strip()}\""
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        classification = response.choices[0].message.content.strip().upper()
        return "YES" in classification
    except Exception as e:
        print("GPT classification error:", e)
        return False

# ================= TRANSCRIBE + CENSOR =================
for idx, chunk in chunks:
    # Save chunk to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, chunk, SAMPLE_RATE)
        filename = f.name

    # Transcribe with Whisper
    try:
        with open(filename, "rb") as af:
            result = client.audio.transcriptions.create(
                file=af,
                model="whisper-1",
                response_format="verbose_json"
            )
        segments = getattr(result, "segments", [])
    except Exception as e:
        print("Whisper error:", e)
        segments = []
    finally:
        try:
            os.remove(filename)
        except:
            pass

    # Censor segments flagged by GPT
    for seg in segments:
        text = seg.text.strip()
        start_sample = idx + max(0, int(seg.start * SAMPLE_RATE))
        end_sample = idx + min(len(audio_array), int(seg.end * SAMPLE_RATE))

        if should_censor(text):
            t = np.linspace(0, (end_sample - start_sample) / SAMPLE_RATE,
                            end_sample - start_sample, endpoint=False)
            beep = BEEP_AMPLITUDE * np.sin(2 * np.pi * BEEP_FREQ * t)
            audio_censored[start_sample:end_sample] = beep
            print(f" Context-censored segment: '{text}'")

# ================= PLAYBACK =================
print(" Playing censored audio…")
sd.play(audio_censored, SAMPLE_RATE)
sd.wait()
print(" Done.")
