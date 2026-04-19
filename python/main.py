from src.downloads import ensure_model
from src.cactus import (
    cactus_init,
   cactus_destroy,
    cactus_transcribe,
)
import json
import subprocess
import wave
from pathlib import Path


AUDIO_PATH = Path("/Users/rossqin/Development/cactus-hack/voice-agents-hack/data/single_person2.16k_mono.wav")
RESAMPLED_PATH = AUDIO_PATH.with_name(AUDIO_PATH.stem + ".16k_mono.wav")




def ensure_16k_mono(src: Path, dst: Path) -> Path:
   with wave.open(str(src), "rb") as w:
       if w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2:
           return src
   if not dst.exists():
       subprocess.run(
           ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst)],
           check=True,
       )
   return dst




audio_path = ensure_16k_mono(AUDIO_PATH, RESAMPLED_PATH)
with wave.open(str(audio_path), "rb") as w:
   assert w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2, \
       "Gemma expects 16kHz mono 16-bit PCM"
   pcm = w.readframes(w.getnframes())


llm_weights = ensure_model("Cactus-Compute/gemma-4-E4B-it")
llm = cactus_init(str(llm_weights), None, False)
options = json.dumps({"max_tokens": 16384, "temperature": 0.0})

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_SECONDS = 30
CHUNK_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_SECONDS
CONTEXT_WORDS = 50

BASE_PROMPT = "Transcribe the audio."


def build_prompt(prior_text: str) -> str:
    if not prior_text:
        return BASE_PROMPT
    tail = " ".join(prior_text.split()[-CONTEXT_WORDS:])
    return (
        f"Continue transcribing. Previous context: \"{tail}\". "
        f"Transcribe only the new audio that follows, without repeating the context."
    )


transcripts = []
try:
    for offset in range(0, len(pcm), CHUNK_BYTES):
        chunk = pcm[offset:offset + CHUNK_BYTES]
        if len(chunk) < SAMPLE_RATE * BYTES_PER_SAMPLE:
            break
        start_sec = offset // (SAMPLE_RATE * BYTES_PER_SAMPLE)
        prompt = build_prompt(" ".join(transcripts))
        print(f"Transcribing chunk starting at {start_sec}s ({len(chunk)} bytes)...")
        result = json.loads(cactus_transcribe(
            llm, None, prompt, options, None, chunk))
        if not result.get("success", True) or result.get("error"):
            print(f"  Error at {start_sec}s:", result.get("error"))
            continue
        piece = result.get("response", result.get("text", "")).strip()
        if piece:
            transcripts.append(piece)
finally:
    cactus_destroy(llm)

full_text = " ".join(transcripts)
print("Reply:", full_text)
