from src.downloads import ensure_model
from src.cactus import (
   cactus_init,
   cactus_complete,
   cactus_destroy,
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


llm_weights = ensure_model("Cactus-Compute/gemma-4-E2B-it")
llm = cactus_init(str(llm_weights), None, False)
messages = json.dumps([{"role": "user", "content": "Transcribe the audio."}])
options = json.dumps({"max_tokens": 512, "temperature": 0.0})
result = json.loads(cactus_complete(llm, messages, options, None, None, pcm_data=pcm))
cactus_destroy(llm)


if not result.get("success", True) or result.get("error"):
   print("Error:", result.get("error"))
print("Reply:", result.get("response", ""))
