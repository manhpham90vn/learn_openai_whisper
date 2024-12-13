from time import time
from faster_whisper import WhisperModel

# https://github.com/openai/whisper
# https://github.com/SYSTRAN/faster-whisper
# size: tiny | base | small | medium | large | turbo

model = WhisperModel("tiny", device="cpu")
segments, info = model.transcribe("data/vi.mp3")

start = time()
print(
    "Detected language '%s' with probability %f"
    % (info.language, info.language_probability)
)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

end = time()
print(f"Total time: {end - start:2f} seconds")

# tiny => 73.786270
# base 113
