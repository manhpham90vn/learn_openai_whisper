import json
from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel

from .transcription_response import TranscriptionResponse

app = FastAPI()
model = WhisperModel("base", device="cpu")


@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    audio_file = BytesIO(audio_bytes)

    return StreamingResponse(generate_transcription_stream(audio_file), media_type="application/json")


def generate_transcription_stream(audio_file: BytesIO):
    segments, info = model.transcribe(audio_file)

    yield json.dumps({
        "language": info.language,
        "language_probability": info.language_probability,
        "segment": None,
    }) + '\n'

    for segment in segments:
        yield json.dumps({
            "language": info.language,
            "language_probability": info.language_probability,
            "segment": {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            },
        }) + '\n'