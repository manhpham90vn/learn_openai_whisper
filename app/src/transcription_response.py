from pydantic import BaseModel

class TranscriptionResponse(BaseModel):
    language: str
    language_probability: float
    segments: list
    processing_time: float