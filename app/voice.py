"""Spoken questions: local speech-to-text (faster-whisper), then the same pipeline as a typed question.

The audio is read from memory and never written to disk; the model runs on the CPU, on Mac and Windows.
"""
import io
import os
from functools import lru_cache

# small: multilingual, about 460 MB, good enough for French and English questions; base is faster, worse.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")


@lru_cache(maxsize=1)
def whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe(audio: bytes) -> dict:
    """Text and detected language ("fr" or "en", else the code Whisper returns) of a spoken question.

    The initial prompt gives Whisper the vocabulary of the domain, so "titres-restaurant" or "CFA" are
    spelled as in the texts we search.
    """
    segments, info = whisper().transcribe(
        io.BytesIO(audio), beam_size=5, vad_filter=True,
        initial_prompt="Apprentissage, contrat d'apprentissage, CFA, titres-restaurant, congés payés, "
                       "mutuelle, télétravail, titre de séjour, apprenticeship, meal vouchers.")
    text = " ".join(s.text.strip() for s in segments).strip()
    return {"text": text, "language": info.language, "language_probability": round(info.language_probability, 2)}
