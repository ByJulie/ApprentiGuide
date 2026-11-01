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


LANGUAGES = ("fr", "en")  # our users ask in French or English
PROMPT = {"fr": "Apprentissage, contrat d'apprentissage, CFA, titres-restaurant, congés payés, mutuelle, "
                "télétravail, titre de séjour.",
          "en": "Apprenticeship, apprenticeship contract, meal vouchers, paid leave, residence permit."}


def transcribe(audio: bytes) -> dict:
    """Text and language ("fr" or "en") of a spoken question.

    Whisper's language guess is limited to French and English: on a short question it can hesitate
    between a hundred languages (a test sentence came out in Russian). The initial prompt gives the
    vocabulary of the domain, so "titres-restaurant" or "CFA" are spelled as in the texts we search.
    """
    from faster_whisper.audio import decode_audio
    samples = decode_audio(io.BytesIO(audio), sampling_rate=16000)
    _, _, probs = whisper().detect_language(samples)
    scores = dict(probs)
    language = max(LANGUAGES, key=lambda l: scores.get(l, 0.0))
    segments, _ = whisper().transcribe(samples, language=language, beam_size=5, vad_filter=True,
                                       initial_prompt=PROMPT[language])
    text = " ".join(s.text.strip() for s in segments).strip()
    return {"text": text, "language": language, "language_probability": round(scores.get(language, 0.0), 2)}
