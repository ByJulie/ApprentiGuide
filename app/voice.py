"""Spoken questions: local speech-to-text (faster-whisper), then the same pipeline as a typed question.

The audio is read from memory and never written to disk; the model runs on the CPU, on Mac and Windows.
"""
import io
import os
from functools import lru_cache

# medium (about 1.5 GB): measured on 8 spoken questions (2 Nov.), word error rate 3 % clean, 10 % at 10 dB
# of noise, 23 % at 0 dB, about 9 s per question on the CPU; small: 10 %, 14 %, 55 %, about 3 s; Open WebUI's
# default (base, no voice activity detection, any language): 12 %, 39 %, 81 % (and Portuguese at 0 dB).
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")


@lru_cache(maxsize=1)
def whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


LANGUAGES = ("fr", "en")  # our users ask in French or English
PROMPT = {"fr": "Apprentissage, contrat d'apprentissage, CFA, titres-restaurant, congés payés, mutuelle, "
                "télétravail, titre de séjour.",
          "en": "Apprenticeship, apprenticeship contract, meal vouchers, paid leave, residence permit."}


# Whisper's own rule for "no speech here": high no-speech probability and low confidence. Such segments are
# dropped (with the voice activity detection, the two defences of the course against invented text).
NO_SPEECH_PROB = 0.6
LOW_LOGPROB = -1.0


def speech_segments(segments) -> list:
    return [s for s in segments if not (s.no_speech_prob > NO_SPEECH_PROB and s.avg_logprob < LOW_LOGPROB)]


def transcribe(audio: bytes, language: str | None = None) -> dict:
    """Text and language ("fr" or "en") of a spoken question.

    Whisper's language guess is limited to French and English: on a short question it can hesitate
    between a hundred languages (a test sentence came out in Russian). The initial prompt gives the
    vocabulary of the domain, so "titres-restaurant" or "CFA" are spelled as in the texts we search.
    """
    from faster_whisper.audio import decode_audio
    samples = decode_audio(io.BytesIO(audio), sampling_rate=16000)
    if language not in LANGUAGES:  # asked language kept only if French or English
        _, _, probs = whisper().detect_language(samples)
        scores = dict(probs)
        language = max(LANGUAGES, key=lambda l: scores.get(l, 0.0))
    else:
        scores = {language: 1.0}
    segments, _ = whisper().transcribe(samples, language=language, beam_size=5, vad_filter=True,
                                       initial_prompt=PROMPT[language])
    text = " ".join(s.text.strip() for s in speech_segments(segments)).strip()
    return {"text": text, "language": language, "language_probability": round(scores.get(language, 0.0), 2)}
