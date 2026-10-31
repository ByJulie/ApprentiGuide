"""Spoken questions: does the voice input change what retrieval finds?

For each question: synthesised speech (macOS `say`, or your own recordings in --audio-dir named <id>.wav),
transcription by app/voice.py (faster-whisper, local), then
  - word error rate of the transcript against the written question (jiwer, after lowercasing and
    removing punctuation),
  - rank of the gold source with the transcript against the rank with the written question.

    python eval/voice_eval.py                       # 10 questions, synthetic voices (Mac only)
    python eval/voice_eval.py --audio-dir my_wavs   # real recordings: <id>.wav for each question id
"""
import argparse, json, os, re, statistics, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import jiwer
from app import pipeline
from eval.eval import golds, first_gold_rank, rate, wilson
from voice import transcribe

VOICES = {"fr": "Thomas", "en": "Samantha"}  # macOS voices; "Flo" (robotic) was transcribed as Russian
DEFAULT_IDS = ["A01", "Q04", "Q08", "Q21", "A06", "Q09", "Q32", "A07", "Q60", "A04"]


def normalise(text: str) -> str:
    text = re.sub(r"[^\w\s'-]", " ", text.lower().replace("’", "'"))
    return re.sub(r"\s+", " ", text).strip()


def speak(text: str, lang: str, path: str) -> None:
    subprocess.run(["say", "-v", VOICES[lang], "-o", path, "--data-format=LEI16@16000", text], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(DEFAULT_IDS))
    ap.add_argument("--audio-dir", help="folder of real recordings <id>.wav (default: synthetic voices)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs", "voice.json"))
    args = ap.parse_args()
    cache_path = os.path.join(os.path.dirname(__file__), ".rewrite_cache.json")
    pipeline.REWRITE_CACHE = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    tests = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "testset.jsonl"), encoding="utf-8")}
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for qid in args.ids.split(","):
            t = tests[qid]
            path = os.path.join(args.audio_dir, f"{qid}.wav") if args.audio_dir else os.path.join(tmp, f"{qid}.wav")
            if not args.audio_dir:
                speak(t["question"], t["lang"], path)
            heard = transcribe(open(path, "rb").read())
            gold = golds(t)
            typed = pipeline.retrieve(t["question"])
            spoken = pipeline.retrieve(heard["text"])
            rank = lambda found: first_gold_rank(list(dict.fromkeys(s["id"] for s in found.sources)), gold)
            rows.append({"id": qid, "lang": t["lang"], "question": t["question"], "transcript": heard["text"],
                         "detected_language": heard["language"],
                         "wer": round(jiwer.wer(normalise(t["question"]), normalise(heard["text"])), 3),
                         "rank_typed": rank(typed), "rank_spoken": rank(spoken)})
            print(f"{qid} wer {rows[-1]['wer']:.2f} rank {rows[-1]['rank_typed']} -> {rows[-1]['rank_spoken']} | {heard['text']}")
    json.dump(pipeline.REWRITE_CACHE, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    same = [r["rank_spoken"] == r["rank_typed"] for r in rows]
    report = {"n": len(rows), "voices": "synthetic" if not args.audio_dir else "recorded",
              "median_wer": statistics.median(r["wer"] for r in rows),
              "language_detected_right": rate([r["detected_language"] == r["lang"] for r in rows]),
              "same_gold_rank_as_typed": {"rate": round(sum(same) / len(same), 3), "n": len(same), "ci95": wilson(sum(same), len(same))},
              "gold_in_top5_typed": rate([bool(r["rank_typed"]) for r in rows]),
              "gold_in_top5_spoken": rate([bool(r["rank_spoken"]) for r in rows])}
    json.dump({"report": report, "rows": rows}, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
