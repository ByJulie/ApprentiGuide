"""LLM judge of the answers, calibrated on human labels. Replaces contains_reference (exact text match).

The judge reads the question, the reference answer, the text of the gold sources (from data/raw, no
database needed) and the system's answer, and decides one thing: is the answer correct and complete for
the user? Citations are checked in code (cited_gold in eval.py), NOT FOUND too (not_found_ok): the judge
is only used where a rule cannot decide.

    python eval/judge.py rejudge eval/runs/<run>.json          # add a verdict to every answer of a run
    python eval/judge.py export eval/runs/<a>.json eval/runs/<b>.json --n 40   # sheet to label by hand
    python eval/judge.py calibrate eval/labels/labels.csv      # agreement judge vs human (Cohen's kappa)

The judge model is set in config.yaml (judge section). It should not be the answer model: an LLM tends
to prefer its own answers (self-preference), and long answers (verbosity bias).
"""
import argparse, csv, json, os, random, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
TESTSET = os.path.join(ROOT, "eval", "testset.jsonl")
MAX_SOURCE_CHARS = 6000  # per gold source, chosen chunk by chunk (relevant_text)

JUDGE_PROMPT = """You grade the answers of ApprentiGuide, an assistant for apprentices in France that answers
from the French labour code and Service-Public.fr. You get the question, the expected behaviour, a reference
answer written by the evaluation lead, the official texts that state the rule, and the assistant's answer.

Decide whether the assistant's answer is CORRECT:
- answer_with_citation: the answer gives the key facts of the reference (figures, durations, conditions)
  for the user's situation, and states nothing that contradicts the official texts. Missing the key figure,
  answering a different question (the legal duration when the maximum is asked), or a wrong condition is
  incorrect. Extra correct detail is fine. Wording and language do not matter.
- general_rule_with_source_and_ask_hr: the answer gives what the law sets (or says plainly that the law sets
  no general rule), consistent with the texts, AND says what to ask HR, the employer or the school. If one of
  the two is missing, it is incorrect. A refusal or "not found" is incorrect when the texts cover the topic.
Judge against the official texts when the reference answer is less precise than them.
Do not reward length. Do not judge the citation format.

Reply with JSON only: {"reason": "<one or two sentences>", "correct": true or false}"""


def as_list(gold) -> list[str]:
    return [] if gold is None else [gold] if isinstance(gold, str) else list(gold)


def gold_ids(t: dict) -> list[str]:
    return as_list(t.get("gold_source")) + [g for g in t.get("gold_added") or [] if g not in as_list(t.get("gold_source"))]


_TEXTS = None


def source_texts() -> dict[str, list[str]]:
    """Citation id -> its chunks in document order, built from data/raw exactly as ingest.load does (no database)."""
    global _TEXTS
    if _TEXTS is None:
        from ingest.load import build_chunks
        _TEXTS = {}
        for c in build_chunks():
            _TEXTS.setdefault(c.citation_id.upper(), []).append(c.text)
    return _TEXTS


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 3 or w.isdigit()}


def relevant_text(chunks: list[str], query: str) -> str:
    """The chunks of one source that share the most words with the question, reference and answer, in
    document order, up to MAX_SOURCE_CHARS. Long pages (F2918 is 32 000 characters) do not fit whole, and
    their first characters often miss the rule: cut at 2500, F2918 lost the salary table and S01 lost
    "aucune tolérance" for apprentices, and the judge marked right answers wrong."""
    q = words(query)
    ranked = sorted(range(len(chunks)), key=lambda i: -len(q & words(chunks[i])))
    keep, size = [], 0
    for i in ranked:
        if size + len(chunks[i]) > MAX_SOURCE_CHARS and keep:
            continue
        keep.append(i)
        size += len(chunks[i])
    return "\n[...]\n".join(chunks[i][:MAX_SOURCE_CHARS] for i in sorted(keep))


def judge_messages(t: dict, answer: str, texts: dict[str, list[str]]) -> list[dict]:
    query = f"{t['question']} {t.get('answer') or ''} {answer}"
    sources = "\n\n".join(f"[{g}] " + (relevant_text(texts[g.upper()], query) if g.upper() in texts
                                       else "(text not in corpus)") for g in gold_ids(t)) or "(none)"
    user = (f"Question: {t['question']}\nExpected behaviour: {t.get('expected_behaviour')}\n"
            f"Reference answer: {t.get('answer')}\n\nOfficial texts:\n{sources}\n\n"
            f"Assistant's answer:\n{answer}")
    return [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}]


def parse_verdict(text: str) -> dict:
    """{"correct": bool | None, "reason": str} from the judge's reply; None when it cannot be read."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data.get("correct"), bool):
                return {"correct": data["correct"], "reason": str(data.get("reason", ""))}
        except json.JSONDecodeError:
            pass
    m = re.search(r'"correct"\s*:\s*(true|false)', text, re.I)
    if m:
        return {"correct": m.group(1).lower() == "true", "reason": "parsed from malformed JSON"}
    return {"correct": None, "reason": "unreadable verdict: " + text[:200]}


def client_and_model():
    """OpenAI-compatible client for the judge model of config.yaml ("ollama:<name>" for a local model)."""
    import yaml
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(os.path.join(ROOT, ".env"))
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    j = cfg.get("judge") or cfg["model"]
    name = os.environ.get("JUDGE_MODEL") or j["name"]
    if name.startswith("ollama:"):
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"), name.split(":", 1)[1]
    env = j.get("api_key_env", "none")
    key = "ollama" if env == "none" else os.environ.get(env) or "missing"
    return OpenAI(base_url=j["base_url"], api_key=key, max_retries=6), name


def judge(t: dict, answer: str, client=None, model: str | None = None) -> dict:
    """Verdict on one answer. Only for questions with an answer (NOT FOUND is checked in code)."""
    if client is None:
        client, model = client_and_model()
    # 1000: output tokens per minute allowed for the judge model on Groq (D30); a verdict is 50 to 100 tokens
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=1000,
                                       messages=judge_messages(t, answer, source_texts()))
    return parse_verdict(r.choices[0].message.content)


def judge_rows(rows: list[dict], tests: dict[str, dict]) -> None:
    """Add judge_correct and correct_and_cited to the rows of an answer run (in place)."""
    client, model = client_and_model()
    for r in rows:
        t = tests.get(r["id"])
        if t is None or t.get("gold_source") is None and not t.get("gold_added"):
            r["judge"], r["judge_correct"], r["correct_and_cited"] = None, None, None
            continue
        if r.get("not_found_reason") or r.get("answered_when_answerable") is False:
            v = {"correct": False, "reason": "NOT FOUND on a question with an answer"}
        else:
            v = judge(t, r["answer"], client, model)
        r["judge"], r["judge_correct"] = v, v["correct"]
        r["correct_and_cited"] = None if v["correct"] is None else bool(v["correct"] and r.get("cited_gold"))
    for r in rows:
        r.setdefault("judge_model", model)


def load_tests() -> dict[str, dict]:
    return {t["id"]: t for t in (json.loads(l) for l in open(TESTSET, encoding="utf-8") if l.strip())}


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(v) / n) * (b.count(v) / n) for v in (True, False))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def cmd_rejudge(args):
    from eval.eval import rate
    from app.pipeline import cited_ids
    run, tests = json.load(open(args.run, encoding="utf-8")), load_tests()
    for r in run["rows"]:  # gold sources may have changed since the run (testset_review.csv): cited_gold again
        gold = {g.upper() for g in gold_ids(tests[r["id"]])} if r["id"] in tests else set()
        if gold and r.get("cited_gold") is not None:
            r["cited_gold"] = bool(gold & set(cited_ids(r["answer"])))
    run["report"]["cited_gold"] = rate([r.get("cited_gold") for r in run["rows"]])
    judge_rows(run["rows"], tests)
    for key in ("judge_correct", "correct_and_cited"):
        run["report"][key] = rate([r.get(key) for r in run["rows"]])
    out = args.out or args.run.replace(".json", "_judged.json")
    json.dump(run, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
    print(json.dumps({k: run["report"][k] for k in ("judge_correct", "correct_and_cited", "cited_gold")}, indent=1))
    print("written", out)


def cmd_export(args):
    """A shuffled sheet of answers to label by hand, without the judge's verdict (so it cannot anchor you)."""
    tests, items = load_tests(), []
    for path in args.runs:
        for r in json.load(open(path, encoding="utf-8"))["rows"]:
            t = tests.get(r["id"])
            if t and t.get("gold_source") is not None and not r.get("not_found_reason"):
                items.append((os.path.basename(path), r, t))
    random.Random(args.seed).shuffle(items)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # utf-8-sig so that Excel shows the accents; Excel in French may save it back with ";" (read either way)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "id", "family", "question", "reference_answer", "gold_sources", "answer", "human_correct", "human_note"])
        for run, r, t in items[:args.n]:
            w.writerow([run, r["id"], t["family"], t["question"], t.get("answer"), ";".join(gold_ids(t)), r["answer"], "", ""])
    print(f"{min(args.n, len(items))} answers to label in {args.out}: write 1 (correct) or 0 in human_correct")


def cmd_calibrate(args):
    tests = load_tests()
    text = open(args.labels, encoding="utf-8-sig").read()
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;")
    rows = [r for r in csv.DictReader(text.splitlines(True), dialect=dialect)
            if (r.get("human_correct") or "").strip() in ("0", "1")]
    client, model = client_and_model()
    human, machine, disagree = [], [], []
    for r in rows:
        v = judge(tests[r["id"]], r["answer"], client, model)
        if v["correct"] is None:
            continue
        human.append(r["human_correct"].strip() == "1")
        machine.append(v["correct"])
        if human[-1] != machine[-1]:
            disagree.append({"id": r["id"], "run": r["run"], "human": human[-1], "judge": machine[-1], "reason": v["reason"]})
    n = len(human)
    agree = sum(x == y for x, y in zip(human, machine))
    result = {"judge_model": model, "n": n, "agreement": round(agree / n, 3), "kappa": round(cohen_kappa(human, machine), 3),
              "human_correct_rate": round(sum(human) / n, 3), "judge_correct_rate": round(sum(machine) / n, 3),
              "judge_too_lenient": sum(m and not h for h, m in zip(human, machine)),
              "judge_too_strict": sum(h and not m for h, m in zip(human, machine)), "disagreements": disagree}
    out = args.labels.replace(".csv", "_calibration.json")
    json.dump(result, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "disagreements"}, indent=1))
    print("kappa >= 0.6 substantial, >= 0.8 almost perfect; read the disagreements in", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("rejudge"); p.add_argument("run"); p.add_argument("--out"); p.set_defaults(f=cmd_rejudge)
    p = sub.add_parser("export"); p.add_argument("runs", nargs="+"); p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=3); p.add_argument("--out", default=os.path.join(ROOT, "eval", "labels", "to_label.csv"))
    p.set_defaults(f=cmd_export)
    p = sub.add_parser("calibrate"); p.add_argument("labels"); p.set_defaults(f=cmd_calibrate)
    args = ap.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
