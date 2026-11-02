"""Apply the evaluation lead's review of the test set (eval/testset_review.csv) to eval/testset.jsonl.

    python eval/review_testset.py            # apply, write eval/testset.jsonl
    python eval/review_testset.py --check    # only check every quote of gold_added against data/raw

One row per reviewed question. decision:
  accept           gold_source = lead's gold minus `remove`, plus gold_added and `add`
  reject           gold_added dropped, lead's gold kept (minus `remove`)
  remove_question  the question leaves the test set
`answer`, when filled, replaces the reference answer (written from the text of the gold source).
A question with no row is left unchanged. The review history stays in git, and the reason of each
decision in the `review_note` field of the question.
"""
import argparse, csv, json, os, re, sys, unicodedata

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TESTSET = os.path.join(ROOT, "eval", "testset.jsonl")
REVIEW = os.path.join(ROOT, "eval", "testset_review.csv")


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).replace("’", "'").split()).lower()


def corpus_texts() -> dict[str, str]:
    """Citation id -> normalised text, from the raw files (no database needed)."""
    raw, texts = os.path.join(ROOT, "data", "raw"), {}
    for name in os.listdir(raw):
        if not name.endswith(".jsonl"):
            continue
        for line in open(os.path.join(raw, name), encoding="utf-8"):
            r = json.loads(line)
            if name.startswith("legifrance"):
                key, body = (r.get("num") or "").strip(), (r.get("texte") or "") + " " + (r.get("nota") or "")
            else:
                key, body = r["id"], " ".join(s["title"] + " " + s["text"] for s in r["sections"])
            texts[key.upper()] = texts.get(key.upper(), "") + " " + norm(body)
    # School regulations: private PDF files in data/private (skipped when absent, see data/README.md).
    sys.path.insert(0, ROOT)
    from ingest import load
    for c in load.regulation_chunks():
        texts[c.citation_id.upper()] = texts.get(c.citation_id.upper(), "") + " " + norm(c.text)
    return texts


def check_evidence(tests: list[dict]) -> list[str]:
    """Quotes of gold_added_evidence not found word for word in the text of their source."""
    texts, bad = corpus_texts(), []
    for t in tests:
        for cid, quote in (t.get("gold_added_evidence") or {}).items():
            parts = [p for p in re.split(r"\.\.\.|…", quote) if len(p.strip()) > 8]
            if not all(norm(p) in texts.get(cid.upper(), "") for p in parts):
                bad.append(f"{t['id']} {cid}")
    return bad


def ids(cell: str) -> list[str]:
    return [x.strip() for x in cell.split(";") if x.strip()]


def as_list(gold) -> list[str]:
    return [] if gold is None else [gold] if isinstance(gold, str) else list(gold)


def apply(tests: list[dict], review: dict[str, dict]) -> list[dict]:
    out = []
    for t in tests:
        row = review.get(t["id"])
        if row is None:
            out.append(t)
            continue
        if row["decision"] == "remove_question":
            continue
        gold = [g for g in as_list(t.get("gold_source")) if g not in ids(row["remove"])]
        if row["decision"] == "accept":
            gold += [g for g in (t.get("gold_added") or []) + ids(row["add"]) if g not in gold]
        elif row["decision"] != "reject":
            sys.exit(f"{t['id']}: unknown decision {row['decision']!r}")
        t = {k: v for k, v in t.items() if k not in ("gold_added", "gold_added_by", "gold_issue")}
        if t.get("family") == "no_answer":
            t["gold_source"] = None
        else:
            t["gold_source"] = gold
        t["status"] = "validated"
        if row.get("answer"):  # reference answer rewritten from the text of the source
            t["answer"] = row["answer"]
        if row["note"]:
            t["review_note"] = row["note"]
        out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    tests = [json.loads(l) for l in open(TESTSET, encoding="utf-8") if l.strip()]
    bad = check_evidence(tests)
    print(f"evidence quotes not found in their source: {bad or 'none'}")
    if args.check:
        return
    review = {r["id"]: r for r in csv.DictReader(open(REVIEW, encoding="utf-8"))}
    result = apply(tests, review)
    empty = [t["id"] for t in result if t.get("family") != "no_answer" and not t["gold_source"]]
    if empty:
        sys.exit(f"answerable questions left without gold source: {empty}")
    with open(TESTSET, "w", encoding="utf-8") as f:
        for t in result:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"{len(tests)} questions -> {len(result)}; {sum(t.get('status') == 'validated' for t in result)} validated")


if __name__ == "__main__":
    main()
