"""Run the test set, compute metrics with 95 % intervals, write a JSON report. Run it after every change.

    python eval/eval.py                      # full system (retrieval + answer)
    python eval/eval.py --baseline           # same model, no documents
    python eval/eval.py --retrieval-only     # no LLM answer: Recall@k, MRR, abstention threshold
    python eval/eval.py --retrieval-only --signals dense,fulltext   # ablation: one change at a time

Test set line: {"id", "question", "gold_source", "answer", "tags"}; gold_source is an id, a list of
ids (any of them counts) or null for a question without an answer (NOT FOUND expected).
"""
import argparse, json, math, os, re, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import pipeline
from app.pipeline import CFG

K_VALUES = (1, 3, 5)


def wilson(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    """95 % interval of a rate: 16/20 is "between 58 and 92 %", not "80 %"."""
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 3), round(centre + half, 3)]


def rate(values: list) -> dict:
    v = [x for x in values if x is not None]
    return {"rate": round(sum(v) / len(v), 3) if v else None, "n": len(v), "ci95": wilson(sum(v), len(v))}


def golds(t: dict) -> set[str]:
    g = t.get("gold_source")
    return set() if g is None else {g.upper()} if isinstance(g, str) else {x.upper() for x in g}


def first_gold_rank(ids: list[str], gold: set[str]) -> int | None:
    return next((i for i, s in enumerate(ids, 1) if s.upper() in gold), None)


def run_retrieval(tests: list[dict]) -> dict:
    rows = []
    for t in tests:
        question_fr = pipeline.rewrite_fr(t["question"]) if CFG["retrieval"].get("rewrite_fr") else None
        found = pipeline.retriever()(t["question"], question_fr)
        ids = list(dict.fromkeys(s["id"] for s in found.sources))
        gold = golds(t)
        rows.append({"id": t["id"], "tags": t.get("tags", []), "question_fr": question_fr,
                     "answerable": bool(gold), "sources": ids, "best_score": found.best_score,
                     "abstain": found.abstain, "missing_ids": found.missing_ids,
                     "gold_rank": first_gold_rank(ids, gold) if gold else None})
    answerable = [r for r in rows if r["answerable"]]
    report = {f"recall@{k}": rate([r["gold_rank"] is not None and r["gold_rank"] <= k for r in answerable])
              for k in K_VALUES}
    report["mrr"] = round(statistics.mean(1 / r["gold_rank"] if r["gold_rank"] else 0 for r in answerable), 3) if answerable else None
    report["threshold"] = suggest_threshold(rows)
    # With the configured threshold and the missing-article rule (what the product actually does).
    report["abstention"] = {"answerable_kept": rate([not r["abstain"] for r in answerable]),
                            "unanswerable_abstained": rate([r["abstain"] for r in rows if not r["answerable"]])}
    return {"report": report, "rows": rows}


def suggest_threshold(rows: list[dict]) -> dict:
    """Reranker score that best separates answerable from unanswerable questions (to set abstain_threshold).

    Chosen on the test set itself, so it is optimistic: report it as such, or set it on a separate split.
    """
    yes = [r["best_score"] for r in rows if r["answerable"]]
    no = [r["best_score"] for r in rows if not r["answerable"]]
    if not yes or not no:
        return {"note": "needs answerable and unanswerable questions"}
    best = max(sorted(set(yes + no)), key=lambda th: sum(s >= th for s in yes) + sum(s < th for s in no))
    return {"value": best, "answerable_kept": rate([s >= best for s in yes]),
            "unanswerable_abstained": rate([s < best for s in no])}


def run_answers(tests: list[dict]) -> dict:
    rows = []
    for t in tests:
        out = pipeline.answer(t["question"])
        gold = golds(t)
        said_nf = out["answer"].strip().upper().startswith(pipeline.NOT_FOUND)
        cited = {c.upper() for c in re.findall(r"\[([^\]]+)\]", out["answer"])}
        rows.append({
            "id": t["id"], "tags": t.get("tags", []), "answer": out["answer"], "sources": out["sources"],
            "best_score": out.get("best_score"), "seconds": out["seconds"],
            "retrieved_gold": bool(gold & {s.upper() for s in out["sources"]}) if gold else None,
            "cited_gold": bool(gold & cited) if gold else None,
            # crude: replace by a judge calibrated on human labels
            "contains_reference": (t["answer"].lower() in out["answer"].lower()) if gold else None,
            "answered_when_answerable": (not said_nf) if gold else None,
            "not_found_ok": said_nf if not gold else None})
    report = {k: rate([r[k] for r in rows]) for k in
              ("retrieved_gold", "cited_gold", "contains_reference", "answered_when_answerable", "not_found_ok")}
    report["median_seconds"] = statistics.median(r["seconds"] for r in rows) if rows else None
    return {"report": report, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="no documents: what the model knows alone")
    ap.add_argument("--retrieval-only", action="store_true", help="measure retrieval without answering")
    ap.add_argument("--signals", help="comma-separated subset of exact,fulltext,dense,labels")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--out", default=CFG["evaluation"]["report"])
    args = ap.parse_args()
    if args.baseline:
        CFG["retrieval"]["enabled"] = False
    if args.signals:
        CFG["retrieval"]["signals"] = args.signals.split(",")
    if args.no_rerank:
        CFG["retrieval"]["rerank"] = False
    tests = [json.loads(l) for l in open(CFG["evaluation"]["testset"], encoding="utf-8") if l.strip()]
    result = run_retrieval(tests) if args.retrieval_only else run_answers(tests)
    result.update(n=len(tests), mode="retrieval" if args.retrieval_only else "baseline" if args.baseline else "system",
                  config=CFG)
    json.dump(result, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
    print(json.dumps({"n": len(tests), "mode": result["mode"], **result["report"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
