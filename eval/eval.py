"""Run the test set, compute metrics with 95 % intervals, write a JSON report. Run it after every change.

    python eval/eval.py                      # full system (retrieval + answer)
    python eval/eval.py --baseline           # same model, no documents
    python eval/eval.py --retrieval-only     # no LLM answer: Recall@k, MRR, abstention threshold
    python eval/eval.py --retrieval-only --signals dense,fulltext   # ablation: one change at a time

Test set line: {"id", "question", "gold_source", "answer", "tags"}; gold_source is an id, a list of
ids (any of them counts) or null for a question without an answer (NOT FOUND expected).
"""
import argparse, json, math, os, statistics, sys
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import pipeline
from app.pipeline import CFG

K_VALUES = (1, 3, 5, 8, 10)


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


USE_ADDED_GOLD = True


def golds(t: dict) -> set[str]:
    """Gold sources: the evaluation lead's, plus the ones added after review (gold_added) unless
    --lead-gold-only. Any of them counts as a correct retrieval."""
    g = t.get("gold_source")
    out = set() if g is None else {g.upper()} if isinstance(g, str) else {x.upper() for x in g}
    if USE_ADDED_GOLD:
        out |= {x.upper() for x in t.get("gold_added") or []}
    return out


def tags(t: dict) -> list[str]:
    """Tags of the draft format, or family and language of the evaluation lead's format."""
    return t.get("tags") or [x for x in (t.get("family"), t.get("lang")) if x]


def corpus_ids() -> set[str]:
    return {r["citation_id"].upper() for r in pipeline.retriever().conn.execute("SELECT DISTINCT citation_id FROM chunks")}


def split_by_corpus(tests: list[dict]) -> tuple[list[dict], list[dict]]:
    """Questions whose gold sources are all outside the corpus measure the corpus, not retrieval: kept apart.

    Gold ids not in the corpus (placeholders such as "CESEDA (à confirmer)", articles outside the scope)
    are dropped from the others, which stay measurable on the ids that exist.
    """
    have = corpus_ids()
    kept, gaps = [], []
    for t in tests:
        gold = golds(t)
        if gold and not gold & have:
            gaps.append({"id": t["id"], "question": t["question"], "gold_source": sorted(gold)})
            continue
        if gold - have:
            t = {**t, "gold_source": sorted(gold & have), "gold_outside_corpus": sorted(gold - have)}
        kept.append(t)
    return kept, gaps


def first_gold_rank(ids: list[str], gold: set[str]) -> int | None:
    return next((i for i, s in enumerate(ids, 1) if s.upper() in gold), None)


def run_retrieval(tests: list[dict]) -> dict:
    rows = []
    for t in tests:
        found = pipeline.retrieve(t["question"])
        question_fr = found.question_fr
        ids = list(dict.fromkeys(s["id"] for s in found.sources))
        gold = golds(t)
        rows.append({"id": t["id"], "tags": tags(t), "question_fr": question_fr,
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
        said_nf = out.get("not_found", out["answer"].strip().upper().startswith(pipeline.NOT_FOUND))
        cited = set(pipeline.cited_ids(out["answer"]))
        rows.append({
            "id": t["id"], "tags": tags(t), "answer": out["answer"], "sources": out["sources"],
            "best_score": out.get("best_score"), "seconds": out["seconds"],
            "not_found_reason": out.get("not_found_reason"),
            "retrieved_gold": bool(gold & {s.upper() for s in out["sources"]}) if gold else None,
            "cited_gold": bool(gold & cited) if gold else None,
            # crude: replace by a judge calibrated on human labels
            "contains_reference": (t["answer"].lower() in out["answer"].lower()) if gold else None,
            "answered_when_answerable": (not said_nf) if gold else None,
            "not_found_ok": said_nf if not gold else None,
            # ids cited by the model that were not among the retrieved sources: must stay at 0
            "invented_citations": out.get("citations", {}).get("invented", [])})
    report = {k: rate([r[k] for r in rows]) for k in
              ("retrieved_gold", "cited_gold", "contains_reference", "answered_when_answerable", "not_found_ok")}
    report["median_seconds"] = statistics.median(r["seconds"] for r in rows) if rows else None
    report["answers_with_invented_citation"] = rate([bool(r["invented_citations"]) for r in rows])
    return {"report": report, "rows": rows}


def by_tag(rows: list[dict]) -> dict:
    """Main rate per tag: Recall@1 in retrieval mode, cited gold or correct NOT FOUND in answer mode."""
    out = {}
    for tag in sorted({t for r in rows for t in r["tags"]}):
        sub = [r for r in rows if tag in r["tags"]]
        if "gold_rank" in sub[0]:
            vals = [r["gold_rank"] == 1 if r["answerable"] else r["abstain"] for r in sub]
        else:
            vals = [r["cited_gold"] if r["cited_gold"] is not None else r["not_found_ok"] for r in sub]
        out[tag] = rate(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="no documents: what the model knows alone")
    ap.add_argument("--retrieval-only", action="store_true", help="measure retrieval without answering")
    ap.add_argument("--signals", help="comma-separated subset of exact,fulltext,dense")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--out", default=CFG["evaluation"]["report"])
    ap.add_argument("--testset", help="another test set file (default: config.yaml)")
    ap.add_argument("--lead-gold-only", action="store_true", help="ignore gold_added (sources added after review)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a retrieval setting for this run, e.g. --set top_k=10 --set rerank_candidates=60")
    ap.add_argument("--model", help="answer model for this run: openai/gpt-oss-120b (endpoint of config.yaml), "
                                    "or ollama:llama3.2 for a local model")
    ap.add_argument("--no-think", action="store_true", help="ask a thinking local model (Qwen3) not to think")
    ap.add_argument("--prompt", help="system prompt file for this run, e.g. prompts/system_v2.txt")
    ap.add_argument("--judge", action="store_true",
                    help="grade every answer with the LLM judge (eval/judge.py): judge_correct, correct_and_cited")
    args = ap.parse_args()
    global USE_ADDED_GOLD
    USE_ADDED_GOLD = not args.lead_gold_only
    if args.model:
        pipeline.use_model(args.model)
    if args.no_think:
        CFG["model"]["think"] = False
    if args.prompt:
        pipeline.SYSTEM = open(args.prompt, encoding="utf-8").read()
        CFG["prompt_file"] = args.prompt
    for item in args.set:
        key, value = item.split("=", 1)
        CFG["retrieval"][key] = yaml.safe_load(value)
    if args.baseline:
        CFG["retrieval"]["enabled"] = False
    if args.signals:
        CFG["retrieval"]["signals"] = args.signals.split(",")
    if args.no_rerank:
        CFG["retrieval"]["rerank"] = False
    cache_path = os.path.join(os.path.dirname(__file__), ".rewrite_cache.json")
    pipeline.REWRITE_CACHE = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    tests = [json.loads(l) for l in open(args.testset or CFG["evaluation"]["testset"], encoding="utf-8") if l.strip()]
    tests, gaps = split_by_corpus(tests)
    result = run_retrieval(tests) if args.retrieval_only else run_answers(tests)
    if args.judge and not args.retrieval_only:
        from eval import judge
        judge.judge_rows(result["rows"], {t["id"]: t for t in tests})
        # Main metric of the project: correct (judge) and citing a gold source (code). Target 0.8.
        for key in ("judge_correct", "correct_and_cited"):
            result["report"][key] = rate([r.get(key) for r in result["rows"]])
    result["report"]["by_tag"] = by_tag(result["rows"])
    result["report"]["corpus_gaps"] = len(gaps)
    result.update(n=len(tests), corpus_gaps=gaps, mode="retrieval" if args.retrieval_only else "baseline" if args.baseline else "system",
                  config=CFG)
    json.dump(result, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
    json.dump(pipeline.REWRITE_CACHE, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(json.dumps({"n": len(tests), "mode": result["mode"], **result["report"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
