"""Run the test set, compute metrics, write a JSON report. Run it after every change."""
import json, os, sys, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.pipeline import answer, CFG

tests = [json.loads(l) for l in open(CFG["evaluation"]["testset"]) if l.strip()]
rows = []
for t in tests:
    out = answer(t["question"])
    unanswerable = t["gold_source"] is None
    said_nf = out["answer"].strip().upper().startswith("NOT FOUND")
    rows.append({
        "id": t["id"], "tags": t.get("tags", []), "answer": out["answer"], "sources": out["sources"],
        "retrieved_gold": (t["gold_source"] in out["sources"]) if not unanswerable else None,
        "contains_reference": (t["answer"].lower() in out["answer"].lower()) if not unanswerable else None,
        "not_found_ok": said_nf if unanswerable else None,
        "seconds": out["seconds"]})

def rate(key):
    v = [r[key] for r in rows if r[key] is not None]
    return round(sum(v) / len(v), 3) if v else None

report = {"n": len(rows), "retrieval_recall": rate("retrieved_gold"),
          "contains_reference": rate("contains_reference"),   # crude: replace by a judge + human check
          "not_found_rate": rate("not_found_ok"),
          "median_seconds": statistics.median(r["seconds"] for r in rows),
          "config": CFG, "rows": rows}
json.dump(report, open(CFG["evaluation"]["report"], "w"), indent=2, ensure_ascii=False)
print({k: v for k, v in report.items() if k not in ("rows", "config")})
