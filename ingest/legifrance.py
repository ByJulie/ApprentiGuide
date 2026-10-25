"""Fetch the in-force articles of the scope (ingest/scope.yaml) from the Legifrance API (PISTE).

Usage: python -I ingest/legifrance.py [code_travail|ceseda] [--limit N]
Writes data/raw/legifrance_<corpus>.jsonl (one raw article per line); a rerun resumes.
"""
import json
import os
import re
import sys
import time

import requests
import yaml
from dotenv import load_dotenv

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))["legifrance"]
SCOPE = yaml.safe_load(open(os.path.join(ROOT, "ingest", "scope.yaml"), encoding="utf-8"))
RAW = os.path.join(ROOT, "data", "raw")
load_dotenv(os.path.join(ROOT, ".env"))


def get_token() -> str:
    """Return an OAuth access token (client credentials) from PISTE."""
    r = requests.post(CFG["oauth_url"], timeout=30, data={
        "grant_type": "client_credentials", "scope": "openid",
        "client_id": os.environ["PISTE_CLIENT_ID"],
        "client_secret": os.environ["PISTE_CLIENT_SECRET"]})
    r.raise_for_status()
    return r.json()["access_token"]


def call(path: str, body: dict, token: list) -> dict:
    """POST to the API; token is a one-item list so an expired token can be renewed once."""
    for attempt in range(2):
        r = requests.post(CFG["api_url"] + path, json=body, timeout=120,
                          headers={"Authorization": "Bearer " + token[0]})
        if r.status_code == 401 and attempt == 0:
            token[0] = get_token()
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("unreachable")


def list_article_ids(corpus: str, token: list) -> list[tuple[str, str, str]]:
    """(id, number, section path) of the in-force articles matching the scope patterns."""
    toc_path = os.path.join(RAW, f"tm_{corpus}.json")
    if os.path.exists(toc_path):
        toc = json.load(open(toc_path, encoding="utf-8"))
    else:
        toc = call("/consult/legi/tableMatieres", {"textId": SCOPE[corpus]["text_id"],
                   "date": time.strftime("%Y-%m-%d"), "nature": "CODE"}, token)
        json.dump(toc, open(toc_path, "w", encoding="utf-8"), ensure_ascii=False)
    patterns = [re.compile(p["regex"]) for p in SCOPE[corpus]["patterns"]]
    found, seen = [], set()

    def walk(section: dict, path: list[str]):
        for a in section.get("articles") or []:
            num = a.get("num") or ""
            in_scope = any(p.match(num) for p in patterns)
            if a.get("etat") == "VIGUEUR" and a["id"] not in seen and in_scope:
                seen.add(a["id"])
                found.append((a["id"], num, " > ".join(path)))
        for s in section.get("sections") or []:
            walk(s, path + [" ".join(s["title"].split())])
    walk(toc, [])
    return found


def main(corpus: str, limit: int | None):
    """Fetch the in-scope articles of one corpus, skipping those already on disk."""
    os.makedirs(RAW, exist_ok=True)
    token = [get_token()]
    ids = list_article_ids(corpus, token)[:limit]
    out_path = os.path.join(RAW, f"legifrance_{corpus}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(line)["id"] for line in open(out_path, encoding="utf-8")}
    print(f"{corpus}: {len(ids)} in-force articles in scope, {len(done)} already fetched")
    with open(out_path, "a", encoding="utf-8") as out:
        for i, (art_id, num, path) in enumerate(ids, 1):
            if art_id in done:
                continue
            article = call("/consult/getArticle", {"id": art_id}, token)["article"]
            article.pop("articleVersions", None)  # history of versions: not needed, large
            article["sectionPath"] = path
            out.write(json.dumps(article, ensure_ascii=False) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(ids)} ({num})")
            time.sleep(CFG["request_delay_s"])


if __name__ == "__main__":
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    corpora = [a for a in args if a in ("code_travail", "ceseda")] or ["code_travail", "ceseda"]
    for c in corpora:
        main(c, limit)
