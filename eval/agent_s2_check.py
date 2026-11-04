"""S2 check of the three tool schemas (docs/agent_sheet.md), with the answer model of config.yaml.

1. first_call: the five S1 requests, the first response only. Current schemas against a version where
   search_texts had a `source` parameter (removed: the model restricted searches to the Labour Code).
2. deadline: two requests in a small agent loop, search_texts replaced by a fixed test passage (a delay of
   48 hours, fictional) so only the date handling changes. Before: a days_between tool and today's weekday in
   the prompt. After: deadline, a calendar of the last days in the prompt and a check in code (an answer that
   states a delay without a deadline call is sent back once).
3. real: the five S1 requests on the real corpus, the Generative AI pipeline against the agent (--real).

Today is fixed (Wednesday 4 November 2026) so runs can be compared. Temperature 1 to see run-to-run changes.
python -m eval.agent_s2_check --runs 5 [--real]   →   eval/runs/<date>_s2_tools_check.json"""
import argparse, copy, json, os, re, sys, time
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openai import BadRequestError

from app import pipeline as P, tools

TODAY = date(2026, 11, 4)
WD = tools.WEEKDAYS_FR
REQUESTS = [
    "Combien de jours de congés j'ai pour réviser mes examens ?",
    "J'ai raté mon examen mardi à cause d'un train en retard. Est-ce que je peux encore le justifier, et à qui ?",
    "J'ai le droit à combien ?",
    "J'ai 22 ans et je suis en deuxième année d'apprentissage : combien je vais gagner par mois, en euros ?",
    "Prépare un email à mes RH pour poser mes 5 jours de congé d'examen avant ma session de juin.",
]
EXAM, EMAIL = REQUESTS[1], REQUESTS[4]
TEST_RULE = {"id": "TEST-ABS-48H", "text": "(test passage) Toute absence à un examen doit être justifiée dans un délai de "
             "48 heures auprès de la scolarité, par un justificatif (certificat médical, attestation de retard SNCF...)."}
LEAVE = {"id": "L6222-35", "text": "Pour la préparation directe des épreuves, l'apprenti a droit à un congé supplémentaire "
         "de cinq jours ouvrables. Il est pris dans le mois qui les précède."}

CALENDAR = "; ".join(f"{WD[(TODAY - timedelta(days=i)).weekday()]} {TODAY - timedelta(days=i)}" for i in range(7, -1, -1))
AGENT = ("You are ApprentiGuide, an assistant for apprentices and international students. Use the tools; never guess "
         "rules, dates or amounts. Call search_texts before stating a rule and cite only ids it returned. Call deadline "
         "for every delay or date limit. If the request is vague or a date is missing, ask one question instead of "
         "guessing. Show an email draft before send_email.")
PROMPT_BEFORE = AGENT + " Today is mercredi 2026-11-04 (yesterday was mardi 2026-11-03)."
PROMPT_AFTER = AGENT + f" Today is mercredi 2026-11-04. Last days: {CALENDAR}."

# The two schemas that were replaced, kept here to rerun the comparison.
WITH_SOURCE = copy.deepcopy(tools.schema("search_texts"))
WITH_SOURCE["function"]["parameters"]["properties"]["source"] = {
    "type": "string", "enum": ["all", "labour_code", "ceseda", "service_public", "school_regulations"],
    "description": "Limit the search to one source. Use all unless the user asks about one source."}
WITH_SOURCE["function"]["parameters"]["required"].append("source")
DAYS_BETWEEN = {"type": "function", "function": {"name": "days_between", "strict": True,
    "description": "Number of days between two dates. Use it for any deadline or duration. Never compute dates yourself.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["start", "end"], "properties": {
        "start": {"type": "string", "description": "YYYY-MM-DD"}, "end": {"type": "string", "description": "YYYY-MM-DD"}}}}}

# A conclusion about time ("you can still", "1 day left", "before Thursday"), not the word "délai" alone:
# the real school rule on a missed exam mentions a delay, but the answer is "transport is not accepted".
DAY = r"(\*\*)?(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|\d)"
DELAY = re.compile(r"(pouvez|pourrez|êtes) encore|ne pouvez plus|n['’]est plus possible|il (vous )?reste|dans les temps|"
                   r"dans les délais|délai (est )?(dépassé|expiré)|trop tard|hors délai|date limite|"
                   rf"avant le\s+{DAY}|jusqu['’]au\s+{DAY}", re.I)
DATE_LIMIT = re.compile(r"\b5\s+novembre|2026[-\u2011]11[-\u2011]05|05/11", re.I)   # gpt-oss writes non-breaking hyphens


MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
ISO = re.compile(r"\b(20\d\d)[-‑](\d\d)[-‑](\d\d)\b")
FR = re.compile(r"\b(?:(\d{1,2})(?:er)?\s+)?(" + "|".join(MONTHS) + r")\s+(20\d\d)\b", re.I)


def dates_in(text):
    """Absolute dates (YYYY-MM-DD) and months (YYYY-MM) written in a text, ISO or French."""
    days = {f"{y}-{m}-{d}" for y, m, d in ISO.findall(text)}
    months = set()
    for d, month, y in FR.findall(text):
        m = f"{MONTHS.index(month.lower()) + 1:02d}"
        if d:
            days.add(f"{y}-{m}-{int(d):02d}")
        else:
            months.add(f"{y}-{m}")
    return days, months


def checks(context, text, trace):
    """Rules checked in code on a final answer (sent back once). Only after a search: a clarifying question
    without any rule found is not checked. context: what the model was given (prompt and request)."""
    if not any(t["name"] == "search_texts" for t in trace):
        return None
    if DELAY.search(text) and not any(t["name"] == "deadline" for t in trace):
        return "Your answer states a delay but you did not call deadline"
    given_days, given_months = dates_in(context + " " + " ".join(t.get("full", "") for t in trace))
    given_months |= {d[:7] for d in given_days}
    days, months = dates_in(text)
    invented = sorted((days - given_days) | (months - given_months))
    if invented:
        return f"Your answer writes {', '.join(invented)}, a date the user and the tools did not give"
    return None


REJECTED = {"count": 0}


def chat(messages, schemas, temperature):
    """With strict schemas, Groq refuses a tool call that does not match (400 "tool call validation failed"):
    the call is retried twice, as in the course lab, and the refusals are counted."""
    for attempt in range(3):
        try:
            r = P.client.chat.completions.create(messages=messages, tools=schemas, **{**P._params(), "temperature": temperature})
            return r.choices[0].message
        except BadRequestError as e:
            if "validation failed" not in str(e) or attempt == 2:
                raise
            REJECTED["count"] += 1


def first_call(schemas, temperature):
    rows = []
    for q in REQUESTS:
        m = chat([{"role": "system", "content": PROMPT_AFTER}, {"role": "user", "content": q}], schemas, temperature)
        calls = [{"name": c.function.name, "arguments": json.loads(c.function.arguments or "{}")} for c in m.tool_calls or []]
        rows.append({"request": q, "calls": calls, "text": None if calls else (m.content or "")[:300]})
    return rows


def agent(question, schemas, prompt, search, temperature, check=False, max_steps=6):
    """Small loop for the check only (the agent loop of the project comes in a later session)."""
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]
    trace, known, checked = [], set(), False
    for _ in range(max_steps):
        m = chat(messages, schemas, temperature)
        messages.append({"role": "assistant", "content": m.content or "", **({"tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in m.tool_calls]} if m.tool_calls else {})})
        if not m.tool_calls:
            text = P.THINK.sub("", m.content or "").strip()
            problem = check and not checked and checks(prompt + " " + question, text, trace)
            if problem:
                checked = True
                trace.append({"name": f"(check in code: {problem}, sent back)"})
                fix = ("Call deadline, or ask the user for the missing date." if "deadline" in problem else
                       "Remove that date, or ask the user for it only if the answer cannot be given without it. Rules "
                       "and amounts returned by the tools can be given as they are.")
                messages.append({"role": "user", "content": f"[system check] {problem}. {fix}"})
                continue
            return {"answer": text, "trace": [{k: v for k, v in t.items() if k != "full"} for t in trace]}
        for c in m.tool_calls:
            if c.function.name == "search_texts":
                out = search(json.loads(c.function.arguments or "{}").get("query", ""))
                known |= set(re.findall(r'"id":\s*"([^"]+)"', out))
            elif c.function.name == "days_between":
                a = json.loads(c.function.arguments or "{}")
                try:
                    out = str((date.fromisoformat(a["end"]) - date.fromisoformat(a["start"])).days)
                except (KeyError, ValueError) as e:
                    out = f"Error: {e}"
            else:
                out = tools.call(c.function.name, c.function.arguments, known_ids=known, today=TODAY)
            trace.append({"name": c.function.name, "arguments": c.function.arguments, "result": out[:300], "full": out})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
    return {"answer": "(max steps)", "trace": [{k: v for k, v in t.items() if k != "full"} for t in trace]}


def test_passages(query):
    p = TEST_RULE if re.search(r"justif|absence|retard|rat", query, re.I) else LEAVE
    return json.dumps({"not_found": False, "missing_ids": [], "passages": [
        {"id": p["id"], "title": None, "source": "test", "version_date": None, "url": None, "text": p["text"]}]},
        ensure_ascii=False)


def real_passages(query):
    return tools.search_texts(tools.SearchTextsArgs(query=query)).model_dump_json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--real", action="store_true", help="also compare with the pipeline on the real corpus")
    a = ap.parse_args()
    t0, out = time.time(), {"date": str(date.today()), "today_fixed": str(TODAY), "model": P.CFG["model"]["name"],
                            "temperature": a.temperature, "runs": a.runs}

    now = tools.schemas()
    before_source = [WITH_SOURCE, now[1], now[2]]
    fc = {"current": [first_call(now, a.temperature) for _ in range(a.runs)],
          "with_source": [first_call(before_source, a.temperature) for _ in range(a.runs)]}
    s = {}
    for k, runs in fc.items():
        searches = [c for run in runs for row in run for c in row["calls"] if c["name"] == "search_texts"]
        s[k] = {"search_first": sum(bool(row["calls"]) and row["calls"][0]["name"] == "search_texts"
                                    for run in runs for i, row in enumerate(run) if i != 2),
                "out_of": 4 * a.runs,
                "request_3_no_tool": sum(not run[2]["calls"] for run in runs),
                "searches_restricted": sum(c["arguments"].get("source", "all") != "all" for c in searches),
                "searches": len(searches)}
    out["first_call"] = {"summary": s, "runs": fc}

    variants = {"before": ([now[0], DAYS_BETWEEN, now[2]], PROMPT_BEFORE, False),
                "after": (now, PROMPT_AFTER, True)}
    dl = {k: {"exam": [], "email": []} for k in variants}
    for k, (schemas, prompt, check) in variants.items():
        for _ in range(a.runs):
            dl[k]["exam"].append(agent(EXAM, schemas, prompt, test_passages, a.temperature, check))
            dl[k]["email"].append(agent(EMAIL, schemas, prompt, test_passages, a.temperature, check))
    summary = {}
    for k, r in dl.items():
        summary[k] = {
            "exam_date_tool_called": sum(any(t["name"] in ("deadline", "days_between") for t in x["trace"]) for x in r["exam"]),
            "exam_date_limit_stated": sum(bool(DATE_LIMIT.search(x["answer"])) for x in r["exam"]),
            "exam_check_triggered": sum(any(t["name"].startswith("(check") for t in x["trace"]) for x in r["exam"]),
            "email_asks_for_the_date": sum(not re.search(r"objet", x["answer"], re.I) for x in r["email"]),
            "email_june_2026": sum("2026-06" in {d[:7] for d in dates_in(x["answer"])[0]} | dates_in(x["answer"])[1]
                                   for x in r["email"]),
            "email_check_triggered": sum(any(t["name"].startswith("(check") for t in x["trace"]) for x in r["email"]),
            "email_sent_without_approval": sum(any(t["name"] == "send_email" and "pending_approval" not in t["result"]
                                                   and not t["result"].startswith("Error") for t in x["trace"]) for x in r["email"]),
            "out_of": a.runs}
    out["deadline"] = {"summary": summary, "runs": dl}

    if a.real:
        rows = []
        for q in REQUESTS:
            w = P.answer(q)
            ag = agent(q, now, PROMPT_AFTER, real_passages, 0.0, check=True)
            for t in ag["trace"]:          # the school regulations are private: keep the ids, not the texts
                if t["name"] == "search_texts":
                    t["result"] = "passage ids: " + ", ".join(re.findall(r'"id":\s*"([^"]+)"', t["result"]))
            rows.append({"request": q, "pipeline": {"answer": w["answer"], "sources": w["sources"]}, "agent": ag})
        out["real"] = rows

    out["tool_calls_refused_by_provider"] = REJECTED["count"]
    out["seconds"] = round(time.time() - t0)
    path = os.path.join(os.path.dirname(__file__), "runs", f"{date.today()}_s2_tools_check.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"first_call": s, "deadline": summary}, indent=1), "\n→", path)


if __name__ == "__main__":
    main()
