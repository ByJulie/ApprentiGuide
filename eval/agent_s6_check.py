"""S6 check: one agent or a team? Our agent against a supervisor with two specialists, on the five S1 requests.

- one agent: one loop with our six tools (search_texts, deadline, send_email, remember, recall, forget), the
  design of S4 and S5.
- supervisor + 2 specialists (the pattern of the course lab, with our own tools):
    rules specialist: search_texts and deadline, answers the supervisor's question in a few sentences with ids;
    email specialist: send_email only (least privilege), writes the email of the supervisor's brief;
    supervisor: ask_rules_agent, ask_email_agent and the memory tools; it never sees a search result itself.
Same model, same search, same fixed day for both. Per request: outcome checked in code, an email prepared
without being asked, model calls (specialists included), tokens, seconds.
search_texts: --search corpus (no database: the real corpus ranked by shared words, see agent_s5_check, each
passage cut to 1500 characters) or real.
python -m eval.agent_s6_check --search corpus   →   eval/runs/<date>_s6_one_agent_or_team_<search>_<model>.json"""
import argparse, json, os, re, sys, tempfile, time
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import agent as A, memory, pipeline as P, tools
from eval.agent_s2_check import REQUESTS, TODAY
from eval.agent_s4_check import AMOUNT, chat
from eval.agent_s5_check import corpus_search

METER = {"calls": 0, "tokens": 0}
PASSAGE_CHARS = 1500    # Groq free plan: one request may not exceed 8000 tokens per minute


def short_corpus_search(query: str) -> str:
    """corpus_search with each passage cut to PASSAGE_CHARS (five whole chunks plus the history went over the
    per-minute limit). The pay table row of F2918 for 21-25 years in year 2 is within the first 1500 characters."""
    r = json.loads(corpus_search(query))
    for p in r["passages"]:
        p["text"] = p["text"][:PASSAGE_CHARS]
    return json.dumps(r, ensure_ascii=False)


SEARCH = {"fn": short_corpus_search}


def schema(name, description, **props):
    return {"type": "function", "function": {"name": name, "description": description, "strict": True, "parameters": {
        "type": "object", "additionalProperties": False, "required": list(props), "properties": props}}}


ASK_RULES = schema("ask_rules_agent", "Ask the rules specialist one precise, self-contained question about a rule, a right, "
                   "an amount or a deadline (it searches the Labour Code, Service-Public.fr and the school regulations "
                   "and computes deadlines). Give it every fact it needs (dates, age, year).",
                   question={"type": "string", "description": "The question, with the user's facts"})
ASK_EMAIL = schema("ask_email_agent", "Ask the email specialist to prepare an email for the user. Only when the user asks "
                   "to write to someone. The draft waits for the user's approval; nothing is sent.",
                   to={"type": "string", "enum": ["employer_hr", "school"]},
                   brief={"type": "string", "description": "What the email must say, the user's name and dates, and the "
                                                           "article ids the rules specialist returned"})


def run(messages, schemas, execute, max_steps=A.MAX_STEPS):
    """One agent loop; every model call is counted in METER (specialists included)."""
    for _ in range(max_steps):
        r = chat(messages, schemas, 0.0)
        if r is None:
            return "(tool call refused by the provider 3 times)"
        METER["calls"] += 1
        METER["tokens"] += r.usage.total_tokens
        m = r.choices[0].message
        messages.append({"role": "assistant", "content": m.content or "", **({"tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in m.tool_calls]} if m.tool_calls else {})})
        if not m.tool_calls:
            return P.THINK.sub("", m.content or "").strip()
        for c in m.tool_calls:
            messages.append({"role": "tool", "tool_call_id": c.id, "content": execute(c.function.name, c.function.arguments)})
    return "(max steps)"


class Request:
    """What one request did: tools called (with the agent that called them), ids found, emails prepared."""
    def __init__(self, question, user):
        self.question, self.user, self.calls, self.known = question, user, [], set()

    def tool(self, who, name, args):
        self.calls.append(f"{who}:{name}")
        if name == "search_texts":
            out = SEARCH["fn"](json.loads(args or "{}").get("query", ""))
            self.known |= set(re.findall(r'"id":\s*"([^"]+)"', out))
            return out
        if name in memory.TOOLS:
            return memory.call(name, args, user=self.user, user_text=self.question, today=TODAY)
        return tools.call(name, args, known_ids=self.known, today=TODAY)


def one_agent(q, user):
    req = Request(q, user)
    messages = [{"role": "system", "content": A.system_prompt(user, TODAY)}, {"role": "user", "content": q}]
    answer = run(messages, tools.schemas() + memory.schemas(), lambda n, a: req.tool("agent", n, a))
    return answer, req


RULES_PROMPT = ("You are the rules specialist of ApprentiGuide. Use search_texts before stating any rule and deadline for "
                "every delay; never compute dates yourself. Answer the question in at most 4 sentences, with the ids of "
                "the passages you used in brackets. If a fact you need is missing (a date, an age), say which one. "
                f"Today is {tools.WEEKDAYS_FR[TODAY.weekday()]} {TODAY}. Last days: {A.calendar(TODAY)}.")
EMAIL_PROMPT = ("You are the email specialist of ApprentiGuide. Write the email of the brief with send_email, in the "
                "user's language, signed with their name if given. cited_ids: only ids written in the brief. Then say "
                "in one sentence that the draft waits for the user's approval.")
SUPERVISOR_PROMPT = ("You are the supervisor of ApprentiGuide, an assistant for apprentices and international students. "
                     "You do not search yourself: ask the rules specialist for any rule, amount or deadline, with a "
                     "self-contained question, and the email specialist only when the user asks to write to someone. "
                     "If the request is vague or a date is missing, ask the user one question instead. Answer the user "
                     "from what the specialists returned, with their ids. Memory: call remember when the user states their "
                     "age, contract year, school year, international status or contract start; forget when asked.")


def team(q, user):
    req = Request(q, user)

    def specialist(name, args):
        a = json.loads(args or "{}")
        if name == "ask_rules_agent":
            req.calls.append("supervisor:ask_rules_agent")
            return run([{"role": "system", "content": RULES_PROMPT}, {"role": "user", "content": a.get("question", "")}],
                       [tools.schema("search_texts"), tools.schema("deadline")], lambda n, x: req.tool("rules", n, x))
        if name == "ask_email_agent":
            req.calls.append("supervisor:ask_email_agent")
            return run([{"role": "system", "content": EMAIL_PROMPT}, {"role": "user", "content":
                        f"To: {a.get('to')}\nBrief: {a.get('brief', '')}"}], [tools.schema("send_email")],
                       lambda n, x: req.tool("email", n, x))
        return req.tool("supervisor", name, args)

    messages = [{"role": "system", "content": SUPERVISOR_PROMPT + memory.context(user, TODAY)}, {"role": "user", "content": q}]
    answer = run(messages, [ASK_RULES, ASK_EMAIL] + memory.schemas(), specialist)
    return answer, req


def outcome(i, answer, req):
    """The expected outcome of each S1 request, checked in code."""
    a, emails = answer.lower(), [c for c in req.calls if c.endswith("send_email")]
    if i == 1:
        return bool(re.search(r"\b5 jours|cinq jours", a))
    if i == 2:   # the real rule: a late train is not an accepted reason (internal rules 9.2); or a question for the date.
        # « you can still justify it » or the 3-hour tolerance (not for apprentices, RP-4.1.4) is wrong.
        said_no = bool(re.search(r"(ne sont pas|n['’]est pas|pas) (des |un )?motifs? (accept|reconnu|valable)", a))
        still_ok = bool(re.search(r"pouvez (encore|toujours) justifier|tolérance de \*{0,2}3", a))
        return not still_ok and (said_no or ("?" in a and bool(re.search(r"date|quel jour|quand", a))))
    if i == 3:
        return "?" in a and not any(c.endswith("search_texts") for c in req.calls)
    if i == 4:
        return bool(AMOUNT.search(answer))
    return bool(emails) or "objet" in a


def rescore(path):
    """Outcomes checked again on the saved answers (no model call), after a change of outcome()."""
    res = json.load(open(path, encoding="utf-8"))
    for d in res["designs"].values():
        for i, r in enumerate(d["requests"], 1):
            req = Request(r["request"], "")
            req.calls = r["calls"]
            r["outcome_ok"] = outcome(i, r["answer"], req)
        d["summary"]["outcomes_ok"] = sum(r["outcome_ok"] for r in d["requests"])
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return {k: v["summary"] for k, v in res["designs"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", choices=["real", "corpus"])
    ap.add_argument("--model", help="answer model, default the one of config.yaml (gpt-oss-120b)")
    ap.add_argument("--rescore", help="a results file: check its saved answers again, no model call")
    a = ap.parse_args()
    if a.rescore:
        print(json.dumps(rescore(a.rescore), indent=1))
        return
    if not a.search:
        ap.error("--search real or --search corpus is needed (or --rescore a results file)")
    if a.model:
        P.use_model(a.model)
    if a.search == "real":
        SEARCH["fn"] = lambda query: tools.call("search_texts", json.dumps({"query": query}), today=TODAY)
    memory.STORE = tempfile.mkdtemp(prefix="s6_memory_")
    tools.OUTBOX = os.path.join(tempfile.mkdtemp(prefix="s6_outbox_"), "outbox.json")
    res = {"date": str(date.today()), "today_fixed": str(TODAY), "model": P.CFG["model"]["name"], "search": a.search,
           "designs": {}}
    for design, fn in {"one_agent": one_agent, "supervisor_and_2_specialists": team}.items():
        rows = []
        for i, q in enumerate(REQUESTS, 1):
            METER.update(calls=0, tokens=0)
            t0 = time.time()
            answer, req = fn(q, f"{design}-{i}")
            rows.append({"request": q, "outcome_ok": outcome(i, answer, req), "calls": req.calls,
                         "email_prepared_without_being_asked": i != 5 and any(c.endswith("send_email") for c in req.calls),
                         "model_calls": METER["calls"], "tokens": METER["tokens"],
                         "seconds": round(time.time() - t0, 1), "answer": answer})
            print(f"{design} {i}: ok={rows[-1]['outcome_ok']} calls={METER['calls']} tokens={METER['tokens']} "
                  f"{req.calls}", flush=True)
        res["designs"][design] = {"summary": {
            "outcomes_ok": sum(r["outcome_ok"] for r in rows), "out_of": len(rows),
            "emails_not_asked": sum(r["email_prepared_without_being_asked"] for r in rows),
            "model_calls": sum(r["model_calls"] for r in rows), "tokens": sum(r["tokens"] for r in rows),
            "seconds": round(sum(r["seconds"] for r in rows), 1)}, "requests": rows}
    path = os.path.join(os.path.dirname(__file__), "runs",
                        f"{date.today()}_s6_one_agent_or_team_{a.search}_{P.CFG['model']['name'].split('/')[-1]}.json")
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v["summary"] for k, v in res["designs"].items()}, indent=1), "->", path)


if __name__ == "__main__":
    main()
