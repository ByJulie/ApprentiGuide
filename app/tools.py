"""Agent tools (Agentic AI, S2). Each tool has a pydantic model for its arguments and one for its result:
the JSON schemas given to the model are generated from them (docs/agent_sheet.md is printed by
`python -m app.tools`), and the model's arguments are validated before a tool runs. A validation error goes
back to the model as the tool result, so it can correct the call."""
import json, os, re, sys, uuid
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
OUTBOX = os.path.join(ROOT, "data", "private", "outbox.json")   # gitignored: emails hold personal data
WEEKDAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


class Args(BaseModel):
    # No field outside the schema: the same rule as "additionalProperties": false in strict mode.
    model_config = ConfigDict(extra="forbid")


# search_texts

class SearchTextsArgs(Args):
    query: str = Field(min_length=3, max_length=500,
                       description="The question or a few keywords, in French or English. Include an article id "
                                   "(for example L6222-35) if the user names one.")


class Passage(BaseModel):
    id: str = Field(description="Article or fiche id, the only ids that may be cited")
    title: str | None = Field(description="Title of the article or fiche")
    source: str = Field(description="Labour Code, CESEDA, Service-Public.fr or school regulations")
    version_date: str | None = Field(description="Date of the version in the corpus, YYYY-MM-DD")
    url: str | None = Field(description="Link to the official page")
    text: str = Field(description="The passage, as indexed")


class SearchTextsResult(BaseModel):
    not_found: bool = Field(description="True when no passage is relevant enough (decided in code, reranker threshold)")
    passages: list[Passage] = Field(description="At most 5 passages, empty when not_found")
    missing_ids: list[str] = Field(description="Article ids named in the query that are not in the corpus")


def search_texts(args: SearchTextsArgs, **_) -> SearchTextsResult:
    """The Generative AI retrieval, unchanged: French rewrite, three signals, RRF, reranker, threshold."""
    from app import pipeline
    found = pipeline.retrieve(args.query)
    if found.abstain:
        return SearchTextsResult(not_found=True, passages=[], missing_ids=found.missing_ids)
    return SearchTextsResult(not_found=False, missing_ids=found.missing_ids, passages=[
        Passage(id=s["id"], title=s.get("title"), source=s.get("source_label") or "", version_date=s.get("version_date"),
                url=s.get("url"), text=s["text"]) for s in found.sources])


# deadline

def easter(year: int) -> date:
    """Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l) // 433
    month = (h + l - 7 * m + 90) // 25
    return date(year, month, (h + l - 7 * m + 33 * month + 19) % 32)


def public_holidays(year: int) -> set[date]:
    """The 11 public holidays of metropolitan France (Labour Code L3133-1)."""
    e = easter(year)
    fixed = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    return {date(year, m, d) for m, d in fixed} | {e + timedelta(days=1), e + timedelta(days=39), e + timedelta(days=50)}


def counts(day: date, unit: str) -> bool:
    if unit == "calendar_days":
        return True
    last = 5 if unit == "working_days_mon_sat" else 4          # Saturday is a working day only Mon-Sat
    return day.weekday() <= last and day not in public_holidays(day.year)


def add_months(day: date, n: int) -> date:
    m = day.month - 1 + n
    y, m = day.year + m // 12, m % 12 + 1
    for d in (day.day, 30, 29, 28):                            # 31 January + 1 month = end of February
        try:
            return date(y, m, d)
        except ValueError:
            continue


DEADLINE_RULE = ("The day of the event is not counted: counting starts the next day. calendar_days counts every day. "
                 "working_days_mon_sat (jours ouvrables) counts Monday to Saturday, working_days_mon_fri (jours ouvrés) "
                 "Monday to Friday, both without the 11 public holidays of metropolitan France. months gives the same "
                 "day number n months later (the last day of the month if it does not exist). hours counts from "
                 "event_time; without event_time, the limit is the same time of day as the event.")


class DeadlineArgs(Args):
    event_date: date = Field(description="Exact date of the event, YYYY-MM-DD, as given by the user or read in the "
                                         "calendar of the system prompt. If the user did not give it, ask for it, never guess.")
    event_time: str | None = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
                                   description="Time of the event, HH:MM, only if the user gave it, else null. Used with hours.")
    amount: int = Field(ge=1, le=400, description="The number written in the rule, for example 48 for 48 hours.")
    unit: Literal["hours", "calendar_days", "working_days_mon_sat", "working_days_mon_fri", "months"] = Field(
        description="The unit written in the rule returned by search_texts: working_days_mon_sat for jours ouvrables, "
                    "working_days_mon_fri for jours ouvrés.")


class DeadlineResult(BaseModel):
    deadline_date: date = Field(description="Date limit, YYYY-MM-DD")
    deadline_weekday: str = Field(description="Day of the week in French, to copy as it is")
    deadline_time: str | None = Field(description="HH:MM for hours with event_time; 'same time as the event' without it; null for days and months")
    today: date = Field(description="Today, from the code")
    passed: bool = Field(description="True when the date limit is before today")
    days_left: int = Field(description="Calendar days from today to the deadline, negative when passed")
    counting_rule: str = Field(description="How the days were counted")


def deadline(args: DeadlineArgs, today: date | None = None, **_) -> DeadlineResult:
    """Today comes from the code, never from the model."""
    today = today or date.today()
    time = None
    if args.unit == "hours":
        start = datetime.combine(args.event_date, datetime.strptime(args.event_time or "00:00", "%H:%M").time())
        end = start + timedelta(hours=args.amount)
        limit, time = end.date(), end.strftime("%H:%M") if args.event_time else "same time as the event"
    elif args.unit == "months":
        limit = add_months(args.event_date, args.amount)
    else:
        limit, n = args.event_date, 0
        while n < args.amount:
            limit += timedelta(days=1)
            n += counts(limit, args.unit)
    return DeadlineResult(deadline_date=limit, deadline_weekday=WEEKDAYS_FR[limit.weekday()], deadline_time=time,
                          today=today, passed=limit < today, days_left=(limit - today).days, counting_rule=DEADLINE_RULE)


# send_email

class SendEmailArgs(Args):
    to: Literal["employer_hr", "school"] = Field(description="Who receives the email.")
    subject: str = Field(min_length=3, max_length=150, description="Short subject line.")
    body: str = Field(min_length=20, max_length=5000,
                      description="The email text, in the language of the user, signed with the user's name.")
    cited_ids: list[str] = Field(max_length=10, description="Article ids returned by search_texts that the email relies on.")

    @field_validator("cited_ids")
    @classmethod
    def ids_were_returned(cls, ids: list[str], info: ValidationInfo) -> list[str]:
        """An id the model did not receive from search_texts in this conversation is invented."""
        known = (info.context or {}).get("known_ids", set())
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise ValueError(f"not returned by search_texts in this conversation: {', '.join(unknown)}")
        return ids


class SendEmailResult(BaseModel):
    status: Literal["pending_approval"] = Field(description="Always pending: the tool never sends")
    draft_id: str = Field(description="Id the interface uses to approve or refuse the draft")
    message: str = Field(description="What happens next, for the model")


PENDING: dict[str, dict] = {}


def send_email(args: SendEmailArgs, **_) -> SendEmailResult:
    """Never sends: the draft waits for approve(), which only the interface calls. The model cannot approve."""
    draft_id = uuid.uuid4().hex[:8]
    PENDING[draft_id] = args.model_dump()
    return SendEmailResult(status="pending_approval", draft_id=draft_id,
                           message="Shown to the user for approval. Not sent until the user approves it in the interface.")


def approve(draft_id: str, approved: bool, edited_body: str | None = None, outbox: str = OUTBOX) -> dict:
    """Called by the interface after the user's click, not a tool. An approved email is written to the outbox
    (no mailbox is connected yet); a refused one is dropped. Both decisions are kept."""
    draft = PENDING.pop(draft_id)
    record = {**draft, "body": edited_body or draft["body"], "draft_id": draft_id,
              "decision": "approved" if approved else "refused", "at": datetime.now().isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(outbox), exist_ok=True)
    log = json.load(open(outbox)) if os.path.exists(outbox) else []
    json.dump(log + [record], open(outbox, "w"), ensure_ascii=False, indent=1)
    return record


# Registry, schemas and calls

TOOLS = {
    "search_texts": dict(kind="read", args=SearchTextsArgs, result=SearchTextsResult, run=search_texts, description=(
        "Search the official texts (Labour Code, CESEDA, Service-Public.fr) and the school regulations together. "
        "Returns up to 5 passages, each with its article id, title, version date and link, or not_found when no "
        "passage is relevant enough. Use it for any question about a rule, a right, a deadline or a procedure, "
        "before answering.")),
    "deadline": dict(kind="read", args=DeadlineArgs, result=DeadlineResult, run=deadline, description=(
        "Date limit of a delay written in a rule, counted from the date of an event, and whether it has passed "
        "today. The tool knows today's date. Use it for every deadline or delay (48 hours, 5 jours ouvrables, "
        "2 months...). Never compute or compare dates yourself. " + DEADLINE_RULE)),
    "send_email": dict(kind="write", args=SendEmailArgs, result=SendEmailResult, run=send_email, description=(
        "Prepare an email for the user to their employer's HR or to the school. WRITE action: the email is shown "
        "to the user and is sent only after the user approves it in the interface. Use it only when the user "
        "asks to write to someone. cited_ids must come from search_texts.")),
}


def _clean(schema: dict) -> dict:
    """Pydantic adds a title to every field; the model does not need them."""
    if isinstance(schema, dict):
        return {k: _clean(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_clean(v) for v in schema]
    return schema


def schema(name: str, strict: bool = True) -> dict:
    t = TOOLS[name]
    f = {"name": name, "description": t["description"], "parameters": _clean(t["args"].model_json_schema())}
    if strict:
        f["strict"] = True
    return {"type": "function", "function": f}


def schemas(strict: bool = True) -> list[dict]:
    return [schema(n, strict) for n in TOOLS]


def call(name: str, arguments: str, known_ids: set[str] = frozenset(), today: date | None = None) -> str:
    """Runs one tool call of the model. The result, or a helpful error, is the content of the "tool" message."""
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'. Tools: {', '.join(TOOLS)}."
    t = TOOLS[name]
    try:
        args = t["args"].model_validate_json(arguments or "{}", context={"known_ids": set(known_ids)})
    except ValidationError as e:
        errors = "; ".join(f"{'.'.join(map(str, x['loc'])) or 'arguments'}: {x['msg']}" for x in e.errors())
        return f"Error: invalid arguments for {name}: {errors}. Fix them and call again."
    return t["run"](args, today=today).model_dump_json()


def _type(annotation) -> str:
    return re.sub(r"\b(app\.tools\.|__main__\.|typing\.|datetime\.)", "", str(getattr(annotation, "__name__", None)
                  if isinstance(annotation, type) else annotation)).replace(" | ", " or ")


def result_table(model: type[BaseModel]) -> str:
    """The result model as a table of fields (the full JSON schema is model.model_json_schema())."""
    rows = []
    for name, f in model.model_fields.items():
        rows.append(f"| `{name}` | {_type(f.annotation)} | {f.description or ''} |")
        sub = getattr(f.annotation, "__args__", [None])[0]
        if isinstance(sub, type) and issubclass(sub, BaseModel):
            rows += [f"| `{name}[].{k}` | {_type(g.annotation)} | {g.description or ''} |" for k, g in sub.model_fields.items()]
    return "| Field | Type | Meaning |\n|---|---|---|\n" + "\n".join(rows)


if __name__ == "__main__":
    # The schemas of docs/agent_sheet.md: python -m app.tools
    for n, t in TOOLS.items():
        print(f"### `{n}` ({t['kind']})\n\nArguments, the schema given to the model:\n\n```json\n"
              f"{json.dumps(schema(n), indent=2, ensure_ascii=False)}\n```\n\nResult (`{t['result'].__name__}`):\n\n"
              f"{result_table(t['result'])}\n")
