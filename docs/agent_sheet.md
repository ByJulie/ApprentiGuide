# Agent sheet: ApprentiGuide

From an assistant that answers to an agent that acts. The Generative AI product (cited answers from the Labour Code, the CESEDA, Service-Public.fr and the school regulations, with NOT FOUND when the texts do not answer) is the base of the agent built in the Agentic AI course. This sheet is completed after each session; the deliverables are listed at the end.

## User and goal

- **User:** apprentices, students looking for an apprenticeship and international students of the school, such as Inès (22, international student, starting year 4 of the PGE).
- **What the agent adds:** the Generative AI product cites the rule but cannot act on the user's situation. It does not know today's date, cannot compute a deadline or an amount, guesses when a question is vague (our main measured weakness), and cannot prepare a message to HR or the school.

## S1. Five real requests: workflow or agent?

A workflow follows steps fixed in code; an agent decides at each step which tool to call, from what it has observed. We keep a workflow whenever the steps are always the same: it is cheaper, faster and easier to test.

| # | Request (as a user would write it) | Decision | Why |
|---|---|---|---|
| 1 | « Combien de jours de congés j'ai pour réviser mes examens ? » | **Workflow** | One search and one cited answer: the steps are always the same, which is exactly our current RAG pipeline. |
| 2 | « J'ai raté mon examen mardi à cause d'un train en retard. Est-ce que je peux encore le justifier, et à qui ? » | **Agent** | It needs today's date, the school rule and a deadline computation, and which tools to call next depends on what the rule says. |
| 3 | « J'ai le droit à combien ? » | **Agent** | The request is vague, so the agent must first ask what the user means instead of guessing, which our RAG cannot do. |
| 4 | « J'ai 22 ans et je suis en deuxième année d'apprentissage : combien je vais gagner par mois, en euros ? » | **Workflow** | The steps are fixed: one search, and the amount is read in the official pay table (Service-Public F2918), never computed by the model. |
| 5 | « Prépare un email à mes RH pour poser mes 5 jours de congé d'examen avant ma session de juin. » | **Agent** | It combines the rule, the user's dates and a draft email, and sending it is a write action that must wait for the user's approval. |



## S2. The three main tools

| Tool | Read or write | What it does |
|---|---|---|
| `search_texts` | **Read** | Our Generative AI retrieval as a tool: it always searches the law and the school regulations together, and returns the 5 best passages with their article id, or not_found. |
| `deadline` | **Read** | From the date of an event and the delay written in the rule, returns the date limit and whether it has passed. The tool knows today's date, so the model never computes or compares dates. |
| `send_email` | **Write** | Prepares the email to HR or the school. It never sends: the draft waits for the user's approval in the interface. |

**Pydantic, in `app/tools.py`.** Each tool has a pydantic model for its arguments and one for its result. The schemas below are generated from these models (`python -m app.tools`), so the code and this sheet cannot drift. Before a tool runs, the model's arguments are parsed with `model_validate_json`; a validation error (an unknown unit, a missing `event_date`, an extra field, an invented article id) goes back to the model as the tool result, so it can correct the call. `"strict": true` is set: Groq checks each tool call against the schema and refuses one that does not match (400 "tool call validation failed"), and the call is then retried, as in the course lab. In the run of the results file, Groq refused no call (`tool_calls_refused_by_provider`: 0); an earlier run, before the retry was added, stopped on one refused call that was missing `event_time`. Tests: `tests/test_tools.py`.

**Rules written in code, not only in the description:**
- `deadline` takes today from the code. The day of the event is not counted. `working_days_mon_sat` (jours ouvrables) and `working_days_mon_fri` (jours ouvrés) skip the 11 public holidays of metropolitan France. `hours` without `event_time` gives the same time of day as the event: a missed exam on Tuesday with 48 hours gives Thursday, at the time of the exam.
- `send_email` only stores a draft and returns `pending_approval`. The email is written to the outbox (`data/private/outbox.json`, gitignored, no mailbox connected yet) only when the interface calls `approve()` after the user's click; the model has no tool to approve. This is the human approval asked in Session 7.
- `cited_ids` is checked by a pydantic validator: an id that `search_texts` did not return in the conversation is refused.

### `search_texts` (read)

Arguments, the schema given to the model:

```json
{
  "type": "function",
  "function": {
    "name": "search_texts",
    "description": "Search the official texts (Labour Code, CESEDA, Service-Public.fr) and the school regulations together. Returns up to 5 passages, each with its article id, title, version date and link, or not_found when no passage is relevant enough. Use it for any question about a rule, a right, a deadline or a procedure, before answering.",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "query": {
          "description": "The question or a few keywords, in French or English. Include an article id (for example L6222-35) if the user names one.",
          "maxLength": 500,
          "minLength": 3,
          "type": "string"
        }
      },
      "required": [
        "query"
      ],
      "type": "object"
    },
    "strict": true
  }
}
```

Result (`SearchTextsResult`):

| Field | Type | Meaning |
|---|---|---|
| `not_found` | bool | True when no passage is relevant enough (decided in code, reranker threshold) |
| `passages` | list[Passage] | At most 5 passages, empty when not_found |
| `passages[].id` | str | Article or fiche id, the only ids that may be cited |
| `passages[].title` | str or None | Title of the article or fiche |
| `passages[].source` | str | Labour Code, CESEDA, Service-Public.fr or school regulations |
| `passages[].version_date` | str or None | Date of the version in the corpus, YYYY-MM-DD |
| `passages[].url` | str or None | Link to the official page |
| `passages[].text` | str | The passage, as indexed |
| `missing_ids` | list[str] | Article ids named in the query that are not in the corpus |

### `deadline` (read)

Arguments, the schema given to the model:

```json
{
  "type": "function",
  "function": {
    "name": "deadline",
    "description": "Date limit of a delay written in a rule, counted from the date of an event, and whether it has passed today. The tool knows today's date. Use it for every deadline or delay (48 hours, 5 jours ouvrables, 2 months...). Never compute or compare dates yourself. The day of the event is not counted: counting starts the next day. calendar_days counts every day. working_days_mon_sat (jours ouvrables) counts Monday to Saturday, working_days_mon_fri (jours ouvrés) Monday to Friday, both without the 11 public holidays of metropolitan France. months gives the same day number n months later (the last day of the month if it does not exist). hours counts from event_time; without event_time, the limit is the same time of day as the event.",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "event_date": {
          "description": "Exact date of the event, YYYY-MM-DD, as given by the user or read in the calendar of the system prompt. If the user did not give it, ask for it, never guess.",
          "format": "date",
          "type": "string"
        },
        "event_time": {
          "anyOf": [
            {
              "pattern": "^([01]\\d|2[0-3]):[0-5]\\d$",
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "description": "Time of the event, HH:MM, only if the user gave it, else null. Used with hours."
        },
        "amount": {
          "description": "The number written in the rule, for example 48 for 48 hours.",
          "maximum": 400,
          "minimum": 1,
          "type": "integer"
        },
        "unit": {
          "description": "The unit written in the rule returned by search_texts: working_days_mon_sat for jours ouvrables, working_days_mon_fri for jours ouvrés.",
          "enum": [
            "hours",
            "calendar_days",
            "working_days_mon_sat",
            "working_days_mon_fri",
            "months"
          ],
          "type": "string"
        }
      },
      "required": [
        "event_date",
        "event_time",
        "amount",
        "unit"
      ],
      "type": "object"
    },
    "strict": true
  }
}
```

Result (`DeadlineResult`):

| Field | Type | Meaning |
|---|---|---|
| `deadline_date` | date | Date limit, YYYY-MM-DD |
| `deadline_weekday` | str | Day of the week in French, to copy as it is |
| `deadline_time` | str or None | HH:MM for hours with event_time; 'same time as the event' without it; null for days and months |
| `today` | date | Today, from the code |
| `passed` | bool | True when the date limit is before today |
| `days_left` | int | Calendar days from today to the deadline, negative when passed |
| `counting_rule` | str | How the days were counted |

### `send_email` (write)

Arguments, the schema given to the model:

```json
{
  "type": "function",
  "function": {
    "name": "send_email",
    "description": "Prepare an email for the user to their employer's HR or to the school. WRITE action: the email is shown to the user and is sent only after the user approves it in the interface. Use it only when the user asks to write to someone. cited_ids must come from search_texts.",
    "parameters": {
      "additionalProperties": false,
      "properties": {
        "to": {
          "description": "Who receives the email.",
          "enum": [
            "employer_hr",
            "school"
          ],
          "type": "string"
        },
        "subject": {
          "description": "Short subject line.",
          "maxLength": 150,
          "minLength": 3,
          "type": "string"
        },
        "body": {
          "description": "The email text, in the language of the user, signed with the user's name.",
          "maxLength": 5000,
          "minLength": 20,
          "type": "string"
        },
        "cited_ids": {
          "description": "Article ids returned by search_texts that the email relies on.",
          "items": {
            "type": "string"
          },
          "maxItems": 10,
          "type": "array"
        }
      },
      "required": [
        "to",
        "subject",
        "body",
        "cited_ids"
      ],
      "type": "object"
    },
    "strict": true
  }
}
```

Result (`SendEmailResult`):

| Field | Type | Meaning |
|---|---|---|
| `status` | Literal['pending_approval'] | Always pending: the tool never sends |
| `draft_id` | str | Id the interface uses to approve or refuse the draft |
| `message` | str | What happens next, for the model |


## S2 check

Script `eval/agent_s2_check.py`, results `eval/runs/2026-11-04_s2_tools_check.json` (gpt-oss-120b, temperature 1, 5 runs, today fixed to Wednesday 4 November 2026). These are indications on a few requests, not a proof; the full evaluation of the agent comes later.

- **First call on the five S1 requests:** in 5 runs, requests 1, 2, 4 and 5 started with `search_texts` in 20 of 20 first calls, and request 3 (« J'ai le droit à combien ? ») got a clarifying question with no tool call in 5 of 5 runs. With a `source` parameter (removed), 13 of 20 searches were restricted to one source.
- **Deadline, missed-exam request** (search_texts replaced by a test passage giving 48 hours): with a `days_between` tool, the model called it 0 times in 5 runs and computed the date limit itself: its answers gave Thursday 5, Friday 6 or Saturday 7 November. With `deadline` and the calendar of the last days in the prompt, it called `deadline` in 4 of 5 runs and the date limit returned by the tool was always right for the date it received; but in 2 runs the model turned « mardi » into Monday 2 November. The error left is turning a weekday into a date, not counting. Next fix: let `deadline` take a weekday (« mardi ») and resolve it in code.
- **Email request:** the exam date is not given. In 5 runs, 1 answer wrote « juin 2026 » (the session is in June 2027) before, and 1 after. After, the check sent an answer back once and 1 answer asked for the date instead of writing a draft. The check is not enough here: it runs once per answer and the model still drafts without the date. `send_email` never sent anything without approval.
- **Real corpus, five S1 requests, one run each** (`real` in the results): the agent and the Generative AI pipeline agree on requests 1, 2 (transport is not an accepted reason in the school regulations, so there is no delay to compute) and 5, and both give 1 138,88 € for request 4 in this run. On request 3, the agent asks what the user means; the pipeline guesses and lists several rights.
- The check in code (a conclusion about time without a `deadline` call, or a date the user and the tools did not give, sends the answer back once) lives in the test script for now; it moves to the agent loop in a later session.

**Not in the S2 tools:** the pay table of F2918 was read in the wrong column by the Generative AI pipeline (952,18 € instead of 1 138,88 € at 22 in year 2). It is a data problem, fixed by Julie (tables written one line per cell with their headers, commit `30a0a03`), not a missing tool.

## S3. Our tools as an MCP server

`server.py` exposes the two read tools of `app/tools.py` with FastMCP: `deadline` and `search_texts`. Their parameters reuse the pydantic fields of S2, so a client lists the same schemas as this sheet, and each result is the pydantic result model (MCP structured output). `send_email` is not exposed: it is a write tool and its approval belongs to the interface (Session 7).

| | |
|---|---|
| Transport | stdio by default (`python server.py`); Streamable HTTP with `MCP_PORT=8765 python server.py` (`127.0.0.1:8765/mcp`, for n8n) |
| Needs | `deadline`: fastmcp and pydantic only, so it runs in Colab. `search_texts`: the PostgreSQL database and the local models of the RAG; without them it returns a clear error. |
| Client config | `{"mcpServers": {"apprentiguide": {"command": "python", "args": ["server.py"]}}}`, the same JSON as Claude Desktop, VS Code and OpenCode |
| Tests | `tests/test_server.py` (in memory) and `eval/s3_mcp_check.py` (stdio, results in `eval/runs/2026-11-04_s3_mcp_check.json`) |

**Called from the Session 3 notebook.** After the cell "Server configurations", add:

```python
!git clone -q https://github.com/ByJulie/ApprentiGuide.git
!pip install -q fastmcp pydantic
SERVERS["apprentiguide"] = {"command": sys.executable, "args": ["ApprentiGuide/server.py"], "env": QUIET}
async with Client(config("apprentiguide")) as mc:
    print([t.name for t in await mc.list_tools()])
    print(text_of(await mc.call_tool("deadline", {"event_date": "2026-11-03", "event_time": None, "amount": 48, "unit": "hours"})))
answer = await mcp_agent("Mon examen était le mardi 3 novembre 2026 et le règlement donne 48 heures pour le justifier. "
                         "Jusqu'à quand ?", servers=("apprentiguide",))
```

**What the check showed (4 November):** the server lists 2 tools, both with an input and an output schema, and no resource or prompt. `deadline` with 48 hours from Tuesday 3 November returns jeudi 2026-11-05. A call with `event_date: "mardi"` and `unit: "jours"` is refused by the server's validation before our code runs. The notebook's agent loop, with the tools taken from the server: for the exam leave, it called `search_texts` and answered 5 jours ouvrables with L6222-35; for the trial period of a contract started 20 days ago, it found the rule of the first 45 days of practical training in the company and did not compute a date (they are days of presence, not calendar days). A fresh environment with only fastmcp and pydantic (as in Colab) runs `deadline`, and `search_texts` returns its error message.

