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
| 4 | « J'ai 22 ans et je suis en deuxième année d'apprentissage : combien je vais gagner par mois, en euros ? » | **Workflow** | The steps are fixed: find the pay rule, then compute the amount with a calculation tool, never by the model. |
| 5 | « Prépare un email à mes RH pour poser mes 5 jours de congé d'examen avant ma session de juin. » | **Agent** | It combines the rule, the user's dates and a draft email, and sending it is a write action that must wait for the user's approval. |


## S2. The three main tools

| Tool | Read or write | What it does |
|---|---|---|
| `search_texts` | **Read** | Our Generative AI retrieval as a tool: it always searches the law and the school regulations together, and returns the 5 best passages with their article id, or NOT FOUND. |
| `deadline` | **Read** | From the date of an event and the delay written in the rule, returns the date limit and whether it has passed. The tool knows today's date, so the model never computes or compares dates. |
| `send_email` | **Write** | Sends the email to HR or the school. It runs only after the user has read and approved the draft. |

### `search_texts`

```json
{
  "type": "function",
  "function": {
    "name": "search_texts",
    "description": "Search the official texts (Labour Code, CESEDA, Service-Public.fr) and the school regulations together. Returns up to 5 passages, each with its article id, title, version date and link, or NOT FOUND when no passage is relevant enough. Use it for any question about a rule, a right, a deadline or a procedure, before answering.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The question or a few keywords, in French or English. Include an article id (for example L6222-35) if the user names one."
        }
      },
      "required": [
        "query"
      ],
      "additionalProperties": false
    }
  }
}
```

### `deadline`

```json
{
  "type": "function",
  "function": {
    "name": "deadline",
    "description": "Date limit of a delay written in a rule, counted from the date of an event, and whether it has passed today. The tool knows today's date. Use it for every deadline or delay (48 hours, 5 jours ouvrables, 2 months...). Never compute or compare dates yourself.",
    "parameters": {
      "type": "object",
      "properties": {
        "event_date": {
          "type": "string",
          "description": "Exact date of the event, YYYY-MM-DD, as given by the user or read in the calendar of the system prompt. If the user did not give it, ask for it, never guess."
        },
        "amount": {
          "type": "number",
          "description": "The number written in the rule, for example 48 for 48 hours."
        },
        "unit": {
          "type": "string",
          "enum": [
            "hours",
            "calendar_days",
            "jours_ouvrables",
            "jours_ouvres",
            "months"
          ],
          "description": "The unit written in the rule returned by search_texts."
        }
      },
      "required": [
        "event_date",
        "amount",
        "unit"
      ],
      "additionalProperties": false
    }
  }
}
```

### `send_email`

```json
{
  "type": "function",
  "function": {
    "name": "send_email",
    "description": "Send an email written for the user to their employer's HR or to the school. WRITE action: it runs only after the user has read the draft and approved it. Use it only when the user asks to write to someone. The body must cite the article ids it relies on.",
    "parameters": {
      "type": "object",
      "properties": {
        "to": {
          "type": "string",
          "enum": [
            "employer_hr",
            "school"
          ],
          "description": "Who receives the email."
        },
        "subject": {
          "type": "string",
          "description": "Short subject line."
        },
        "body": {
          "type": "string",
          "description": "The email text, in the language of the user, signed with the user's name."
        },
        "cited_ids": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Article ids returned by search_texts that the email relies on."
        }
      },
      "required": [
        "to",
        "subject",
        "body",
        "cited_ids"
      ],
      "additionalProperties": false
    }
  }
}
```

**Check (gpt-oss-120b, temperature 0, the five S1 requests, first response only):** requests 1, 2, 4 and 5 start with `search_texts`, request 3 (« J'ai le droit à combien ? ») gets a clarifying question and no tool call. A call with the three schemas, the system prompt and the request costs about 610 prompt tokens. A first version let the model choose one source: it restricted 3 searches out of 4 to the Labour Code and missed the school regulations, so the parameter was removed.

**Why `deadline` and not `days_between`:** with a `days_between(start, end)` tool, the model still had to turn « mardi » into a date and compare it with today. On the missed-exam request (5 runs at temperature 1, with a test passage giving a delay of 48 hours), it called the tool 2 times out of 5 and never stated the date limit. With `deadline`, a calendar of the last days in the system prompt and a check in code (an answer that states a delay without a `deadline` call is sent back once), it called `deadline` 5 times out of 5 and gave the right date limit, jeudi 5 novembre, 5 times out of 5. For the email request, it asked for the exam date 5 times out of 5 instead of guessing it (before: one draft dated « juin 2026 »).
