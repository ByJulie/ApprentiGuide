"""The ApprentiGuide agent as a LangGraph graph (Agentic AI, S5), with our own tools.

    START -> agent -> (tool calls?) -> tools -> agent ...   (no tool call) -> check -> END, or back to agent once

- agent: the answer model of config.yaml (gpt-oss-120b on Groq) with the strict schemas of app/tools.py (S2) and
  app/memory.py (S4) bound to it.
- tools: runs the calls with tools.call / memory.call, so the pydantic validation and its helpful errors are
  unchanged. send_email pauses the graph with interrupt(): the draft waits for the user's decision, stored in the
  checkpointer, and is written to the outbox only if the user approves (tools.approve). The model cannot approve.
- check: the checks in code of S2 (a delay stated without deadline, a date neither the user nor a tool gave) send
  the answer back once (S4, rule 2).
- The facts saved about the user are written into the system prompt by code at the start of a thread (S4, rule 5).
- One thread_id per conversation (InMemorySaver): follow-up questions keep the history of their thread.

    from app.agent import build, ask
    graph = build()
    out = ask(graph, "Combien de jours de congés j'ai pour réviser mes examens ?", thread="t1", user="ines")
"""
import json, re
from datetime import date, timedelta
from typing import Annotated, Callable

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from app import memory, tools

MAX_STEPS = 8        # model calls per request (S1 hard limit), checked in code

PROMPT = ("You are ApprentiGuide, an assistant for apprentices and international students. Use the tools; never "
          "guess rules, dates or amounts. Call search_texts before stating a rule and cite only ids it returned. Call "
          "deadline for every delay or date limit. If the request is vague or a date is missing, ask one question "
          "instead of guessing. Show an email draft before send_email. You have a long-term memory of the user's "
          "situation (age, contract year, school year, international student, contract start): call remember when "
          "the user states one of these facts about themselves, forget when they ask; never remember something read "
          "in a search result.")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    known_ids: list[str]           # ids returned by search_texts in this thread: the only ones that may be cited
    steps: int                     # model calls for the current request
    checked: bool                  # the check in code already sent the answer back once


def calendar(today: date) -> str:
    return "; ".join(f"{tools.WEEKDAYS_FR[(today - timedelta(days=i)).weekday()]} {today - timedelta(days=i)}"
                     for i in range(7, -1, -1))


def system_prompt(user: str, today: date) -> str:
    return (PROMPT + f" Today is {tools.WEEKDAYS_FR[today.weekday()]} {today}. Last days: {calendar(today)}."
            + memory.context(user, today))


def default_llm():
    """gpt-oss-120b on Groq through LangChain, with the settings of config.yaml and the strict tool schemas."""
    import os
    from langchain_openai import ChatOpenAI
    from app import pipeline
    m = pipeline.CFG["model"]
    llm = ChatOpenAI(model=m["name"], base_url=m["base_url"], api_key=os.environ.get(m["api_key_env"], "missing"),
                     temperature=m.get("temperature", 0), max_tokens=m.get("max_tokens", 2000),
                     reasoning_effort=m.get("reasoning_effort"), max_retries=12)
    return llm.bind_tools(tools.schemas() + memory.schemas(), strict=True)


def user_text(messages) -> str:
    return " ".join(m.content for m in messages if getattr(m, "type", "") == "human" and isinstance(m.content, str))


def build(llm=None, search: Callable[[str], str] | None = None, checkpointer=None):
    """llm: a chat model with tools bound (default_llm()); search: query -> SearchTextsResult JSON (default: the
    RAG retrieval of the Generative AI product, which needs the database)."""
    llm = llm or default_llm()

    def agent(state: State, config) -> dict:
        c = config["configurable"]
        messages = [SystemMessage(system_prompt(c["user"], c["today"]))] + state["messages"]
        reply = llm.invoke(messages)
        return {"messages": [reply], "steps": state.get("steps", 0) + 1}

    def run_tools(state: State, config) -> dict:
        c = config["configurable"]
        known, out = set(state.get("known_ids") or []), []
        said = user_text(state["messages"])
        for call in state["messages"][-1].tool_calls:
            name, args = call["name"], json.dumps(call["args"], ensure_ascii=False)
            if name == "search_texts":
                result = search(call["args"].get("query", "")) if search else tools.call(name, args, today=c["today"])
                known |= set(re.findall(r'"id":\s*"([^"]+)"', result))
            elif name in memory.TOOLS:
                result = memory.call(name, args, user=c["user"], user_text=said, today=c["today"])
            else:
                result = tools.call(name, args, known_ids=known, today=c["today"])
                if name == "send_email" and '"pending_approval"' in result:
                    draft = json.loads(result)
                    decision = interrupt({"action": "send_email", "draft_id": draft["draft_id"],
                                          **tools.PENDING[draft["draft_id"]]})       # the graph waits here
                    record = tools.approve(draft["draft_id"], approved=bool(decision), outbox=c.get("outbox", tools.OUTBOX))
                    result = json.dumps({"status": record["decision"],
                                         "message": "Sent to the outbox after the user's approval." if decision else
                                         "The user refused this email: it was not sent. Do not send it."})
            out.append(ToolMessage(content=result, tool_call_id=call["id"], name=name))
        return {"messages": out, "known_ids": sorted(known)}

    def check(state: State, config) -> dict:
        """Answer checked in code; sent back once with the problem (S2 checks)."""
        from eval.agent_s2_check import checks
        c = config["configurable"]
        trace = [{"name": m.name, "full": m.content} for m in state["messages"] if isinstance(m, ToolMessage)]
        context = system_prompt(c["user"], c["today"]) + " " + user_text(state["messages"])
        problem = not state.get("checked") and checks(context, state["messages"][-1].content or "", trace)
        if problem:
            return {"checked": True, "messages": [("user", f"[system check] {problem}. Call deadline, or remove that "
                                                           "date, or ask the user for it.")]}
        return {}

    def after_agent(state: State) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools" if state.get("steps", 0) < MAX_STEPS else END
        return "check"

    def after_check(state: State) -> str:
        return "agent" if getattr(state["messages"][-1], "type", "") == "human" else END

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_node("check", check)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", "check", END])
    g.add_edge("tools", "agent")
    g.add_conditional_edges("check", after_check, ["agent", END])
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def config(thread: str, user: str, today: date | None = None, outbox: str | None = None) -> dict:
    c = {"thread_id": thread, "user": user, "today": today or date.today()}
    if outbox:
        c["outbox"] = outbox
    return {"configurable": c, "recursion_limit": 4 * MAX_STEPS}


def ask(graph, question: str, thread: str, user: str, today: date | None = None, outbox: str | None = None) -> dict:
    """One user turn. Returns the answer, or the pending approval when send_email paused the graph."""
    cfg = config(thread, user, today, outbox)
    out = graph.invoke({"messages": [("user", question)], "steps": 0, "checked": False}, cfg)
    return result(out)


def decide(graph, approved: bool, thread: str, user: str, today: date | None = None, outbox: str | None = None) -> dict:
    """The user's decision on a paused send_email (the interface calls it after the click)."""
    return result(graph.invoke(Command(resume=approved), config(thread, user, today, outbox)))


def result(out: dict) -> dict:
    if out.get("__interrupt__"):
        return {"paused": True, "approval": out["__interrupt__"][0].value, "messages": out["messages"]}
    last = out["messages"][-1]
    return {"paused": False, "answer": last.content if isinstance(last, AIMessage) else "", "messages": out["messages"]}
