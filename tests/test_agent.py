"""The LangGraph agent (S5) with a scripted model: routing, our tools, human approval, the check in code."""
import json, os, sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from langchain_core.messages import AIMessage

from app import agent, memory

TODAY = date(2026, 11, 4)
RULE = json.dumps({"not_found": False, "missing_ids": [], "passages": [{"id": "L6222-35", "title": None, "source": "test",
                   "version_date": None, "url": None, "text": "cinq jours ouvrables"}]})


class Scripted:
    """Returns the planned messages in order, and records what it was sent."""
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def invoke(self, messages):
        self.seen.append(messages)
        return self.replies.pop(0)


def call(name, args, i=1):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}", "type": "tool_call"}])


def test_search_then_answer_and_threads_keep_history(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "STORE", str(tmp_path))
    llm = Scripted(call("search_texts", {"query": "congé examens"}), AIMessage(content="5 jours ouvrables [L6222-35]."),
                   AIMessage(content="Oui, payés."))
    g = agent.build(llm=llm, search=lambda q: RULE)
    out = agent.ask(g, "Combien de jours pour réviser ?", thread="t1", user="ines", today=TODAY)
    assert out["answer"].startswith("5 jours") and not out["paused"]
    agent.ask(g, "Et ils sont payés ?", thread="t1", user="ines", today=TODAY)
    assert any("Combien de jours" in getattr(m, "content", "") for m in llm.seen[-1])     # same thread: history kept


def test_send_email_waits_for_the_user_and_refused_is_not_sent(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "STORE", str(tmp_path))
    outbox = str(tmp_path / "outbox.json")
    llm = Scripted(call("search_texts", {"query": "congé examens"}),
                   call("send_email", {"to": "employer_hr", "subject": "Congé examens", "cited_ids": ["L6222-35"],
                                       "body": "Bonjour, je souhaite poser mes 5 jours de congé d'examen. Inès"}, 2),
                   AIMessage(content="Email non envoyé."))
    g = agent.build(llm=llm, search=lambda q: RULE)
    out = agent.ask(g, "Écris à mes RH pour mes 5 jours.", thread="t2", user="ines", today=TODAY, outbox=outbox)
    assert out["paused"] and out["approval"]["action"] == "send_email" and not os.path.exists(outbox)
    done = agent.decide(g, False, thread="t2", user="ines", today=TODAY, outbox=outbox)
    assert done["answer"] == "Email non envoyé."
    assert json.load(open(outbox))[0]["decision"] == "refused"


def test_the_check_sends_an_unchecked_delay_back_once(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "STORE", str(tmp_path))
    llm = Scripted(call("search_texts", {"query": "absence examen"}),
                   AIMessage(content="Vous pouvez encore le justifier jusqu'au jeudi."),
                   call("deadline", {"event_date": "2026-11-03", "event_time": None, "amount": 48, "unit": "hours"}, 3),
                   AIMessage(content="Vous pouvez encore le justifier jusqu'au jeudi 5 novembre."))
    g = agent.build(llm=llm, search=lambda q: RULE)
    out = agent.ask(g, "J'ai raté mon examen mardi, je peux le justifier ?", thread="t3", user="ines", today=TODAY)
    assert "5 novembre" in out["answer"]
    assert any("[system check]" in getattr(m, "content", "") for m in out["messages"])


def test_saved_facts_are_in_the_system_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "STORE", str(tmp_path))
    memory.call("remember", json.dumps({"kind": "age", "value": "22"}), user="ines", user_text="j'ai 22 ans", today=TODAY)
    llm = Scripted(AIMessage(content="ok"))
    agent.ask(agent.build(llm=llm, search=lambda q: RULE), "Combien je gagne ?", thread="t4", user="ines", today=TODAY)
    assert "age = 22" in llm.seen[0][0].content
