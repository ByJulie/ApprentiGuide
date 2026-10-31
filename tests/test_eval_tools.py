"""Tests of the evaluation lead's tools: judge parsing, agreement, test set review."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval import judge, review_testset


def test_verdict_plain_json():
    assert judge.parse_verdict('{"reason": "gives 43 %", "correct": true}') == {"correct": True, "reason": "gives 43 %"}


def test_verdict_with_thinking_and_text_around():
    text = '<think>{"correct": true}</think>Here is my grade:\n```json\n{"reason": "wrong figure", "correct": false}\n```'
    assert judge.parse_verdict(text)["correct"] is False


def test_verdict_malformed_json():
    assert judge.parse_verdict('{"reason": "missing "quote", "correct": true}')["correct"] is True


def test_verdict_unreadable():
    assert judge.parse_verdict("I think it is fine")["correct"] is None


def test_kappa():
    assert judge.cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert round(judge.cohen_kappa([True, True, False, False], [True, False, True, False]), 3) == 0.0


def test_judge_messages_contain_texts_and_answer():
    t = {"question": "Q ?", "expected_behaviour": "answer_with_citation", "answer": "ref", "gold_source": ["L1"]}
    msgs = judge.judge_messages(t, "the answer", {"L1": "texte officiel"})
    assert "texte officiel" in msgs[1]["content"] and "the answer" in msgs[1]["content"]
    assert "not in corpus" in judge.judge_messages({**t, "gold_source": ["X9"]}, "a", {})[1]["content"]


def row(**kw):
    return {"decision": "accept", "remove": "", "add": "", "answer": "", "note": "", **kw}


def test_review_accept_merges_and_removes():
    t = {"id": "Q1", "family": "in_corpus", "gold_source": ["A", "B"], "gold_added": ["C"], "gold_issue": "B wrong"}
    out = review_testset.apply([t], {"Q1": row(remove="B", add="D", answer="new", note="why")})[0]
    assert out["gold_source"] == ["A", "C", "D"] and out["answer"] == "new" and out["review_note"] == "why"
    assert "gold_added" not in out and "gold_issue" not in out and out["status"] == "validated"


def test_review_reject_and_remove_question():
    t = {"id": "Q1", "family": "in_corpus", "gold_source": "A", "gold_added": ["C"]}
    assert review_testset.apply([t], {"Q1": row(decision="reject")})[0]["gold_source"] == ["A"]
    assert review_testset.apply([t], {"Q1": row(decision="remove_question")}) == []


def test_review_no_answer_stays_null():
    t = {"id": "Q1", "family": "no_answer", "gold_source": None}
    assert review_testset.apply([t], {"Q1": row()})[0]["gold_source"] is None
