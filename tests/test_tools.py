import json
from datetime import date

import pytest

from app import tools
from app.tools import DeadlineArgs, call, deadline, public_holidays

TODAY = date(2026, 11, 4)          # a Wednesday


def run(**kw):
    args = {"event_time": None, **kw}
    return json.loads(call("deadline", json.dumps(args), today=TODAY))


def test_48_hours_from_tuesday_is_thursday_same_time():
    r = run(event_date="2026-11-03", amount=48, unit="hours")
    assert (r["deadline_date"], r["deadline_weekday"], r["deadline_time"]) == ("2026-11-05", "jeudi", "same time as the event")
    assert not r["passed"] and r["days_left"] == 1


def test_48_hours_with_a_time():
    r = run(event_date="2026-11-03", event_time="10:00", amount=48, unit="hours")
    assert (r["deadline_date"], r["deadline_time"]) == ("2026-11-05", "10:00")


def test_working_days_skip_sundays_and_11_november():
    # From Friday 6 Nov.: Sat 7, Mon 9, Tue 10, (Wed 11 is a holiday), Thu 12, Fri 13
    assert run(event_date="2026-11-06", amount=5, unit="working_days_mon_sat")["deadline_date"] == "2026-11-13"
    # Monday to Friday also skips Saturday 7: Mon 9, Tue 10, Thu 12, Fri 13, Mon 16
    assert run(event_date="2026-11-06", amount=5, unit="working_days_mon_fri")["deadline_date"] == "2026-11-16"
    assert run(event_date="2026-11-06", amount=5, unit="calendar_days")["deadline_date"] == "2026-11-11"


def test_months_end_of_month_and_passed():
    assert run(event_date="2027-01-31", amount=1, unit="months")["deadline_date"] == "2027-02-28"
    r = run(event_date="2026-09-01", amount=1, unit="months")
    assert r["deadline_date"] == "2026-10-01" and r["passed"] and r["days_left"] == -34


def test_public_holidays_include_easter_based_days():
    h = public_holidays(2027)
    assert len(h) == 11
    assert {date(2027, 3, 29), date(2027, 5, 6), date(2027, 5, 17)} <= h   # Easter Monday, Ascension, Whit Monday


def test_today_comes_from_the_code_not_the_model():
    with pytest.raises(Exception):
        DeadlineArgs.model_validate({"event_date": "2026-11-03", "event_time": None, "amount": 2,
                                     "unit": "calendar_days", "today": "2026-11-01"})


@pytest.mark.parametrize("args, field", [
    ({"event_date": "2026-11-03", "event_time": None, "amount": 5, "unit": "jours"}, "unit"),
    ({"event_time": None, "amount": 48, "unit": "hours"}, "event_date"),
    ({"event_date": "mardi", "event_time": None, "amount": 48, "unit": "hours"}, "event_date"),
    ({"event_date": "2026-11-03", "event_time": "25:00", "amount": 48, "unit": "hours"}, "event_time"),
])
def test_invalid_arguments_go_back_to_the_model(args, field):
    out = call("deadline", json.dumps(args), today=TODAY)
    assert out.startswith("Error: invalid arguments for deadline") and field in out


def test_unknown_tool():
    assert call("delete_file", "{}").startswith("Error: unknown tool")


def email(ids):
    return json.dumps({"to": "employer_hr", "subject": "Congé d'examen",
                       "body": "Bonjour, je souhaite poser mes cinq jours de congé d'examen. Inès", "cited_ids": ids})


def test_email_cannot_cite_an_id_search_texts_did_not_return():
    out = call("send_email", email(["L6222-35", "L9999-1"]), known_ids={"L6222-35"})
    assert out.startswith("Error") and "L9999-1" in out and "L6222-35" not in out.split(":", 2)[2]


def test_email_waits_for_the_interface(tmp_path):
    out = json.loads(call("send_email", email(["L6222-35"]), known_ids={"L6222-35"}))
    assert out["status"] == "pending_approval"
    box = tmp_path / "outbox.json"
    assert not box.exists()                                    # nothing written before the user decides
    record = tools.approve(out["draft_id"], approved=True, outbox=str(box))
    assert record["decision"] == "approved" and json.loads(box.read_text())[0]["to"] == "employer_hr"


def test_email_to_any_other_address_is_refused():
    out = call("send_email", email([]).replace("employer_hr", "someone@example.com"))
    assert out.startswith("Error") and "to" in out


def test_schemas_are_generated_from_the_models():
    s = {x["function"]["name"]: x["function"] for x in tools.schemas()}
    assert set(s) == {"search_texts", "deadline", "send_email"}
    for f in s.values():
        p = f["parameters"]
        assert f["strict"] is True and p["additionalProperties"] is False
        assert set(p["required"]) == set(p["properties"])     # strict mode: every field is required
    assert s["deadline"]["parameters"]["properties"]["unit"]["enum"] == \
        ["hours", "calendar_days", "working_days_mon_sat", "working_days_mon_fri", "months"]


def test_search_texts_without_retrieval_is_not_found(monkeypatch):
    from app import pipeline
    monkeypatch.setattr(pipeline, "retrieve", lambda question: None)     # retrieval.enabled: false (baseline)
    assert json.loads(call("search_texts", json.dumps({"query": "congés d'examen"}))) == \
        {"not_found": True, "passages": [], "missing_ids": []}
