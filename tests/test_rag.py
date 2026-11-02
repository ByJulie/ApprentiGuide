from rag.retrieve import cited_ids, rrf


def test_cited_ids_normalises_article_and_fiche_numbers():
    assert cited_ids("Que disent l'article L 6222-35 et la fiche f2918 ?") == ["L6222-35", "F2918"]
    assert cited_ids("R5221-7 puis R5221-7 encore") == ["R5221-7"]
    assert cited_ids("How many days of leave?") == []


def test_rrf_rewards_agreement_between_rankings():
    assert rrf([["a", "b", "c"], ["b", "a"], ["b"]])[0] == "b"
    assert set(rrf([["a"], ["z"]])) == {"a", "z"}


def test_wilson_interval_matches_the_course_table():
    from eval.eval import wilson
    assert wilson(16, 20) == [0.584, 0.919]
    assert wilson(48, 60) == [0.682, 0.882]


def test_rewrite_loses_invented_article_numbers_keeps_asked_ones():
    from rag.retrieve import strip_unasked_ids
    invented = "Combien de jours de congés pour préparer mes examens en apprentissage ? (article L. 6323-1)"
    assert strip_unasked_ids("How many days off for exams?", invented) == \
        "Combien de jours de congés pour préparer mes examens en apprentissage ?"
    assert strip_unasked_ids("What does L6222-35 say?", "Que dit l'article L6222-35 ?") == "Que dit l'article L6222-35 ?"
    assert cited_ids("voir l'article L. 6222-35") == ["L6222-35"]


def test_a_refusal_is_not_used_as_a_search_query():
    from app.pipeline import REFUSAL
    assert REFUSAL.search("I’m sorry, but I can’t help with that.")
    assert REFUSAL.search("Je ne peux pas répondre")
    assert not REFUSAL.search("salaire minimum apprenti 22 ans première année")


def test_citations_come_from_retrieved_sources_never_from_the_model():
    from app.pipeline import citations
    sources = [{"id": "L6222-35", "text": "cinq jours", "title": "Article L6222-35", "url": "https://legifrance/x",
                "source_label": "Code du travail, Legifrance", "version_date": "2009-11-26"},
               {"id": "F2918", "text": "section 1", "title": "a", "url": "https://sp/F2918",
                "source_label": "Service-Public.fr/DILA", "version_date": "2026-09-08"},
               {"id": "F2918", "text": "section 2", "title": "b", "url": "https://sp/F2918",
                "source_label": "Service-Public.fr/DILA", "version_date": "2026-09-08"}]
    out = citations("Tu as 5 jours [L6222-35] et la fiche le confirme [F2918; L9999-1].", sources)
    assert [c["id"] for c in out["cited"]] == ["L6222-35", "F2918"]
    assert len(out["cited"][1]["passages"]) == 2
    assert out["invented"] == ["L9999-1"]
    assert out["consulted"] == []
    assert citations("NOT FOUND", sources)["consulted"][0]["id"] == "L6222-35"


def test_not_found_message_is_fixed_and_in_the_language_of_the_question():
    from app.pipeline import language, not_found_message
    assert language("Quel est le menu de la cantine ?", "Quel est le menu de la cantine ?") == "fr"
    assert language("Who is the director of aivancity?", "Qui est le directeur d'aivancity ?") == "en"
    assert language("How many days can I take?", None) == "en"
    msg = not_found_message("fr", ["L9999-1"])
    assert "école ou à tes RH" in msg and "L'article L9999-1 ne fait pas partie" in msg
    assert not_found_message("en").startswith("I could not find")


def test_cited_ids_read_every_citation_format_models_write():
    from app.pipeline import cited_ids
    text = "a [ F2918 ] b 【F33754】 c [F2918 ; D6222-1] d [​F21059​] e [L6222-1][R5221-7]"
    assert cited_ids(text) == ["F2918", "F33754", "D6222-1", "F21059", "L6222-1", "R5221-7"]
    assert cited_ids("la portabilité [L911-7 III] et [L. 6222-35]") == ["L911-7", "L6222-35"]


def test_diversify_caps_passages_per_document():
    from rag.retrieve import diversify
    ranked = [({"citation_id": c}, s) for c, s in
              [("F21059", .9), ("F21059", .8), ("F21059", .7), ("R3262-7", .6), ("L3262-1", .5)]]
    assert [r["citation_id"] for r, _ in diversify(ranked, 3, 2)] == ["F21059", "F21059", "R3262-7"]
    assert len(diversify(ranked, 3, None)) == 3


def test_public_sector_texts_go_last_unless_the_question_asks_about_it():
    from rag.retrieve import sector_priority
    pub = ({"citation_id": "F3059", "title": "Apprentissage dans la fonction publique : congés", "section_path": ""}, .9)
    pen = ({"citation_id": "R6227-7", "title": "Article R6227-7",
            "section_path": "Chapitre VII : Dispositions pénales"}, .8)
    priv = ({"citation_id": "F2918", "title": "Contrat d'apprentissage : congés", "section_path": ""}, .7)
    assert [r["citation_id"] for r, _ in sector_priority("Combien de congés ?", [pub, pen, priv])] == ["R6227-7", "F2918", "F3059"]
    assert sector_priority("Et dans une mairie ?", [pub, priv])[0][0]["citation_id"] == "F3059"


def test_think_blocks_never_reach_the_user():
    from app.pipeline import THINK
    assert THINK.sub("", "<think>\nlet me see\n</think>\nTu as 5 jours [L6222-35].").strip() == "Tu as 5 jours [L6222-35]."


def test_api_openai_compatible_with_tasks_history_and_streaming(monkeypatch):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
    from fastapi.testclient import TestClient
    import api, pipeline
    seen = {}
    def fake_standalone(q, history):
        seen["history"] = history
        return q + " (complete)"
    def fake_answer(q):
        seen["question"] = q
        return {"answer": "Tu as 5 jours [L6222-35].", "citations": {
            "cited": [{"id": "L6222-35", "url": "https://legifrance/x", "source_label": "Code du travail, Legifrance",
                       "version_date": "2009-11-26", "passages": [{"title": "t", "text": "cinq jours"}]}],
            "consulted": [], "invented": []}}
    monkeypatch.setattr(pipeline, "standalone", fake_standalone)
    monkeypatch.setattr(pipeline, "answer", fake_answer)
    client = TestClient(api.app)
    assert client.get("/v1/models").json()["data"][0]["id"] == "apprentiguide"
    # an interface task never reaches the pipeline
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "### Task: Generate a title"}]})
    assert "title" in r.json()["choices"][0]["message"]["content"] and "question" not in seen
    # a follow-up gets the previous turns, without the sources block of the previous answer
    msgs = [{"role": "user", "content": "Congés pour examens ?"},
            {"role": "assistant", "content": "5 jours.\n**Sources**\n- [L6222-35](x)"},
            {"role": "user", "content": [{"type": "text", "text": "Et si je suis mineur ?"}]}]
    out = client.post("/v1/chat/completions", json={"messages": msgs}).json()["choices"][0]["message"]["content"]
    assert seen["question"] == "Et si je suis mineur ? (complete)"
    assert seen["history"][1]["content"] == "5 jours."
    assert "[L6222-35](https://legifrance/x)" in out and "Licence Ouverte" in out


def _stream_client(monkeypatch, pieces, abstain=False):
    import sys, os, json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    import api, pipeline
    src = {"id": "L6222-35", "text": "cinq jours", "title": "t", "url": "https://legifrance/x",
           "source_label": "Code du travail, Legifrance", "version_date": "2009-11-26"}
    monkeypatch.setattr(pipeline, "retrieve", lambda q: SimpleNamespace(
        abstain=abstain, sources=[src], missing_ids=[], question_fr=q, best_score=0.9))
    monkeypatch.setattr(pipeline, "generate_stream", lambda *a, **k: iter(pieces))
    body = {"stream": True, "messages": [{"role": "user", "content": "Congés pour examens ?"}]}
    raw = TestClient(api.app).post("/v1/chat/completions", json=body).text
    chunks = [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: {")]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks if c["choices"])
    events = [c["event"] for c in chunks if "event" in c]
    return raw, text, reasoning, events


def test_stream_shows_status_reasoning_answer_and_source_cards(monkeypatch):
    pieces = [("reasoning", "Je lis L6222-35."), ("content", "Tu as 5 jours "), ("content", "[L6222-35].\n\nSour"),
              ("content", "ces\n[L6222-35]")]
    raw, text, reasoning, events = _stream_client(monkeypatch, pieces)
    assert raw.strip().endswith("data: [DONE]")
    assert text.startswith("Tu as 5 jours [1].") and "Sources" not in text and "Licence Ouverte" in text
    assert reasoning == "Je lis L6222-35."
    assert [e["type"] for e in events][0] == "status" and events[-1]["data"]["done"] is True
    cards = [e["data"] for e in events if e["type"] == "source"]
    assert cards[0]["source"]["name"] == "L6222-35" and cards[0]["metadata"][0]["name"] == "L6222-35"
    doc = cards[0]["document"][0]
    assert doc.startswith("Article L6222-35 · Code du travail, Legifrance, version du 2009-11-26")
    assert "Ce qui répond à ta question :\n« cinq jours »" in doc


def test_stream_never_shows_the_models_not_found(monkeypatch):
    _, text, _, _ = _stream_client(monkeypatch, [("content", "NOT "), ("content", "FOUND")])
    assert "NOT FOUND" not in text and text.startswith("Je n'ai pas trouvé")
    _, text, _, _ = _stream_client(monkeypatch, [], abstain=True)
    assert text.startswith("Je n'ai pas trouvé")


def test_model_sources_line_is_replaced_by_the_database_block():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
    import api
    out = {"answer": "43 % du Smic [F2918].\n\nSources :  \n[F2918]  \n[D6222-26]",
           "citations": {"cited": [], "consulted": [], "invented": []}}
    text = api.render(out)
    assert text.startswith("43 % du Smic [F2918].") and "[D6222-26]" not in text


def test_citations_become_numbers_in_order_and_partial_brackets_wait():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
    import api
    known = {"L6222-35", "F2918"}
    text = "5 jours [L6222-35], rappelé par la fiche [F2918] et [L6222-35 ; F2918]. Voir [L9999-1]"
    assert api.numbered(text, known, final=True) == "5 jours [1], rappelé par la fiche [2] et [1][2]. Voir [L9999-1]"
    assert api.numbered("5 jours [L62", known) == "5 jours "


def test_passage_cleaned_of_heading_and_entry_into_force_note():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
    import api
    title, body = api.clean_passage("Article R5221-7\nL'étudiant étranger   peut conclure. Nota : Conformément au décret.", "R5221-7")
    assert title == "" and body == "L'étudiant étranger peut conclure."
    title, _ = api.clean_passage("Contrat d'apprentissage : Quels sont les congés ?\nTexte.", "F2918")
    assert title == "Contrat d'apprentissage : Quels sont les congés ?"


def test_chat_title_is_the_start_of_the_first_question():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
    import api
    task = "### Task:\nGenerate a concise title\n### Chat History:\n<chat_history>\nUSER: combien de jours de congés pour mes examens en tant qu'apprenti ?\nASSISTANT: 5 jours\n</chat_history>"
    assert api.chat_title(task) == "Combien de jours de congés pour mes examens en…"
    assert api.chat_title("### Task: title") == "ApprentiGuide"


def test_rotating_client_switches_key_on_daily_limit_only(monkeypatch):
    import httpx
    from openai import RateLimitError
    from app.keys import RotatingClient, keys_from_env
    monkeypatch.setenv("TEST_KEY", "a"); monkeypatch.setenv("TEST_KEY_2", ""); monkeypatch.setenv("TEST_KEY_3", "c")
    assert keys_from_env("TEST_KEY") == ["a", "c"]
    response = httpx.Response(429, request=httpx.Request("POST", "https://x"))
    calls = []
    def fake_create(key, message):
        def create(**kwargs):
            calls.append(key)
            if message:
                raise RateLimitError(message, response=response, body=None)
            return "ok"
        return create
    client = RotatingClient("https://x", ["a", "c"])
    client._clients[0].chat.completions.create = fake_create("a", "Rate limit reached on tokens per day (TPD): Limit 200000")
    client._clients[1].chat.completions.create = fake_create("c", None)
    assert client.chat.completions.create(model="m") == "ok" and calls == ["a", "c"] and client.index == 1
    minute = RotatingClient("https://x", ["a", "c"])
    minute._clients[0].chat.completions.create = fake_create("a", "Rate limit reached on tokens per minute (TPM)")
    try:
        minute.chat.completions.create(model="m")
        assert False, "a per-minute limit must not switch keys"
    except RateLimitError:
        assert minute.index == 0
