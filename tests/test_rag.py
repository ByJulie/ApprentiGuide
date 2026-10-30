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
