from rag import labels
from rag.retrieve import cited_ids, rrf


def test_cited_ids_normalises_article_and_fiche_numbers():
    assert cited_ids("Que disent l'article L 6222-35 et la fiche f2918 ?") == ["L6222-35", "F2918"]
    assert cited_ids("R5221-7 puis R5221-7 encore") == ["R5221-7"]
    assert cited_ids("How many days of leave?") == []


def test_rrf_rewards_agreement_between_rankings():
    assert rrf([["a", "b", "c"], ["b", "a"], ["b"]])[0] == "b"
    assert set(rrf([["a"], ["z"]])) == {"a", "z"}


def test_postprocess_applies_limits_and_cross_type_dedup():
    data = {"categories": ["Paid Leave", "Apprenticeship", "Training", "Extra"],
            "keywords": ["congés payés", "Paid leave", "cinq jours ouvrables",
                         "une phrase beaucoup trop longue pour être un mot clé utile"] + [f"mot {i}" for i in range(20)],
            "concepts": [{"concept": "Exam Preparation Leave", "keyword": "cinq jours ouvrables"},
                         {"concept": "apprenticeship"}, "Extra Concept"]}
    pairs = labels.postprocess(data)
    kinds = [k for k, _ in pairs]
    assert kinds.count("category") == 3
    assert kinds.count("keyword") == 12
    assert ("keyword", "Paid leave") not in pairs           # already a category
    assert ("concept", "apprenticeship") not in pairs       # already a category
    assert ("concept", "Exam Preparation Leave") in pairs
    assert all(len(l.split()) <= labels.MAX_KEYWORD_WORDS for k, l in pairs if k == "keyword")


def test_postprocess_drops_noise_and_survives_bad_output():
    pairs = labels.postprocess({"categories": ["Article", "Employer"], "keywords": [None, 3, "Légifrance"]})
    assert pairs == []
    assert labels.parse("not json") == {}
    assert labels.parse('```json\n{"categories": ["Paid Leave"]}\n```') == {"categories": ["Paid Leave"]}
