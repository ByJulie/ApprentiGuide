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


def test_postprocess_drops_legal_plumbing_keeps_rules():
    data = {"keywords": ["article R. 422-9", "décret n° 2020-1734", "1er mai 2021", "L. 422-10",
                         "cinq jours ouvrables", "moins de 18 ans", "50 % du SMIC"],
            "concepts": ["Effective Date", "Decree 2020-1734", "Exam Preparation Leave"]}
    labels_kept = [l for _, l in labels.postprocess(data)]
    assert labels_kept == ["cinq jours ouvrables", "moins de 18 ans", "50 % du SMIC", "Exam Preparation Leave"]


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


def test_generic_categories_are_dropped_and_known_ones_listed_alphabetically():
    pairs = labels.postprocess({"categories": ["Notification", "Caregiver Leave"]})
    assert pairs == [("category", "Caregiver Leave")]
    prompt = labels.build_prompt({"citation_id": "L1", "title": "", "text": "x"}, ["Paid Leave", "Immigration"])
    assert "Known categories: Immigration, Paid Leave" in prompt


def test_copied_section_path_is_not_a_category():
    pairs = labels.postprocess({"categories": ["Paid Leave > Section 5 : Caisses", "Sabbatical Leave"]})
    assert pairs == [("category", "Sabbatical Leave")]
