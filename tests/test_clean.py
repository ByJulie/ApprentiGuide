from ingest import clean

PATTERNS = [{"regex": r"^[LRD]62\d\d-", "theme": "apprentissage"},
            {"regex": r"^L433-3$", "theme": "etranger"}]


def article(num, etat="VIGUEUR", texte="Texte.", date=1546300800000):
    return {"id": "LEGIARTI1", "num": num, "etat": etat, "texte": texte, "dateDebut": date,
            "sectionPath": "Partie législative > Livre II"}


def test_legifrance_keeps_in_force_and_deferred_repeal_only():
    articles = [article("L6222-1"), article("L6222-2", etat="ABROGE"),
                article("L6222-3", etat="MODIFIE"), article("L6222-4", etat="ABROGE_DIFF")]
    kept = [c.citation_id for c in clean.legifrance_chunks(articles, "code_travail", PATTERNS)]
    assert kept == ["L6222-1", "L6222-4"]


def test_legifrance_drops_duplicate_numbers_and_empty_text():
    articles = [article("L6222-1"), article("L6222-1", texte="Autre."),
                article("L6222-2", texte=" ")]
    chunks = clean.legifrance_chunks(articles, "code_travail", PATTERNS)
    assert [c.citation_id for c in chunks] == ["L6222-1"]
    assert chunks[0].text == "Texte."


def test_legifrance_chunk_fields():
    chunk = clean.legifrance_chunks([article("L433-3")], "ceseda", PATTERNS)[0]
    assert chunk.version_date == "2019-01-01"
    assert chunk.source_label == "CESEDA, Legifrance"
    assert chunk.scope == {"theme": "etranger", "audience": "international"}
    assert chunk.section_path == "Partie législative > Livre II"


def test_legifrance_appends_nota_to_the_text():
    with_nota = {**article("L6222-1"), "nota": "Ne s'applique pas\n  avant 2019."}
    chunks = clean.legifrance_chunks([with_nota, article("L6222-2")], "code_travail", PATTERNS)
    assert chunks[0].text == "Texte.\n\nNota : Ne s'applique pas avant 2019."
    assert chunks[1].text == "Texte."


def test_scope_is_empty_without_theme():
    assert clean.build_scope(None) == {}
    assert clean.theme_for("L9999-1", PATTERNS) is None


def test_split_text_respects_limit_and_keeps_all_lines():
    text = "\n".join(["ligne " + str(i) * 10 for i in range(50)])
    pieces = clean.split_text(text, max_chars=100)
    assert all(len(p) <= 100 for p in pieces)
    assert "\n".join(pieces) == text


def test_split_text_keeps_a_long_line_whole():
    assert clean.split_text("x" * 500, max_chars=100) == ["x" * 500]


def fiche(fiche_id="F1", corpus="sp_particuliers", type_="Fiche pratique", texts=("a",)):
    return {"id": fiche_id, "corpus": corpus, "theme": "conges", "type": type_, "title": "Titre",
            "sectionPath": "Accueil > Travail", "url": "https://x", "versionDate": "2026-06-18",
            "sourceLabel": "Service-Public.fr/DILA",
            "sections": [{"title": f"S{i}", "text": t} for i, t in enumerate(texts)]}


def test_service_public_drops_menus_and_identical_fiches():
    fiches = [fiche("F1", type_="Comment faire si..."), fiche("F2"),
              fiche("F2", corpus="sp_entreprendre"), fiche("F2", texts=("autre",))]
    chunks = clean.service_public_chunks(fiches)
    assert [(c.citation_id, c.corpus, c.text) for c in chunks] == [
        ("F2", "sp_particuliers", "a"), ("F2", "sp_particuliers", "autre")]


def test_service_public_numbers_parts_and_titles():
    long_section = "\n".join(["y" * 1000] * 5)
    chunks = clean.service_public_chunks([fiche(texts=("court", long_section))])
    assert [c.part for c in chunks] == list(range(len(chunks)))
    assert chunks[0].title == "Titre : S0"
    assert chunks[2].title == "Titre : S1 (suite 1)"
    keys = {(c.corpus, c.citation_id, c.part) for c in chunks}
    assert len(chunks) > 2 and len(keys) == len(chunks)
