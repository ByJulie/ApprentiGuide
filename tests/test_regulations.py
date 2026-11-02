from ingest import clean, regulations

RI_TEXT = """PREAMBULE
Texte d'introduction.
ARTICLE 3 Dossier administratif ______ 5
ARTICLE 5 Demission ______ 6
ARTICLE 3 DOSSIER ADMINISTRATIF
L'Ecole tient un dossier.
12
ARTICLE 5 DEMISSION
Intro de l'article.
5.1. Desinscription d'un cours
Demande expresse.
5.2. Demission de fait
Absence de 15 jours.
ARTICLE 8 COMPORTEMENT
Hors selection.
"""

RP_TEXT = """SOMMAIRE
1 DISPOSITIONS GENERALES ................ 4
3 ORGANISATION ................ 10
3.3 PERIODES EN ENTREPRISE ................ 12
3.3.4 Absence pendant le stage ................ 14
3.3.5 ADMISSION DANS LE CADRE DE LA VAE .. 8
1 DISPOSITIONS GENERALES
Le cursus dure 5 ans.
6 ou autre diplome admis en equivalence.
3 ORGANISATION
3.3 PERIODES EN ENTREPRISE
Entre 10 et 18 mois.
3.3.4 Absence pendant le stage
Les absences sont justifiees.
3.3.5 ADMISSION DANS LE CADRE DE LA
VAE
Texte de la VAE.
"""


def chunks(text, kind, **extra):
    return regulations.regulation_chunks(
        text, school="aivancity", kind=kind, doc_label="Doc", corpus="aivancity_x",
        source_label="Reglement 2025-2026", version_date=None, **extra)


def test_ri_cuts_articles_at_subsections_and_keeps_selected_articles():
    result = chunks(RI_TEXT, "RI", articles=["3", "5"])
    assert [c.citation_id for c in result] == [
        "aivancity-RI-3", "aivancity-RI-5", "aivancity-RI-5.1", "aivancity-RI-5.2"]
    assert "12" not in result[0].text.split()


def test_rp_headings_come_from_the_table_of_contents():
    result = chunks(RP_TEXT, "RP")
    assert [c.citation_id for c in result] == [
        "aivancity-RP-1", "aivancity-RP-3.3", "aivancity-RP-3.3.4", "aivancity-RP-3.3.5"]
    assert "6 ou autre diplome" in result[0].text
    assert result[2].section_path == "Organisation > Periodes en entreprise"
    assert result[3].text == "Texte de la VAE."


def test_chunk_fields_for_citation_and_scope():
    chunk = chunks(RP_TEXT, "RP")[2]
    assert chunk.source == "reglement" and chunk.source_label == "Reglement 2025-2026"
    assert chunk.title == "Doc 3.3.4 : Absence pendant le stage"
    assert chunk.scope == {"theme": "ecole", "school": "aivancity"}
    assert chunk.url is None


def test_clean_lines_expands_ligatures_and_drops_page_numbers():
    lines = regulations.clean_lines("   apprentissage ﬁnal\n  12  \n puce")
    assert lines == ["apprentissage final", "- puce"]


def test_build_scope_school_is_optional():
    assert clean.build_scope("ecole", "aivancity") == {"theme": "ecole", "school": "aivancity"}
    assert clean.build_scope("ecole") == {"theme": "ecole"}
