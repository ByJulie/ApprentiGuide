from lxml import etree

from ingest import service_public


def node(xml):
    return etree.fromstring(xml)


def test_table_cell_paragraphs_do_not_stick_together():
    table = node("<Tableau><Colonne><Paragraphe>Cas</Paragraphe></Colonne>"
                 "<Rangée><Cellule><Paragraphe>100 % du Smic</Paragraphe>"
                 "<Paragraphe>Salaire le + élevé</Paragraphe></Cellule></Rangée></Tableau>")
    assert service_public.table_lines(table) == ["Cas", "100 % du Smic Salaire le + élevé"]


def test_wide_table_gets_one_line_per_cell_with_both_headers():
    table = node("<Tableau>"
                 "<Rangée type='header'><Cellule><Paragraphe>Situation</Paragraphe></Cellule>"
                 "<Cellule><Paragraphe>18-20 ans</Paragraphe></Cellule>"
                 "<Cellule><Paragraphe>21-25 ans</Paragraphe></Cellule></Rangée>"
                 "<Rangée type='normal'><Cellule><Paragraphe>2e année</Paragraphe></Cellule>"
                 "<Cellule><Paragraphe>51 % du Smic, soit 952,18 €</Paragraphe></Cellule>"
                 "<Cellule><Paragraphe>61 % du Smic, soit 1 138,88 €</Paragraphe></Cellule>"
                 "</Rangée></Tableau>")
    assert service_public.table_lines(table) == [
        "Situation | 18-20 ans | 21-25 ans",
        "2e année, 18-20 ans : 51 % du Smic, soit 952,18 €",
        "2e année, 21-25 ans : 61 % du Smic, soit 1 138,88 €"]


def test_wide_table_row_of_another_width_stays_on_one_line():
    table = node("<Tableau>"
                 "<Rangée type='header'><Cellule><Paragraphe>A</Paragraphe></Cellule>"
                 "<Cellule><Paragraphe>B</Paragraphe></Cellule>"
                 "<Cellule><Paragraphe>C</Paragraphe></Cellule></Rangée>"
                 "<Rangée type='normal'><Cellule><Paragraphe>Titre</Paragraphe></Cellule>"
                 "</Rangée></Tableau>")
    assert service_public.table_lines(table) == ["A | B | C", "Titre"]


def test_online_service_is_kept_with_its_publisher():
    service = node("<ServiceEnLigne><Titre>Faire une demande - ANEF</Titre>"
                   "<Source>Ministère de l'intérieur</Source></ServiceEnLigne>")
    assert service_public.render(service) == [
        "Service en ligne : Faire une demande - ANEF (Ministère de l'intérieur)"]


def test_online_service_without_publisher():
    service = node("<ServiceEnLigne><Titre>Simulateur</Titre></ServiceEnLigne>")
    assert service_public.render(service) == ["Service en ligne : Simulateur"]


def test_layout_blocks_are_dropped():
    assert service_public.render(node("<TitreRiche><Paragraphe>x</Paragraphe></TitreRiche>")) == []
