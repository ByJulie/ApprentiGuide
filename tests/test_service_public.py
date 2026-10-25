from lxml import etree

from ingest import service_public


def node(xml):
    return etree.fromstring(xml)


def test_table_cell_paragraphs_do_not_stick_together():
    table = node("<Tableau><Colonne><Paragraphe>Cas</Paragraphe></Colonne>"
                 "<Rangée><Cellule><Paragraphe>100 % du Smic</Paragraphe>"
                 "<Paragraphe>Salaire le + élevé</Paragraphe></Cellule></Rangée></Tableau>")
    assert service_public.table_lines(table) == ["Cas", "100 % du Smic Salaire le + élevé"]


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
