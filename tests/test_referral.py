"""Referral to the international office (professor's review): added in code, never left to the model."""
import os
import sys
import types

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
# The embedding and reranker models (torch) are not needed here: no retrieval runs in these tests.
sys.modules.setdefault("rag.models", types.ModuleType("rag.models"))

from app import pipeline


def source(cid: str, international: bool) -> dict:
    return {"id": cid, "text": "...", "source_label": "CESEDA" if international else "Code du travail",
            "url": "https://www.legifrance.gouv.fr", "version_date": "2021-05-01", "international": international}


def test_referral_when_an_immigration_source_is_cited():
    refs = pipeline.citations("La carte permet de travailler [L422-1].", [source("L422-1", True)])
    assert "bureau international" in pipeline.international_referral(refs, "fr")
    assert "international office" in pipeline.international_referral(refs, "en")


def test_no_referral_for_other_sources_or_uncited_ones():
    sources = [source("F2918", False), source("L422-1", True)]
    refs = pipeline.citations("Le salaire est de 43 % du SMIC [F2918].", sources)  # L422-1 retrieved, not cited
    assert pipeline.international_referral(refs, "fr") == ""
