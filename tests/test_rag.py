import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag import Assistant  # noqa: E402

A = Assistant()


def test_trouve_la_bonne_section():
    titre = A.index.chercher("Combien de jours de réflexion après l'offre de prêt ?", 1)[0][0].titre
    assert titre.endswith("Offre de prêt")


def test_refuse_hors_sujet(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert A.repondre("Qui a gagné le match hier ?")["sources"] == []


def test_cite_ses_sources(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "[05_epargne.md]" in A.repondre("Quel est le plafond du Livret A ?")["reponse"]
