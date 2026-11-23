"""Assistant question-réponse sur une base documentaire (RAG).

1. Découpage des documents Markdown en passages (une section = un passage).
2. Recherche : TF-IDF mots + caractères, similarité cosinus.
3. Réponse :
   - avec une clé ANTHROPIC_API_KEY : un LLM rédige la réponse à partir des
     seuls passages retrouvés, avec citation des sources ;
   - sans clé : réponse extractive (les phrases les plus pertinentes), sources citées.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

DOCS = Path(__file__).resolve().parents[1] / "docs"
STOP = set("""le la les un une des de du d l à au aux et ou en est sont pour par sur dans
ce cette ces qui que quoi quel quelle quels quelles comment combien je j me mon ma mes
il elle on nous vous ils a peut peux puis faire fait si ne pas plus avec""".split())


@dataclass
class Passage:
    source: str
    titre: str
    texte: str


def decouper(dossier: Path = DOCS) -> list[Passage]:
    passages = []
    for f in sorted(dossier.glob("*.md")):
        doc = f.read_text(encoding="utf-8")
        titre_doc = doc.splitlines()[0].lstrip("# ").strip()
        for bloc in re.split(r"\n(?=## )", doc):
            lignes = bloc.strip().splitlines()
            if not lignes or lignes[0].startswith("# "):
                continue
            passages.append(Passage(f.name, f"{titre_doc} › {lignes[0].lstrip('# ')}",
                                    " ".join(lignes[1:]).strip()))
    return passages


class Index:
    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.vec = FeatureUnion([
            ("mots", TfidfVectorizer(stop_words=list(STOP), sublinear_tf=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])
        self.M = self.vec.fit_transform([f"{p.titre} {p.texte}" for p in passages])

    def score_mots(self, question: str) -> float:
        """Meilleure similarité sur les seuls mots : sert à détecter les questions hors sujet."""
        mots = self.vec.transformer_list[0][1]
        return float(cosine_similarity(mots.transform([question]), mots.transform(
            [f"{p.titre} {p.texte}" for p in self.passages])).max())

    def chercher(self, question: str, k: int = 3) -> list[tuple[Passage, float]]:
        sims = cosine_similarity(self.vec.transform([question]), self.M)[0]
        top = np.argsort(-sims)[:k]
        return [(self.passages[i], float(sims[i])) for i in top]


def reponse_extractive(question: str, resultats, n_phrases: int = 2) -> str:
    phrases = [(s, p, sc) for p, sc in resultats for s in re.split(r"(?<=[.!?])\s+", p.texte) if s]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit([question] + [s for s, _, _ in phrases])
    sims = cosine_similarity(vec.transform([question]), vec.transform([s for s, _, _ in phrases]))[0]
    sims = sims * np.array([sc for _, _, sc in phrases])  # favorise le passage le mieux classé
    meilleures = sorted(np.argsort(-sims)[:n_phrases])
    return " ".join(f"{phrases[i][0]} [{phrases[i][1].source}]" for i in meilleures)


def reponse_llm(question: str, resultats) -> str:
    import anthropic  # dépendance optionnelle

    contexte = "\n\n".join(f"[{p.source} – {p.titre}]\n{p.texte}" for p, _ in resultats)
    msg = anthropic.Anthropic().messages.create(
        model=os.getenv("RAG_MODEL", "claude-haiku-5-5"),
        max_tokens=400,
        system=("Tu es un assistant de conseiller bancaire. Réponds en français, en 2 à 4 phrases, "
                "uniquement à partir des extraits fournis, et cite la source entre crochets. "
                "Si la réponse n'est pas dans les extraits, dis-le."),
        messages=[{"role": "user", "content": f"Extraits :\n{contexte}\n\nQuestion : {question}"}],
    )
    return msg.content[0].text


class Assistant:
    def __init__(self, dossier: Path = DOCS, seuil: float = 0.1):
        self.index = Index(decouper(dossier))
        self.seuil = seuil

    def repondre(self, question: str, k: int = 3) -> dict:
        res = self.index.chercher(question, k)
        if self.index.score_mots(question) < self.seuil:
            return {"reponse": "Je ne trouve pas cette information dans la base documentaire.",
                    "sources": []}
        mode = "llm" if os.getenv("ANTHROPIC_API_KEY") else "extractif"
        texte = reponse_llm(question, res) if mode == "llm" else reponse_extractive(question, res)
        return {"reponse": texte, "mode": mode,
                "sources": [{"source": p.source, "section": p.titre, "score": round(s, 3)} for p, s in res]}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Quel est le plafond du Livret A ?"
    r = Assistant().repondre(q)
    print("Q :", q, "\nR :", r["reponse"])
    for s in r["sources"]:
        print("  -", s)
