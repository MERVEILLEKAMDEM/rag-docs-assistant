# 💬 RAG Docs Assistant – assistant GenAI pour conseiller bancaire

[![Tests](https://github.com/MERVEILLEKAMDEM/rag-docs-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/MERVEILLEKAMDEM/rag-docs-assistant/actions)

Un conseiller cherche une règle dans des dizaines de procédures internes. Cet assistant **retrouve le bon passage et répond en citant sa source**. S'il ne trouve rien, il refuse de répondre au lieu d'inventer.

```text
$ python src/rag.py "Combien de jours de réflexion après l'offre de prêt ?"
R : Après acceptation du dossier, l'offre de prêt est envoyée à l'emprunteur. [03_credit_immobilier.md]
    Celui-ci dispose d'un délai de réflexion légal de 10 jours avant de pouvoir l'accepter. [03_credit_immobilier.md]

$ python src/rag.py "Qui a gagné le match hier ?"
R : Je ne trouve pas cette information dans la base documentaire.
```

## Architecture (Retrieval-Augmented Generation)

```
docs/*.md ──► découpage par section ──► index TF-IDF (mots + n-grammes de caractères)
                                                   │
question ──► recherche (similarité cosinus, top 3) ┘
                 │
                 ├─ score trop faible ──► « je ne trouve pas » (anti-hallucination)
                 ├─ ANTHROPIC_API_KEY présente ──► le LLM rédige à partir des seuls extraits, sources citées
                 └─ sinon ──► réponse extractive (phrases les plus pertinentes + source)
```

## Évaluation de la recherche

24 questions annotées à la main, avec la section attendue :

| Métrique | Score |
|---|---|
| hit@1 (bonne section en 1re position) | 0,83 |
| hit@3 (bonne section dans le top 3 envoyé au LLM) | **1,00** |
| MRR | 0,91 |

Le hit@3 compte le plus : le LLM reçoit les 3 meilleurs passages, donc la bonne information lui parvient pour les 24 questions. Les erreurs de 1re position sont détaillées dans `results/evaluation.json`.

## Lancer le projet

```bash
pip install -r requirements.txt
python src/rag.py "J'ai perdu ma carte, que faire ?"
python src/evaluate.py
streamlit run app.py                  # interface web
export ANTHROPIC_API_KEY=...          # optionnel : réponses rédigées par un LLM
python -m pytest -q
```

> Les documents de `docs/` sont des **procédures fictives** écrites pour la démonstration. Ce ne sont pas des documents réels.

**Pistes** : passer à des embeddings et une base vectorielle (FAISS, pgvector), ajouter un re-ranking, et évaluer la fidélité des réponses du LLM.

---
Pascale KAMDEM · Data & IA · [LinkedIn](https://www.linkedin.com/in/pascale-merveille-kamdem-562aa0284/)
