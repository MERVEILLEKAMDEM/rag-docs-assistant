"""Évalue la recherche documentaire sur 24 questions annotées (hit@1, hit@3, MRR)."""
import json
from pathlib import Path

from rag import Assistant

QUESTIONS = [
    ("J'ai perdu ma CB, que faire ?", "Conditions d'utilisation de la carte bancaire › Opposition"),
    ("Combien de temps pour recevoir une nouvelle carte après opposition ?", "Conditions d'utilisation de la carte bancaire › Opposition"),
    ("Je peux bloquer ma carte juste quelques heures ?", "Conditions d'utilisation de la carte bancaire › Blocage temporaire"),
    ("Combien puis-je retirer par semaine ?", "Conditions d'utilisation de la carte bancaire › Plafonds"),
    ("J'ai oublié mon code secret", "Conditions d'utilisation de la carte bancaire › Code confidentiel"),
    ("Un paiement que je n'ai pas fait, je serai remboursé quand ?", "Procédure en cas de fraude › Opération non autorisée"),
    ("Jusqu'à quand puis-je contester une opération frauduleuse ?", "Procédure en cas de fraude › Opération non autorisée"),
    ("J'ai reçu un mail qui me demande mon mot de passe", "Procédure en cas de fraude › Hameçonnage (phishing)"),
    ("Quelqu'un se disant de la banque m'appelle pour valider un virement", "Procédure en cas de fraude › Faux conseiller"),
    ("Quel taux d'endettement maximum pour un prêt ?", "Crédit immobilier › Taux d'endettement"),
    ("Sur combien d'années peut-on emprunter ?", "Crédit immobilier › Taux d'endettement"),
    ("Combien de jours de réflexion après l'offre de prêt ?", "Crédit immobilier › Offre de prêt"),
    ("Quelles pénalités si je rembourse mon crédit plus tôt ?", "Crédit immobilier › Remboursement anticipé"),
    ("Puis-je baisser mes mensualités ?", "Crédit immobilier › Modulation"),
    ("Combien coûte une commission d'intervention ?", "Tarifs et réclamations sur les frais › Commission d'intervention"),
    ("Je suis étudiant de 22 ans, dois-je payer la tenue de compte ?", "Tarifs et réclamations sur les frais › Frais de tenue de compte"),
    ("Sous quel délai répondez-vous à une réclamation ?", "Tarifs et réclamations sur les frais › Réclamation"),
    ("Quel est le plafond du Livret A ?", "Épargne et placements › Livret A"),
    ("Combien peut-on mettre sur un LDDS ?", "Épargne et placements › LDDS"),
    ("Quelle fiscalité pour l'assurance vie après 8 ans ?", "Épargne et placements › Assurance vie"),
    ("Au bout de combien de temps les gains du PEA sont exonérés ?", "Épargne et placements › PEA"),
    ("Je déménage, comment changer mon adresse ?", "Gestion du compte › Changement d'adresse"),
    ("Où télécharger mon RIB ?", "Gestion du compte › RIB"),
    ("Comment donner procuration à mon fils ?", "Gestion du compte › Procuration"),
]


def main():
    idx = Assistant().index
    hit1 = hit3 = mrr = 0
    erreurs = []
    for q, attendu in QUESTIONS:
        titres = [p.titre for p, _ in idx.chercher(q, k=5)]
        rang = titres.index(attendu) + 1 if attendu in titres else None
        hit1 += rang == 1
        hit3 += bool(rang and rang <= 3)
        mrr += 1 / rang if rang else 0
        if rang != 1:
            erreurs.append({"question": q, "attendu": attendu, "trouve": titres[0], "rang": rang})
    n = len(QUESTIONS)
    res = {"questions": n, "hit@1": round(hit1 / n, 3), "hit@3": round(hit3 / n, 3),
           "MRR": round(mrr / n, 3), "erreurs": erreurs}
    out = Path(__file__).resolve().parents[1] / "results" / "evaluation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
