# Scoring Crédit Cameroun

Modèle de scoring prédictif pour l'évaluation du risque de défaut de paiement (Machine Learning), adapté au contexte camerounais. Projet de fin d'études — groupe de 3 (Aristide, Andy, Marie-Thérèse). Deadline : **31 août 2026**.

## Équipe et rôles

| Membre | Rôle principal | Branche |
|---|---|---|
| Aristide | Data Engineering, ML, code Python, intégration | `aristide` |
| Andy | EDA, visualisation, Power BI, rédaction technique | `andy` |
| Marie-Thérèse | Contextualisation Cameroun, formulaire métier, rapport final | `marie-therese` |

## Structure du dépôt

```
scripts/       scripts Python (format "cellules" # %%, ouvrables dans VS Code ou Jupyter)
notebooks/     versions .ipynb des scripts (générées avec jupytext, utilisables dans Colab)
data/          NON versionné (voir .gitignore) — données Kaggle + fichiers générés
requirements.txt
```

Le dossier `data/` n'est pas versionné : le dataset Kaggle ne doit pas être redistribué, et les fichiers générés sont reproductibles en quelques secondes via les scripts.

## Installation (environnement local — recommandé)

```
git clone https://github.com/ARIS7177/scoring-credit-cameroun.git
cd scoring-credit-cameroun
pip install -r requirements.txt
```

Il faut Python 3.10+ installé. Chacun peut aussi utiliser Google Colab en secours pour une séance de travail synchrone (upload manuel des notebooks depuis `notebooks/`) — les scripts détectent automatiquement l'environnement Colab et demandent alors votre propre token Kaggle (`kaggle.json`, via kaggle.com/settings > API).

## Récupérer le dataset

1. Créez un compte Kaggle (gratuit) si besoin.
2. Téléchargez : https://www.kaggle.com/datasets/yasserh/loan-default-dataset
3. Dézippez `Loan_Default.csv` dans `data/raw/`.

## Exécuter le pipeline

```
cd scripts
python 01_chargement_nettoyage.py       # verifie colonnes, supprime doublons
python 02_traduction_camerounaise.py    # traduit colonnes/valeurs, adapte l'echelle FCFA
```

Résultat : `data/processed/Loan_Default_Cameroun.csv` (dataset traduit et adapté au contexte camerounais, prêt pour l'exploration et la modélisation).

## Workflow Git (une branche par personne)

- **`main`** : code commun stable. On n'y commite jamais directement.
- Chacun travaille sur sa propre branche (`aristide`, `andy`, `marie-therese`).
- Quand une partie du travail est prête à être partagée avec le groupe :
  ```
  git add <fichiers>
  git commit -m "Description du changement"
  git push
  ```
  puis ouvrir une **Pull Request** vers `main` sur GitHub (un lien est généré automatiquement après le premier push de la branche). Un autre membre relit rapidement avant de merger, pour éviter les conflits et garder `main` toujours fonctionnel.
- Avant de commencer une session de travail, mettre sa branche à jour avec `main` :
  ```
  git checkout main
  git pull
  git checkout <sa-branche>
  git merge main
  ```

## Feuille de route (7 semaines, 14 juillet → 31 août 2026)

1. **Semaine 1** — Données & cadrage (dataset camerounais + EDA initiale)
2. **Semaine 2** — Nettoyage & feature engineering
3. **Semaine 3** — Modélisation ML (régression logistique, Random Forest, XGBoost)
4. **Semaine 4** — Application Streamlit (formulaire, score, décision)
5. **Semaine 5** — Rapport + finalisation app (export PDF, README, requirements)
6. **Semaine 6** — Déploiement (GitHub + Streamlit Cloud) & relecture
7. **Semaine 7** — Buffer & soumission finale
