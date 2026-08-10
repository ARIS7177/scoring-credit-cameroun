# %% [markdown]
# ## Semaine 3 - Etape 1 : modelisation ML (Regression Logistique, Random Forest, XGBoost)
# Entree  : ../data/processed/Loan_Default_Cameroun_Encode.csv (sortie du script 06)
# Sortie  : tableau comparatif des 3 modeles (aucune sauvegarde de modele ici -
#           choix du modele final au point d'equipe du jeudi 13 aout, sauvegarde
#           prevue vendredi 14 aout, voir docs/Analyse_et_Plan_Projet.docx section 5)
#
# Taches (lundi 10 - mercredi 12 aout 2026) :
# 1) Separation features (categorie A, 16 colonnes) / cible, split train/test
#    80/20 stratifie (la cible est desequilibree, 75,4% Rembourse / 24,5% Defaut)
# 2) Regression Logistique (reference) + validation croisee stratifiee 5-fold
# 3) Random Forest + validation croisee stratifiee 5-fold
# 4) XGBoost + validation croisee stratifiee 5-fold (demarre mercredi, tache
#    prevue mercredi-jeudi dans le plan)
# 5) Tableau comparatif des metriques (AUC-ROC, F1, Precision, Rappel) des 3
#    modeles sur le jeu de test - version de travail pour le point d'equipe de
#    jeudi, le tableau/graphiques officiels (courbes ROC, matrices de confusion)
#    restent la tache d'Andy (notebook 04_evaluation_modeles.ipynb)
#
# Desequilibre de classe traite par class_weight="balanced" (Regression
# Logistique, Random Forest) et scale_pos_weight (XGBoost) plutot que par
# sur/sous-echantillonnage, pour ne pas alterer la distribution reelle du jeu
# d'entrainement.

# %%
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

IN_COLAB = "google.colab" in sys.modules
BASE_DIR = "/content/ProjetScoringCredit" if IN_COLAB else ".."

INPUT_PATH = f"{BASE_DIR}/data/processed/Loan_Default_Cameroun_Encode.csv"

RANDOM_SEED = 42

df = pd.read_csv(INPUT_PATH)
print("Dimensions en entree :", df.shape)

# %% [markdown]
# ### 1. Separation features / cible et split train/test
# Les colonnes descriptives (genre, tranche_age, niveau_education,
# membre_tontine, activite_saisonniere, utilisation_mobile_money) et
# id_client sont exclues des features ML (Classification_Variables_
# Consolidee.txt, section 3.B) - seules les 16 colonnes de la categorie A
# (section 3.A, apres encodage) entrent dans le modele.

# %%
CIBLE = "statut_remboursement"
COLONNES_NON_FEATURES = [
    "id_client",
    "genre",
    "tranche_age",
    "niveau_education",
    "membre_tontine",
    "activite_saisonniere",
    "utilisation_mobile_money",
]

FEATURES = [c for c in df.columns if c not in COLONNES_NON_FEATURES and c != CIBLE]
print(f"{len(FEATURES)} features retenues :")
print(FEATURES)

X = df[FEATURES]
y = df[CIBLE]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)
print(f"\nTrain : {X_train.shape[0]} lignes, Test : {X_test.shape[0]} lignes")
print("Repartition de la cible (train) :")
print(y_train.value_counts(normalize=True).round(4) * 100)
print("Repartition de la cible (test) :")
print(y_test.value_counts(normalize=True).round(4) * 100)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
resultats = {}

# %% [markdown]
# ### 2. Regression Logistique (modele de reference)
# class_weight="balanced" pour compenser le desequilibre 75,4%/24,5% sans
# sous/sur-echantillonner. Validation croisee 5-fold stratifiee sur le train,
# scoring AUC-ROC (metrique la moins sensible au desequilibre de classe parmi
# les 4 retenues pour la comparaison finale).

# %%
modele_logreg = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
)

scores_cv_logreg = cross_val_score(
    modele_logreg, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1
)
print(
    f"Regression Logistique - AUC-ROC (5-fold CV) : "
    f"{scores_cv_logreg.mean():.4f} +/- {scores_cv_logreg.std():.4f}"
)

modele_logreg.fit(X_train, y_train)
y_pred_logreg = modele_logreg.predict(X_test)
y_proba_logreg = modele_logreg.predict_proba(X_test)[:, 1]

resultats["Regression Logistique"] = {
    "AUC-ROC (test)": roc_auc_score(y_test, y_proba_logreg),
    "F1 (test)": f1_score(y_test, y_pred_logreg),
    "Precision (test)": precision_score(y_test, y_pred_logreg),
    "Rappel (test)": recall_score(y_test, y_pred_logreg),
    "AUC-ROC (5-fold CV, moyenne)": scores_cv_logreg.mean(),
    "AUC-ROC (5-fold CV, ecart-type)": scores_cv_logreg.std(),
}
print(pd.Series(resultats["Regression Logistique"]).round(4))

# %% [markdown]
# ### 3. Random Forest
# Meme traitement du desequilibre (class_weight="balanced") et meme protocole
# de validation croisee que la regression logistique, pour une comparaison
# equitable entre les deux modeles.

# %%
modele_rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

scores_cv_rf = cross_val_score(
    modele_rf, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1
)
print(
    f"Random Forest - AUC-ROC (5-fold CV) : "
    f"{scores_cv_rf.mean():.4f} +/- {scores_cv_rf.std():.4f}"
)

modele_rf.fit(X_train, y_train)
y_pred_rf = modele_rf.predict(X_test)
y_proba_rf = modele_rf.predict_proba(X_test)[:, 1]

resultats["Random Forest"] = {
    "AUC-ROC (test)": roc_auc_score(y_test, y_proba_rf),
    "F1 (test)": f1_score(y_test, y_pred_rf),
    "Precision (test)": precision_score(y_test, y_pred_rf),
    "Rappel (test)": recall_score(y_test, y_pred_rf),
    "AUC-ROC (5-fold CV, moyenne)": scores_cv_rf.mean(),
    "AUC-ROC (5-fold CV, ecart-type)": scores_cv_rf.std(),
}
print(pd.Series(resultats["Random Forest"]).round(4))

# %% [markdown]
# ### 4. XGBoost
# scale_pos_weight (ratio negatifs/positifs sur le train) joue le meme role
# que class_weight="balanced" pour les deux modeles precedents - XGBoost n'a
# pas ce parametre nomme ainsi. Tache prevue mercredi 12 - jeudi 13 aout dans
# le plan ; demarree ici des mercredi.

# %%
ratio_desequilibre = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight applique : {ratio_desequilibre:.4f}")

modele_xgb = XGBClassifier(
    n_estimators=300,
    scale_pos_weight=ratio_desequilibre,
    random_state=RANDOM_SEED,
    eval_metric="logloss",
    n_jobs=-1,
)

scores_cv_xgb = cross_val_score(
    modele_xgb, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1
)
print(
    f"XGBoost - AUC-ROC (5-fold CV) : "
    f"{scores_cv_xgb.mean():.4f} +/- {scores_cv_xgb.std():.4f}"
)

modele_xgb.fit(X_train, y_train)
y_pred_xgb = modele_xgb.predict(X_test)
y_proba_xgb = modele_xgb.predict_proba(X_test)[:, 1]

resultats["XGBoost"] = {
    "AUC-ROC (test)": roc_auc_score(y_test, y_proba_xgb),
    "F1 (test)": f1_score(y_test, y_pred_xgb),
    "Precision (test)": precision_score(y_test, y_pred_xgb),
    "Rappel (test)": recall_score(y_test, y_pred_xgb),
    "AUC-ROC (5-fold CV, moyenne)": scores_cv_xgb.mean(),
    "AUC-ROC (5-fold CV, ecart-type)": scores_cv_xgb.std(),
}
print(pd.Series(resultats["XGBoost"]).round(4))

# %% [markdown]
# ### 5. Tableau comparatif (version de travail)
# Sert de base au point d'equipe du jeudi 13 aout pour choisir le modele
# final. Le tableau/graphiques officiels (courbes ROC, matrices de confusion)
# restent la tache d'Andy, a construire des que ces resultats sont
# disponibles (notebooks/04_evaluation_modeles.ipynb).

# %%
tableau_comparatif = pd.DataFrame(resultats).T.round(4)
print(tableau_comparatif)
