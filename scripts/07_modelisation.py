# %% [markdown]
# ## Semaine 3 : modelisation ML (Regression Logistique, Random Forest,
# XGBoost, CatBoost) et sauvegarde du modele final
# Entree  : ../data/processed/Loan_Default_Cameroun_Encode.csv (sortie du script 06)
# Sortie  : tableau comparatif des 4 modeles + modele final sauvegarde avec
#           joblib dans ../models/, reutilisable par l'app Streamlit
#
# Etape 1 (lundi 10 - mercredi 12 aout 2026) :
# 1) Separation features (categorie A, 16 colonnes) / cible, split train/test
#    80/20 stratifie (la cible est desequilibree, 75,4% Rembourse / 24,5% Defaut)
# 2) Regression Logistique (reference) + validation croisee stratifiee 5-fold
# 3) Random Forest + validation croisee stratifiee 5-fold
# 4) XGBoost + validation croisee stratifiee 5-fold
# Desequilibre de classe traite par class_weight="balanced" (Regression
# Logistique, Random Forest) et scale_pos_weight (XGBoost) plutot que par
# sur/sous-echantillonnage, pour ne pas alterer la distribution reelle du jeu
# d'entrainement. Resultats (test) : AUC-ROC 0.590 / 0.776 / 0.793 - XGBoost
# meilleur des 3 prevus au planning initial.
#
# Etape 2 (vendredi 14 aout 2026) : test complementaire d'un 4e modele,
# CatBoost, non prevu au planning initial mais teste avant la sauvegarde
# definitive pour verifier qu'aucune alternative simple ne fait mieux que
# XGBoost. Meme protocole exact que les 3 premiers modeles (meme split, meme
# seed, memes 16 features, scale_pos_weight equivalent pour le desequilibre)
# afin d'avoir une comparaison equitable. CatBoost devance XGBoost sur les 4
# metriques (AUC-ROC test 0.805 vs 0.793) et avec une variance plus faible en
# validation croisee (+/- 0.0036 vs +/- 0.0047) - retenu comme modele final.
# Le modele CatBoost entraine ci-dessous (sur X_train, celui dont les
# metriques sont rapportees dans le tableau comparatif) est sauvegarde avec
# joblib, accompagne de la liste des features, pour reutilisation par l'app
# Streamlit sans reentrainement.

# %%
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
# ### 5. CatBoost (test complementaire, vendredi 14 aout)
# Meme traitement du desequilibre (scale_pos_weight, equivalent a
# class_weight="balanced") et meme protocole de validation croisee que les
# 3 modeles precedents. allow_writing_files=False evite que CatBoost cree un
# dossier de travail catboost_info/ (inutile ici et source de conflits quand
# cross_val_score parallelise les folds).

# %%
modele_catboost = CatBoostClassifier(
    iterations=300,
    scale_pos_weight=ratio_desequilibre,
    random_seed=RANDOM_SEED,
    verbose=False,
    allow_writing_files=False,
)

scores_cv_catboost = cross_val_score(
    modele_catboost, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1
)
print(
    f"CatBoost - AUC-ROC (5-fold CV) : "
    f"{scores_cv_catboost.mean():.4f} +/- {scores_cv_catboost.std():.4f}"
)

modele_catboost.fit(X_train, y_train)
y_pred_catboost = modele_catboost.predict(X_test)
y_proba_catboost = modele_catboost.predict_proba(X_test)[:, 1]

resultats["CatBoost"] = {
    "AUC-ROC (test)": roc_auc_score(y_test, y_proba_catboost),
    "F1 (test)": f1_score(y_test, y_pred_catboost),
    "Precision (test)": precision_score(y_test, y_pred_catboost),
    "Rappel (test)": recall_score(y_test, y_pred_catboost),
    "AUC-ROC (5-fold CV, moyenne)": scores_cv_catboost.mean(),
    "AUC-ROC (5-fold CV, ecart-type)": scores_cv_catboost.std(),
}
print(pd.Series(resultats["CatBoost"]).round(4))

# %% [markdown]
# ### 6. Tableau comparatif final
# Le tableau/graphiques officiels (courbes ROC, matrices de confusion)
# restent la tache d'Andy (notebook 04_evaluation_modeles.ipynb). CatBoost
# devance les 3 autres modeles sur les 4 metriques : retenu comme modele
# final.

# %%
tableau_comparatif = pd.DataFrame(resultats).T.round(4)
print(tableau_comparatif)

# %% [markdown]
# ### 7. Sauvegarde du modele final (joblib)
# CatBoost (section 5) est le modele final : meilleur des 4 sur AUC-ROC, F1,
# Precision et Rappel (test), et validation croisee la plus stable. On
# sauvegarde le modele deja entraine sur X_train ci-dessus (celui dont les
# metriques sont rapportees dans le tableau comparatif, pour que le fichier
# reflete exactement les performances annoncees) avec joblib, accompagne de
# la liste des features et du nom de la cible pour que l'app Streamlit
# reconstruise l'entree dans le bon ordre sans avoir a redupliquer cette
# logique.

# %%
import os

import joblib

os.makedirs(f"{BASE_DIR}/models", exist_ok=True)

OUTPUT_PATH = f"{BASE_DIR}/models/modele_scoring_credit.joblib"

joblib.dump(
    {
        "modele": modele_catboost,
        "features": FEATURES,
        "cible": CIBLE,
        "algorithme": "CatBoost",
    },
    OUTPUT_PATH,
)
print(f"Modele final sauvegarde : {OUTPUT_PATH}")

# %% [markdown]
# Verification : rechargement du fichier et controle que les predictions
# sont identiques a celles calculees plus haut (aucune perte a la
# serialisation/deserialisation).

# %%
objet_recharge = joblib.load(OUTPUT_PATH)
y_proba_recharge = objet_recharge["modele"].predict_proba(X_test[objet_recharge["features"]])[:, 1]
assert np.allclose(y_proba_recharge, y_proba_catboost), "Predictions differentes apres rechargement !"
print("Rechargement joblib verifie : predictions identiques.")
