# %% [markdown]
# ## Semaine 2 - Etape 2 : encodage des variables categorielles et normalisation
# Entree  : ../data/processed/Loan_Default_Cameroun_Traite.csv (sortie du script 05)
# Sortie  : ../data/processed/Loan_Default_Cameroun_Encode.csv
#
# Taches (mercredi 5 - jeudi 6 aout 2026) :
# 1) Suppression de mode_soumission (decision du 06/08/2026, voir note ci-dessous)
# 2) Encodage de la cible statut_remboursement (binaire, 0/1)
# 3) Encodage binaire (Oui/Non -> 1/0) de usage_professionnel et credit_ouvert
# 4) One-Hot Encoding de type_pret, objet_pret, secteur_activite
# 5) Normalisation (StandardScaler) des variables numeriques du modele
#
# Note du 06/08/2026 (Aristide) - suppression de mode_soumission :
# la liste finale d'Andy (Liste_finale.docx, categorie A section 4) et sa
# documentation Power BI recommandent mode_soumission comme variable prioritaire
# (28,4% de defaut en agence contre 17,5% en ligne). Decision revisee malgre ce
# signal reel confirme le 30/07 : la configuration retenue pour l'application ne
# traite que des demandes de credit physique en agence (perimetre produit, pas un
# constat statistique) - la colonne serait donc constante ("En_agence") pour
# toute nouvelle demande en production, sans aucune valeur predictive ni meme
# descriptive a ce stade. Meme principe que la suppression de region_cameroun le
# 31/07 (section 1.C de Classification_Variables_Consolidee.txt) : une colonne
# retiree du perimetre applicatif est supprimee du dataset dans son ensemble,
# pas seulement exclue des features du modele. A signaler a Andy avant le point
# d'equipe du dimanche 9 aout (sa liste finale devra etre mise a jour en
# consequence).
#
# Note du 06/08/2026 (Aristide) - divergences avec la liste finale d'Andy :
# la liste d'Andy reintroduit genre et tranche_age comme features ML, et exclut
# credit_ouvert. Les deux premiers points contredisent des decisions deja
# tranchees et documentees (genre : motif ethique, 29/07 ; tranche_age :
# regeneree independamment de la cible pour eviter la fuite, reclassee
# descriptive le 04/08). Le troisieme (credit_ouvert) contredit le statut de
# feature ML retenu le 30/07 (signal reel mais rare, a surveiller). Ce script
# suit la classification consolidee (Classification_Variables_Consolidee.txt),
# pas la liste d'Andy, sur ces 3 points precis - egalement a signaler avant le
# 9 aout.

# %%
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

IN_COLAB = "google.colab" in sys.modules
BASE_DIR = "/content/ProjetScoringCredit" if IN_COLAB else ".."

INPUT_PATH = f"{BASE_DIR}/data/processed/Loan_Default_Cameroun_Traite.csv"
OUTPUT_PATH = f"{BASE_DIR}/data/processed/Loan_Default_Cameroun_Encode.csv"

df = pd.read_csv(INPUT_PATH)
print("Dimensions en entree :", df.shape)

# %% [markdown]
# ### 1. Suppression de mode_soumission
# Colonne retiree du perimetre applicatif (voir note ci-dessus) : supprimee du
# dataset dans son ensemble, pas seulement des features du modele.

# %%
df = df.drop(columns=["mode_soumission"])
print("mode_soumission supprimee. Dimensions :", df.shape)

# %% [markdown]
# ### 2. Encodage de la cible (statut_remboursement)
# Encodage binaire simple : 1 = Defaut (classe positive, celle que le modele doit
# detecter), 0 = Rembourse. La colonne est ecrasee en place (meme nom), il n'y a
# pas de doublon a gerer (statut_remboursement_label a deja ete supprime au
# script 03).

# %%
MAPPING_CIBLE = {"Rembourse": 0, "Defaut": 1}
valeurs_avant = set(df["statut_remboursement"].unique())
assert valeurs_avant == set(MAPPING_CIBLE), f"Valeurs inattendues : {valeurs_avant}"
df["statut_remboursement"] = df["statut_remboursement"].map(MAPPING_CIBLE)
print("Cible encodee :", MAPPING_CIBLE)
print(df["statut_remboursement"].value_counts(normalize=True).round(4) * 100)

# %% [markdown]
# ### 3. Colonnes exclues des features du modele (conservees telles quelles)
# Ces colonnes restent dans le fichier de sortie pour le profil client / le
# rapport, mais ne sont ni encodees ni normalisees : elles ne sont pas destinees
# a alimenter l'entrainement (voir Classification_Variables_Consolidee.txt,
# section 3.B). id_client suit la meme logique (identifiant technique, jamais
# une feature).

# %%
COLONNES_NON_FEATURES = [
    "id_client",
    "genre",
    "tranche_age",
    "niveau_education",
    "membre_tontine",
    "activite_saisonniere",
    "utilisation_mobile_money",
]
print("Colonnes conservees sans transformation :", COLONNES_NON_FEATURES)

# %% [markdown]
# ### 4. Encodage binaire (Oui/Non -> 1/0)
# usage_professionnel et credit_ouvert n'ont que 2 categories sans ordre naturel
# a preserver : un encodage binaire direct suffit, pas besoin de One-Hot (qui
# ne ferait que dupliquer l'information sur 2 colonnes).

# %%
COLONNES_BINAIRES = ["usage_professionnel", "credit_ouvert"]
MAPPING_BINAIRE = {"Non": 0, "Oui": 1}

for col in COLONNES_BINAIRES:
    valeurs = set(df[col].unique())
    assert valeurs == set(MAPPING_BINAIRE), f"{col} : valeurs inattendues {valeurs}"
    df[col] = df[col].map(MAPPING_BINAIRE)
    print(f"{col} encodee (Non=0, Oui=1)")

# %% [markdown]
# ### 5. One-Hot Encoding des variables categorielles nominales
# type_pret, objet_pret et secteur_activite n'ont pas d'ordre naturel entre leurs
# categories : le One-Hot Encoding evite d'imposer une hierarchie artificielle
# qu'un Label Encoding introduirait (ex. Banque=0 < Microfinance=1 n'aurait
# aucun sens pour un modele lineaire).

# %%
COLONNES_ONEHOT = ["type_pret", "objet_pret", "secteur_activite"]

for col in COLONNES_ONEHOT:
    print(f"{col} : {df[col].nunique()} categories -> {sorted(df[col].unique())}")

df = pd.get_dummies(df, columns=COLONNES_ONEHOT, prefix=COLONNES_ONEHOT, dtype=int)
print("\nDimensions apres One-Hot Encoding :", df.shape)

# %% [markdown]
# ### 6. Normalisation des variables numeriques (StandardScaler)
# Centrage-reduction (moyenne 0, ecart-type 1) : necessaire pour les modeles
# sensibles a l'echelle des variables (regression logistique) et sans effet
# negatif pour les modeles bases sur des arbres (Random Forest, XGBoost) prevus
# en semaine 3 - un seul dataset encode sert donc aux 3 modeles.

# %%
COLONNES_NUMERIQUES = [
    "montant_pret_fcfa",
    "duree_mois",
    "revenu_mensuel_fcfa",
    "ratio_endettement",
]

scaler = StandardScaler()
df[COLONNES_NUMERIQUES] = scaler.fit_transform(df[COLONNES_NUMERIQUES])
print("Variables normalisees :", COLONNES_NUMERIQUES)
print(df[COLONNES_NUMERIQUES].describe().round(3))

# %% [markdown]
# ### 7. Verification finale

# %%
print("Dimensions finales :", df.shape)
assert df.isna().sum().sum() == 0, "Des valeurs manquantes subsistent"
print("Aucune valeur manquante restante : OK")

colonnes_features = [
    c
    for c in df.columns
    if c not in COLONNES_NON_FEATURES and c != "statut_remboursement"
]
print(f"\n{len(colonnes_features)} colonnes de features pretes pour la modelisation :")
print(colonnes_features)

# %% [markdown]
# ### 8. Sauvegarde du dataset encode

# %%
df.to_csv(OUTPUT_PATH, index=False)
print(f"Dataset sauvegarde : {OUTPUT_PATH}")
