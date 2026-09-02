# %% [markdown]
# ## Semaine 1 - Etape 1 : Chargement et nettoyage initial
# Dataset attendu : Kaggle "Loan Default Dataset" (yasserh), ~148 670 lignes.
# A executer en local (VS Code / Jupyter) ou en copiant les cellules dans Google Colab.

# %%
import os
import sys

import pandas as pd

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    # Chaque membre du groupe doit avoir son propre token Kaggle (gratuit) :
    # https://www.kaggle.com/settings -> API -> "Create New Token" (telecharge kaggle.json)
    BASE_DIR = "/content/ProjetScoringCredit"
    os.makedirs(f"{BASE_DIR}/data/raw", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/data/processed", exist_ok=True)
    if not os.path.exists(f"{BASE_DIR}/data/raw/Loan_Default.csv"):
        from google.colab import files
        print("Uploadez votre kaggle.json (Kaggle > Settings > API > Create New Token) :")
        files.upload()
        os.makedirs("/root/.kaggle", exist_ok=True)
        os.system("cp kaggle.json /root/.kaggle/kaggle.json && chmod 600 /root/.kaggle/kaggle.json")
        os.system("pip install -q kaggle")
        os.system(
            f"kaggle datasets download -d yasserh/loan-default-dataset "
            f"-p {BASE_DIR}/data/raw --unzip"
        )
else:
    BASE_DIR = ".."

RAW_PATH = f"{BASE_DIR}/data/raw/Loan_Default.csv"

df = pd.read_csv(RAW_PATH)
print("Dimensions :", df.shape)
print("\nColonnes trouvees :")
print(list(df.columns))

# %% [markdown]
# ### Verification des colonnes attendues
# Si le fichier telecharge correspond au dataset "Loan Default Dataset" de yasserh,
# on doit retrouver approximativement ces colonnes. On compare pour detecter un ecart
# (nom different, colonnes manquantes/en plus) avant d'aller plus loin.

# %%
colonnes_attendues = [
    "ID", "year", "loan_limit", "Gender", "approv_in_adv", "loan_type",
    "loan_purpose", "Credit_Worthiness", "open_credit", "business_or_commercial",
    "loan_amount", "rate_of_interest", "Interest_rate_spread", "Upfront_charges",
    "term", "Neg_ammortization", "interest_only", "lump_sum_payment",
    "property_value", "construction_type", "occupancy_type", "Secured_by",
    "total_units", "income", "credit_type", "Credit_Score",
    "co-applicant_credit_type", "age", "submission_of_application", "LTV",
    "Region", "Security_Type", "Status", "dtir1",
]

manquantes = [c for c in colonnes_attendues if c not in df.columns]
en_plus = [c for c in df.columns if c not in colonnes_attendues]
print("Colonnes attendues manquantes :", manquantes)
print("Colonnes presentes non attendues :", en_plus)

# %% [markdown]
# ### Types et valeurs manquantes

# %%
print(df.dtypes)
print("\nValeurs manquantes par colonne (top 15) :")
print(df.isna().sum().sort_values(ascending=False).head(15))

# %% [markdown]
# ### Valeurs manquantes deguisees
# 1260 lignes ont income == 0 (jamais negatif). Un revenu mensuel exactement nul
# pour un demandeur de credit n'est pas plausible : c'est tres probablement un
# encodage de valeur manquante par 0 plutot qu'un NaN explicite.
# On le convertit en NaN pour que l'imputation de la Semaine 2 (mediane) le
# traite comme les 9150 valeurs deja manquantes, plutot que de laisser un "0
# FCFA de revenu mensuel" une fois la mise a l'echelle camerounaise appliquee
# (risque de division par zero dans les ratios derives, ex. ratio_endettement).

# %%
n_income_zero = (df["income"] == 0).sum()
print(f"Lignes avec income == 0 (traitees comme manquantes) : {n_income_zero}")
df.loc[df["income"] == 0, "income"] = pd.NA

# %% [markdown]
# ### Doublons

# %%
n_doublons = df.duplicated().sum()
print(f"Doublons exacts detectes : {n_doublons}")
df = df.drop_duplicates()

if "ID" in df.columns:
    n_doublons_id = df.duplicated(subset="ID").sum()
    print(f"Doublons sur la colonne ID : {n_doublons_id}")
    df = df.drop_duplicates(subset="ID")

print("Dimensions apres suppression des doublons :", df.shape)

# %% [markdown]
# ### Sauvegarde intermediaire (avant traduction camerounaise)

# %%
OUTPUT_PATH = f"{BASE_DIR}/data/processed/loan_default_clean.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"Fichier nettoye sauvegarde : {OUTPUT_PATH}")
