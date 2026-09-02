# %% [markdown]
# ## Semaine 1 (ajustement) - Etape 4 : nettoyage avance avant la Semaine 2
# Entree  : ../data/processed/Loan_Default_Cameroun_Enrichi.csv (sortie du script 03)
# Sortie  : ../data/processed/Loan_Default_Cameroun_Modele.csv (dataset pret pour la
#           Semaine 2 : imputation, encodage, entrainement)
#
# Objectif : retirer physiquement les colonnes sans aucune utilite (ni modele, ni
# formulaire, ni description), listees dans
# docs/Classification_Variables_Consolidee.txt (sections 1.B et 1.C) :
# - fuite / post-decision banque : plafond_pret, approbation_anticipee, solvabilite,
#   ecart_taux_interet, taux_interet, frais_initiaux_fcfa, amortissement_negatif,
#   interet_seul, paiement_forfaitaire ;
# - infrastructure de bureau de credit inexistante au Cameroun : type_credit_bureau,
#   score_credit_bureau, type_credit_coemprunteur ;
# - constante sans information : annee ;
# - region_cameroun : decision revisee le 2026-07-31 (Aristide) - aucune valeur meme
#   descriptive (couverture partielle 4/10 regions, gradient herite du remapping des
#   regions US d'origine) -> supprimee completement, alors qu'elle etait jusque-la
#   conservee en formulaire a titre descriptif uniquement.

# %%
import sys

import pandas as pd

IN_COLAB = "google.colab" in sys.modules
BASE_DIR = "/content/ProjetScoringCredit" if IN_COLAB else ".."

INPUT_PATH = f"{BASE_DIR}/data/processed/Loan_Default_Cameroun_Enrichi.csv"
OUTPUT_PATH = f"{BASE_DIR}/data/processed/Loan_Default_Cameroun_Modele.csv"

df = pd.read_csv(INPUT_PATH)
print("Dimensions en entree :", df.shape)

# %% [markdown]
# ### 1. Colonnes de fuite / post-decision banque
# Fixees ou connues par l'etablissement seulement apres (ou pendant) l'evaluation du
# risque : jamais declarables par un nouveau demandeur au moment de la demande.

# %%
COLONNES_FUITE_POST_DECISION = [
    "plafond_pret",
    "approbation_anticipee",
    "solvabilite",
    "ecart_taux_interet",
    "taux_interet",
    "frais_initiaux_fcfa",
    "amortissement_negatif",
    "interet_seul",
    "paiement_forfaitaire",
]

# %% [markdown]
# ### 2. Infrastructure de bureau de credit inexistante au Cameroun
# Le public vise (secteur informel, primo-emprunteurs) n'a le plus souvent aucun
# historique de bureau de credit individuel de type Experian/Equifax.

# %%
COLONNES_BUREAU_CREDIT = [
    "type_credit_bureau",
    "score_credit_bureau",
    "type_credit_coemprunteur",
]

# %% [markdown]
# ### 3. Constante et geographie sans valeur
# `annee` est constante (2019 sur toutes les lignes). `region_cameroun` etait
# jusqu'ici conservee a titre descriptif ; decision revisee le 2026-07-31 : aucune
# valeur meme descriptive, retiree entierement (dataset + formulaire).

# %%
COLONNES_SANS_VALEUR = [
    "annee",
    "region_cameroun",
]

COLONNES_A_RETIRER = COLONNES_FUITE_POST_DECISION + COLONNES_BUREAU_CREDIT + COLONNES_SANS_VALEUR

colonnes_absentes = [c for c in COLONNES_A_RETIRER if c not in df.columns]
assert not colonnes_absentes, f"Colonnes attendues mais deja absentes : {colonnes_absentes}"

df = df.drop(columns=COLONNES_A_RETIRER)
print(f"{len(COLONNES_A_RETIRER)} colonnes retirees : {COLONNES_A_RETIRER}")

# %% [markdown]
# ### 4. Verification finale

# %%
print("Dimensions finales :", df.shape)
print("\nColonnes restantes :")
print(df.columns.tolist())

for col in COLONNES_A_RETIRER:
    assert col not in df.columns, f"{col} n'aurait pas du etre presente"

print("\nValeurs manquantes restantes (top 10) :")
print(df.isna().sum().sort_values(ascending=False).head(10))

# %% [markdown]
# ### 5. Sauvegarde du dataset pret pour la Semaine 2

# %%
df.to_csv(OUTPUT_PATH, index=False)
print(f"Dataset sauvegarde : {OUTPUT_PATH}")
