# %% [markdown]
# ## Semaine 2 - Etape 1 : valeurs manquantes, valeurs aberrantes, recontextualisation
# Entree  : ../data/processed/Loan_Default_Cameroun_Modele.csv (sortie du script 04)
# Sortie  : ../data/processed/Loan_Default_Cameroun_Traite.csv
#
# Taches (lundi 3 - mardi 4 aout 2026) :
# 1) Imputation des valeurs manquantes : mediane (numeriques), mode (categorielles)
# 2) Traitement des valeurs aberrantes par la methode IQR (capping/winsorisation)
# 3) Regeneration complete de duree_mois : le dataset source (pret immobilier
#    americain) a 360 mois (30 ans) comme valeur dominante (81,8% des lignes), ce
#    qui rend la methode IQR degeneree sur cette colonne (Q1 = Q3 = 360) et ne
#    correspond a aucune realite de credit general/microfinance au Cameroun.
#    Decision du 04/08/2026 (Aristide) : reouverture assumee du point tranche le
#    29/07 (section 4.5 du plan, "duree_mois conservee telle quelle") - remplace
#    par des durees tirees aleatoirement dans des tranches realistes, conditionnees
#    sur objet_pret et montant_pret_fcfa (jamais sur statut_remboursement, pour ne
#    pas fabriquer un proxy de la cible comme categorie_risque).
# 4) Plancher de plausibilite metier sur revenu_mensuel_fcfa (99 lignes < 10 000
#    FCFA/mois, non plausible pour solliciter un credit) - decision du 04/08/2026.
# 5) Regeneration complete de tranche_age : le dataset source reflete une
#    population de refinancement hypothecaire americain (0,9% de <25 ans, 18,8%
#    de plus de 65 ans, defaut croissant avec l'age) plutot qu'une clientele
#    active de credit general/microfinance camerounaise. Meme principe que
#    duree_mois : tirage independant de statut_remboursement, decision du
#    04/08/2026.
# 6) Relabellisation de type_pret (Type_1/2/3 -> Banque/Microfinance/
#    Cooperative_epargne_credit) : simple renommage des categories (aucune valeur
#    ne change de ligne), pour rattacher ce signal reel (22,8%/34,5%/25,1% de
#    defaut) au contexte camerounais explicitement couvert par le perimetre du
#    projet - banques traditionnelles ET institutions de microfinance (section 1
#    du plan, tranche le 29/07). Decision du 04/08/2026.

# %%
import sys

import numpy as np
import pandas as pd

IN_COLAB = "google.colab" in sys.modules
BASE_DIR = "/content/ProjetScoringCredit" if IN_COLAB else ".."

INPUT_PATH = f"{BASE_DIR}/data/processed/Loan_Default_Cameroun_Modele.csv"
OUTPUT_PATH = f"{BASE_DIR}/data/processed/Loan_Default_Cameroun_Traite.csv"

RANDOM_SEED = 42

df = pd.read_csv(INPUT_PATH)
print("Dimensions en entree :", df.shape)
print("\nValeurs manquantes en entree :")
print(df.isna().sum()[df.isna().sum() > 0])

# %% [markdown]
# ### 1. Imputation des valeurs manquantes
# Mode pour les variables categorielles, mediane pour les numeriques. `duree_mois`
# est exclue de l'imputation par mediane : elle est integralement regeneree en
# etape 3, ses valeurs manquantes sont donc traitees par la meme occasion.

# %%
COLONNES_CATEGORIELLES_A_IMPUTER = ["objet_pret", "tranche_age", "mode_soumission"]
COLONNES_NUMERIQUES_A_IMPUTER = ["revenu_mensuel_fcfa", "ratio_endettement"]

for col in COLONNES_CATEGORIELLES_A_IMPUTER:
    mode = df[col].mode(dropna=True)[0]
    n_manquantes = df[col].isna().sum()
    df[col] = df[col].fillna(mode)
    print(f"{col} : {n_manquantes} valeurs imputees par le mode ({mode!r})")

for col in COLONNES_NUMERIQUES_A_IMPUTER:
    mediane = df[col].median()
    n_manquantes = df[col].isna().sum()
    df[col] = df[col].fillna(mediane)
    print(f"{col} : {n_manquantes} valeurs imputees par la mediane ({mediane})")

# %% [markdown]
# ### 2. Traitement des valeurs aberrantes (methode IQR)
# Bornes `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` calculees sur les valeurs deja imputees.
# Capping (winsorisation) plutot que suppression de lignes : dans un contexte de
# scoring credit, les valeurs extremes de revenu/montant/endettement peuvent etre
# un vrai signal de risque, pas seulement du bruit - les supprimer ferait perdre
# des cas de defaut reels. `duree_mois` est exclue : elle est regeneree en etape 3.

# %%
COLONNES_IQR = ["montant_pret_fcfa", "revenu_mensuel_fcfa", "ratio_endettement"]

for col in COLONNES_IQR:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    borne_basse, borne_haute = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_aberrantes = ((df[col] < borne_basse) | (df[col] > borne_haute)).sum()
    df[col] = df[col].clip(lower=borne_basse, upper=borne_haute)
    print(
        f"{col} : bornes [{borne_basse:.0f}, {borne_haute:.0f}], "
        f"{n_aberrantes} valeurs cappees ({100 * n_aberrantes / len(df):.2f}%)"
    )

# %% [markdown]
# ### 2 bis. Plancher de plausibilite metier sur revenu_mensuel_fcfa
# La borne basse IQR est negative (aucun revenu n'est donc capte par la methode
# statistique), mais un revenu declare sous 15 000 FCFA/mois n'est pas plausible
# pour un demandeur de credit. Plancher metier fixe a 15 000 FCFA/mois (sous le
# SMIG formel camerounais de 36 270 FCFA, pour rester compatible avec un profil
# informel/rural), distinct du capping statistique ci-dessus.

# %%
PLANCHER_REVENU_FCFA = 15_000
n_sous_plancher = (df["revenu_mensuel_fcfa"] < PLANCHER_REVENU_FCFA).sum()
df["revenu_mensuel_fcfa"] = df["revenu_mensuel_fcfa"].clip(lower=PLANCHER_REVENU_FCFA)
print(
    f"revenu_mensuel_fcfa : {n_sous_plancher} lignes remontees au plancher "
    f"de {PLANCHER_REVENU_FCFA} FCFA/mois"
)

# %% [markdown]
# ### 3. Regeneration de duree_mois avec des tranches realistes camerounaises
# Tranches de montant (calibrees sur les quantiles reels de montant_pret_fcfa) et
# durees en mois par objet_pret x tranche de montant, validees le 04/08/2026.
# Tirage aleatoire (entier, loi uniforme) dans la fourchette correspondante, sans
# jamais lire statut_remboursement : aucune fuite possible vers la cible.

# %%
def tranche_montant(montant):
    if montant < 2_000_000:
        return "petit"
    if montant < 8_000_000:
        return "moyen"
    if montant < 20_000_000:
        return "grand"
    return "tres_grand"


DUREE_BORNES_MOIS = {
    ("Autre", "petit"): (3, 12),
    ("Autre", "moyen"): (6, 18),
    ("Autre", "grand"): (12, 24),
    ("Autre", "tres_grand"): (18, 36),
    ("Achat", "petit"): (6, 12),
    ("Achat", "moyen"): (12, 24),
    ("Achat", "grand"): (24, 36),
    ("Achat", "tres_grand"): (24, 48),
    ("Investissement_activite", "petit"): (6, 18),
    ("Investissement_activite", "moyen"): (12, 36),
    ("Investissement_activite", "grand"): (24, 48),
    ("Investissement_activite", "tres_grand"): (36, 60),
    ("Refinancement", "petit"): (6, 12),
    ("Refinancement", "moyen"): (12, 24),
    ("Refinancement", "grand"): (18, 36),
    ("Refinancement", "tres_grand"): (24, 48),
}

rng = np.random.default_rng(RANDOM_SEED)
tranches = df["montant_pret_fcfa"].apply(tranche_montant)

bornes = pd.Series(
    list(zip(df["objet_pret"], tranches)), index=df.index
).map(DUREE_BORNES_MOIS)
assert bornes.isna().sum() == 0, "Combinaison objet_pret/tranche_montant non couverte"

bornes_basses = bornes.map(lambda b: b[0])
bornes_hautes = bornes.map(lambda b: b[1])
df["duree_mois"] = rng.integers(bornes_basses, bornes_hautes + 1)

print("Nouvelle distribution de duree_mois :")
print(df["duree_mois"].describe())
print("\nDuree mediane par objet_pret :")
print(df.groupby("objet_pret")["duree_mois"].median())

# %% [markdown]
# ### 4. Regeneration de tranche_age avec une distribution active camerounaise
# Distribution cible validee le 04/08/2026, recentree sur une population active
# (25-44 ans = 57% au lieu de 35% dans le dataset source) plutot que sur un profil
# de refinancement hypothecaire americain. Tirage aleatoire independant de toute
# autre colonne (donc de statut_remboursement) : aucune correlation fabriquee avec
# la cible.
#
# Consequence assumee (verifiee le 04/08/2026) : le taux de defaut devient plat
# sur toutes les tranches (24,0% a 25,1%, cf. verification finale ci-dessous),
# puisque le tirage est independant de tout le reste. Decision d'Aristide : on
# accepte cette perte de signal (deja artificiel, herite du profil americain) au
# profit du realisme demographique. tranche_age passe donc de "feature ML"
# (section 3.A de Classification_Variables_Consolidee.txt) a "descriptive
# uniquement" (section 3.B), au meme titre que niveau_education - mise a jour du
# document a faire en consequence.

# %%
DISTRIBUTION_AGE_CIBLE = {
    "<25": 0.08,
    "25-34": 0.30,
    "35-44": 0.27,
    "45-54": 0.18,
    "55-64": 0.11,
    "65-74": 0.04,
    ">74": 0.02,
}
assert abs(sum(DISTRIBUTION_AGE_CIBLE.values()) - 1.0) < 1e-9

tranches_age = list(DISTRIBUTION_AGE_CIBLE.keys())
probabilites_age = list(DISTRIBUTION_AGE_CIBLE.values())
df["tranche_age"] = rng.choice(tranches_age, size=len(df), p=probabilites_age)

print("Nouvelle distribution de tranche_age :")
print(df["tranche_age"].value_counts(normalize=True).reindex(tranches_age) * 100)

# %% [markdown]
# ### 5. Relabellisation de type_pret (Banque / Microfinance / Cooperative)
# Renommage pur des 3 categories existantes (Type_1/2/3), sans regeneration de
# valeurs : aucune ligne ne change de categorie, donc aucun risque de fuite. Le
# rattachement au type d'etablissement s'appuie sur l'ordre de risque deja observe
# dans les donnees (Type_2 > Type_3 > Type_1 en taux de defaut), coherent avec des
# profils reels camerounais (banque : dossiers plus formels/garantis, risque plus
# faible ; microfinance : clientele plus informelle, risque plus eleve ;
# cooperative d'epargne-credit : caution mutuelle entre membres, risque
# intermediaire). Decision du 04/08/2026, perimetre banques + microfinance deja
# valide le 29/07 (section 1 du plan).

# %%
LIBELLES_TYPE_PRET = {
    "Type_1": "Banque",
    "Type_2": "Microfinance",
    "Type_3": "Cooperative_epargne_credit",
}
df["type_pret"] = df["type_pret"].map(LIBELLES_TYPE_PRET)
assert df["type_pret"].isna().sum() == 0, "Valeur de type_pret non couverte par le mapping"

print("Nouvelles categories de type_pret :")
print(df["type_pret"].value_counts())

# %% [markdown]
# ### 6. Verification finale

# %%
print("Dimensions finales :", df.shape)
assert df.isna().sum().sum() == 0, "Des valeurs manquantes subsistent"
print("Aucune valeur manquante restante : OK")

print("\nTaux de defaut par tranche_age (verification de la perte de signal assumee) :")
print(
    (df.groupby("tranche_age")["statut_remboursement"].apply(lambda s: (s == "Defaut").mean() * 100))
    .round(2)
)

# %% [markdown]
# ### 7. Sauvegarde du dataset traite

# %%
df.to_csv(OUTPUT_PATH, index=False)
print(f"Dataset sauvegarde : {OUTPUT_PATH}")
