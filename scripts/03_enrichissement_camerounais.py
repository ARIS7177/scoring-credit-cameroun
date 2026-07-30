# %% [markdown]
# ## Semaine 1 - Etape 3 : Assainissement et enrichissement camerounais
# Entree  : ../data/processed/Loan_Default_Cameroun.csv (sortie du script 02)
#           ../data/processed/secteur_activite_niveau_education.csv (variables
#           de Marie-Therese conservees telles quelles, voir section 2)
# Sortie  : ../data/processed/Loan_Default_Cameroun_Enrichi.csv (livrable Semaine 1)
#
# Objectif : appliquer les decisions actees le 2026-07-29 (voir
# docs/Analyse_et_Plan_Projet.docx, section 3) sur les 8 variables camerounaises
# ajoutees par Marie-Therese en Semaine 2 initiale, et sur le perimetre metier
# (credit en general, pas credit immobilier) :
# - supprimer les colonnes de fuite/redondance (statut_remboursement_label,
#   zone_geographique) ;
# - regenerer 3 variables bruitees en les conditionnant sur des colonnes reelles
#   deja correlees a la cible (secteur_activite, region_cameroun), au lieu d'un
#   tirage uniforme sans rapport avec le risque de defaut ;
# - retirer les colonnes specifiques au credit immobilier (fuite + hors perimetre
#   produit) ;
# - ne PAS recreer categorie_risque ici : c'est une sortie calculee par l'app a
#   partir du score du modele, jamais une variable d'entree.

# %%
import sys

import numpy as np
import pandas as pd

IN_COLAB = "google.colab" in sys.modules
BASE_DIR = "/content/ProjetScoringCredit" if IN_COLAB else ".."

INPUT_PATH = f"{BASE_DIR}/data/processed/Loan_Default_Cameroun.csv"
LOOKUP_PATH = f"{BASE_DIR}/data/processed/secteur_activite_niveau_education.csv"
OUTPUT_PATH = f"{BASE_DIR}/data/processed/Loan_Default_Cameroun_Enrichi.csv"

GRAINE_ALEATOIRE = 42
rng = np.random.default_rng(GRAINE_ALEATOIRE)

df = pd.read_csv(INPUT_PATH)
print("Dimensions en entree :", df.shape)

# %% [markdown]
# ### 1. Suppression des colonnes specifiques au credit immobilier
# `valeur_bien_fcfa` est manquant pour 10.2% des lignes, et ces lignes ont un
# taux de defaut de 100.0% (contre 16.1% quand la valeur est presente) : un
# artefact de collecte du dataset source (la valorisation du bien s'arrete
# probablement d'etre enregistree une fois le pret en defaut/saisie), pas un
# vrai signal de risque transferable a un nouveau demandeur camerounais.
# Ces colonnes sont de toute facon specifiques a un credit immobilier, hors du
# perimetre produit (credit en general) : leur suppression resout a la fois la
# fuite et le desalignement de perimetre. Elles restent disponibles dans
# Loan_Default_Cameroun.csv pour un futur module optionnel "credit immobilier"
# hors perimetre V1.

# %%
COLONNES_IMMOBILIER = [
    "valeur_bien_fcfa",
    "ratio_pret_valeur",
    "type_construction",
    "type_occupation",
    "nombre_unites",
    "garanti_par",
    "type_garantie",
]

colonnes_a_retirer = [c for c in COLONNES_IMMOBILIER if c in df.columns]
df = df.drop(columns=colonnes_a_retirer)
print(f"{len(colonnes_a_retirer)} colonnes immobilieres retirees : {colonnes_a_retirer}")

# %% [markdown]
# ### 2. Ajout des variables camerounaises conservees telles quelles
# `secteur_activite` et `niveau_education` (variables de Marie-Therese) ne sont
# pas regenerees : l'audit du 2026-07-29 a confirme que `secteur_activite`
# porte un vrai gradient de risque (23.5-24.2% pour les 3 secteurs dominants vs
# 34-37% pour Commerce/Negoce, Profession liberale, Artisanat), et
# `niveau_education` est conservee pour la description du profil dans le
# formulaire (elle n'entre pas dans les variables du modele ML, decide en
# feature selection en Semaine 2, pas dans ce script).

# %%
lookup = pd.read_csv(LOOKUP_PATH)
n_avant = len(df)
df = df.merge(lookup, on="id_client", how="left")
assert len(df) == n_avant, "Le merge a modifie le nombre de lignes"
assert df["secteur_activite"].isna().sum() == 0, "secteur_activite manquant apres merge"
print("secteur_activite / niveau_education ajoutees. Repartition secteur_activite :")
print(df["secteur_activite"].value_counts())

# %% [markdown]
# ### 3. Regeneration des 3 variables bruitees
# L'audit du 2026-07-29 a montre que ces 3 variables, telles que generees
# initialement, etaient quasi non correlees a la cible (bruit ~aleatoire),
# malgre une documentation annoncant des "distributions realistes". On les
# regenere ici conditionnees sur des colonnes reelles deja correlees au risque
# (`secteur_activite`, `region_cameroun`), sans jamais lire la cible.
#
# Limite assumee (verifiee le 2026-07-30, decision d'Aristide) : le
# regroupement de `membre_tontine` (secteur informel/traditionnel : Agriculture,
# Petit commerce, Artisanat vs secteur formel/liberal : Salarie formel,
# Commerce/Negoce, Profession liberale) reflete une segmentation socio-
# economique reelle, mais melange des secteurs a risque de defaut oppose
# (Artisanat est le secteur le PLUS risque - 37.3% - tout en etant classe dans
# le groupe "forte tontine" avec Agriculture/Petit commerce, qui sont les
# secteurs les MOINS risques). Consequence verifiee : le taux de defaut par
# `membre_tontine` reste quasi plat (24.8% Non vs 24.5% Oui), tout comme pour
# `activite_saisonniere` et `utilisation_mobile_money`. Choix delibere : ne pas
# reforcer artificiellement une correlation avec la cible pour ces 3 variables
# - elles restent des signaux socio-economiques plausibles pour le profil
# camerounais, meme faiblement predictifs du defaut. A documenter comme limite
# honnete dans le rapport plutot que de forcer un signal.

# %%
PROBA_TONTINE = {
    "Agriculture": 0.55,
    "Petit commerce": 0.55,
    "Artisanat": 0.55,
    "Salarié formel": 0.25,
    "Commerce/Négoce": 0.25,
    "Profession libérale": 0.25,
}
proba_tontine = df["secteur_activite"].map(PROBA_TONTINE).fillna(0.40)
df["membre_tontine"] = np.where(rng.random(len(df)) < proba_tontine, "Oui", "Non")

PROBA_SAISONNIERE = {
    "Agriculture": 0.85,
    "Petit commerce": 0.35,
    "Commerce/Négoce": 0.35,
}
proba_saisonniere = df["secteur_activite"].map(PROBA_SAISONNIERE).fillna(0.10)
df["activite_saisonniere"] = np.where(rng.random(len(df)) < proba_saisonniere, "Oui", "Non")

# Mobile money : penetration plus forte en zone urbaine (Littoral/Centre) et
# chez les demandeurs plus instruits. Le mapping region -> urbanite est le
# meme que l'ancienne colonne zone_geographique (deterministe, donc pas besoin
# de la stocker comme colonne a part - voir section 4).
BASE_REGION_MOBILE_MONEY = {
    "Littoral": 0.75,
    "Centre": 0.75,
    "Ouest": 0.55,
    "Nord": 0.30,
}
AJUSTEMENT_EDUCATION_MOBILE_MONEY = {
    "Sans diplôme": -0.15,
    "Primaire": -0.05,
    "Secondaire": 0.05,
    "Supérieur": 0.15,
}
base_region = df["region_cameroun"].map(BASE_REGION_MOBILE_MONEY).fillna(0.50)
ajustement_education = df["niveau_education"].map(AJUSTEMENT_EDUCATION_MOBILE_MONEY).fillna(0.0)
proba_mobile_money = (base_region + ajustement_education).clip(0.05, 0.97)
df["utilisation_mobile_money"] = np.where(rng.random(len(df)) < proba_mobile_money, "Oui", "Non")

print("Variables regenerees : membre_tontine, activite_saisonniere, utilisation_mobile_money")

# %% [markdown]
# ### 4. Colonnes volontairement absentes de ce livrable
# - `statut_remboursement_label` : doublon exact de la cible `statut_remboursement`
#   (fuite pure). `statut_remboursement` est deja lisible en francais
#   ("Rembourse"/"Defaut"), donc aucune reconstruction n'est necessaire pour
#   l'affichage dans l'app.
# - `zone_geographique` : redondante a 100% avec `region_cameroun` (mapping
#   deterministe 1:1, aucune variance intra-region).
# - `categorie_risque` : ce n'est pas une variable d'entree. Elle sera calculee
#   par l'app a partir du score predit par le modele (>=70 Faible, 55-69
#   Modere, 40-54 Eleve, <40 Tres haut risque), en Semaine 4.

# %% [markdown]
# ### 5. Verification finale

# %%
print("Dimensions finales :", df.shape)

print("\nTaux de defaut par secteur_activite (doit conserver le gradient) :")
print(df.groupby("secteur_activite")["statut_remboursement"].apply(lambda s: (s == "Defaut").mean()).sort_values())

print("\nTaux de defaut par membre_tontine (attendu quasi plat, limite assumee - voir section 3) :")
print(df.groupby("membre_tontine")["statut_remboursement"].apply(lambda s: (s == "Defaut").mean()))

for col in COLONNES_IMMOBILIER:
    assert col not in df.columns, f"{col} n'aurait pas du etre presente"
for col in ["statut_remboursement_label", "zone_geographique", "categorie_risque"]:
    assert col not in df.columns, f"{col} n'aurait pas du etre presente"

print("\nValeurs manquantes restantes (top 10) :")
print(df.isna().sum().sort_values(ascending=False).head(10))

# %% [markdown]
# ### 6. Sauvegarde du livrable Semaine 1

# %%
df.to_csv(OUTPUT_PATH, index=False)
print(f"Livrable sauvegarde : {OUTPUT_PATH}")
