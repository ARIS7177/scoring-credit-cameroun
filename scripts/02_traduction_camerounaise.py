# %% [markdown]
# ## Semaine 1 - Etape 2 : Traduction camerounaise du dataset
# Entree  : ../data/processed/loan_default_clean.csv (sortie du script 01)
# Sortie  : ../data/processed/Loan_Default_Cameroun.csv  (livrable Semaine 1 d'Aristide)
#
# Objectif : renommer les colonnes en francais et adapter les valeurs a un contexte
# camerounais (montants en FCFA, regions du Cameroun, libelles en francais).
# Les variables METIER specifiques (tontine, mobile money, secteur informel, etc.)
# seront ajoutees par Marie-Therese en Semaine 2 : ce script ne fait QUE la traduction
# du dataset original, pas l'enrichissement.

# %%
import sys

import pandas as pd

IN_COLAB = "google.colab" in sys.modules
BASE_DIR = "/content/ProjetScoringCredit" if IN_COLAB else ".."

INPUT_PATH = f"{BASE_DIR}/data/processed/loan_default_clean.csv"
OUTPUT_PATH = f"{BASE_DIR}/data/processed/Loan_Default_Cameroun.csv"

# IMPORTANT : ce n'est PAS un taux de change USD/FCFA. Ce dataset represente des prets
# immobiliers americains ; ses revenus/montants sont structurellement bien superieurs
# a un contexte camerounais. Un taux de change brut (~600) donnerait un revenu mensuel
# median de ~3,45M FCFA, alors que le salaire moyen FORMEL au Cameroun est ~120 000
# FCFA/mois et le SMIG ~60 000 FCFA/mois (sources : africarrieres.com, infospratiques.cm).
#
# FACTEUR_ECHELLE est donc calibre pour que le revenu mensuel median obtenu (5760 x
# FACTEUR_ECHELLE) se rapproche du salaire moyen formel camerounais, plutot que de
# refleter un taux de change reel. Applique de facon UNIFORME a toutes les colonnes
# monetaires, il preserve les ratios internes du dataset (LTV, ratio_endettement,
# pret/revenu), qui restent la variable predictive interessante pour le modele.
# -> A documenter explicitement comme hypothese methodologique dans le rapport.
FACTEUR_ECHELLE = 20

df = pd.read_csv(INPUT_PATH)

# %% [markdown]
# ### 1. Renommage des colonnes (anglais -> francais)

# %%
RENOMMAGE_COLONNES = {
    "ID": "id_client",
    "year": "annee",
    "loan_limit": "plafond_pret",
    "Gender": "genre",
    "approv_in_adv": "approbation_anticipee",
    "loan_type": "type_pret",
    "loan_purpose": "objet_pret",
    "Credit_Worthiness": "solvabilite",
    "open_credit": "credit_ouvert",
    "business_or_commercial": "usage_professionnel",
    "loan_amount": "montant_pret_fcfa",
    "rate_of_interest": "taux_interet",
    "Interest_rate_spread": "ecart_taux_interet",
    "Upfront_charges": "frais_initiaux_fcfa",
    "term": "duree_mois",
    "Neg_ammortization": "amortissement_negatif",
    "interest_only": "interet_seul",
    "lump_sum_payment": "paiement_forfaitaire",
    "property_value": "valeur_bien_fcfa",
    "construction_type": "type_construction",
    "occupancy_type": "type_occupation",
    "Secured_by": "garanti_par",
    "total_units": "nombre_unites",
    "income": "revenu_mensuel_fcfa",
    "credit_type": "type_credit_bureau",
    "Credit_Score": "score_credit_bureau",
    "co-applicant_credit_type": "type_credit_coemprunteur",
    "age": "tranche_age",
    "submission_of_application": "mode_soumission",
    "LTV": "ratio_pret_valeur",
    "Region": "region_cameroun",
    "Security_Type": "type_garantie",
    "Status": "statut_remboursement",
    "dtir1": "ratio_endettement",
}

df = df.rename(columns=RENOMMAGE_COLONNES)

colonnes_traduites = [v for k, v in RENOMMAGE_COLONNES.items() if k in df.columns or v in df.columns]
print(f"{len(colonnes_traduites)} colonnes renommees en francais.")

# %% [markdown]
# ### 2. Mise a l'echelle des montants (contexte camerounais)
# Facteur unique applique uniformement a toutes les colonnes monetaires pour
# preserver leurs ratios internes (LTV, ratio_endettement, pret/revenu).

# %%
for col in ["montant_pret_fcfa", "frais_initiaux_fcfa", "valeur_bien_fcfa", "revenu_mensuel_fcfa"]:
    if col in df.columns:
        df[col] = (df[col] * FACTEUR_ECHELLE).round(0)

# %% [markdown]
# ### 3. Traduction des valeurs categorielles

# %%
MAPPINGS_VALEURS = {
    "plafond_pret": {"cf": "Conforme", "ncf": "Non_conforme"},
    "genre": {
        "Male": "Homme",
        "Female": "Femme",
        "Joint": "Conjoint",
        "Sex Not Available": "Non_specifie",
    },
    "approbation_anticipee": {"pre": "Oui", "nopre": "Non"},
    "type_pret": {"type1": "Type_1", "type2": "Type_2", "type3": "Type_3"},
    "objet_pret": {"p1": "Achat", "p2": "Refinancement", "p3": "Amelioration_habitat", "p4": "Autre"},
    "solvabilite": {"l1": "Standard", "l2": "Sous_standard"},
    "credit_ouvert": {"nopc": "Non", "opc": "Oui"},
    "usage_professionnel": {"nob/c": "Non", "b/c": "Oui"},
    "amortissement_negatif": {"neg_amm": "Oui", "not_neg": "Non"},
    "interet_seul": {"int_only": "Oui", "not_int": "Non"},
    "paiement_forfaitaire": {"lpsm": "Oui", "not_lpsm": "Non"},
    "type_occupation": {"pr": "Residence_principale", "sr": "Residence_secondaire", "ir": "Investissement"},
    "garanti_par": {"home": "Logement", "land": "Terrain"},
    "mode_soumission": {"to_inst": "En_agence", "not_inst": "En_ligne"},
    # NB: le dataset source contient la faute de frappe "Indriect" (pas "Indirect")
    "type_garantie": {"direct": "Directe", "Indriect": "Indirecte"},
    "statut_remboursement": {0: "Rembourse", 1: "Defaut"},
    # Regions US du dataset source -> regions du Cameroun (mapping fixe 1:1 pour la trace)
    "region_cameroun": {
        "south": "Littoral",
        "North": "Centre",
        "central": "Ouest",
        "North-East": "Nord",
    },
}

for col, mapping in MAPPINGS_VALEURS.items():
    if col in df.columns:
        df[col] = df[col].replace(mapping)

# %% [markdown]
# ### 4. Verification finale

# %%
print("Dimensions finales :", df.shape)
print("\nApercu :")
print(df.head(3).T)

nb_colonnes_traduites = sum(1 for c in RENOMMAGE_COLONNES.values() if c in df.columns)
print(f"\nColonnes traduites presentes dans le fichier final : {nb_colonnes_traduites} (objectif >= 15)")

# %% [markdown]
# ### 5. Sauvegarde du livrable Semaine 1

# %%
df.to_csv(OUTPUT_PATH, index=False)
print(f"Livrable sauvegarde : {OUTPUT_PATH}")
