"""
=====================================================================
 SYSTÈME DE SCORING CRÉDIT CAMEROUN — Application Streamlit V2
=====================================================================
Intégration du modèle CatBoost sauvegardé (v1) :
    • Prédiction en temps réel basée sur la section 4 (Activité professionnelle)
    • Score de 0 à 100 avec jauge 3 couleurs (vert ≥ 70, orange 40-70, rouge < 40)
    • Calcul de la catégorie de risque à partir du score
    • Simulation du montant maximum recommandé selon score, revenu et durée

=====================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import joblib
import os
from catboost import Pool

# fpdf2 optionnel pour export PDF
try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    FPDF_DISPONIBLE = True
except ImportError:
    FPDF_DISPONIBLE = False


# =====================================================================
# 0. CHARGEMENT DU MODÈLE ML
# =====================================================================
@st.cache_resource
def load_model():
    """Charge le modèle CatBoost sauvegardé une seule fois."""
    try:
        model_path = "../models/modele_scoring_credit.joblib"
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            st.warning(f"⚠️ Modèle non trouvé à {model_path}")
            return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

MODEL_DATA = load_model()
MODEL = MODEL_DATA['modele'] if MODEL_DATA else None
FEATURES_NAMES = MODEL_DATA['features'] if MODEL_DATA else []


# =====================================================================
# 1. CONFIGURATION GÉNÉRALE DE LA PAGE
# =====================================================================
st.set_page_config(
    page_title="Système de Scoring Crédit Cameroun",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS commun à toute l'application ---
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }

        /* --- Sidebar en vert --- */
        section[data-testid="stSidebar"] {
            background-color: #1f9d55 !important;
        }
        
        /* --- Textes blancs dans le sidebar --- */
        section[data-testid="stSidebar"] {
            color: white !important;
            background-color: #1f9d55 !important;
        }
        section[data-testid="stSidebar"] * {
            color: white !important;
            background-color: #178449 !important;
        }
        
        /* --- Dashboard/Contenu principal --- */
        .main {
            background-color: #ffffff !important;
        }
        .main p, .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
        .main span, .main label, .main div {
            color: #000000 !important;
        }

        /* --- Boutons : thème vert cohérent --- */
        button[data-testid="stBaseButton-secondary"] {
            color: #1f9d55;
            border: 1px solid #1f9d55;
            background-color: #ffffff;
        }
        button[data-testid="stBaseButton-secondary"]:hover {
            color: #ffffff;
            border-color: #178449;
            background-color: #1f9d55;
        }
        button[data-testid="stBaseButton-secondary"]:disabled,
        button[data-testid="stBaseButton-secondary"]:disabled:hover {
            color: #94a3b8;
            border-color: #cbd5e1;
            background-color: #f8fafc;
        }
        button[data-testid="stBaseButton-primary"] {
            background-color: #1f9d55;
            border-color: #1f9d55;
            color: #ffffff;
        }
        button[data-testid="stBaseButton-primary"]:hover {
            background-color: #178449;
            border-color: #178449;
        }
        button[data-testid="stBaseButton-primary"]:disabled,
        button[data-testid="stBaseButton-primary"]:disabled:hover {
            background-color: #cbd5e1;
            border-color: #cbd5e1;
            color: #64748b;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# 2. CONSTANTES MÉTIER
# =====================================================================
OPTIONS_GENRE = ["Masculin", "Féminin"]
OPTIONS_AGE = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
OPTIONS_EDUCATION = ["Sans diplôme", "Primaire", "Secondaire", "Supérieur"]
OPTIONS_LOGEMENT = ["Propriétaire", "Locataire", "Hébergé", "Autre"]
OPTIONS_DUREE = [6, 12, 18, 24, 36, 48, 60]
OPTIONS_OBJET = ["Investissement (activité)", "Achat d'équipement", "Trésorerie", "Autre"]
OPTIONS_SECTEUR = ["Salarié formel", "Fonctionnaire", "Commerçant indépendant",
                    "Agriculteur", "Activité saisonnière", "Autre"]
OPTIONS_GARANT = ["Oui (logement en hypothèque)", "Oui (caution personnelle)", "Non"]

# Mapping objet_pret pour le modèle (one-hot encoding)
OBJET_PRET_MAPPING = {
    "Investissement (activité)": "Investissement_activite",
    "Achat d'équipement": "Achat",
    "Trésorerie": "Autre",  # Pas dans le modèle, utilise "Autre"
    "Autre": "Autre",
    "Refinancement": "Refinancement",  # Au cas où
}

# Mapping secteur pour le modèle (one-hot encoding)
SECTEUR_MAPPING = {
    "Salarié formel": "Salarié formel",
    "Fonctionnaire": "Salarié formel",  # Traité comme salarié
    "Commerçant indépendant": "Petit commerce",
    "Agriculteur": "Agriculture",
    "Activité saisonnière": "Petit commerce",  # Traité comme petit commerce
    "Autre": "Commerce/Négoce",
}

# Profils d'exemple
EXEMPLES = {
    "favorable": {
        "nom": "MBARGA", "prenom": "Jean", "adresse": "Quartier Bastos, Yaoundé",
        "genre": "Masculin", "age": "35-44", "education": "Supérieur",
        "revenu": 250000, "charges": 80000, "ligne_credit": "Non", "usage_credit": "Professionnel",
        "personnes_charge": 3, "logement": "Propriétaire", "anciennete": 36,
        "montant_demande": 2000000, "duree": 24, "objet": "Investissement (activité)",
        "secteur": "Salarié formel", "activite_saisonniere": "Non",
        "mobile_money": "Oui",
        "garant": "Oui (logement en hypothèque)",
    },
    "moyen": {
        "nom": "NGONO", "prenom": "Marie", "adresse": "Quartier Akwa, Douala",
        "genre": "Féminin", "age": "25-34", "education": "Secondaire",
        "revenu": 150000, "charges": 50000, "ligne_credit": "Oui", "usage_credit": "Personnel",
        "personnes_charge": 2, "logement": "Locataire", "anciennete": 26,
        "montant_demande": 800000, "duree": 18, "objet": "Trésorerie",
        "secteur": "Commerçant indépendant", "activite_saisonniere": "Non",
        "mobile_money": "Oui",
        "garant": "Non",
    },
    "risque": {
        "nom": "ABDOULAYE", "prenom": "Oumar", "adresse": "Quartier Domayo, Maroua",
        "genre": "Masculin", "age": "55-64", "education": "Primaire",
        "revenu": 65000, "charges": 35000, "ligne_credit": "Oui", "usage_credit": "Personnel",
        "personnes_charge": 5, "logement": "Hébergé", "anciennete": 8,
        "montant_demande": 600000, "duree": 12, "objet": "Autre",
        "secteur": "Activité saisonnière", "activite_saisonniere": "Oui",
        "mobile_money": "Non",
        "garant": "Non",
    },
}

VALEURS_PAR_DEFAUT_FORMULAIRE = {
    "revenu": 250000, "charges": 80000, "personnes_charge": 3, "anciennete": 36,
    "montant_demande": 2000000, "duree": 24, "objet": OPTIONS_OBJET[0],
    "activite_saisonniere": "Non", "mobile_money": "Oui",
}


# =====================================================================
# 3. ÉTAT DE SESSION
# =====================================================================
def init_session_state():
    """Initialise les variables de session."""
    valeurs_par_defaut = {
        "page": "connexion",
        "authenticated": False,
        "agent_nom": "KOM Olivier",
        "institution": "Microfinance XYZ",
        "demande_id_counter": 13,
        "demande_data": {},
        "dernier_score_model": None,
        "dernier_score_categ": None,
        "dernier_montant_recommande": None,
        "dernier_facteurs_model": None,
    }
    for cle, valeur in valeurs_par_defaut.items():
        if cle not in st.session_state:
            st.session_state[cle] = valeur

init_session_state()


# =====================================================================
# 4. FONCTIONS UTILITAIRES
# =====================================================================
def go_to(nom_page):
    """Change la page active."""
    st.session_state.page = nom_page
    st.rerun()


def format_fcfa(montant):
    """Formate un montant avec séparateur de milliers + FCFA."""
    return f"{montant:,.0f}".replace(",", " ") + " FCFA"


def calc_ratio_endettement(revenu, charges):
    """Calcule le ratio d'endettement (%)."""
    if not revenu:
        return 0.0
    return (charges / revenu) * 100


def calculer_mensualite(montant, taux_annuel_pct, duree_mois):
    """Calcule une mensualité par amortissement classique."""
    if montant <= 0 or duree_mois <= 0:
        return 0
    taux_mensuel = (taux_annuel_pct / 100) / 12
    if taux_mensuel == 0:
        return montant / duree_mois
    facteur = (1 + taux_mensuel) ** duree_mois
    return montant * (taux_mensuel * facteur) / (facteur - 1)


def charger_exemple(nom_profil):
    """Pré-remplit le formulaire avec un profil d'exemple."""
    for champ, valeur in EXEMPLES[nom_profil].items():
        st.session_state[f"f_{champ}"] = valeur
    st.rerun()


def init_formulaire_defaults():
    """Pré-initialise les champs par défaut une seule fois."""
    for champ, valeur in VALEURS_PAR_DEFAUT_FORMULAIRE.items():
        st.session_state.setdefault(f"f_{champ}", valeur)


def reinitialiser_formulaire():
    """Vide tous les champs du formulaire."""
    for cle in list(st.session_state.keys()):
        if cle.startswith("f_"):
            del st.session_state[cle]
    st.rerun()


def construire_features_pour_modele(data):
    """
    Construit le vecteur de features pour le modèle ML à partir des données du formulaire.
    Retourne un array numpy prêt pour prediction.
    """
    features_array = np.zeros(len(FEATURES_NAMES))
    feature_dict = {}
    
    # --- Features numériques et booléennes ---
    feature_dict['credit_ouvert'] = 1 if data.get('ligne_credit') == 'Oui' else 0
    feature_dict['usage_professionnel'] = 1 if data.get('usage_credit') == 'Professionnel' else 0
    feature_dict['montant_pret_fcfa'] = data.get('montant_demande', 0)
    feature_dict['duree_mois'] = data.get('duree', 12)
    feature_dict['revenu_mensuel_fcfa'] = data.get('revenu', 0)
    feature_dict['ratio_endettement'] = data.get('ratio_endettement', 0)
    
    # --- One-hot encoding objet_pret ---
    objet_raw = data.get('objet', 'Autre')
    objet_mapped = OBJET_PRET_MAPPING.get(objet_raw, 'Autre')
    for feat_name in FEATURES_NAMES:
        if feat_name.startswith('objet_pret_'):
            objet_val = feat_name.replace('objet_pret_', '')
            feature_dict[feat_name] = 1 if objet_val == objet_mapped else 0
    
    # --- One-hot encoding secteur_activite ---
    secteur_raw = data.get('secteur', 'Autre')
    secteur_mapped = SECTEUR_MAPPING.get(secteur_raw, 'Commerce/Négoce')
    for feat_name in FEATURES_NAMES:
        if feat_name.startswith('secteur_activite_'):
            secteur_val = feat_name.replace('secteur_activite_', '').replace('_', ' ')
            feature_dict[feat_name] = 1 if secteur_val == secteur_mapped else 0
    
    # --- Remplir l'array dans l'ordre des FEATURES_NAMES ---
    for i, feat_name in enumerate(FEATURES_NAMES):
        if feat_name in feature_dict:
            features_array[i] = feature_dict[feat_name]

    return features_array.reshape(1, -1)


def calculer_facteurs_shap(features, data):
    """
    Calcule les facteurs explicatifs de la prédiction ML à partir des valeurs
    SHAP natives de CatBoost (get_feature_importance, type="ShapValues").

    Les valeurs SHAP sont dans l'espace du score brut du modèle (log-odds,
    avant sigmoïde) : leur somme + le biais reconstruit exactement la marge
    utilisée par predict_proba. Pour les exprimer en "points" sur le score
    0-100 affiché à l'agent (même unité que evaluer_demande_heuristique),
    on les multiplie par la dérivée locale de la sigmoïde au point de
    prédiction (linéarisation). Cette conversion préserve exactement le
    classement des facteurs par importance ; seule l'amplitude affichée se
    comprime pour les dossiers très tranchés (score proche de 0 ou 100),
    ce qui est attendu : à ce niveau de confiance, aucun facteur pris
    isolément ne suffirait à faire basculer la décision.

    Retourne la même structure que evaluer_demande_heuristique : liste de
    (nom, valeur, impact_points, explication), triée par impact absolu,
    limitée aux 5 facteurs les plus déterminants.
    """
    try:
        pool = Pool(features, feature_names=FEATURES_NAMES)
        shap_row = MODEL.get_feature_importance(pool, type="ShapValues")[0]
        shap_par_feature = dict(zip(FEATURES_NAMES, shap_row[:-1]))

        marge = shap_row.sum()
        proba_defaut = 1 / (1 + np.exp(-marge))
        points_par_unite_marge = -100 * proba_defaut * (1 - proba_defaut)

        def impact(*noms_features):
            return sum(shap_par_feature[n] for n in noms_features) * points_par_unite_marge

        facteurs = []

        imp = impact("ratio_endettement")
        facteurs.append((
            "Ratio d'endettement", f"{data['ratio_endettement']:.0f} %", round(imp),
            "Bonne capacité résiduelle." if imp >= 0 else "Charges élevées par rapport au revenu.",
        ))

        imp = impact("revenu_mensuel_fcfa")
        facteurs.append((
            "Revenu mensuel", format_fcfa(data["revenu"]), round(imp),
            "Revenu qui rassure sur la capacité de remboursement." if imp >= 0
            else "Revenu qui pèse sur la capacité de remboursement.",
        ))

        imp = impact("montant_pret_fcfa")
        facteurs.append((
            "Montant du prêt", format_fcfa(data["montant_demande"]), round(imp),
            "Montant raisonnable au vu du profil." if imp >= 0 else "Montant élevé au vu du profil.",
        ))

        imp = impact("duree_mois")
        facteurs.append((
            "Durée du prêt", f"{data['duree']} mois", round(imp),
            "Durée qui limite l'exposition au risque." if imp >= 0
            else "Durée longue qui augmente l'exposition au risque.",
        ))

        imp = impact("credit_ouvert")
        facteurs.append((
            "Ligne de crédit ouverte", data["ligne_credit"], round(imp),
            "Aucune autre ligne de crédit en cours." if imp >= 0
            else "Une autre ligne de crédit déjà ouverte augmente le risque.",
        ))

        imp = impact("usage_professionnel")
        facteurs.append((
            "Usage du crédit", data["usage_credit"], round(imp),
            "Crédit professionnel, susceptible de générer du revenu." if imp >= 0
            else "Crédit personnel, sans revenu généré directement.",
        ))

        noms_objet = [f for f in FEATURES_NAMES if f.startswith("objet_pret_")]
        imp = impact(*noms_objet)
        facteurs.append((
            "Objet du prêt", data["objet"], round(imp),
            "Cet usage du crédit est statistiquement plus sûr." if imp >= 0
            else "Cet usage du crédit est statistiquement plus risqué.",
        ))

        noms_secteur = [f for f in FEATURES_NAMES if f.startswith("secteur_activite_")]
        imp = impact(*noms_secteur)
        facteurs.append((
            "Secteur d'activité", data["secteur"], round(imp),
            "Secteur au profil de risque favorable." if imp >= 0
            else "Secteur au profil de risque plus élevé.",
        ))

        return sorted(facteurs, key=lambda f: abs(f[2]), reverse=True)[:5]

    except Exception:
        return []


def predire_score_ml(data):
    """
    Effectue une prédiction avec le modèle ML.
    Retourne : (score_0_100, categorie_risque, couleur, proba_defaut, facteurs)
    """
    if MODEL is None:
        return None, None, None, None, []

    try:
        features = construire_features_pour_modele(data)

        # Prédiction : retourne [proba_0, proba_1]
        proba = MODEL.predict_proba(features)[0]
        proba_defaut = proba[1]  # Probabilité de défaut (classe 1)

        # Convertir en score 0-100 : score = (1 - proba_defaut) * 100
        # Plus la probabilité de défaut est basse, plus le score est haut
        score_0_100 = round((1 - proba_defaut) * 100)
        score_0_100 = max(0, min(100, score_0_100))  # Clamp à [0, 100]

        # Catégorie et couleur
        if score_0_100 >= 70:
            categorie = "FAIBLE"
            couleur = "#16a34a"  # Vert
        elif score_0_100 >= 40:
            categorie = "MODÉRÉ"
            couleur = "#d97706"  # Orange
        else:
            categorie = "ÉLEVÉ"
            couleur = "#dc2626"  # Rouge

        facteurs = calculer_facteurs_shap(features, data)

        return score_0_100, categorie, couleur, proba_defaut * 100, facteurs

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None, None, None, None, []


def recommander_montant_maximum(score, revenu, duree_mois):
    """
    Recommande un montant maximum de prêt selon le score, le revenu et la durée.
    Formule : montant_max = revenu * ratio_capacite * duree_mois / 12
    où ratio_capacite dépend du score.
    """
    if score is None or revenu <= 0:
        return 0
    
    # Ratio de capacité de remboursement selon le score
    if score >= 70:
        ratio = 12  # Peut emprunter jusqu'à 12 mois de revenu
    elif score >= 55:
        ratio = 9
    elif score >= 40:
        ratio = 6
    else:
        ratio = 3
    
    # Ajustement selon la durée (pour un crédit plus court, moins de montant)
    duree_factor = min(duree_mois / 12, 1.5)  # Max 1.5x pour durée longue
    
    montant_max = revenu * ratio * duree_factor
    return int(montant_max / 10000) * 10000  # Arrondir à 10k FCFA


def evaluer_demande_heuristique(data):
    """
    Fallback heuristique simple (au cas où le modèle ne serait pas disponible).
    """
    facteurs = []
    score = 40
    
    secteur = data.get("secteur", "Autre")
    poids_secteur = {
        "Salarié formel": 12, "Fonctionnaire": 12, "Commerçant indépendant": 4,
        "Agriculteur": -2, "Activité saisonnière": -10, "Autre": 0,
    }.get(secteur, 0)
    score += poids_secteur
    facteurs.append((
        "Secteur d'activité", secteur, poids_secteur,
        "Les revenus de ce secteur sont stables." if poids_secteur > 0 else "Les revenus irréguliers augmentent le risque.",
    ))
    
    revenu = data.get("revenu", 0)
    if revenu >= 500000:
        poids_revenu = 10
    elif revenu >= 120000:
        poids_revenu = 6
    elif revenu >= 60000:
        poids_revenu = 0
    else:
        poids_revenu = -8
    score += poids_revenu
    facteurs.append((
        "Revenu mensuel", format_fcfa(revenu), poids_revenu,
        "Revenu supérieur à la médiane nationale." if poids_revenu > 0 else "Revenu faible.",
    ))
    
    ratio = data.get("ratio_endettement", 0)
    if ratio < 20:
        poids_ratio = 12
    elif ratio < 35:
        poids_ratio = 6
    elif ratio < 50:
        poids_ratio = -6
    else:
        poids_ratio = -14
    score += poids_ratio
    facteurs.append((
        "Ratio d'endettement", f"{ratio:.0f} %", poids_ratio,
        "Bonne capacité résiduelle." if poids_ratio > 0 else "Charges élevées.",
    ))
    
    montant = data.get("montant_demande", 0)
    if montant > 100_000_000:
        poids_montant = -8
    elif montant > 20_000_000:
        poids_montant = -3
    else:
        poids_montant = 2
    score += poids_montant
    facteurs.append((
        "Montant du prêt", format_fcfa(montant), poids_montant,
        "Montant élevé." if poids_montant < 0 else "Montant raisonnable.",
    ))
    
    usage = data.get("usage_credit", "Personnel")
    poids_usage = 4 if usage == "Professionnel" else -2
    score += poids_usage
    facteurs.append((
        "Usage du crédit", usage, poids_usage,
        "Crédit professionnel génère du revenu." if usage == "Professionnel" else "Crédit personnel.",
    ))
    
    garant = data.get("garant", "Non")
    poids_garant = 5 if garant != "Non" else -3
    score += poids_garant
    facteurs.append((
        "Garant / caution", garant, poids_garant,
        "Garantie présente." if poids_garant > 0 else "Pas de garantie.",
    ))
    
    anciennete = data.get("anciennete", 0)
    if anciennete >= 24:
        poids_anciennete = 4
    elif anciennete >= 12:
        poids_anciennete = 0
    else:
        poids_anciennete = -4
    score += poids_anciennete
    facteurs.append((
        "Ancienneté", f"{anciennete} mois", poids_anciennete,
        "Activité stable depuis longtemps." if poids_anciennete > 0 else "Activité récente.",
    ))
    
    score = max(0, min(100, round(score)))
    
    if score >= 70:
        categorie, couleur = "FAIBLE", "#16a34a"
    elif score >= 55:
        categorie, couleur = "MODÉRÉ", "#d97706"
    elif score >= 40:
        categorie, couleur = "ÉLEVÉ", "#dc2626"
    else:
        categorie, couleur = "TRÈS HAUT", "#1f2937"
    
    if score >= 65:
        decision = "ACCORDÉ"
    elif score >= 45:
        decision = "ÉTUDE APPROFONDIE"
    else:
        decision = "REFUSÉ"
    
    proba_defaut = max(2, min(96, 98 - score))
    
    facteurs_tries = sorted(facteurs, key=lambda f: abs(f[2]), reverse=True)[:5]
    
    return {
        "score": score, "categorie": categorie, "couleur": couleur,
        "decision": decision, "proba_defaut": proba_defaut,
        "facteurs": facteurs_tries,
    }


def get_historique_demandes():
    """Génère un historique de démonstration."""
    data = [
        {"id": "#20260815-0020", "date": "2026-08-15", "profil": "Salarié formel", "age": "35-44 ans", "montant": 2000000, "statut": "Accordé", "score": 72},
        {"id": "#20260815-0019", "date": "2026-08-15", "profil": "Salarié formel", "age": "35-44 ans", "montant": 3000000, "statut": "Accordé", "score": 72},
        {"id": "#20260815-0018", "date": "2026-08-15", "profil": "Commerçant indépendant", "age": "25-34 ans", "montant": 1500000, "statut": "Étude approfondie", "score": 58},
        {"id": "#20260815-0017", "date": "2026-08-15", "profil": "Activité saisonnière", "age": "45-54 ans", "montant": 800000, "statut": "Refusé", "score": 37},
        {"id": "#20260814-0016", "date": "2026-08-14", "profil": "Fonctionnaire", "age": "45-54 ans", "montant": 5000000, "statut": "Accordé", "score": 81},
    ]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generer_pdf(data, resultat, montant_accorde, taux, mensualite, score_model=None):
    """Génère le rapport PDF."""
    if not FPDF_DISPONIBLE:
        return None
    
    def texte(s):
        return str(s).encode("latin-1", "replace").decode("latin-1")
    
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    
    def titre_section(t):
        pdf.ln(1.5)
        pdf.set_font("Helvetica", "B", 11.5)
        pdf.set_text_color(27, 42, 74)
        pdf.cell(0, 7, texte(t), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(2.5)
    
    def deux_colonnes(label1, val1, label2, val2):
        pdf.cell(90, 6, texte(f"{label1} : {val1}"))
        pdf.cell(90, 6, texte(f"{label2} : {val2}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, texte("RAPPORT D'ANALYSE - DEMANDE DE PRET"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, texte(f"{st.session_state.institution} - Systeme de Scoring Credit"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    pdf.set_draw_color(27, 42, 74)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)
    
    titre_section("INFORMATIONS DE LA DEMANDE")
    deux_colonnes("ID demande", data["id"], "Date d'analyse", datetime.now().strftime("%d/%m/%Y %Hh%M"))
    deux_colonnes("Agent", st.session_state.agent_nom, "Institution", st.session_state.institution)
    deux_colonnes("Score", f"{score_model}/100" if score_model else "N/A", "Risque estimé", resultat["categorie"])
    pdf.ln(3)
    
    titre_section("PROFIL DU DEMANDEUR")
    deux_colonnes("M/Mme", f"{data['prenom']} {data['nom']}", "Adresse", data["adresse"])
    deux_colonnes("Genre", data["genre"], "Tranche d'age", f"{data['age']} ans")
    deux_colonnes("Niveau d'éducation", data["education"], "Secteur d'activité", data["secteur"])
    pdf.ln(3)
    
    titre_section("DEMANDE DE CREDIT")
    deux_colonnes("Montant demandé", format_fcfa(data["montant_demande"]), "Montant accordé", format_fcfa(montant_accorde))
    deux_colonnes("Objet du prêt", f"{data['objet']} mois", "Durée", f"{data['duree']} mois")
    pdf.ln(3)
    
    titre_section("ANALYSE DU RISQUE (MODELE ML)")
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, texte(f"Score ML : {score_model}/100  |  Risque : {resultat['categorie']}  |  {resultat['decision']}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    deux_colonnes("Probabilité de défaut", f"{resultat['proba_defaut']:.1f} %",
                  "Ratio d'endettement", f"{data['ratio_endettement']:.0f} %")
    pdf.ln(3)

    if resultat.get("facteurs"):
        titre_section("FACTEURS EXPLICATIFS DU SCORE")
        for nom, valeur, impact, explication in resultat["facteurs"]:
            sens = "reduit" if impact >= 0 else "augmente"
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.cell(0, 5.5, texte(f"{nom} : {valeur}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 4.3, texte(
                f"   -> {sens} le score de {abs(impact)} pt(s) sur 100 - {explication}"
            ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1.5)

    titre_section("AVERTISSEMENT LEGAL")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.multi_cell(0, 4.3, texte(
        "Cet outil est un support a la decision uniquement. La decision finale reste du "
        "ressort du comite de credit de l'institution. Tous les facteurs contextuels et humains doivent "
        "etre pris en compte dans la deliberation finale."
    ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "", 10)
    deux_colonnes("Signature agent", "______________", "Tampon institution", "______________")
    pdf.ln(3)
    pdf.cell(0, 6, texte(f"Date : {datetime.now().strftime('%d/%m/%Y')}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, texte("Généré par Système de Scoring Credit Cameroun"), align="C")
    
    return bytes(pdf.output())


# =====================================================================
# 5. COMPOSANTS D'INTERFACE
# =====================================================================
def render_sidebar():
    """Menu latéral."""
    with st.sidebar:
        st.markdown("## Évaluer intelligemment le risque de crédit ")
        
        st.divider()
        pages_menu = [
            ("Tableau de bord", "tableau_de_bord"),
            ("Nouvelle demande", "nouvelle_demande"),
            ("Historique", "historique"),
            ("Paramètres", "parametres"),
        ]
        for label, cle_page in pages_menu:
            type_bouton = "primary" if st.session_state.page == cle_page else "secondary"
            if st.button(label, width="content", key=f"nav_{cle_page}", type=type_bouton):
                go_to(cle_page)
        
        st.divider()
        if st.button("Déconnexion", width="stretch", key="nav_deconnexion"):
            st.session_state.authenticated = False
            go_to("connexion")
        
        st.divider()
        st.caption(f"**Agent :** {st.session_state.agent_nom}")
        st.caption(st.session_state.institution)
        
        # Status du modèle
        st.divider()
        if MODEL:
            st.success("Modèle ML chargé")
        else:
            st.error("❌ Modèle ML non disponible")


def render_entete(sous_titre="Évaluation du risque de defaut de crédit"):
    """Bandeau d'en-tête."""
    st.markdown(
        f"""
        <div style="background-color:#16233f; padding:14px 22px; border-radius:8px; margin-bottom:20px;">
            <span style="color:white; font-size:1.25em; font-weight:700;">
                SYSTÈME DE SCORING CRÉDIT CAMEROUN
            </span><br>
            <span style="color:#c9d3e3; font-size:0.85em;">{sous_titre}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_jauge_score(score, couleur):
    """Jauge circulaire Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": couleur, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#dcfce7"},
            ],
        },
    ))
    fig.update_layout(height=230, margin=dict(l=15, r=15, t=35, b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_graphique_facteurs(facteurs):
    """Diagramme en barres des facteurs."""
    noms = [f[0] for f in facteurs]
    impacts = [f[2] for f in facteurs]
    couleurs = ["#16a34a" if v >= 0 else "#dc2626" for v in impacts]
    textes = [f"−{v}pp" if v >= 0 else f"+{abs(v)}pp" for v in impacts]

    fig = go.Figure(go.Bar(
        x=impacts, y=noms, orientation="h",
        marker_color=couleurs, text=textes, textposition="outside",
    ))
    fig.update_layout(
        title="Impact sur le risque (points de pourcentage)",
        height=260, margin=dict(l=10, r=30, t=40, b=10),
        xaxis_title=None, yaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# =====================================================================
# 6. PAGE 1 — CONNEXION
# =====================================================================
def page_connexion():
    """Écran de connexion."""
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(160deg, #0f2b21 0%, #14243f 100%); }
        [data-testid="collapsedControl"] { display: none; }
        section[data-testid="stSidebar"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    _, col_centre, _ = st.columns([1, 1.8, 1])
    with col_centre:
        st.markdown(
            """
            <div style='text-align:center; margin-top:20px;'>
                <h2 style='color:white; margin-bottom:0;'>SYSTÈME DE SCORING CRÉDIT</h2>
                <span style='color:#c9d3e3;'>Cameroun — Modèle Catboost intégré</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        
        with st.container(border=True):
            st.markdown(
                "<h3 style='text-align: center;'>Connexion agent</h3>",
                unsafe_allow_html=True
            )
            identifiant = st.text_input("Email ou Nom d'utilisateur", placeholder="exemple@imf.cm")
            mot_de_passe = st.text_input("Mot de passe", type="password")
            st.checkbox("Rester connecté")
            
            if st.button("CONNEXION", width="stretch", type="primary"):
                if identifiant.strip() and mot_de_passe.strip():
                    st.session_state.authenticated = True
                    st.session_state.login_erreur = False
                    go_to("tableau_de_bord")
                else:
                    st.session_state.login_erreur = True
            
            if st.session_state.get("login_erreur"):
                st.error("Veuillez renseigner votre identifiant et votre mot de passe.")
            
            st.markdown(
                "<p style='text-align:center; color:#64748b; font-size:0.9em;'>"
                "Mot de passe oublié? · Aide</p>",
                unsafe_allow_html=True,
            )
        
        st.markdown(
            "<p style='text-align:center; color:#c9d3e3; font-size:0.85em; margin-top:14px;'>"
            "Connexion sécurisée</p>",
            unsafe_allow_html=True,
        )


# =====================================================================
# 7. PAGE 2 — TABLEAU DE BORD
# =====================================================================
def page_tableau_de_bord():
   
    """Tableau de bord principal."""
    render_sidebar()
    render_entete()
    
    col_titre, col_bouton = st.columns([3, 1])
    with col_titre:
        prenom = st.session_state.agent_nom.split()[0] if st.session_state.agent_nom else "Agent"
        st.title(f"Bienvenue, Agent {prenom} 👋")
        st.caption(datetime.now().strftime("%A %d %B %Y — %Hh%M"))
    with col_bouton:
        st.write("")
        st.write("")
        if st.button("Nouvelle demande", type="primary", width="stretch"):
            go_to("nouvelle_demande")
    
    st.subheader("Aujourd'hui")
    df = get_historique_demandes()
    jour_recent = df["date"].max()
    du_jour = df[df["date"] == jour_recent]
    
    nb_total = len(du_jour)
    nb_accordees = int((du_jour["statut"] == "Accordé").sum())
    nb_refusees = int((du_jour["statut"] == "Refusé").sum())
    nb_etude = int((du_jour["statut"] == "Étude approfondie").sum())
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.container(border=True):
            st.metric("Demandes du jour", nb_total, f"{nb_total} demandes analysées" if nb_total else "Aucune demande")
    with c2:
        with st.container(border=True):
            st.metric("Accordées", nb_accordees, f"{nb_accordees / nb_total * 100:.0f}% des demandes" if nb_total else None)
    with c3:
        with st.container(border=True):
            st.metric("Refusées", nb_refusees, f"{nb_refusees / nb_total * 100:.0f}% des demandes" if nb_total else None)
    with c4:
        with st.container(border=True):
            st.metric("En étude", nb_etude, f"{nb_etude / nb_total * 100:.0f}% des demandes" if nb_total else None)
    
    st.write("")
    col_gauche, col_droite = st.columns([2, 1])
    
    with col_gauche:
        with st.container(border=True):
            st.subheader("Action rapide")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("NOUVELLE DEMANDE", width="stretch", type="primary", key="qa_nouvelle"):
                    go_to("nouvelle_demande")
            with b2:
                if st.button("DEMANDES RÉCENTES", width="stretch", key="qa_historique"):
                    go_to("historique")
            
            st.write("**Demandes récentes**")
            recentes = df.sort_values("date", ascending=False).head(3)[["id", "profil", "age", "statut", "score"]]
            st.dataframe(
                recentes,
                column_config={
                    "id": "ID", "profil": "Profil", "age": "Âge", "statut": "Statut",
                    "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d/100"),
                },
                hide_index=True, use_container_width=True,
            )
    
    with col_droite:
        with st.container(border=True):
            st.subheader("Dernière demande analysée")
            derniere = df.sort_values("date", ascending=False).iloc[0]
            st.metric(f"ID {derniere['id']}", f"{derniere['score']}/100")
            emoji = "✅" if derniere["statut"] == "Accordé" else ("⏳" if derniere["statut"] == "Étude approfondie" else "❌")
            st.write(f"{emoji} **Décision : {derniere['statut'].upper()}**")
            if st.button("Voir le détail →", width="stretch"):
                go_to("historique")


# =====================================================================
# 8. PAGE 3 — NOUVELLE DEMANDE DE PRÊT
# =====================================================================
def page_nouvelle_demande():
    """Formulaire de nouvelle demande avec prédiction ML en temps réel."""
    render_sidebar()
    render_entete()
    
    st.title("Nouvelle demande de prêt")
    nouvel_id = f"#{datetime.now().strftime('%Y%m%d')}-{st.session_state.demande_id_counter:04d}"
    st.caption(f"ID demande : {nouvel_id} · Statut : Saisie en cours · Mode : ML Prédictif")
    
    init_formulaire_defaults()
    
    # --- Chargement rapide d'un exemple ---
    with st.expander("🎯 Charger un exemple pour tester le formulaire"):
        e1, e2, e3 = st.columns(3)
        with e1:
            if st.button("😀 Profil favorable", width="stretch"):
                charger_exemple("favorable")
        with e2:
            if st.button("😐 Profil moyen", width="stretch"):
                charger_exemple("moyen")
        with e3:
            if st.button("⚠️ Profil à risque", width="stretch"):
                charger_exemple("risque")
    
    # --- SECTION 1 : IDENTITÉ ---
    with st.expander("1. IDENTITÉ & PROFIL DEMANDEUR ", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            nom = st.text_input("Nom *", placeholder="Ex : MBARGA", key="f_nom")
        with c2:
            prenom = st.text_input("Prénom *", placeholder="Ex : Jean", key="f_prenom")
        adresse = st.text_input("Adresse *", placeholder="Quartier, ville", key="f_adresse")
        st.caption("Identité complète du demandeur")
        
        c3, c4 = st.columns(2)
        with c3:
            genre = st.selectbox("Genre du demandeur *", OPTIONS_GENRE,
                                  index=None, placeholder="Sélectionner...", key="f_genre")
        with c4:
            age = st.selectbox("Tranche d'âge *", OPTIONS_AGE,
                                index=None, placeholder="Sélectionner...", key="f_age")
        education = st.radio("Niveau d'éducation *", OPTIONS_EDUCATION,
                              index=None, horizontal=True, key="f_education")
    
    # --- SECTION 2 : CAPACITÉ FINANCIÈRE ---
    with st.expander("2. CAPACITÉ FINANCIÈRE ", expanded=True):
        revenu = st.number_input("Revenu mensuel déclaré (FCFA) *", min_value=15000, max_value=50000000,
                                  step=5000, key="f_revenu")
        
        charges = st.number_input("Charges mensuelles déclarées (FCFA)", min_value=0, max_value=20000000,
                                   step=5000, key="f_charges")
        st.caption("Min : 0 — Max : 20 000 000 FCFA")
        
        ratio = calc_ratio_endettement(revenu, charges)
        st.write(f" **Ratio d'endettement (calculé) : {ratio:.0f} %**")
        st.progress(min(ratio / 100, 1.0))
        if ratio < 35:
            st.success("Bon (< 35 %)")
        elif ratio < 50:
            st.warning("Modéré (35–50 %)")
        else:
            st.error("Élevé (≥ 50 %)")
        
        ligne_credit = st.radio("A-t-il une ligne de crédit ouverte ? *", ["Oui", "Non"],
                                 index=None, horizontal=True, key="f_ligne_credit")
        usage_credit = st.radio("Quelle est l'utilisation du crédit ? *", ["Professionnel", "Personnel"],
                                 index=None, horizontal=True, key="f_usage_credit")
        personnes_charge = st.number_input("Nombre de personnes à charge", min_value=0, max_value=20,
                                            step=1, key="f_personnes_charge")
        logement = st.radio("Situation de logement", OPTIONS_LOGEMENT,
                             index=None, horizontal=True, key="f_logement")
    
    # --- SECTION 3 : DEMANDE DE CRÉDIT ---
    with st.expander("3. DEMANDE DE CRÉDIT", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            montant_demande = st.number_input("Montant demandé (FCFA) *", min_value=50000, max_value=500000000,
                                                step=50000, key="f_montant_demande")
            duree = st.selectbox("Durée souhaitée (mois)", OPTIONS_DUREE, key="f_duree")
        with c2:
            objet = st.selectbox("Objet du prêt", OPTIONS_OBJET, key="f_objet")
    
    # --- SECTION 4 : ACTIVITÉ PROFESSIONNELLE (+ PRÉDICTION ML EN TEMPS RÉEL) ---
    with st.expander("4. ACTIVITÉ PROFESSIONNELLE — PRÉDICTION ML EN TEMPS RÉEL", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            secteur = st.selectbox("Secteur d'activité *", OPTIONS_SECTEUR,
                                        index=None, placeholder="Sélectionner...", key="f_secteur")
        with c2:
            anciennete = st.number_input("Ancienneté dans l'activité (mois)", min_value=0, max_value=600,
                                        step=1, key="f_anciennete")
        
        c3, c4, c5 = st.columns(3)
        with c3:
            activite_saisonniere = st.radio("Activité saisonnière ?", ["Oui", "Non"],
                                        horizontal=True, key="f_activite_saisonniere")
        with c4:
            mobile_money = st.radio("Utilise Mobile Money ?", ["Oui", "Non"],
                                        horizontal=True, key="f_mobile_money")
        with c5:
            membre_tontine = st.radio("Membre de tontine ?", ["Oui", "Non"],
                                        horizontal=True, key="f_membre_tontine")
        
        # --- PRÉDICTION ML EN TEMPS RÉEL ---
        st.divider()
        st.markdown("### 🤖 Prédiction ML (temps réel)")
        
        # Vérifier si on peut faire une prédiction
        champs_pour_ml = {
            "revenu": revenu,
            "charges": charges,
            "ligne_credit": ligne_credit,
            "usage_credit": usage_credit,
            "montant_demande": montant_demande,
            "duree": duree,
            "objet": objet,
            "secteur": secteur,
            "ratio_endettement": ratio,
        }
        
        champs_ml_manquants = [k for k, v in champs_pour_ml.items() if v is None or v == ""]
        
        if champs_ml_manquants:
            st.info(f"⏳ Complétez les champs obligatoires pour activer la prédiction ML : {', '.join(champs_ml_manquants)}")
        else:
            # Effectuer la prédiction ML
            data_ml = {
                "revenu": revenu,
                "charges": charges,
                "ligne_credit": ligne_credit,
                "usage_credit": usage_credit,
                "montant_demande": montant_demande,
                "duree": duree,
                "objet": objet,
                "secteur": secteur,
                "ratio_endettement": ratio,
            }

            score_model, categorie_model, couleur_model, proba_defaut_model, facteurs_model = predire_score_ml(data_ml)

            if score_model is not None:
                # Afficher les résultats ML
                col_score, col_info = st.columns([1, 1.5])

                with col_score:
                    st.markdown("**Score ML**")
                    render_jauge_score(score_model, couleur_model)

                with col_info:
                    st.markdown("**Résultat du modèle**")
                    st.metric("Score", f"{score_model} / 100")
                    st.metric("Catégorie de risque", categorie_model)
                    st.metric("Prob. défaut estimée", f"{proba_defaut_model:.1f} %")

                    # Décision basée sur le score
                    if score_model >= 65:
                        decision_ml = "✅ ACCORDÉ"
                        decision_color = "green"
                    elif score_model >= 45:
                        decision_ml = "⏳ ÉTUDE APPROFONDIE"
                        decision_color = "orange"
                    else:
                        decision_ml = "❌ REFUSÉ"
                        decision_color = "red"

                    st.markdown(f"<p style='color:{decision_color}; font-weight:bold;'>{decision_ml}</p>", unsafe_allow_html=True)

                # Montant recommandé
                st.divider()
                montant_recommande = recommander_montant_maximum(score_model, revenu, duree)

                st.markdown("### 💰 Montant maximum recommandé (selon le score ML)")
                col_montant_1, col_montant_2, col_montant_3 = st.columns(3)
                with col_montant_1:
                    st.metric("Montant demandé", format_fcfa(montant_demande))
                with col_montant_2:
                    st.metric("Montant recommandé", format_fcfa(montant_recommande))
                with col_montant_3:
                    ratio_accord = (montant_recommande / montant_demande * 100) if montant_demande > 0 else 0
                    st.metric("% du montant demandé", f"{ratio_accord:.0f}%")

                # Stockage pour la page résultats
                st.session_state.dernier_score_model = score_model
                st.session_state.dernier_score_categ = categorie_model
                st.session_state.dernier_montant_recommande = montant_recommande
                st.session_state.dernier_facteurs_model = facteurs_model

                st.caption(
                    "📊 La recommandation est basée sur le modèle CatBoost v2 entraîné sur l'historique "
                    "de remboursement. Elle prend en compte le score, le revenu mensuel et la durée du prêt."
                )
    
    # --- SECTION 5 : LEVIERS DE DÉCISION ---
    with st.expander("5. LEVIERS DE DÉCISION", expanded=False):
        garant = st.radio("Garant / caution *", OPTIONS_GARANT, index=None, key="f_garant")
    
    # --- VALIDATION ---
    champs_requis = {
        "Nom": nom, "Prénom": prenom, "Adresse": adresse,
        "Genre du demandeur": genre, "Tranche d'âge": age, "Niveau d'éducation": education,
        "Ligne de crédit ouverte": ligne_credit, "Utilisation du crédit": usage_credit,
        "Secteur d'activité": secteur, "Garant / caution": garant,
    }
    champs_manquants = [
        nom_champ for nom_champ, valeur in champs_requis.items()
        if valeur is None or (isinstance(valeur, str) and not valeur.strip())
    ]
    
    if champs_manquants:
        st.error(
            f"{len(champs_manquants)} champ(s) obligatoire(s) manque(nt) : "
            f"{', '.join(champs_manquants)}. Complétez avant d'analyser."
        )
    
    # --- ACTIONS ---
    st.write("")
    b1, b2, b3, b4 = st.columns([1, 1, 1.3, 1.6])
    with b1:
        if st.button("Réinitialiser", width="stretch"):
            reinitialiser_formulaire()
    with b2:
        if st.button("← Retour", width="stretch"):
            go_to("tableau_de_bord")
    with b3:
        if st.button("Enregistrer brouillon", width="stretch"):
            st.toast("Brouillon enregistré ✅")
    with b4:
        if st.button("Analyser la demande", type="primary", width="stretch",
                      disabled=bool(champs_manquants)):
            st.session_state.demande_data = {
                "id": nouvel_id, "nom": nom.strip(), "prenom": prenom.strip(), "adresse": adresse.strip(),
                "genre": genre, "age": age, "education": education,
                "revenu": revenu, "charges": charges, "ratio_endettement": ratio,
                "ligne_credit": ligne_credit, "usage_credit": usage_credit,
                "personnes_charge": personnes_charge, "logement": logement, "anciennete": anciennete,
                "montant_demande": montant_demande, "duree": duree, "objet": objet,
                "secteur": secteur, "activite_saisonniere": activite_saisonniere,
                "mobile_money": mobile_money, "membre_tontine": membre_tontine, "garant": garant,
            }
            st.session_state.demande_id_counter += 1
            go_to("resultats")


# =====================================================================
# 9. PAGE 4 — RÉSULTAT DE L'ANALYSE
# =====================================================================
def page_resultats():
    """Affiche les résultats avec score ML."""
    render_sidebar()
    render_entete()
    
    data = st.session_state.demande_data
    if not data:
        st.warning("Aucune demande à afficher.")
        if st.button("Nouvelle demande"):
            go_to("nouvelle_demande")
        return
    
    # Utiliser le score ML si disponible, sinon fallback heuristique
    score_model = st.session_state.dernier_score_model
    
    if score_model is not None:
        # Résultats ML
        categorie = st.session_state.dernier_score_categ
        if score_model >= 70:
            couleur = "#16a34a"
        elif score_model >= 40:
            couleur = "#d97706"
        else:
            couleur = "#dc2626"

        if score_model >= 65:
            decision = "ACCORDÉ"
        elif score_model >= 45:
            decision = "ÉTUDE APPROFONDIE"
        else:
            decision = "REFUSÉ"

        proba_defaut = max(2, min(96, (1 - score_model / 100) * 100))

        resultat = {
            "score": score_model,
            "categorie": categorie,
            "couleur": couleur,
            "decision": decision,
            "proba_defaut": proba_defaut,
            "facteurs": st.session_state.dernier_facteurs_model or [],
        }
        montant_recommande = st.session_state.dernier_montant_recommande or st.session_state.demande_data.get("montant_demande", 0)
        score_source = "🤖 Modèle ML (CatBoost)"
    else:
        # Fallback heuristique
        resultat = evaluer_demande_heuristique(data)
        montant_recommande = st.session_state.demande_data.get("montant_demande", 0)
        score_source = "📊 Système heuristique"
    
    st.title("Résultat de l'analyse")
    st.caption(f"ID : {data['id']} · Source : {score_source} · Statut : OK")
    
    col_score, col_decision = st.columns([1, 1.6])
    
    with col_score:
        with st.container(border=True):
            st.markdown("**SCORE PRÉDIT**")
            render_jauge_score(resultat["score"], resultat["couleur"])
            st.caption(f"Probabilité de défaut : {resultat['proba_defaut']:.1f} %")
    
    with col_decision:
        with st.container(border=True):
            if resultat["decision"] == "ACCORDÉ":
                st.success(f"DÉCISION : {resultat['decision']}")
                montant_accorde = montant_recommande
            elif resultat["decision"] == "ÉTUDE APPROFONDIE":
                st.warning(f"DÉCISION : {resultat['decision']}")
                montant_accorde = round(montant_recommande * 0.50 / 1000) * 1000
            else:
                st.error(f"DÉCISION : {resultat['decision']}")
                montant_accorde = 0
            
            taux_indicatif = 18.5 if resultat["score"] >= 55 else 22.0
            mensualite = calculer_mensualite(montant_accorde, taux_indicatif, data["duree"]) if montant_accorde else 0
            
            cc1, cc2 = st.columns(2)
            with cc1:
                st.metric("Montant accordé", format_fcfa(montant_accorde))
                st.metric("Taux indicatif", f"{taux_indicatif} %")
                st.metric("Mensualité estimée", format_fcfa(mensualite))
            with cc2:
                st.metric("Montant demandé", format_fcfa(data["montant_demande"]))
                st.metric("Durée", f"{data['duree']} mois")
                st.metric("Conditions", "Avec garant" if data["garant"] != "Non" else "Sans garant")
        
        if resultat.get("facteurs"):
            st.write("")
            with st.container(border=True):
                st.subheader("Facteurs influençants")
                for i, (nom, valeur, impact, explication) in enumerate(resultat["facteurs"], 1):
                    fc1, fc2 = st.columns([3, 1])
                    with fc1:
                        st.markdown(f"**{i}. {nom}** = {valeur}")
                        st.caption(explication)
                    with fc2:
                        if impact >= 0:
                            st.markdown(f"<span style='color:#16a34a; font-weight:600;'> RÉDUIT de {impact}pp</span>",
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color:#dc2626; font-weight:600;'> AUGMENTE de {abs(impact)}pp</span>",
                                        unsafe_allow_html=True)
                    if i < len(resultat["facteurs"]):
                        st.divider()
    
    st.write("")
    with st.container(border=True):
        st.subheader("👤 Profil du demandeur")
        st.write(f"**{data['prenom']} {data['nom']}** · {data['adresse']}")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Genre :** {data['genre']}")
            st.write(f"**Éducation :** {data['education']}")
        with c2:
            st.write(f"**Âge :** {data['age']} ans")
            st.write(f"**Secteur :** {data['secteur']}")
    
    st.info(
        "**Cet outil est un support à la décision uniquement.** La décision finale reste du "
        "ressort du comité de crédit. Tous les facteurs contextuels et humains doivent être "
        "pris en compte."
    )
    
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Modifier", width="stretch"):
            go_to("nouvelle_demande")
    with b2:
        if st.button("Nouvelle demande", width="stretch"):
            reinitialiser_formulaire()
    with b3:
        if st.button("Exporter en PDF", type="primary", width="stretch"):
            st.session_state.dernier_resultat = resultat
            st.session_state.montant_accorde = montant_accorde
            st.session_state.taux_indicatif = taux_indicatif
            st.session_state.mensualite = mensualite
            go_to("export_pdf")


# =====================================================================
# 10. PAGE 5 — EXPORT PDF
# =====================================================================
def page_export_pdf():
    """Aperçu et export PDF."""
    render_sidebar()
    render_entete()
    
    
    data = st.session_state.demande_data
    if not data:
        st.warning("Aucune demande à exporter.")
        return
    
    resultat = st.session_state.get("dernier_resultat") or evaluer_demande_heuristique(data)
    montant_accorde = st.session_state.get("montant_accorde", 0)
    taux_indicatif = st.session_state.get("taux_indicatif", 18.5)
    mensualite = st.session_state.get("mensualite", 0)
    
    st.title("Export PDF — Aperçu avant impression")
    st.caption(f"Demande {data['id']} · Statut : {resultat['decision']}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Retour aux résultats", width="stretch"):
            go_to("resultats")
    with col2:
        if FPDF_DISPONIBLE:
            score_model = st.session_state.dernier_score_model
            pdf_bytes = generer_pdf(data, resultat, montant_accorde, taux_indicatif, mensualite, score_model)
            st.download_button(
                "Télécharger le PDF", data=pdf_bytes,
                file_name=f"rapport_{data['id'].strip('#')}.pdf", mime="application/pdf",
                type="primary", width="stretch",
            )
    
    st.markdown(
        """
        <style>
            .st-key-apercu_a4 {
                max-width: 794px;
                margin: 0 auto 24px auto;
                padding: 56px 64px !important;
                box-shadow: 0 0 0 1px #e2e8f0, 0 12px 32px rgba(15, 23, 42, 0.10);
                border-radius: 3px;
                background-color: #ecf0f1;
                color: #1e293b;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    with st.container(border=True, key="apercu_a4"):
        st.markdown("<h3 style='text-align:center;'>RAPPORT D'ANALYSE — DEMANDE DE PRÊT</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align:center; color:#64748b;'>🏦 {st.session_state.institution} "
            f"— Système de Scoring Crédit</p>",
            unsafe_allow_html=True,
        )
        st.divider()
        
        st.markdown("**📋 INFORMATIONS**")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"ID : {data['id']}")
            st.write(f"Agent : {st.session_state.agent_nom}")
        with c2:
            st.write(f"Date : {datetime.now().strftime('%d %B %Y, %Hh%M')}")
            st.write(f"Score ML : {resultat['score']} / 100")
        st.divider()
        
        st.markdown("**👤 PROFIL**")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"M/Mme **{data['prenom']} {data['nom']}**")
        with c2:
            st.write(f"Adresse : {data['adresse']}")
        st.divider()
        
        st.markdown("**💰 DEMANDE**")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"Montant demandé : {format_fcfa(data['montant_demande'])}")
            st.write(f"Durée : {data['duree']} mois")
        with c2:
            st.write(f"Montant accordé : {format_fcfa(montant_accorde)}")
            st.write(f"Taux : {taux_indicatif} %")
        st.divider()
        
        st.markdown("**✅ ANALYSE DU RISQUE**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Score", f"{resultat['score']}/100")
        with c2:
            st.metric("Catégorie", resultat["categorie"])
        with c3:
            st.metric("Décision", resultat["decision"])

        if resultat.get("facteurs"):
            st.divider()
            st.markdown("**🔍 FACTEURS EXPLICATIFS DU SCORE**")
            for nom, valeur, impact, explication in resultat["facteurs"]:
                signe = "🟢 réduit" if impact >= 0 else "🔴 augmente"
                st.write(f"- **{nom}** ({valeur}) — {signe} le score de {abs(impact)} pt(s) · {explication}")

        st.divider()
        st.caption("Généré par Système de Scoring Crédit Cameroun", text_alignment="center")


# =====================================================================
# 11. PAGE — HISTORIQUE
# =====================================================================
def page_historique():
    """Historique des demandes."""
    render_sidebar()
    render_entete()
    
    st.title("Historique des demandes")
    df = get_historique_demandes()
    
    c1, c2, c3 = st.columns([2, 2, 1.3])
    with c1:
        statuts = st.multiselect("Filtrer par statut", options=sorted(df["statut"].unique().tolist()),
                                  default=sorted(df["statut"].unique().tolist()))
    with c2:
        score_min, score_max = st.slider("Plage de score", 0, 100, (0, 100))
    with c3:
        recherche = st.text_input("Rechercher un ID", "")
    
    df_filtre = df[df["statut"].isin(statuts)]
    df_filtre = df_filtre[(df_filtre["score"] >= score_min) & (df_filtre["score"] <= score_max)]
    if recherche:
        df_filtre = df_filtre[df_filtre["id"].str.contains(recherche, case=False)]
    df_filtre = df_filtre.sort_values("date", ascending=False)
    
    st.caption(f"{len(df_filtre)} demande(s) trouvée(s) sur {len(df)}")
    
    st.dataframe(
        df_filtre,
        column_config={
            "id": "ID Demande",
            "date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
            "profil": "Profil",
            "age": "Tranche d'âge",
            "montant": st.column_config.NumberColumn("Montant demandé", format="%d FCFA"),
            "statut": "Statut",
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d/100"),
        },
        use_container_width=True,
        hide_index=True,
    )


# =====================================================================
# 12. PAGE — PARAMÈTRES
# =====================================================================
def page_parametres():
    """Paramètres de l'application."""
    render_sidebar()
    render_entete()
    
    st.title("Paramètres")
    
    st.subheader("Profil de l'agent")
    c1, c2 = st.columns(2)
    with c1:
        nouveau_nom = st.text_input("Nom de l'agent", value=st.session_state.agent_nom)
    with c2:
        nouvelle_institution = st.text_input("Institution", value=st.session_state.institution)
    if st.button("Enregistrer"):
        st.session_state.agent_nom = nouveau_nom
        st.session_state.institution = nouvelle_institution
        st.success("Profil mis à jour.")
    
    st.divider()
    st.subheader("Modèle ML")
    if MODEL:
        st.success("✅ Modèle CatBoost chargé avec succès")
        st.metric("Nombre de features", len(FEATURES_NAMES))
        st.write("**Features utilisées:**")
        cols = st.columns(2)
        for i, feat in enumerate(FEATURES_NAMES):
            cols[i % 2].caption(feat)
    else:
        st.error("❌ Modèle non disponible")
    
    st.divider()
    st.subheader("Seuils de catégorie de risque")
    st.caption("Ces seuils déterminent les catégories affichées.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🟢 Faible", "≥ 70")
    with c2:
        st.metric("🟠 Modéré", "40 – 69")
    with c3:
        st.metric("🔴 Élevé", "< 40")
    
    st.divider()
    st.subheader("À propos")
    st.info(
        "**Système de Scoring Crédit Cameroun V2.0**\n\n"
        "Modèle ML : CatBoost Classifier (16 features)\n\n"
        "Ce système utilise un modèle de machine learning entraîné sur l'historique de remboursement "
        "pour prédire le risque de crédit et recommander un montant maximum.\n\n"
        "⚠️ Cet outil est un support à la décision uniquement."
    )


# =====================================================================
# 13. ROUTAGE PRINCIPAL
# =====================================================================
def main():
    """Point d'entrée principal."""
    if not st.session_state.authenticated and st.session_state.page != "connexion":
        st.session_state.page = "connexion"
    
    routes = {
        "connexion": page_connexion,
        "tableau_de_bord": page_tableau_de_bord,
        "nouvelle_demande": page_nouvelle_demande,
        "resultats": page_resultats,
        "export_pdf": page_export_pdf,
        "historique": page_historique,
        "parametres": page_parametres,
    }
    page_active = routes.get(st.session_state.page, page_connexion)
    page_active()


if __name__ == "__main__":
    main()