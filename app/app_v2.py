"""
=====================================================================
 CREDORA — Système de scoring crédit (Cameroun) — Application Streamlit
=====================================================================
Credora = Credit + Aurora : apporter de la clarté sur la décision de
credit grace aux donnees et au scoring.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import joblib
import os
import sys
import uuid
import base64
import smtplib
from email.message import EmailMessage
from urllib.parse import urlencode

# streamlit-cookies-manager (non maintenue) decore une fonction avec l'ancien
# @st.cache, retire depuis peu de certaines versions recentes de Streamlit
# (AttributeError bloquant au demarrage sur Streamlit Cloud, la version
# locale utilisee pour les tests l'a encore mais avec un avertissement de
# depreciation). st.cache_data est le remplacement direct recommande par
# Streamlit lui-meme pour ce cas d'usage (fonction pure, mise en cache par
# arguments) : on le pose sous cet ancien nom avant l'import si besoin,
# pour rester compatible avec les deux familles de versions.
if not hasattr(st, "cache"):
    st.cache = st.cache_data

from streamlit_cookies_manager import EncryptedCookieManager

# Ajouter la racine du projet au chemin Python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shap_view  # Module SHAP d'Andy (à la racine)
from catboost import Pool


# fpdf2 optionnel pour export PDF
try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    FPDF_DISPONIBLE = True
except ImportError:
    FPDF_DISPONIBLE = False

from db_manager import (
    login_user,
    register_user,
    logout_user,
    save_demande,
    get_demandes,
    get_demande_detail,
    archiver_demande,
    restaurer_demande,
    get_agent_info,
    create_password_reset_token,
    reset_password_with_token,
    get_user_by_session,
)

COOKIES_PASSWORD = (
    st.secrets.get("cookies", {}).get("password")
    or st.secrets.get("supabase", {}).get("db_password")
)
COOKIES = (
    EncryptedCookieManager(prefix="credora_", password=COOKIES_PASSWORD)
    if COOKIES_PASSWORD else None
)

# =====================================================================
# 0. CHARGEMENT DU MODÈLE ML
# =====================================================================
@st.cache_resource
def load_model():
    """Charge le modèle CatBoost sauvegardé une seule fois."""
    try:
        # Construire le chemin absolu vers models/modele_scoring_credit.joblib
        # __file__ = app/app_v2.py, on remonte d'un niveau avec dirname deux fois
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "modele_scoring_credit.joblib")
        
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


@st.cache_data
def charger_logo_base64(nom_fichier):
    """Charge un asset du logo (dossier app/assets/) et l'encode en base64
    pour l'incorporer directement dans le HTML (pas de serveur de fichiers
    statiques a configurer)."""
    chemin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", nom_fichier)
    try:
        with open(chemin, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


LOGO_ICONE_B64 = charger_logo_base64("credora-icon.svg")


# =====================================================================
# 1. CONFIGURATION GÉNÉRALE DE LA PAGE
# =====================================================================
NOM_APP = "Credora"
VERSION_APP = "1.0.0"

# --- Palette de marque (identité visuelle Credora, validée en equipe) ---
COULEUR_PRIMAIRE = "#1B5E3F"        # vert foret - accents institutionnels (logo, icones, nav active)
COULEUR_PRIMAIRE_SOMBRE = "#163f2c"  # variante sombre du vert
COULEUR_ACCENT = "#E8A33D"          # ambre/jaune - couleur interactive principale (boutons, survol)
COULEUR_ACCENT_SOMBRE = "#C87F1F"   # ambre fonce - survol des boutons pleins
COULEUR_ACCENT_2 = "#D96C4A"        # corail - accent secondaire
COULEUR_FOND = "#EFF6F1"            # vert tres clair - fond de page (hors cartes), distinct des titres
COULEUR_FOND_SIDEBAR = "#E3EFE6"    # vert sauge pale - sidebar (theme "Feuillage clair")
COULEUR_BORDURE_SIDEBAR = "#D3E5D8"
COULEUR_FOND_CARTE = "#FFFBF3"      # creme - cartes/sections du contenu principal
COULEUR_TEXTE = "#2B2B2B"           # anthracite - plus doux qu'un noir pur
COULEUR_BORDURE = "#EDE3D0"

# --- Palette tonale (derivee des 3 couleurs de marque, methode Material
# Design : plusieurs nuances par couleur plutot que des teintes choisies
# a la main une par une) - "50" = tres clair (fonds de puce/badge),
# "700" = fonce (texte sur fond colore). Utilisee pour les pastilles
# d'icones et badges de statut.
VERT_50 = "#DDE7E2"
VERT_700 = COULEUR_PRIMAIRE_SOMBRE
AMBRE_50 = "#FBF1E2"
AMBRE_700 = COULEUR_ACCENT_SOMBRE
CORAIL_50 = "#F9E9E4"
CORAIL_700 = "#A34E30"

_FAVICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "credora-icon.svg")

st.set_page_config(
    page_title=f"{NOM_APP}",
    page_icon=_FAVICON_PATH if os.path.exists(_FAVICON_PATH) else "🌅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS commun à toute l'application ---
st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Poppins', 'Segoe UI', system-ui, sans-serif;
        }}

        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        div[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}

        /* --- App et contenu principal : vert tres clair, distinct des titres --- */
        .stApp {{
            background-color: {COULEUR_FOND} !important;
        }}
        .main h1 {{
            color: {COULEUR_PRIMAIRE} !important;
        }}
        .main p, .main h2, .main h3, .main h4, .main h5, .main h6,
        .main span, .main label, .main div {{
            color: {COULEUR_TEXTE} !important;
        }}

        /* --- Sidebar : vert plein + accent ambre --- */
        section[data-testid="stSidebar"] {{
            background-color: {COULEUR_PRIMAIRE} !important;
            border-right: 3px solid {COULEUR_ACCENT};
        }}
        section[data-testid="stSidebar"] * {{
            color: #ffffff !important;
        }}
        /* Les boutons gardent leurs propres couleurs (regles plus specifiques
           que le wildcard ci-dessus, donc non affectees) - texte non touche ici. */
        /* st.success/st.error du sidebar ont un fond pastel clair (natif
           Streamlit) - texte blanc y serait illisible, on le repasse fonce
           uniquement a cet endroit. */
        section[data-testid="stSidebar"] [data-testid="stAlert"] * {{
            color: {COULEUR_TEXTE} !important;
        }}
        /* Texte des boutons du sidebar en vert (regle a forte specificite,
           gagne sur le wildcard blanc ci-dessus quel que soit le testid
           exact du bouton selon la version de Streamlit). */
        section[data-testid="stSidebar"] button,
        section[data-testid="stSidebar"] button * {{
            color: {COULEUR_PRIMAIRE} !important;
        }}

        /* --- Cartes/sections du contenu principal : blocs creme, coins et ombre coherents ---
               Streamlit >=1.61 n'expose plus de wrapper dedie (stVerticalBlockBorderWrapper a
               disparu du DOM) : chaque conteneur borde a etudier doit donc porter un key=
               explicite, cible ici directement via la classe .st-key-<nom> que Streamlit lui
               appose (verifie : cette classe est bien sur le div data-testid="stVerticalBlock"
               qui porte deja la bordure native). --- */
        .st-key-carte_connexion, .st-key-carte_register,
        .st-key-carte_score, .st-key-carte_decision, .st-key-carte_facteurs, .st-key-carte_profil,
        .st-key-card_metric_0, .st-key-card_metric_1, .st-key-card_metric_2, .st-key-card_metric_3,
        .st-key-card_action_rapide, .st-key-card_derniere_demande {{
            background-color: {COULEUR_FOND_CARTE} !important;
            border-color: {COULEUR_BORDURE} !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 2px rgba(43, 43, 43, 0.04), 0 4px 14px rgba(43, 43, 43, 0.05);
        }}

        /* --- Cartes du tableau de bord : liseré extérieur ambre uniquement
               (le fond reste creme, pas de remplissage jaune) --- */
        .st-key-card_metric_0, .st-key-card_metric_1, .st-key-card_metric_2, .st-key-card_metric_3,
        .st-key-card_action_rapide, .st-key-card_derniere_demande {{
            border-color: {COULEUR_ACCENT} !important;
            border-width: 2px !important;
        }}

        /* --- Rayon coherent sur les boutons et badges --- */
        button[data-testid^="stBaseButton"] {{
            border-radius: 8px !important;
        }}

        /* --- Cadre complet des 5 sections en ambre, coins arrondis marques --- */
        .st-key-exp_identite [data-testid="stExpander"],
        .st-key-exp_capacite [data-testid="stExpander"],
        .st-key-exp_credit [data-testid="stExpander"],
        .st-key-exp_activite [data-testid="stExpander"],
        .st-key-exp_leviers [data-testid="stExpander"],
        .st-key-exp_identite details,
        .st-key-exp_capacite details,
        .st-key-exp_credit details,
        .st-key-exp_activite details,
        .st-key-exp_leviers details {{
            border: 2px solid {COULEUR_ACCENT} !important;
            border-radius: 12px !important;
            overflow: hidden;
        }}

        /* --- En-tetes des 5 sections du formulaire "Nouvelle demande" en ambre
               (uniquement la barre de titre cliquable, pas les champs a l'interieur) --- */
        .st-key-exp_identite summary,
        .st-key-exp_capacite summary,
        .st-key-exp_credit summary,
        .st-key-exp_activite summary,
        .st-key-exp_leviers summary,
        .st-key-exp_identite [data-testid="stExpanderHeader"],
        .st-key-exp_capacite [data-testid="stExpanderHeader"],
        .st-key-exp_credit [data-testid="stExpanderHeader"],
        .st-key-exp_activite [data-testid="stExpanderHeader"],
        .st-key-exp_leviers [data-testid="stExpanderHeader"] {{
            background-color: {COULEUR_ACCENT} !important;
            border-radius: 8px;
        }}
        .st-key-exp_identite summary *, .st-key-exp_identite [data-testid="stExpanderHeader"] *,
        .st-key-exp_capacite summary *, .st-key-exp_capacite [data-testid="stExpanderHeader"] *,
        .st-key-exp_credit summary *, .st-key-exp_credit [data-testid="stExpanderHeader"] *,
        .st-key-exp_activite summary *, .st-key-exp_activite [data-testid="stExpanderHeader"] *,
        .st-key-exp_leviers summary *, .st-key-exp_leviers [data-testid="stExpanderHeader"] * {{
            color: {COULEUR_TEXTE} !important;
        }}
        .credora-badge {{
            display: inline-block;
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 0.85em;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .credora-link {{
            color: {COULEUR_PRIMAIRE} !important;
            font-size: 0.9em;
            font-weight: 500;
            text-decoration: underline;
            text-underline-offset: 3px;
        }}
        .credora-link:hover {{
            color: {COULEUR_ACCENT_SOMBRE} !important;
        }}
        .credora-chip {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            border-radius: 50%;
            font-size: 1.1em;
            flex-shrink: 0;
        }}

        /* --- Boutons secondaires (la plupart des boutons) : blancs, jaunissent au survol --- */
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="stBaseButtonSecondary"] {{
            color: {COULEUR_PRIMAIRE} !important;
            border: 1px solid {COULEUR_PRIMAIRE} !important;
            background-color: {COULEUR_FOND} !important;
            transition: background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        }}
        button[data-testid="stBaseButton-secondary"]:hover,
        button[data-testid="stBaseButtonSecondary"]:hover {{
            color: {COULEUR_TEXTE} !important;
            border-color: {COULEUR_ACCENT} !important;
            background-color: {COULEUR_ACCENT} !important;
        }}
        button[data-testid="stBaseButton-secondary"]:disabled,
        button[data-testid="stBaseButtonSecondary"]:disabled {{
            color: #b0aca3 !important;
            border-color: #e2ddd2 !important;
            background-color: #faf8f3 !important;
        }}

        /* --- Boutons primaires : jaune/ambre plein (couleur du logo) --- */
        button[data-testid="stBaseButton-primary"],
        button[data-testid="stBaseButtonPrimary"] {{
            background-color: {COULEUR_ACCENT} !important;
            border-color: {COULEUR_ACCENT} !important;
            color: {COULEUR_TEXTE} !important;
            font-weight: 600;
            transition: background-color 0.15s ease, border-color 0.15s ease;
        }}
        button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="stBaseButtonPrimary"]:hover {{
            background-color: {COULEUR_ACCENT_SOMBRE} !important;
            border-color: {COULEUR_ACCENT_SOMBRE} !important;
            color: #ffffff !important;
        }}
        button[data-testid="stBaseButton-primary"]:disabled,
        button[data-testid="stBaseButtonPrimary"]:disabled {{
            background-color: #f1ecdf !important;
            border-color: #f1ecdf !important;
            color: #b0aca3 !important;
        }}
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
        "nom": "MANDENG", "prenom": "Francois", "adresse": "Bépanda, Douala",
        "genre": "Masculin", "age": "35-44", "education": "Supérieur",
        "revenu": 250000, "charges": 150000, "ligne_credit": "Oui", "usage_credit": "Professionnel",
        "personnes_charge": 3, "logement": "Propriétaire", "anciennete": 36,
        "montant_demande": 2000000, "duree": 24, "objet": "Investissement (activité)",
        "secteur": "Autre", "activite_saisonniere": "Non",
        "garant": "Oui (logement en hypothèque)",
    },
    "moyen": {
        "nom": "NGONO", "prenom": "Manie", "adresse": "Akwa, Douala",
        "genre": "Féminin", "age": "25-34", "education": "Secondaire",
        "revenu": 150000, "charges": 60000, "ligne_credit": "Non", "usage_credit": "Professionnel",
        "personnes_charge": 2, "logement": "Locataire", "anciennete": 25,
        "montant_demande": 800000, "duree": 18, "objet": "Investissement (activité)",
        "secteur": "Commerçant indépendant", "activite_saisonniere": "Non",
        "garant": "Non",
    },
    "risque": {
        "nom": "MABO", "prenom": "Oumar", "adresse": "Newbell, Maroua",
        "genre": "Masculin", "age": "55-64", "education": "Primaire",
        "revenu": 65000, "charges": 55000, "ligne_credit": "Oui", "usage_credit": "Personnel",
        "personnes_charge": 5, "logement": "Locataire", "anciennete": 8,
        "montant_demande": 600000, "duree": 12, "objet": "Autre",
        "secteur": "Activité saisonnière", "activite_saisonniere": "Oui",
        "garant": "Non",
    },
}

VALEURS_PAR_DEFAUT_FORMULAIRE = {
    "revenu": 250000, "charges": 150000, "personnes_charge": 3, "anciennete": 36,
    "montant_demande": 2000000, "duree": 24, "objet": OPTIONS_OBJET[0],
    "activite_saisonniere": "Non",
}


# =====================================================================
# 3. ÉTAT DE SESSION (AVEC SUPABASE)
# =====================================================================
def init_session_state():
    """Initialise les variables de session."""
    defaults = {
        "page": "connexion",
        "authenticated": False,
        "registration_message": None,
        "user": None,
        "agent_nom": None,
        "institution": "Microfinance",
        "demande_id_counter": 1,
        "demande_data": {},
        "dernier_score_model": None,
        "dernier_score_categ": None,
        "dernier_montant_recommande": None,
    }
    for cle, valeur in defaults.items():
        if cle not in st.session_state:
            st.session_state[cle] = valeur
 
init_session_state()


def restaurer_session_persistante():
    """Restaure la session depuis le cookie sans conserver le mot de passe."""
    if (
        COOKIES is None
        or st.session_state.get("authenticated")
        or st.session_state.pop("deconnexion_en_cours", False)
    ):
        return
    if not COOKIES.ready():
        # Le composant cookies n'a pas encore renvoye les cookies du
        # navigateur (aller-retour asynchrone). st.stop() ici bloquait
        # l'app indefiniment si ce round-trip echouait ou tardait (page
        # blanche, obligeant a relancer le serveur) : on renonce plutot
        # a l'auto-connexion pour cette execution et on laisse la page de
        # connexion s'afficher normalement. Si le cookie est bien present,
        # streamlit-cookies-manager redeclenchera un rerun automatique des
        # que le composant repond, et la connexion sera restauree alors.
        return

    session_id = COOKIES.get("session_id")
    user = get_user_by_session(session_id) if session_id else None
    if user:
        st.session_state.authenticated = True
        st.session_state.user = user
        st.session_state.agent_nom = user.get("nom_complet") or user.get("email", "Agent")
        st.session_state.institution = user.get("institution") or "Microfinance"
        st.session_state.page = "tableau_de_bord"
    elif session_id:
        COOKIES.pop("session_id", None)
        COOKIES.save()
 

# =====================================================================
# 4. FONCTIONS UTILITAIRES
# =====================================================================
def go_to(nom_page):
    """Change la page active."""
    st.session_state.page = nom_page
    st.rerun()


# Code couleur unique pour la decision, partage par le tableau de bord et
# l'historique : vert = accorde (faible risque), orange = etude approfondie,
# rouge = refuse. Coherent avec la jauge de score de la page Resultat.
COULEUR_CELLULE_DECISION = {
    "ACCORDÉ": "background-color: #dcfce7; color: #16a34a; font-weight: 600;",
    "ÉTUDE APPROFONDIE": "background-color: #fef3c7; color: #d97706; font-weight: 600;",
    "REFUSÉ": "background-color: #fee2e2; color: #dc2626; font-weight: 600;",
}


def style_ligne_selon_decision(row, col_decision, colonnes_a_colorer):
    """Retourne la liste des styles CSS par colonne pour une ligne de
    dataframe : les colonnes de `colonnes_a_colorer` prennent la couleur
    associee a `row[col_decision]`, les autres restent neutres. Fonction
    commune au tableau de bord et a l'historique."""
    style = COULEUR_CELLULE_DECISION.get(str(row.get(col_decision, "")).upper(), "")
    return [style if c in colonnes_a_colorer else "" for c in row.index]


def envoyer_email_reinitialisation(email, token):
    """Envoie le lien de réinitialisation via SMTP configuré dans secrets."""
    smtp = st.secrets.get("smtp", {})
    host = smtp.get("host")
    username = smtp.get("username")
    password = smtp.get("password")
    sender = smtp.get("sender") or username
    if not all((host, username, password, sender)):
        return False

    base_url = smtp.get("app_url", "http://localhost:8501")   # Changer cet URL en production
    lien = f"{base_url}?{urlencode({'reset_token': token})}"
    message = EmailMessage()
    message["Subject"] = "Réinitialisation de votre mot de passe Credora"
    message["From"] = sender
    message["To"] = email
    message.set_content(
        "Bonjour,\n\n"
        "Utilisez ce lien dans l'heure pour définir un nouveau mot de passe :\n"
        f"{lien}\n\n"
        "Si vous n'êtes pas à l'origine de cette demande, ignorez cet email.\n"
    )
    try:
        port = int(smtp.get("port", 587))
        with smtplib.SMTP(host, port, timeout=10) as serveur:
            serveur.starttls()
            serveur.login(username, password)
            serveur.send_message(message)
        return True
    except (OSError, smtplib.SMTPException):
        return False


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


def _libelle_correspond(valeur_depuis_feature, valeur_attendue):
    """
    Compare une valeur de catégorie extraite d'un nom de colonne one-hot
    (ex. "Salarié formel" tiré de "secteur_activite_Salarié formel") à la
    valeur attendue, en tolérant un double encodage UTF-8 des accents.

    Bug constaté sur models/modele_scoring_credit.joblib : les colonnes
    secteur_activite_Commerce/Négoce, _Profession libérale et _Salarié
    formel sont enregistrées mal encodées ("SalariÃ© formel" au lieu de
    "Salarié formel") - corruption déjà présente dans le CSV source
    (data/processed/Loan_Default_Cameroun_Encode.csv), pas introduite ici.
    Sans cette tolérance, ces 3 secteurs (dont "Salarié formel", un des
    plus courants) ne sont jamais reconnus : la colonne reste à 0 quel
    que soit le secteur réellement déclaré par le client, ce qui fausse
    silencieusement le score pour ces clients.
    """
    if valeur_depuis_feature == valeur_attendue:
        return True
    try:
        if valeur_depuis_feature.encode("latin-1").decode("utf-8") == valeur_attendue:
            return True
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return False


def construire_features_pour_modele(data):
    """
    Construit le vecteur de features pour le modèle ML à partir des données du formulaire.
    Retourne un array numpy prêt pour prediction.
    """
    features_array = np.zeros(len(FEATURES_NAMES))
    feature_dict = {}
    
    # --- Features numériques et booléennes ---
    # Note : credit_ouvert (ligne de crédit déjà ouverte) n'est plus une
    # feature du modèle depuis le 30/08/2026 (présente sur seulement 0,4%
    # des dossiers d'entraînement, le modèle sur-apprenait un effet ~10x
    # plus fort que ce que les données réelles justifient - cf.
    # scripts/07_modelisation.py). Le champ reste dans le formulaire à
    # titre informatif pour le dossier, mais n'influence plus le score.
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
            feature_dict[feat_name] = 1 if _libelle_correspond(objet_val, objet_mapped) else 0
    
    # --- One-hot encoding secteur_activite ---
    secteur_raw = data.get('secteur', 'Autre')
    secteur_mapped = SECTEUR_MAPPING.get(secteur_raw, 'Commerce/Négoce')
    for feat_name in FEATURES_NAMES:
        if feat_name.startswith('secteur_activite_'):
            secteur_val = feat_name.replace('secteur_activite_', '').replace('_', ' ')
            feature_dict[feat_name] = 1 if _libelle_correspond(secteur_val, secteur_mapped) else 0
    
    # --- Remplir l'array dans l'ordre des FEATURES_NAMES ---
    for i, feat_name in enumerate(FEATURES_NAMES):
        if feat_name in feature_dict:
            features_array[i] = feature_dict[feat_name]

    return features_array.reshape(1, -1)


def calculer_facteurs_shap(features, data):
    """
    Calcule les facteurs explicatifs de la prédiction ML à partir des valeurs
    SHAP natives de CatBoost (get_feature_importance, type="ShapValues").
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


def generer_resume_decision(decision, facteurs, prenom):
    """
    Genere une synthese courte (1-2 phrases), en langage clair et sans
    jargon technique, expliquant la decision a partir de la capacite de
    remboursement du demandeur. Reutilise les explications deja
    redigees dans `facteurs` (calculer_facteurs_shap ou
    evaluer_demande_heuristique), en ne retenant que celles qui vont
    dans le sens de la decision - pas la liste complete des 5 facteurs
    deja affichee par ailleurs, juste l'essentiel pour un agent presse.
    """
    prenom = prenom or "Le demandeur"

    if decision == "ACCORDÉ":
        intro = (
            f"{prenom} présente une situation financière qui permet d'envisager "
            f"sereinement le remboursement de ce crédit dans les délais prévus."
        )
        pertinents = [f for f in facteurs if f[2] >= 0]
    elif decision == "REFUSÉ":
        intro = (
            f"La situation actuelle de {prenom} ne permet pas d'assurer ce "
            f"remboursement dans de bonnes conditions."
        )
        pertinents = [f for f in facteurs if f[2] < 0]
    else:
        intro = (
            f"La capacité de {prenom} à assumer ce remboursement reste incertaine "
            f"et mérite un examen complémentaire."
        )
        pertinents = list(facteurs)

    pertinents = sorted(pertinents, key=lambda f: abs(f[2]), reverse=True)[:2]
    raisons = " ".join(f[3] for f in pertinents)

    return f"{intro} {raisons}".strip()


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
        proba_defaut = proba[1]

        # Convertir en score 0-100 : score = (1 - proba_defaut) * 100
        score_0_100 = round((1 - proba_defaut) * 100)
        score_0_100 = max(0, min(100, score_0_100))

        # Catégorie et couleur
        if score_0_100 >= 70:
            categorie = "FAIBLE"
            couleur = "#16a34a"
        elif score_0_100 >= 40:
            categorie = "MODÉRÉ"
            couleur = "#d97706"
        else:
            categorie = "ÉLEVÉ"
            couleur = "#dc2626"

        facteurs = calculer_facteurs_shap(features, data)

        return score_0_100, categorie, couleur, proba_defaut * 100, facteurs

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None, None, None, None, []


def recommander_montant_maximum(score, revenu, duree_mois):
    """
    Recommande un montant maximum de prêt selon le score, le revenu et la durée.
    """
    if score is None or revenu <= 0:
        return 0
    
    if score >= 70:
        ratio = 12
    elif score >= 55:
        ratio = 9
    elif score >= 40:
        ratio = 6
    else:
        ratio = 3
    
    duree_factor = min(duree_mois / 12, 1.5)
    montant_max = revenu * ratio * duree_factor
    return int(montant_max / 10000) * 10000


def evaluer_demande_heuristique(data):
    """Fallback heuristique simple."""
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


def get_historique_demandes(archivees: bool = False):
    """
    Récupère l'historique Supabase dans le format utilisé par l'interface.

    ⚠️ IMPORTANT : L'historique doit être le MÊME pour tous les agents
    d'une même institution. Chaque agent voit TOUTES les demandes de son
    institution, pas seulement les siennes. C'est une exigence métier clé.

    `archivees=True` renvoie la corbeille (demandes au statut 'archivee')
    au lieu de l'historique normal.
    """
    # Colonnes brutes de get_demandes() + colonnes derivees ajoutees
    # ci-dessous : la liste complete est conservee (pas de df[colonnes] qui
    # tronquerait le resultat) car d'autres pages (Tableau de bord) lisent
    # directement des colonnes brutes comme date_creation.
    colonnes_brutes = [
        "id_demande", "user_id", "nom_demandeur", "prenom_demandeur",
        "age_tranche", "secteur_activite", "montant_demande", "montant_accorde",
        "score_ml", "categorie_risque", "decision", "statut",
        "date_creation", "date_analyse",
    ]
    colonnes_derivees = [
        "id", "date", "nom", "prenom", "demandeur", "age", "profil", "montant", "score",
    ]

    def _construire_df(demandes):
        df = pd.DataFrame(demandes)
        df["id"] = df["id_demande"]
        df["date"] = pd.to_datetime(df["date_creation"])
        df["nom"] = df["nom_demandeur"]
        df["prenom"] = df["prenom_demandeur"]
        df["demandeur"] = (df["prenom"].fillna("") + " " + df["nom"].fillna("")).str.strip()
        df["age"] = df["age_tranche"]
        df["profil"] = df["secteur_activite"]
        df["montant"] = df["montant_demande"]
        df["score"] = pd.to_numeric(df["score_ml"], errors="coerce").fillna(0)
        df["decision"] = df["decision"].fillna("")
        df["statut"] = df["statut"].fillna("")
        return df

    def _df_vide():
        return pd.DataFrame(columns=colonnes_brutes + colonnes_derivees)

    user = st.session_state.get("user")
    if user:
        demandes = get_demandes(
            user_id=user["id"],
            role=user.get("role", "agent"),
            institution=user.get("institution"),
            archivees=archivees,
        )
        # Un utilisateur reel n'a jamais droit aux donnees de demonstration,
        # meme quand son historique (ou sa corbeille) est reellement vide -
        # sinon "0 demande active" retombe a tort sur les exemples fictifs.
        return _construire_df(demandes) if demandes else _df_vide()

    if archivees:
        return _df_vide()

    # Données de démonstration utilisées uniquement quand aucun utilisateur
    # n'est connecté (pas de session Supabase du tout).
    data = [
        {"id": "#20260815-0020", "date": "2026-08-15", "profil": "Salarié formel", "age": "35-44 ans", "montant": 2000000, "statut": "Accordé", "decision": "ACCORDÉ", "score": 72},
        {"id": "#20260815-0019", "date": "2026-08-15", "profil": "Salarié formel", "age": "35-44 ans", "montant": 3000000, "statut": "Accordé", "decision": "ACCORDÉ", "score": 72},
        {"id": "#20260815-0018", "date": "2026-08-15", "profil": "Commerçant indépendant", "age": "25-34 ans", "montant": 1500000, "statut": "Étude approfondie", "decision": "ÉTUDE APPROFONDIE", "score": 58},
        {"id": "#20260815-0017", "date": "2026-08-15", "profil": "Activité saisonnière", "age": "45-54 ans", "montant": 800000, "statut": "Refusé", "decision": "REFUSÉ", "score": 37},
        {"id": "#20260814-0016", "date": "2026-08-14", "profil": "Fonctionnaire", "age": "45-54 ans", "montant": 5000000, "statut": "Accordé", "decision": "ACCORDÉ", "score": 81},
    ]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["date_creation"] = df["date"]
    df["categorie_risque"] = df["profil"]
    df["nom"] = ""
    df["prenom"] = ""
    df["demandeur"] = ""
    df["id_demande"] = df["id"]
    df["nom_demandeur"] = df["nom"]
    df["montant_demande"] = df["montant"]
    df["score_ml"] = df["score"]
    return df


def reconstruire_data_depuis_ligne_db(ligne):
    """Reconstruit le dict `data` (meme forme que celui construit dans
    page_nouvelle_demande) a partir d'une ligne complete de
    public.demandes_credit (get_demande_detail), pour reafficher la page
    Resultat d'une demande passee depuis l'Historique."""
    revenu = float(ligne.get("revenu_mensuel") or 0)
    charges = float(ligne.get("charges_mensuelles") or 0)
    return {
        "id": ligne.get("id_demande"),
        "nom": ligne.get("nom_demandeur") or "",
        "prenom": ligne.get("prenom_demandeur") or "",
        "adresse": ligne.get("adresse_demandeur") or "",
        "genre": ligne.get("genre"),
        "age": ligne.get("age_tranche"),
        "education": ligne.get("education"),
        "revenu": revenu,
        "charges": charges,
        "ratio_endettement": calc_ratio_endettement(revenu, charges),
        "ligne_credit": "Oui" if ligne.get("ligne_credit_ouverte") else "Non",
        "usage_credit": ligne.get("usage_credit"),
        "logement": ligne.get("logement_situation"),
        "anciennete": ligne.get("anciennete_activite") or 0,
        "montant_demande": float(ligne.get("montant_demande") or 0),
        "duree": int(ligne.get("duree_mois") or 0),
        "objet": ligne.get("objet_pret"),
        "objet_justification": ligne.get("objet_pret_justification"),
        "secteur": ligne.get("secteur_activite"),
        "activite_saisonniere": "Oui" if ligne.get("activite_saisonniere") else "Non",
        "secteur_justification": ligne.get("secteur_activite_justification"),
        "garant": ligne.get("garant"),
    }


def rouvrir_demande_sur_resultats(id_demande):
    """Recharge une demande depuis Supabase par son id_demande et bascule
    sur la page Resultat, pour qu'un clic sur une ligne de l'Historique (ou
    du Dashboard) reaffiche exactement ce que l'agent avait vu a l'epoque."""
    detail = get_demande_detail(id_demande)
    if not detail:
        st.warning("Détail introuvable pour cette demande (données de démonstration ou demande supprimée).")
        return
    data = reconstruire_data_depuis_ligne_db(detail)
    score_ml = detail.get("score_ml")
    score_ml = int(score_ml) if score_ml is not None else None
    st.session_state.demande_data = data
    st.session_state.dernier_score_model = score_ml
    st.session_state.dernier_score_categ = detail.get("categorie_risque")
    st.session_state.dernier_montant_recommande = (
        recommander_montant_maximum(score_ml, data["revenu"], data["duree"])
        if score_ml is not None else data["montant_demande"]
    )
    st.session_state.dernier_facteurs_model = []
    st.session_state.resultats_depuis_historique = True
    go_to("resultats")


@st.dialog("Demande sélectionnée")
def dialogue_ligne_historique(id_demande, mode_corbeille):
    """Mini fenetre ouverte quand une ligne de l'Historique (ou de la
    Corbeille) est cochee. Historique normal : Afficher (va sur la page
    Resultat) ou Supprimer (avec confirmation, met en corbeille - jamais
    de suppression definitive). Corbeille : Restaurer uniquement."""
    st.write(f"**{id_demande}**")

    def _fermer(cle_tableau):
        st.session_state.confirmation_suppression_demande = None
        if cle_tableau in st.session_state:
            del st.session_state[cle_tableau]
        st.rerun()

    if mode_corbeille:
        st.caption("Cette demande est dans la corbeille.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Restaurer", type="primary", width="stretch"):
                restaurer_demande(id_demande)
                get_demandes.clear()
                _fermer("tableau_corbeille")
        with col2:
            if st.button("Fermer", width="stretch"):
                _fermer("tableau_corbeille")
        return

    if st.session_state.get("confirmation_suppression_demande") == id_demande:
        st.warning("Êtes-vous sûr de vouloir supprimer cette demande de l'historique ?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Oui, supprimer", type="primary", width="stretch"):
                archiver_demande(id_demande)
                get_demandes.clear()
                _fermer("tableau_historique")
        with col2:
            if st.button("Annuler", width="stretch"):
                st.session_state.confirmation_suppression_demande = None
                st.rerun()
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Afficher", type="primary", width="stretch"):
            rouvrir_demande_sur_resultats(id_demande)
    with col2:
        if st.button("Supprimer", width="stretch"):
            st.session_state.confirmation_suppression_demande = id_demande
            st.rerun()
    with col3:
        if st.button("Fermer", width="stretch"):
            _fermer("tableau_historique")


def generer_pdf(data, resultat, montant_disponible, taux, mensualite, score_model=None):
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
    pdf.cell(0, 6, texte(f"{st.session_state.institution} - {NOM_APP}"),
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
    # Afficher le montant disponible UNIQUEMENT pour ÉTUDE APPROFONDIE et REFUSÉ
    if resultat["decision"] in ("ÉTUDE APPROFONDIE", "REFUSÉ"):
        deux_colonnes(
            "Montant demandé",
            format_fcfa(data["montant_demande"]),
            "Montant disponible",
            format_fcfa(montant_disponible)
        )
    else:
        # Pour ACCORDÉ, afficher uniquement le montant demandé
        pdf.cell(90, 6, texte(f"Montant demandé : {format_fcfa(data['montant_demande'])}"))
        pdf.ln(6)
    
    deux_colonnes(
        "Objet du prêt",
        data["objet"],
        "Durée",
        f"{data['duree']} mois"
    )    
    pdf.ln(3)
    
    titre_section("ANALYSE DU RISQUE")
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, texte(f"Score ML : {score_model}/100  |  Risque : {resultat['categorie']}  |  {resultat['decision']}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    deux_colonnes("Probabilité de défaut", f"{resultat['proba_defaut']:.1f} %",
                  "Ratio d'endettement", f"{data['ratio_endettement']:.0f} %")
    pdf.ln(3)

    if resultat.get("resume"):
        pdf.set_font("Helvetica", "I", 9.5)
        pdf.multi_cell(0, 4.6, texte(resultat["resume"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

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
    pdf.cell(0, 6, texte(f"Généré par {NOM_APP} — Scoring Crédit Cameroun"), align="C")
    
    return bytes(pdf.output())


# =====================================================================
# 5. COMPOSANTS D'INTERFACE
# =====================================================================
def render_sidebar():
    """Menu latéral."""
    with st.sidebar:
        if LOGO_ICONE_B64:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
                    <div style="background:#ffffff; border-radius:14px; padding:8px; width:64px; height:64px;
                                box-sizing:border-box; display:flex; align-items:center; justify-content:center;
                                box-shadow:0 2px 6px rgba(0,0,0,0.18); flex-shrink:0;">
                        <img src="data:image/svg+xml;base64,{LOGO_ICONE_B64}" width="46" height="46">
                    </div>
                    <span style="font-size:1.9em; font-weight:700; color:#ffffff; line-height:1;">{NOM_APP}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"## {NOM_APP}")
        st.caption("Évaluer intelligemment le risque de crédit")
        st.markdown(
            f'<div style="height:3px; width:100%; background:{COULEUR_ACCENT}; '
            f'border-radius:2px; margin:6px 0 12px 0;"></div>',
            unsafe_allow_html=True,
        )

        st.divider()
        pages_menu = [
            ("Tableau de bord", "tableau_de_bord", ":material/space_dashboard:"),
            ("Nouvelle demande", "nouvelle_demande", ":material/note_add:"),
            ("Historique", "historique", ":material/history:"),
            ("Paramètres", "parametres", ":material/settings:"),
        ]
        for label, cle_page, icone in pages_menu:
            type_bouton = "primary" if st.session_state.page == cle_page else "secondary"
            if st.button(label, width="stretch", key=f"nav_{cle_page}", type=type_bouton, icon=icone):
                go_to(cle_page)

        st.divider()
        if st.button("Déconnexion", width="stretch", key="nav_deconnexion", icon=":material/logout:"):
            if st.session_state.get("user"):
                logout_user(
                    st.session_state.user["id"],
                    st.session_state.user.get("session_id", "")
                )
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.deconnexion_en_cours = True
            if COOKIES is not None:
                COOKIES.pop("session_id", None)
                COOKIES.save()
            go_to("connexion")

        st.divider()
        if st.session_state.user:
            st.caption(f"**Agent :** {st.session_state.user['nom_complet']}")
            st.caption(st.session_state.user.get('institution', 'Microfinance'))

        st.divider()
        if MODEL:
            st.success("Modèle ML chargé", icon=":material/check_circle:")
        else:
            st.error("Modèle ML non disponible", icon=":material/error:")


def render_entete(sous_titre="Un accès au crédit plus juste, une décision à la fois."):
    """Bandeau d'en-tête."""
    st.markdown(
        f"""
        <div style="text-align:center; padding:26px 16px 22px 16px; margin-bottom:22px;">
            <span style="color:{COULEUR_PRIMAIRE}; font-size:2.6em; font-weight:700; line-height:1.1; letter-spacing:0.01em;">
                {NOM_APP}
            </span><br>
            <span style="display:inline-block; width:52px; height:3px; background:{COULEUR_ACCENT};
                         border-radius:2px; margin:10px 0 12px 0;"></span><br>
            <span style="color:{COULEUR_TEXTE}; opacity:0.75; font-size:1.15em; font-weight:500;">{sous_titre}</span>
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
            "bgcolor": COULEUR_FOND_CARTE,
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


# Couleurs de statut (badges de decision) - distinctes de la palette de
# marque, ce sont les memes signaux vert/orange/rouge deja utilises pour
# le score ailleurs dans l'app, inchanges.
COULEURS_STATUT = {
    "ACCORDÉ": ("#16a34a", "#eafaf0"),
    "ÉTUDE APPROFONDIE": ("#d97706", "#fef3c7"),
    "REFUSÉ": ("#dc2626", "#fee2e2"),
}


def render_badge(texte, statut=None, couleur_fond=None, couleur_texte="#ffffff"):
    """Pastille arrondie pleine (badge de statut), plus compacte et lisible
    qu'une alerte st.success/warning/error en pleine largeur."""
    if statut and statut in COULEURS_STATUT:
        couleur_fond, _ = COULEURS_STATUT[statut]
    st.markdown(
        f'<span class="credora-badge" style="background:{couleur_fond}; color:{couleur_texte};">{texte}</span>',
        unsafe_allow_html=True,
    )


def render_chip(icone_svg_path, couleur_fond, couleur_icone):
    """Pastille circulaire coloree contenant une icone (mini-motif SVG
    inspire du logo), pour les cartes de metriques."""
    return (
        f'<div class="credora-chip" style="background:{couleur_fond};">'
        f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{couleur_icone}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{icone_svg_path}</svg>'
        f'</div>'
    )


# Petits tracés d'icones (grille 24x24, style Material/Feather - traits
# simples, pas d'emoji) reutilises pour les puces colorees du dashboard.
ICONE_DEMANDES = '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>'
ICONE_CHECK = '<path d="M20 6 9 17l-5-5"/>'
ICONE_CROIX = '<path d="M18 6 6 18"/><path d="M6 6l12 12"/>'
ICONE_HORLOGE = '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/>'

# =====================================================================
# 6. PAGE 1 — CONNEXION
# =====================================================================
def page_connexion():
    """Écran de connexion."""
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {COULEUR_ACCENT} !important; }}
        [data-testid="collapsedControl"] {{ display: none; }}
        section[data-testid="stSidebar"] {{ display: none; }}
        .st-key-carte_connexion div[data-testid="stTextInputRootElement"] {{
            border: 1.5px solid {COULEUR_ACCENT} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, col_centre, _ = st.columns([1, 1.6, 1])
    with col_centre:
        st.markdown(
            f"""
            <div style='text-align:center; margin-top:20px;'>
                <div style='display:flex; align-items:center; justify-content:center; gap:14px;'>
                    {(
                        f'''<div style="background:#ffffff; border-radius:16px; padding:10px; width:70px; height:70px;
                            box-sizing:border-box; display:flex; align-items:center; justify-content:center;
                            box-shadow:0 2px 8px rgba(0,0,0,0.18);">
                            <img src="data:image/svg+xml;base64,{LOGO_ICONE_B64}" width="52" height="52">
                        </div>'''
                        if LOGO_ICONE_B64 else ""
                    )}
                    <h1 style='color:{COULEUR_PRIMAIRE}; margin:0; font-weight:700;'>{NOM_APP}</h1>
                </div>
                <span style='color:{COULEUR_TEXTE}; opacity:0.75;'>Apporter de la clarté sur la décision de
                    crédit grâce aux données et au scoring.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        with st.container(border=True, key="carte_connexion"):
            st.markdown(
                f"<h2 style='text-align:center; color:{COULEUR_PRIMAIRE};'>Connexion</h2>",
                unsafe_allow_html=True,
            )
            identifiant = st.text_input("Email ou Nom d'utilisateur", placeholder="exemple@imf.cm")
            mot_de_passe = st.text_input("Mot de passe", type="password")
            rester_connecte = st.checkbox("Rester connecté")
            
            if st.button("CONNEXION", width="stretch", type="primary"):
                if identifiant.strip() and mot_de_passe.strip():
                    with st.spinner("Connexion en cours..."):
                        user = login_user(identifiant, mot_de_passe, remember_me=rester_connecte)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        # Synchronise les champs plats utilises par le PDF/apercu
                        # (page Parametres peut ensuite les personnaliser pour la session).
                        st.session_state.agent_nom = user.get("nom_complet") or user.get("email", "Agent")
                        st.session_state.institution = user.get("institution") or "Microfinance"
                        if COOKIES is not None:
                            if rester_connecte:
                                COOKIES["session_id"] = user["session_id"]
                            else:
                                COOKIES.pop("session_id", None)
                            COOKIES.save()
                        go_to("tableau_de_bord")
                    else:
                        st.error("Email ou mot de passe incorrect")
                else:
                    st.error("Veuillez renseigner identifiant et mot de passe")

            if st.session_state.registration_message:
                st.success(st.session_state.registration_message)
                st.session_state.registration_message = None

            if st.button("Créer un compte", width="stretch"):
                go_to("register")

            st.markdown(
                "<div style='text-align:right; margin-top:8px;'>"
                "<a class='credora-link' href='?page=mot_de_passe_oublie'>"
                "Mot de passe oublié ?</a></div>",
                unsafe_allow_html=True,
            )
        
        st.markdown(
            f"<p style='text-align:center; color:{COULEUR_TEXTE}; opacity:0.6; font-size:0.85em; margin-top:14px;'>"
            "Connexion sécurisée</p>",
            unsafe_allow_html=True,
        )


def page_mot_de_passe_oublie():
    """Demande l'envoi d'un lien de réinitialisation par email."""
    st.markdown(
        f"""
        <style>
        .stApp {{ background: linear-gradient(160deg, #ffffff 0%, {COULEUR_FOND_SIDEBAR} 55%, #fdecd2 100%) !important; }}
        [data-testid="collapsedControl"] {{ display: none; }}
        section[data-testid="stSidebar"] {{ display: none; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, col_centre, _ = st.columns([1, 1.8, 1])
    with col_centre:
        st.markdown(
            f"<h2 style='text-align:center; color:{COULEUR_PRIMAIRE};'>Mot de passe oublié</h2>",
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="carte_reset_request"):
            st.write("Saisissez l'adresse email associée à votre compte.")
            email = st.text_input("Email professionnel", placeholder="exemple@imf.cm")
            if st.button("Envoyer le lien", width="stretch", type="primary"):
                email = email.strip().lower()
                if not email or "@" not in email:
                    st.error("Veuillez saisir une adresse email valide.")
                else:
                    with st.spinner("Envoi du lien de réinitialisation..."):
                        token = create_password_reset_token(email)
                        email_envoye = token is not None and envoyer_email_reinitialisation(email, token)
                    if token is not None and not email_envoye:
                        st.error("Le service d'envoi d'email n'est pas configuré ou est indisponible.")
                    else:
                        st.success(
                            "Si un compte actif correspond à cette adresse, un lien de réinitialisation "
                            "vient d'être envoyé. Vérifiez votre boîte de réception."
                        )

            if st.button("Retour à la connexion", width="stretch"):
                go_to("connexion")


def page_reinitialiser_mot_de_passe(token):
    """Permet de définir un nouveau mot de passe à partir d'un jeton valide."""
    st.markdown(
        f"""
        <style>
        .stApp {{ background: linear-gradient(160deg, #ffffff 0%, {COULEUR_FOND_SIDEBAR} 55%, #fdecd2 100%) !important; }}
        [data-testid="collapsedControl"] {{ display: none; }}
        section[data-testid="stSidebar"] {{ display: none; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, col_centre, _ = st.columns([1, 1.8, 1])
    with col_centre:
        st.markdown(
            f"<h2 style='text-align:center; color:{COULEUR_PRIMAIRE};'>Nouveau mot de passe</h2>",
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="carte_reset_password"):
            nouveau_mot_de_passe = st.text_input("Nouveau mot de passe", type="password")
            confirmation = st.text_input("Confirmer le mot de passe", type="password")
            if st.button("Réinitialiser le mot de passe", width="stretch", type="primary"):
                if len(nouveau_mot_de_passe) < 8:
                    st.error("Le mot de passe doit contenir au moins 8 caractères.")
                elif nouveau_mot_de_passe != confirmation:
                    st.error("Les mots de passe ne correspondent pas.")
                else:
                    with st.spinner("Réinitialisation du mot de passe..."):
                        reset_reussi = reset_password_with_token(token, nouveau_mot_de_passe)
                    if reset_reussi:
                        st.query_params.clear()
                        st.session_state.registration_message = (
                            "Votre mot de passe a été réinitialisé. Vous pouvez vous connecter."
                        )
                        go_to("connexion")
                    else:
                        st.error("Ce lien est invalide ou expiré. Demandez un nouveau lien.")

            if st.button("Retour à la connexion", width="stretch"):
                st.query_params.clear()
                go_to("connexion")


# =====================================================================
# 6.1 PAGE — INSCRIPTION
# =====================================================================
def page_register():
    """Crée un compte utilisateur avec register_user()."""
    st.markdown(
        f"""
        <style>
        .stApp {{ background: linear-gradient(160deg, #ffffff 0%, {COULEUR_FOND_SIDEBAR} 55%, #fdecd2 100%) !important; }}
        [data-testid="collapsedControl"] {{ display: none; }}
        section[data-testid="stSidebar"] {{ display: none; }}
        .st-key-carte_register div[data-testid="stTextInputRootElement"] {{
            border: 1.5px solid {COULEUR_ACCENT} !important;
            border-radius: 6px !important;
        }}
        .st-key-carte_register div[data-testid="stTextInputRootElement"]:focus-within {{
            border-color: {COULEUR_ACCENT_SOMBRE} !important;
            box-shadow: 0 0 0 2px rgba(232, 163, 61, 0.18) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, col_centre, _ = st.columns([1, 1.6, 1])
    with col_centre:
        st.markdown(
            f"""
            <div style='text-align:center; margin-top:20px;'>
                <div style='display:flex; align-items:center; justify-content:center; gap:14px;'>
                    {(
                        f'''<div style="background:#ffffff; border-radius:16px; padding:10px; width:70px; height:70px;
                                     box-sizing:border-box; display:flex; align-items:center; justify-content:center;
                                     box-shadow:0 2px 8px rgba(0,0,0,0.18);">
                                <img src="data:image/svg+xml;base64,{LOGO_ICONE_B64}" width="52" height="52">
                            </div>'''
                        if LOGO_ICONE_B64 else ""
                    )}
                    <h1 style='color:{COULEUR_PRIMAIRE}; margin:0; font-weight:700;'>{NOM_APP}</h1>
                </div>
                <span style='color:{COULEUR_TEXTE}; opacity:0.75;'>Apporter de la clarté sur la décision de
                    crédit grâce aux données et au scoring.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        with st.container(border=True, key="carte_register"):
            st.markdown(
                f"<h2 style='text-align:center; color:{COULEUR_PRIMAIRE};'>Créer un compte</h2>",
                unsafe_allow_html=True,
            )
            col_nom, col_prenom = st.columns(2)
            with col_nom:
                nom = st.text_input("Nom *", placeholder="Ex : KOM")
            with col_prenom:
                prenom = st.text_input("Prénom *", placeholder="Ex : Olivier")
            email = st.text_input("Email professionnel *", placeholder="exemple@imf.cm")
            institution = st.text_input("Institution *", placeholder="Nom de votre institution")
            col_password, col_confirmation = st.columns(2)
            with col_password:
                mot_de_passe = st.text_input("Mot de passe *", type="password")
            with col_confirmation:
                confirmation = st.text_input("Confirmer le mot de passe *", type="password")

            if st.button("Créer le compte", width="stretch", type="primary"):
                email = email.strip().lower()
                nom = nom.strip()
                prenom = prenom.strip()
                institution = institution.strip()

                if not nom or not prenom or not email or not institution or not mot_de_passe:
                    st.error("Veuillez renseigner tous les champs obligatoires.")
                elif "@" not in email:
                    st.error("Veuillez saisir une adresse email valide.")
                elif mot_de_passe != confirmation:
                    st.error("Les mots de passe ne correspondent pas.")
                elif len(mot_de_passe) < 8:
                    st.error("Le mot de passe doit contenir au moins 8 caractères.")
                else:
                    with st.spinner("Création du compte..."):
                        succes, message = register_user(
                            email, mot_de_passe, nom, prenom, institution, "agent"
                        )
                    if succes:
                        st.session_state.registration_message = message
                        go_to("connexion")
                    else:
                        st.error(message)

            if st.button("Retour à la connexion", width="stretch"):
                go_to("connexion")

    st.markdown(
        f"<p style='text-align:center; color:{COULEUR_TEXTE}; opacity:0.6; font-size:0.85em; margin-top:14px;'>"
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
        nom = st.session_state.user.get("nom_complet", "Agent") if st.session_state.user else "Agent"
        st.title(f"Bienvenue, Agent {nom}")
        st.caption(datetime.now().strftime("%A %d %B %Y — %Hh%M"))
    with col_bouton:
        st.write("")
        st.write("")
        if st.button("Nouvelle demande", type="primary", width="stretch"):
            go_to("nouvelle_demande")
    
    st.subheader("Aujourd'hui")
    df = get_historique_demandes()
    if not df.empty:
        du_jour = df[pd.to_datetime(df["date_creation"]).dt.date == datetime.now().date()]
    
        nb_total = len(du_jour)
        nb_accordees = int((du_jour["decision"] == "ACCORDÉ").sum()) if "decision" in du_jour.columns else 0
        nb_refusees = int((du_jour["decision"] == "REFUSÉ").sum()) if "decision" in du_jour.columns else 0
        nb_etude = int((du_jour["decision"] == "ÉTUDE APPROFONDIE").sum()) if "decision" in du_jour.columns else 0
        
        metriques = [
            ("Demandes", nb_total, ICONE_DEMANDES, VERT_50, COULEUR_PRIMAIRE),
            ("Accordées", nb_accordees, ICONE_CHECK, "#eafaf0", "#16a34a"),
            ("Refusées", nb_refusees, ICONE_CROIX, "#fee2e2", "#dc2626"),
            ("En étude", nb_etude, ICONE_HORLOGE, AMBRE_50, COULEUR_ACCENT_SOMBRE),
        ]
        for i, (col, (label, valeur, icone, fond_puce, couleur_icone)) in enumerate(zip(st.columns(4), metriques)):
            with col:
                with st.container(border=True, key=f"card_metric_{i}"):
                    st.markdown(
                        f'<div style="display:flex; align-items:center; gap:10px;">'
                        f'{render_chip(icone, fond_puce, couleur_icone)}'
                        f'<div><div style="font-size:0.78em; color:{COULEUR_TEXTE}; opacity:0.65;">{label}</div>'
                        f'<div style="font-size:1.5em; font-weight:700; color:{COULEUR_TEXTE};">{valeur}</div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    
    st.write("")
    col_gauche, col_droite = st.columns([2, 1])
    
    with col_gauche:
        with st.container(border=True, key="card_action_rapide"):
            st.subheader("Action rapide")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("NOUVELLE DEMANDE", width="stretch", type="primary", key="qa_nouvelle"):
                    go_to("nouvelle_demande")
            with b2:
                if st.button("DEMANDES RÉCENTES", width="stretch"):
                    if not df.empty:
                        recentes = df.head(5)
                        st.dataframe(
                            recentes[["id", "demandeur", "montant", "decision", "score"]],
                            column_config={"demandeur": "Nom du demandeur"},
                            use_container_width=True, hide_index=True,

                        )
                    else:
                        st.info("Aucune demande enregistrée")
                    go_to("historique")
            
            st.write("**Demandes récentes**")
            recentes = df.sort_values("date", ascending=False).head(3)[["id", "demandeur", "profil", "age", "decision", "score"]]
            st.dataframe(
                recentes.style.apply(
                    style_ligne_selon_decision, col_decision="decision",
                    colonnes_a_colorer=("decision", "score"), axis=1,
                ),
                column_config={
                    "id": "ID", "demandeur": "Nom du demandeur", "profil": "Profil", "age": "Âge", "decision": "Statut",
                    "score": st.column_config.NumberColumn("Score", format="%d/100"),
                },
                hide_index=True, use_container_width=True,
            )
    
    with col_droite:
        with st.container(border=True, key="card_derniere_demande"):
            st.subheader("Dernière demande analysée")
            if df.empty:
                st.caption("Aucune demande analysée pour l'instant.")
            else:
                derniere = df.sort_values("date", ascending=False).iloc[0]
                st.metric(f"ID {derniere['id']}", f"{derniere['score']}/100")
                render_badge(derniere["statut"].upper(), statut=derniere["statut"].upper())
                st.write("")
                if st.button("Voir le détail →", width="stretch"):
                    go_to("historique")


# =====================================================================
# 8. PAGE 3 — NOUVELLE DEMANDE DE PRÊT
# =====================================================================
def page_nouvelle_demande():
    """Formulaire de nouvelle demande. Le score est calculé à la validation ;
    le résultat officiel est affiché sur la page Résultat."""
    render_sidebar()
    render_entete()
    
    st.title("Nouvelle demande de prêt")
    nouvel_id = f"#{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
    st.caption(f"ID demande : {nouvel_id} · Statut : Saisie en cours")
    
    init_formulaire_defaults()
    
    # --- Chargement rapide d'un exemple ---
    with st.expander("Charger un profil pour tester le formulaire"):
        e1, e2, e3 = st.columns(3)
        with e1:
            if st.button("Profil favorable", width="stretch"):
                charger_exemple("favorable")
        with e2:
            if st.button("Profil moyen", width="stretch"):
                charger_exemple("moyen")
        with e3:
            if st.button("Profil à risque", width="stretch"):
                charger_exemple("risque")
    
    # --- SECTION 1 : IDENTITÉ ---
    with st.expander("1. IDENTITÉ & PROFIL DEMANDEUR ", expanded=True, key="exp_identite"):
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
    with st.expander("2. CAPACITÉ FINANCIÈRE ", expanded=True, key="exp_capacite"):
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
    with st.expander("3. DEMANDE DE CRÉDIT", expanded=False, key="exp_credit"):
        c1, c2 = st.columns(2)
        with c1:
            montant_demande = st.number_input("Montant demandé (FCFA) *", min_value=50000, max_value=500000000,
                                                step=50000, key="f_montant_demande")
            duree = st.selectbox("Durée souhaitée (mois)", OPTIONS_DUREE, key="f_duree")
        with c2:
            objet = st.selectbox("Objet du prêt", OPTIONS_OBJET, key="f_objet")
        objet_justification = None
        if objet == "Autre":
            objet_justification = st.text_input(
                "Justification de l'objet du prêt *",
                placeholder="Précisez l'utilisation prévue du crédit",
                key="f_objet_justification",
            )
    
    # --- SECTION 4 : ACTIVITÉ PROFESSIONNELLE ---
    with st.expander("4. ACTIVITÉ PROFESSIONNELLE", expanded=True, key="exp_activite"):
        c1, c2 = st.columns(2)
        with c1:
            secteur = st.selectbox("Secteur d'activité *", OPTIONS_SECTEUR,
                                        index=None, placeholder="Sélectionner...", key="f_secteur")
        with c2:
            anciennete = st.number_input("Ancienneté dans l'activité (mois)", min_value=0, max_value=600,
                                        step=1, key="f_anciennete")

        secteur_justification = None
        if secteur == "Autre":
            secteur_justification = st.text_input(
                "Justification du secteur d'activité *",
                placeholder="Précisez l'activité exercée",
                key="f_secteur_justification",
            )
        
        activite_saisonniere = st.radio(
            "Activité saisonnière ?", ["Oui", "Non"],
            horizontal=True, key="f_activite_saisonniere"
        )
        
        # Calcul du score, silencieux : plus d'aperçu pendant la saisie.
        # Le résultat officiel est affiché sur la page Résultat après
        # validation. Les valeurs sont stockées en session pour cette page
        # et pour l'enregistrement de la demande.
        _champs_pour_ml = [revenu, charges, ligne_credit, usage_credit,
                           montant_demande, duree, objet, secteur]
        if not any(v is None or v == "" for v in _champs_pour_ml):
            data_ml = {
                "revenu": revenu, "charges": charges, "ligne_credit": ligne_credit,
                "usage_credit": usage_credit, "montant_demande": montant_demande,
                "duree": duree, "objet": objet, "secteur": secteur,
                "ratio_endettement": ratio,
            }
            score_model, categorie_model, _couleur_model, _proba_model, facteurs_model = predire_score_ml(data_ml)
            if score_model is not None:
                st.session_state.dernier_score_model = score_model
                st.session_state.dernier_score_categ = categorie_model
                st.session_state.dernier_montant_recommande = recommander_montant_maximum(score_model, revenu, duree)
                st.session_state.dernier_facteurs_model = facteurs_model

    # --- SECTION 5 : LEVIERS DE DÉCISION ---
    with st.expander("5. LEVIERS DE DÉCISION", expanded=False, key="exp_leviers"):
        garant = st.radio("Garant / caution *", OPTIONS_GARANT, index=None, key="f_garant")
    
    # --- VALIDATION ---
    champs_requis = {
        "Nom": nom, "Prénom": prenom, "Adresse": adresse,
        "Genre du demandeur": genre, "Tranche d'âge": age, "Niveau d'éducation": education,
        "Ligne de crédit ouverte": ligne_credit, "Utilisation du crédit": usage_credit,
        "Secteur d'activité": secteur, "Garant / caution": garant,
    }
    if objet == "Autre":
        champs_requis["Justification de l'objet du prêt"] = objet_justification
    if secteur == "Autre":
        champs_requis["Justification du secteur d'activité"] = secteur_justification
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
            demande_data = {
                "id": nouvel_id, "nom": nom.strip(), "prenom": prenom.strip(), "adresse": adresse.strip(),
                "genre": genre, "age": age, "education": education,
                "revenu": revenu, "charges": charges, "ratio_endettement": ratio,
                "ligne_credit": ligne_credit, "usage_credit": usage_credit,
                "personnes_charge": personnes_charge, "logement": logement, "anciennete": anciennete,
                "montant_demande": montant_demande, "duree": duree, "objet": objet,
                "objet_justification": objet_justification.strip() if objet_justification else None,
                "secteur": secteur, "activite_saisonniere": activite_saisonniere,
                "secteur_justification": secteur_justification.strip() if secteur_justification else None,
                "garant": garant,
            }
            demande_data.update({
                "score_ml": st.session_state.get("dernier_score_model"),
                "categorie_risque": st.session_state.get("dernier_score_categ"),
                "proba_defaut": (
                    (100 - st.session_state.dernier_score_model)
                    if st.session_state.get("dernier_score_model") is not None else None
                ),
                "decision": (
                    "ACCORDÉ" if st.session_state.dernier_score_model >= 65
                    else "ÉTUDE APPROFONDIE" if st.session_state.dernier_score_model >= 45
                    else "REFUSÉ"
                ) if st.session_state.get("dernier_score_model") is not None else None,
                "source_score": "ML" if st.session_state.get("dernier_score_model") is not None else "HEURISTIQUE",
            })

            user = st.session_state.get("user")
            with st.spinner("Enregistrement de la demande..."):
                demande_id = save_demande(demande_data, user["id"]) if user else None
            if demande_id:
                get_demandes.clear()
                demande_data["id"] = demande_id
                st.session_state.demande_data = demande_data
                st.session_state.demande_id_counter += 1
                st.session_state.resultats_depuis_historique = False
                go_to("resultats")
            else:
                st.error("La demande n'a pas pu être enregistrée. Vérifiez la connexion à la base de données.")


# =====================================================================
# 9. PAGE 4 — RÉSULTAT DE L'ANALYSE
# =====================================================================
def page_resultats():
    """Affiche les résultats avec score ML + graphiques SHAP."""
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
        score_source = "Modèle ML (CatBoost)"
    else:
        # Fallback heuristique
        resultat = evaluer_demande_heuristique(data)
        montant_recommande = st.session_state.demande_data.get("montant_demande", 0)
        score_source = "Système heuristique"

    resultat["resume"] = generer_resume_decision(
        resultat["decision"], resultat.get("facteurs") or [], data.get("prenom")
    )

    if st.session_state.get("resultats_depuis_historique"):
        if st.button("← Retour à l'historique"):
            go_to("historique")

    st.title("Résultat de l'analyse")
    st.caption(f"ID : {data['id']} · Source : {score_source} · Statut : OK")
    
    col_score, col_decision = st.columns([1, 1.6])
    
    with col_score:
        with st.container(border=True, key="carte_score"):
            st.markdown("**SCORE PRÉDIT**")
            render_jauge_score(resultat["score"], resultat["couleur"])
            st.caption(f"Probabilité de défaut : {resultat['proba_defaut']:.1f} %")
    
    with col_decision:
        with st.container(border=True, key="carte_decision"):
            if resultat["decision"] == "ACCORDÉ":
                montant_disponible = montant_recommande
            elif resultat["decision"] == "ÉTUDE APPROFONDIE":
                montant_disponible = round(montant_recommande * 0.50 / 1000) * 1000
            else:
                montant_disponible = 0

            render_badge(f"DÉCISION : {resultat['decision']}", statut=resultat["decision"])
            st.write("")
            st.caption(resultat["resume"])

            taux_indicatif = 18.5 if resultat["score"] >= 55 else 22.0

            # ACCORDÉ      → montant demandé
            # ÉTUDE        → montant disponible
            # REFUSÉ       → montant disponible (= 0)
            if resultat["decision"] == "ACCORDÉ":
                montant_base_mensualite = data["montant_demande"]
            else:
                montant_base_mensualite = montant_disponible

            mensualite = (calculer_mensualite(montant_base_mensualite, taux_indicatif, data["duree"])
                if montant_base_mensualite > 0
                else 0
            )    

            cc1, cc2 = st.columns(2)
            with cc1:
                # Afficher le montant disponible UNIQUEMENT pour ÉTUDE APPROFONDIE et REFUSÉ
                if resultat["decision"] in ("ÉTUDE APPROFONDIE", "REFUSÉ"):
                    st.metric("Montant disponible", format_fcfa(montant_disponible))

                st.metric("Taux indicatif", f"{taux_indicatif} %")
                st.metric("Mensualité estimée", format_fcfa(mensualite))
            with cc2:
                st.metric("Montant demandé", format_fcfa(data["montant_demande"]))
                st.metric("Durée", f"{data['duree']} mois")
                st.metric("Conditions", "Avec garant" if data["garant"] != "Non" else "Sans garant")
        
        if resultat.get("facteurs"):
            st.write("")
            with st.container(border=True, key="carte_facteurs"):
                st.subheader("Facteurs influençants (SHAP)")
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
    with st.container(border=True, key="carte_profil"):
        st.subheader("👤 Profil du demandeur")
        st.write(f"**{data['prenom']} {data['nom']}** · {data['adresse']}")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Genre :** {data['genre']}")
            st.write(f"**Éducation :** {data['education']}")
        with c2:
            st.write(f"**Âge :** {data['age']} ans")
            st.write(f"**Secteur d'activités :** {data['secteur']}")
    
    # ==========================================================
    # INTÉGRATION SHAP PAR ANDY - GRAPHIQUES EXPLICATIFS
    # Sur une page dédiée (pas affiché directement, jugé peu utile pour un
    # utilisateur non technique) : le bouton y renvoie, potentiellement
    # utile pour un profil plus expert.
    # ==========================================================
    if st.button("Facteurs explicatifs SHAP", type="secondary", width="stretch"):
        go_to("explicabilite_shap")

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
            st.session_state.montant_disponible = montant_disponible            
            st.session_state.taux_indicatif = taux_indicatif
            st.session_state.mensualite = mensualite
            go_to("export_pdf")


# =====================================================================
# 9.1 PAGE — EXPLICABILITÉ SHAP (DÉTAILLÉE, RÉSERVÉE AU BOUTON DÉDIÉ)
# =====================================================================
def page_explicabilite_shap():
    """Page dédiée aux graphiques SHAP détaillés. Accessible uniquement
    via le bouton "Facteurs explicatifs SHAP" de la page Résultat — pas
    affichée par défaut, jugée peu utile pour un utilisateur non
    technique mais potentiellement utile pour un profil plus expert."""
    render_sidebar()
    render_entete()

    data = st.session_state.demande_data
    if not data:
        st.warning("Aucune demande à expliquer.")
        if st.button("← Retour aux résultats"):
            go_to("resultats")
        return

    if st.button("← Retour aux résultats"):
        go_to("resultats")

    st.title("Facteurs explicatifs SHAP")
    st.caption(f"Demande {data['id']} · {data['prenom']} {data['nom']}")

    features_ml = construire_features_pour_modele(data)
    donnees_client = dict(zip(FEATURES_NAMES, features_ml[0]))
    shap_view.afficher_explications(donnees_client)


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
    montant_disponible = st.session_state.get("montant_disponible", 0)
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
            pdf_bytes = generer_pdf(
                data,
                resultat,
                montant_disponible,
                taux_indicatif,
                mensualite,
                score_model
            )            
            st.download_button(
                "Télécharger le PDF", data=pdf_bytes,
                file_name=f"rapport_{data['id'].strip('#')}.pdf", mime="application/pdf",
                type="primary", width="stretch",
            )
    
    st.markdown(
        f"""
        <style>
            .st-key-apercu_a4 {{
                max-width: 794px;
                margin: 0 auto 24px auto;
                padding: 56px 64px !important;
                box-shadow: 0 0 0 1px {COULEUR_BORDURE}, 0 12px 32px rgba(15, 23, 42, 0.10);
                border-radius: 3px;
                background-color: #ffffff;
                color: {COULEUR_TEXTE};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    with st.container(border=True, key="apercu_a4"):
        st.markdown("<h3 style='text-align:center;'>RAPPORT D'ANALYSE — DEMANDE DE PRÊT</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align:center; color:#64748b;'>{st.session_state.institution} "
            f"— {NOM_APP}</p>",
            unsafe_allow_html=True,
        )
        st.divider()
        
        st.markdown("**INFORMATIONS**")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"ID : {data['id']}")
            st.write(f"Agent : {st.session_state.agent_nom}")
        with c2:
            st.write(f"Date : {datetime.now().strftime('%d %B %Y, %Hh%M')}")
            st.write(f"Score ML : {resultat['score']} / 100")
        st.divider()
        
        st.markdown("**PROFIL**")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"M/Mme **{data['prenom']} {data['nom']}**")
        with c2:
            st.write(f"Adresse : {data['adresse']}")
        st.divider()
        
        st.markdown("**INFORMATIONS SUR LA DEMANDE DE CRÉDIT**")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"Montant demandé : {format_fcfa(data['montant_demande'])}")
            st.write(f"Durée : {data['duree']} mois")
        with c2:
            # Afficher le montant disponible UNIQUEMENT pour ÉTUDE APPROFONDIE et REFUSÉ
            if resultat["decision"] in ("ÉTUDE APPROFONDIE", "REFUSÉ"):
                st.write(f"Montant disponible : {format_fcfa(montant_disponible)}")
            st.write(f"Taux indicatif : {taux_indicatif} %")

        st.divider()
        
        st.markdown("**ANALYSE DU RISQUE**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Score", f"{resultat['score']}/100")
        with c2:
            st.metric("Catégorie de risque", resultat["categorie"])
        with c3:
            st.metric("Décision", resultat["decision"])

        if resultat.get("resume"):
            st.caption(resultat["resume"])

        if resultat.get("facteurs"):
            st.divider()
            st.markdown("**FACTEURS EXPLICATIFS DU SCORE**")
            for nom, valeur, impact, explication in resultat["facteurs"]:
                signe = "🟢 réduit" if impact >= 0 else "🔴 augmente"
                st.write(f"- **{nom}** ({valeur}) — {signe} le score de {abs(impact)} pt(s) · {explication}")

        st.divider()
        st.caption(f"Généré par {NOM_APP} — Scoring Crédit Cameroun", text_alignment="center")


# =====================================================================
# 11. PAGE — HISTORIQUE
# =====================================================================
def page_historique():
    """Historique des demandes."""
    render_sidebar()
    render_entete()

    mode_corbeille = st.session_state.get("historique_vue_corbeille", False)

    col_titre, col_bouton = st.columns([5, 1.4])
    with col_titre:
        st.title("Corbeille" if mode_corbeille else "Historique des demandes")
    with col_bouton:
        st.write("")
        st.write("")
        if mode_corbeille:
            if st.button("← Retour à l'historique", width="stretch"):
                st.session_state.historique_vue_corbeille = False
                st.rerun()
        else:
            if st.button("Corbeille", width="stretch"):
                st.session_state.historique_vue_corbeille = True
                st.rerun()

    df = get_historique_demandes(archivees=mode_corbeille)

    if df.empty:
        icone_b64 = charger_logo_base64("credora-icon.svg")
        message = (
            "La corbeille est vide." if mode_corbeille
            else "Aucune demande enregistrée pour l'instant.<br>"
                 "L'historique se remplit automatiquement à chaque analyse."
        )
        st.markdown(
            f"""
            <div style="text-align:center; padding:48px 20px; opacity:0.85;">
                <img src="data:image/svg+xml;base64,{icone_b64}" width="72" height="72" style="opacity:0.35;"><br>
                <p style="color:{COULEUR_TEXTE}; opacity:0.6; margin-top:14px; font-size:1.05em;">
                    {message}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    c1, c2, c3 = st.columns([2, 2, 1.3])
    with c1:
        decisions = sorted(df["decision"].dropna().unique().tolist())
        decisions_selectionnes = st.multiselect(
            "Filtrer par statut", options=decisions , default=decisions
        )
    with c2:
        score_min, score_max = st.slider("Plage de score", 0, 100, (0, 100))
    with c3:
        recherche = st.text_input("Rechercher un ID ou un demandeur", "")
    
    df_filtre = df[df["decision"].isin(decisions_selectionnes)]
    df_filtre = df_filtre[(df_filtre["score"] >= score_min) & (df_filtre["score"] <= score_max)]
    if recherche:
        recherche = recherche.strip()
        correspondance = (
            df_filtre["id"].fillna("").astype(str).str.contains(recherche, case=False, regex=False)
            | df_filtre["demandeur"].fillna("").astype(str).str.contains(recherche, case=False, regex=False)
        )
        df_filtre = df_filtre[correspondance]
    df_filtre = df_filtre.sort_values("date", ascending=False)
    
    st.caption(f"{len(df_filtre)} demande(s) trouvée(s) sur {len(df)}")

    if df_filtre.empty:
        st.info("Aucune demande ne correspond à ces filtres.")
        return

    historique_visible = df_filtre[
        ["id", "demandeur", "date", "profil", "age", "montant", "decision", "score"]
    ].rename(columns={
        "id": "ID demande",
        "demandeur": "Nom du demandeur",
        "date": "Date",
        "profil": "Profil",
        "age": "Tranche d'âge",
        "montant": "Montant demandé",
        "decision": "Décision",
        "score": "Score",
    })

    st.caption(
        "Cochez une demande (case à gauche) pour la restaurer." if mode_corbeille
        else "Cochez une demande (case à gauche) pour l'afficher ou la supprimer."
    )
    cle_tableau = "tableau_corbeille" if mode_corbeille else "tableau_historique"
    evenement = st.dataframe(
        historique_visible.style.apply(
            style_ligne_selon_decision, col_decision="Décision",
            colonnes_a_colorer=("Décision",), axis=1,
        ),
        column_config={
            "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
            "Montant demandé": st.column_config.NumberColumn("Montant demandé", format="%d FCFA"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d/100"),
        },
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=cle_tableau,
    )

    lignes_selectionnees = evenement.selection.rows if evenement and evenement.selection else []
    if lignes_selectionnees:
        id_selectionne = df_filtre.iloc[lignes_selectionnees[0]]["id"]
        dialogue_ligne_historique(id_selectionne, mode_corbeille)


# =====================================================================
# 12. PAGE — PARAMÈTRES
# =====================================================================
def page_parametres():
    """Paramètres de l'application."""
    render_sidebar()
    render_entete()
    
    st.title("Paramètres")
    
    st.subheader("Profil de l'agent")
    utilisateur = st.session_state.get("user") or {}
    agent_db = get_agent_info(utilisateur["id"]) if utilisateur.get("id") else None
    profil_db = agent_db or utilisateur
    nom_agent_db = profil_db.get("nom_complet", "")
    institution_db = profil_db.get("institution", "Microfinance")
    c1, c2 = st.columns(2)
    with c1:
        nouveau_nom = st.text_input("Nom de l'agent", value=nom_agent_db)
    with c2:
        nouvelle_institution = st.text_input("Institution", value=institution_db)
    if st.button("Enregistrer"):
        st.session_state.agent_nom = nouveau_nom
        st.session_state.institution = nouvelle_institution
        st.success("Profil mis à jour.")
    
    st.divider()
    st.subheader("Modèle ML")
    if MODEL:
        st.success("Modèle CatBoost chargé avec succès")
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
        f"**{NOM_APP}** — Scoring Crédit Cameroun\n\n"
        "Modèle ML : CatBoost Classifier (16 features)\n\n"
        "Ce système utilise un modèle de machine learning entraîné sur l'historique de remboursement "
        "pour prédire le risque de crédit et recommander un montant maximum.\n\n"
        "Cet outil est un support à la décision uniquement."
    )


# =====================================================================
# 13. ROUTAGE PRINCIPAL
# =====================================================================
def render_footer():
    """Pied de page global, affiché sur toutes les pages."""
    annee = datetime.now().year
    st.markdown(
        f"""
        <div style="margin-top:48px; padding-top:16px; border-top:1px solid {COULEUR_BORDURE};
                    text-align:center; color:{COULEUR_TEXTE}; opacity:0.6; font-size:0.82em;">
            © {annee} {NOM_APP} · Version {VERSION_APP} · Scoring Crédit Cameroun
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Point d'entrée principal."""
    restaurer_session_persistante()

    reset_token = st.query_params.get("reset_token")
    if reset_token:
        page_reinitialiser_mot_de_passe(reset_token)
        return

    page_param = st.query_params.get("page")
    if page_param in {"connexion", "register", "mot_de_passe_oublie"}:
        st.session_state.page = page_param
        st.query_params.clear()

    pages_publiques = {"connexion", "register", "mot_de_passe_oublie"}
    if not st.session_state.authenticated and st.session_state.page not in pages_publiques:
        st.session_state.page = "connexion"
    
    routes = {
        "connexion": page_connexion,
        "register": page_register,
        "mot_de_passe_oublie": page_mot_de_passe_oublie,
        "tableau_de_bord": page_tableau_de_bord,
        "nouvelle_demande": page_nouvelle_demande,
        "resultats": page_resultats,
        "explicabilite_shap": page_explicabilite_shap,
        "export_pdf": page_export_pdf,
        "historique": page_historique,
        "parametres": page_parametres,
    }
    page_active = routes.get(st.session_state.page, page_connexion)
    page_active()
    render_footer()


if __name__ == "__main__":
    main()