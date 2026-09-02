"""
shap_view.py - Module d'explicabilité SHAP pour CatBoost
Andy - Semaine 4
"""

import joblib
import pandas as pd
import numpy as np
import shap
import streamlit as st
import matplotlib.pyplot as plt
import catboost
import os

# ============================================
# 1. CHARGEMENT DU MODÈLE CatBoost
# ============================================

@st.cache_resource
def charger_modele():
    """Charge le modèle CatBoost sauvegardé."""
    try:
        # Obtenir le chemin absolu du dossier racine
        # __file__ = chemin vers shap_view.py (à la racine)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Le modèle est dans le dossier models/ à la racine
        model_path = os.path.join(base_dir, "models", "modele_scoring_credit.joblib")
        
        # Vérifier si le fichier existe
        if not os.path.exists(model_path):
            st.error(f"❌ Modèle non trouvé à : {model_path}")
            # Essayer un chemin alternatif (si le script est exécuté depuis app/)
            alt_path = os.path.join(base_dir, "..", "models", "modele_scoring_credit.joblib")
            if os.path.exists(alt_path):
                st.write(f"✅ Trouvé à : {alt_path}")
                model_path = alt_path
            else:
                st.error(f"❌ Modèle également introuvable à : {alt_path}")
                return None
        
        # Charger le fichier
        data = joblib.load(model_path)
        
        # Extraire le modèle du dictionnaire si nécessaire
        if isinstance(data, dict):
            if 'model' in data:
                return data['model']
            elif 'modele' in data:
                return data['modele']
            elif 'classifier' in data:
                return data['classifier']
            else:
                for key, value in data.items():
                    if hasattr(value, 'predict'):
                        return value
        return data
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None


@st.cache_resource
def get_feature_names():
    """Récupère les noms des features du modèle CatBoost."""
    modele = charger_modele()
    
    # Méthode 1: feature_names_in_ (si disponible)
    if hasattr(modele, 'feature_names_in_'):
        return list(modele.feature_names_in_)
    
    # Méthode 2: feature_names_ (CatBoost)
    if hasattr(modele, 'feature_names_'):
        return list(modele.feature_names_)
    
    # Méthode 3: get_feature_names() (CatBoost)
    if hasattr(modele, 'get_feature_names'):
        return list(modele.get_feature_names())
    
    # Fallback: utiliser une liste par défaut
    return [
        'usage_professionnel', 'montant_pret_fcfa', 
        'duree_mois', 'revenu_mensuel_fcfa', 'ratio_endettement',
        'objet_pret_Achat', 'objet_pret_Autre', 'objet_pret_Investissement_activite',
        'objet_pret_Refinancement', 'secteur_activite_Agriculture',
        'secteur_activite_Commerce_Negoce', 'secteur_activite_Petit_commerce',
        'secteur_activite_Salarie_formel'
    ]

@st.cache_resource
def charger_explainer():
    """Crée l'explainer SHAP adapté au modèle CatBoost."""
    modele = charger_modele()
    try:
        return shap.TreeExplainer(modele)
    except:
        return shap.Explainer(modele.predict, get_feature_names())

# ============================================
# 2. PRÉPARATION DES DONNÉES POUR SHAP
# ============================================

def preparer_donnees_client(donnees_brutes):
    """
    Transforme les données du formulaire en DataFrame.
    CatBoost est SENSIBLE à l'ordre des colonnes.
    """
    modele = charger_modele()
    features_attendues = get_feature_names()
    
    # Créer le DataFrame avec les données brutes
    df = pd.DataFrame([donnees_brutes])
    
    # Ajouter les colonnes manquantes avec 0
    for col in features_attendues:
        if col not in df.columns:
            df[col] = 0
    
    # Réordonner selon l'ordre attendu
    return df[features_attendues]

# ============================================
# 3. FONCTIONS D'AFFICHAGE DES GRAPHIQUES SHAP
# ============================================

def afficher_shap_waterfall(donnees_client):
    """Affiche un waterfall plot SHAP."""
    try:
        st.info(
            "**Comment lire ce graphique**  \n"
            "🔴 **Barres rouges** : ce qui pousse le dossier vers un risque plus élevé.  \n"
            "🔵 **Barres bleues** : ce qui pousse le dossier vers un risque plus faible.  \n"
            "**E[f(x)]** (en haut) : le point de départ — la moyenne sur l'ensemble des "
            "dossiers, avant de prendre en compte les caractéristiques de ce client précis.  \n"
            "**f(x)** (en bas) : le point d'arrivée — une fois toutes les barres "
            "appliquées, c'est la valeur qui correspond au score final de ce dossier."
        )

        modele = charger_modele()
        explainer = charger_explainer()
        df_client = preparer_donnees_client(donnees_client)
        
        # Calculer les valeurs SHAP
        shap_values = explainer.shap_values(df_client)
        
        # Si shap_values est une liste (modèle multi-classe)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Gérer expected_value
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
        
        # Créer le waterfall plot
        fig, ax = plt.subplots(figsize=(12, 7))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0] if shap_values.ndim > 1 else shap_values,
                base_values=expected_value,
                data=df_client.iloc[0],
                feature_names=df_client.columns
            ),
            show=False
        )
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        st.error(f"❌ Erreur SHAP (Waterfall) : {e}")

def afficher_shap_importance(donnees_client):
    """Affiche un graphique à barres des variables importantes POUR CE CLIENT."""
    try:
        st.info(
            "Classe les variables selon leur poids dans le calcul **pour ce dossier "
            "précis** (pas une règle générale valable pour tous les clients) — "
            "du plus déterminant en haut, au moins déterminant en bas. Plus la "
            "barre est longue, plus la variable a pesé dans la décision."
        )

        modele = charger_modele()
        explainer = charger_explainer()
        df_client = preparer_donnees_client(donnees_client)
        
        # Calculer les valeurs SHAP
        shap_values = explainer.shap_values(df_client)
        
        # Si shap_values est une liste (modèle multi-classe)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Créer le graphique à barres
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.bar_plot(
            shap_values[0] if shap_values.ndim > 1 else shap_values,
            feature_names=df_client.columns,
            show=False
        )
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        st.error(f"❌ Erreur SHAP (Importance) : {e}")
        st.info("💡 Vérifiez que les features du modèle sont correctement définies.")

def afficher_explications(donnees_client):
    """
    Fonction principale appelée par app_v2.py.
    Affiche tous les graphiques SHAP en colonnes.
    """
    st.markdown("---")
    st.header(
        "🔍 Pourquoi ce score ?",
        help=(
            "Ces deux graphiques détaillent, variable par variable, comment le "
            "score de ce dossier a été construit — calculés avec SHAP, la méthode "
            "d'explicabilité utilisée dans l'app."
        ),
    )
    st.caption(
        "Vue détaillée pour approfondir, en complément de la synthèse en langage "
        "courant affichée plus haut sur la page."
    )

    # Waterfall Plot (pleine largeur)
    st.subheader(
        "Décomposition du score",
        help="Montre comment on passe du score moyen de référence au score final de ce client, variable par variable.",
    )
    afficher_shap_waterfall(donnees_client)

    st.write("")  # Ligne vide pour séparer
    st.write("")

    # Impact des variables (pleine largeur)
    st.subheader(
        "Classement par importance",
        help="Les mêmes variables, mais classées de la plus déterminante à la moins déterminante pour ce dossier.",
    )
    afficher_shap_importance(donnees_client)