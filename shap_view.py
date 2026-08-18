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
    
    # Méthode 4: depuis les données d'entraînement (si disponibles)
    if hasattr(modele, 'feature_importances_'):
        # Essayer de récupérer depuis le modèle
        try:
            # Pour CatBoost, on peut utiliser les colonnes du Pool
            # Mais on va utiliser une liste par défaut
            pass
        except:
            pass
    
    # Fallback: utiliser une liste par défaut
    # Ces noms doivent correspondre aux features utilisées pour l'entraînement
    return [
        'credit_ouvert', 'usage_professionnel', 'montant_pret_fcfa', 
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
    
        st.markdown("""
        **Comment lire ce graphique :**
        - 🔴 **Barres rouges** : variables qui AUGMENTENT le risque de défaut
        - 🔵 **Barres bleues** : variables qui DIMINUENT le risque de défaut
        - La flèche en bas montre le score final
        """)
        
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
        
        st.markdown("""
        Ce graphique classe les variables par ordre d'impact sur la décision.
        Plus la barre est longue, plus la variable a influencé le score.
        """)
        
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
    st.header("🔍 Pourquoi ce score ?")
    st.markdown("""
    Voici une explication détaillée du score attribué à ce client.
    Ces graphiques ont été générés avec SHAP (SHapley Additive exPlanations).
    """)
    
    # Waterfall Plot (pleine largeur)
    st.subheader("Waterfall Plot - Décomposition du score")
    afficher_shap_waterfall(donnees_client)
    
    st.write("")  # Ligne vide pour séparer
    st.write("")
    
    # Impact des variables (pleine largeur)
    st.subheader("Impact des variables - Classement par importance")
    afficher_shap_importance(donnees_client)