# Système de Scoring Crédit Cameroun — Streamlit

Cette application reproduit en Python/Streamlit la maquette fournie :

- Écran 1 : Connexion agent
- Écran 2 : Dashboard
- Écran 3 : Formulaire de demande de prêt
- Écran 4 : Résultats du scoring
- Écran 5 : Export PDF

## Installation

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Linux / macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Connexion de démonstration

L'application accepte n'importe quel identifiant et mot de passe non vides.

Exemple :

- Identifiant : `agent@imf.cm`
- Mot de passe : `demo123`

## Fonctionnalités

- Navigation avec `st.sidebar`
- Mise en page responsive avec `st.columns`
- Tableau des demandes avec `st.dataframe`
- Formulaire structuré par sections
- Calcul pédagogique d'un score de 0 à 100
- Catégorie de risque et facteurs explicatifs
- Export d'un rapport PDF A4 avec ReportLab
- Conservation de l'état avec `st.session_state`

## Important

Le scoring fourni est volontairement **pédagogique** afin de reproduire la maquette. 
Il ne constitue pas un modèle de crédit validé et ne doit pas être utilisé pour
prendre automatiquement une décision financière réelle. Une validation humaine,
des règles métier documentées, une gouvernance du modèle, des tests de biais,
la sécurité et un vrai système d'authentification sont nécessaires avant toute
utilisation en production.

## Évolution production suggérée

1. Remplacer les données de démonstration par PostgreSQL.
2. Ajouter un vrai système d'authentification et de gestion des rôles.
3. Versionner les modèles de scoring.
4. Journaliser les analyses et décisions.
5. Ajouter la validation du comité de crédit.
6. Ajouter les tests de qualité et de sécurité.
