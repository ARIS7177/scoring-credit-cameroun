# Credora — Scoring Crédit Cameroun (Streamlit)

Credora = Credit + Aurora : apporter de la clarté sur la décision de crédit
grâce aux données et au scoring.

Cette application couvre le parcours complet d'un agent de microfinance :

- Écran 1 : Connexion / création de compte (authentification réelle, Supabase)
- Écran 2 : Tableau de bord
- Écran 3 : Formulaire de demande de prêt
- Écran 4 : Résultats du scoring (score ML, décision, facteurs explicatifs, graphiques SHAP)
- Écran 5 : Export PDF
- Historique des demandes, partagé entre les agents d'une même institution

## Installation

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_v2.py
```

### Linux / macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_v2.py
```

## Configuration (obligatoire pour l'authentification et l'historique)

L'application se connecte à une base Supabase (PostgreSQL hébergé) pour
l'authentification et la persistance des demandes. Créer un fichier
`.streamlit/secrets.toml` (jamais versionné, voir `.gitignore`) :

```toml
[supabase]
url = "https://xxxx.supabase.co"
pooler_host = "aws-0-xxxx.pooler.supabase.com"
db_user = "postgres"
db_port = 5432
db_name = "postgres"
db_password = "xxxxxxxx"
```

Demander ces identifiants au membre de l'équipe responsable du projet
Supabase. Sans ce fichier, l'application se lance normalement mais
l'authentification échoue proprement (message d'erreur, pas de plantage).

## Fonctionnalités

- Authentification réelle (connexion + inscription) avec mots de passe hachés (bcrypt)
- Prédiction en temps réel avec le modèle CatBoost entraîné (16 features)
- Score de 0 à 100, catégorie de risque, décision (Accordé / Étude approfondie / Refusé)
- Facteurs explicatifs en langage clair + synthèse courte de la décision (sans jargon technique)
- Graphiques d'explicabilité SHAP (waterfall, importance des variables), avec légendes et infobulles
- Simulation du montant disponible et de la mensualité
- Historique des demandes partagé par institution, alimenté en continu
- Export d'un rapport PDF A4 avec fpdf2
- Conservation de l'état avec `st.session_state`

## Important

Cet outil est un support à la décision, pas un système entièrement automatisé.
La décision finale reste du ressort du comité de crédit de l'institution — tous
les facteurs contextuels et humains doivent être pris en compte dans la
délibération finale.

## Évolution production suggérée

1. Journaliser les analyses et décisions (audit trail complet).
2. Ajouter la validation du comité de crédit dans le flux applicatif.
3. Versionner les modèles de scoring.
4. Ajouter des tests de biais et de qualité sur le modèle.
5. Renforcer la sécurité (gestion des rôles plus fine, tests de sécurité).
