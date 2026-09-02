# Projet — Scoring Crédit Cameroun

## Objectif

Développer un modèle de scoring prédictif du risque de défaut de paiement
dans le contexte du crédit au Cameroun.

Le projet comprend :
- la préparation et l'enrichissement des données ;
- l'analyse exploratoire ;
- le feature engineering ;
- l'entraînement et la comparaison de modèles de Machine Learning ;
- la création d'une application Streamlit ;
- l'explicabilité des prédictions ;
- le déploiement.

## Équipe

- Aristide : Data preparation, feature engineering, Machine Learning,
  export PDF et déploiement.
- Andy : EDA avancée, sélection des features, comparaison des modèles,
  SHAP et Power BI.
- Marie-Thérèse : règles métier, maquette et développement de
  l'application Streamlit.

## Utilisateur actuel

Je suis Aristide.

Claude doit principalement m'aider sur les tâches qui me sont attribuées.
Il ne doit pas modifier inutilement le travail des autres membres.

## Stack technique

- Python 3.10+
- pandas
- numpy
- scikit-learn
- XGBoost
- matplotlib
- seaborn
- SHAP
- Streamlit
- joblib
- fpdf2
- Git / GitHub

## Git

Le projet utilise une branche par membre.

Branche d'Aristide :
`aristide`

Les modifications doivent être développées sur cette branche
puis intégrées à `main` via Pull Request.

Ne jamais pousser directement sur `main` sans instruction explicite.

## Données

Le dataset principal utilisé pour le projet est :

`Loan_Default_Cameroun_Encode.csv`

Le dataset a été enrichi et nettoyé pour intégrer le contexte camerounais.

Le projet dispose notamment du script :

`03_enrichissement_camerounais.py`

Ne pas modifier les décisions déjà validées concernant :
- les variables ;
- les transformations ;
- les données camerounaises ;
- la structure du dataset ;

sans vérifier le contexte existant ou demander confirmation.

## Planning

### Semaine 1 — Préparation des données
- préparation du dataset ;
- enrichissement avec les données camerounaises ;
- finalisation du notebook ;
- documentation ;
- versionnement Git.

### Semaine 2 — EDA et Feature Engineering
- analyse exploratoire ;
- traitement des valeurs manquantes ;
- traitement des valeurs aberrantes ;
- encodage ;
- normalisation ;
- validation des features.

### Semaine 3 — Machine Learning
Aristide :
- Logistic Regression ;
- Random Forest ;
- XGBoost ;
- comparaison des modèles ;
- AUC-ROC ;
- F1-score ;
- Precision ;
- Recall ;
- sauvegarde du meilleur modèle avec joblib.

### Semaine 4 — Application et déploiement
- intégration du modèle ;
- application Streamlit ;
- score de risque ;
- décision ;
- explication ;
- export PDF ;
- déploiement.

# État actuel du projet

Le projet est actuellement dans la phase de modélisation Machine Learning.

Les travaux précédents ont déjà été réalisés.

## Tâches actuelles d'Aristide

Les modèles de Machine Learning ont déjà été entraînés et comparés.

Les métriques nécessaires à leur évaluation ont également été calculées :
- AUC-ROC
- F1-score
- Precision
- Recall

La comparaison des modèles a permis d'identifier le meilleur modèle.

### Tâche restante

Il reste actuellement une tâche principale à réaliser :

1. Sauvegarder le meilleur modèle avec `joblib`.

Le modèle sauvegardé devra pouvoir être réutilisé ultérieurement par l'application Streamlit.

Avant de sauvegarder le modèle, vérifier :
- quel modèle a été sélectionné comme meilleur modèle ;
- que le modèle est bien entraîné ;
- que les éventuels objets nécessaires à la prédiction sont correctement pris en compte ;
- que le fichier généré pourra être chargé ultérieurement sans réentraîner le modèle.

Une fois la sauvegarde réalisée, vérifier que le fichier généré est bien présent dans le repository et qu'il peut être rechargé correctement avec `joblib`.

Ne pas réentraîner les modèles ni refaire leur comparaison sauf si un problème est détecté dans le travail existant.

## Principe important pour la reprise du projet

Ne pas considérer les tâches listées dans le planning général comme des tâches encore à effectuer.

Le planning décrit l'ensemble du projet, mais l'état réel du repository fait foi.

Pour Aristide, à ce stade, la seule tâche restante de la semaine en cours est la sauvegarde du meilleur modèle avec `joblib`.

Avant toute action, vérifier l'état réel des fichiers et du code afin de ne pas reproduire un travail déjà effectué.

## Documentation de référence

Le document complet du projet est disponible dans :

`docs/Analyse_et_Plan_Projet.docx`

Ce document constitue la référence détaillée concernant :
- le contexte ;
- les objectifs ;
- le calendrier ;
- les rôles de l'équipe ;
- les livrables.

En cas de besoin d'informations détaillées non présentes dans ce fichier,
consulter ce document.