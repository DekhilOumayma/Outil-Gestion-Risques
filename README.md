# Guide de Déploiement - Application Web Gestion des Risques

## Fichiers inclus :
1. app.py : Code source Streamlit principal
2. requirements.txt : Liste des bibliothèques Python nécessaires
3. .streamlit/config.toml : Configuration du thème graphique

## Déploiement sur Streamlit Community Cloud (Gratuit)
1. Créez un dépôt sur GitHub et déposez-y les fichiers app.py, requirements.txt et .streamlit/config.toml.
2. Connectez-vous sur Streamlit Community Cloud (share.streamlit.io).
3. Cliquez sur 'New app', choisissez votre dépôt GitHub et le fichier app.py.
4. Cliquez sur 'Deploy'. Vous obtiendrez votre lien web d'accès en moins de 2 minutes.

## Exécution locale
Dans votre terminal :
- Installez les dépendances : pip install -r requirements.txt
- Lancez l'application : streamlit run app.py
