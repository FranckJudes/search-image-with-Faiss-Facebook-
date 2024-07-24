
# Application de Recherche d'Images Similaires

## Description
Cette application web permet aux utilisateurs de télécharger une image et de trouver des images similaires dans une base de données pré-indexée. Le système utilise un modèle de réseau neuronal pour extraire des caractéristiques des images et utilise l'indexation FAISS pour une recherche rapide et efficace.

## Fonctionnalités
- Téléchargement d'une image par l'utilisateur
- Extraction de caractéristiques à l'aide de ResNet50
- Recherche d'images similaires avec FAISS
- Interface utilisateur conviviale avec Bootstrap
- Affichage des résultats de recherche avec des images similaires et leurs distances

## Installation
1. **Prérequis :**
   - Python 3.x
   - Pip (gestionnaire de paquets Python)

2. **Clonage du Repository :**
   ```
   git clone https://github.com/FranckJudes/search-image-with-Faiss-Facebook-.git
   cd search-image-with-Faiss-Facebook
   ```

3. **Installation des Dépendances :**
   ```
   pip install -r requirements.txt
   ```

4. **Téléchargement du Modèle Pré-entraîné :**
   - Téléchargez le fichier `resnet50_v2-ecdde353.zip` depuis [lien_de_téléchargement] et extrayez-le dans le répertoire `./models/`.

## Utilisation
1. Lancez l'application Flask :
   ```
   python app.py
   ```
2. Accédez à l'application dans votre navigateur à l'adresse [http://localhost:5000](http://localhost:5000).

## Structure du Projet
```
├── app.py                    # Script principal pour lancer l'application Flask
├── static                    # Répertoire contenant les fichiers statiques (CSS, JS)
│   ├── css
│   │   └── style.css         # Feuille de style personnalisée
│   └── images                # Images statiques utilisées dans l'application
├── templates                 # Répertoire contenant les templates HTML
│   ├── index.html            # Page d'accueil de l'application
│   └── result.html           # Page des résultats de recherche
├── models                    # Répertoire pour stocker les modèles pré-entraînés
│   └── resnet50_v2-ecdde353.zip  # Modèle ResNet50 pré-entraîné (à télécharger)
├── save10.py                 # Script Python pour la préparation des données et l'indexation
├── requirements.txt          # Fichier des dépendances Python
└── README.md                 # Fichier que vous êtes en train de lire
```

## Contributions
Les contributions au projet sont les bienvenues. Pour des suggestions ou des problèmes, veuillez ouvrir une issue ou soumettre une pull request.

## Licence
Sous Licence MIT 

## Auteur
FranckJudes
