
# Deep Learning Plant Disease Detection App 🌿

Un streamer Streamlit pour détecter les maladies des plantes via un modèle de deep learning (TensorFlow + Transfer Learning sur MobileNetV2).

---

## Description

Ce projet est une application web interactive développée avec **Streamlit**, destinée à la détection de maladies sur les feuilles de plantes. Le cœur du modèle repose sur **MobileNetV2** (pré-entraîné sur ImageNet), affiné via Transfer Learning sur un dataset de feuilles malades et saines. L’application permet à l’utilisateur d’uploader une image, puis affiche la prédiction et un score de confiance. Pour le moment c'est fiable que sur les données qu'on a parcequ'on a pas utilisé un enorme dataset.

---

## Fonctionnalités

- **Upload ou capture d’image**  
- **Prédiction en temps réel** de maladies des plantes  
- **Affichage du label** et **score de confiance (%)**  

---

## Architecture

- **Frontend** : Streamlit avec widgets pour upload d’image, affichage de résultats et capture webcam.  
- **Backend / Modèle** :  
  - **Prétraitement** : redimensionnement, normalisation, conversion en tensor  
  - **Modèle** : MobileNetV2 + couches denses personnalisées  
  - **Fine-tuning** : réglage des couches supérieures  
- **Entraînement** (hors app) :  
  - Séparation dataset (train / validation / test)  
  - Callbacks : EarlyStopping, ModelCheckpoint  
  - Optimiseur : Adam, loss = categorical_crossentropy  

---

## Installation

```bash
git clone https://github.com/<ton-username>/<ton-repo>.git
cd <ton-repo>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Fichier `requirements.txt` recommandé :

```text
streamlit
tensorflow>=2.x
pillow
numpy
opencv-python
```

---

## Usage

```bash
streamlit run app.py
```

Puis ouvrir http://localhost:8501 dans ton navigateur.  
L’utilisateur peut uploader une image pour obtenir la prédiction.

---

## Exemple de projet

![image](https://github.com/user-attachments/assets/6346432c-0e8b-4e42-bd92-5d683947213f)


---

## Résultats

Ton modèle **MobileNetV2** a obtenu une précision de **78,96 %** sur le jeu de test.  

---

## Contributions

Contributions bienvenues !  
Pour proposer une amélioration, merci d’ouvrir une issue ou un pull request.

---

## Licence

MIT © 2025 – Bahani Vissiley Thierry
