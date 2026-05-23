# Projet NILM (P2M) — Désagrégation d’énergie avec HMM

## 1) Contexte et objectif
Ce projet de fin d’année (P2M) traite la **désagrégation d’énergie** (NILM — Non‑Intrusive Load Monitoring). L’objectif est de partir d’un **signal électrique total** d’une maison et **estimer l’état et la puissance** de 4 appareils : **kettle (bouilloire)**, **microwave (micro‑ondes)**, **fridge (réfrigérateur)** et **TV**. Le modèle principal repose sur des **Gaussian HMM** entraînés par **Baum‑Welch**, puis un **décodage Viterbi** pour estimer les états cachés.

## 2) Jeu de données
- **REFIT dataset** (20 maisons, UK).
- **Échantillonnage natif : 8 secondes**.
- Colonnes principales : `Aggregate` + `Appliance1…Appliance9`.
- Les appareils changent d’une maison à l’autre : un **mapping** est nécessaire (voir `Projet_NILM/data/refit_metadata.py`).

## 3) Architecture du dépôt
- **`Projet_NILM/`** : cœur Python du pipeline NILM.
  - `run_nilm.py` : point d’entrée (train + désagrégation).
  - `pipeline/` : prétraitement, entraînement HMM, désagrégation.
  - `models/` : modèles HMM sauvegardés (JSON).
  - `plots/` : figures générées.
  - `streaming_demo.py` : simulation temps réel.
  - `api.py` : API Flask pour l’app mobile et la démo web.
- **`Processed_Data_CSV/`** : données REFIT (dont `House_3_demo.csv`).
- **`nilm_monitor/`** : application Flutter (mobile).
- **`templates/`** : interface web minimale (dashboard) utilisée par `api.py`.

## 4) Pipeline NILM (batch)
### 4.1 Prétraitement (`pipeline/preprocessing.py`)
- Chargement CSV et parsing temporel.
- **Rééchantillonnage à 8s**.
- **Filtre de Hampel** (suppression des outliers).
- **Interpolation des NaN** puis écrêtage des valeurs négatives.

### 4.2 Entraînement HMM (`pipeline/train_hmm.py`)
- **Un HMM par appareil** (GaussianHMM — `hmmlearn`).
- **Nombre d’états** par défaut :
  - kettle/microwave/tv → 2 états (OFF/ON)
  - fridge → 3 états (OFF/LOW/HIGH)
- **Initialisation intelligente** via quantiles (stabilise la convergence EM).
- **Sauvegarde JSON** des modèles dans `models/<house>/`.

### 4.3 Désagrégation (`pipeline/disaggregate.py`)
Deux modes :
- **Sub‑metering** : Viterbi sur la colonne réelle de l’appareil (évaluation).
- **NILM pur** : uniquement `Aggregate` → recherche combinatoire des états dont la somme est la plus proche de l’agrégat.

Les états bruts sont **remappés sémantiquement** (OFF/LOW/HIGH/ON) en triant les états par **moyenne de puissance**.

### 4.4 Sorties
- Colonnes générées : `<appliance>_power`, `<appliance>_state`, `<appliance>_state_label`.
- Figures dans `plots/` (états et signatures).

## 5) Simulation temps réel (`streaming_demo.py`)
Une simulation “streaming” lit **un échantillon toutes les 8s** (ou plus vite si `delay_ms=0`) et calcule une prédiction par **fenêtre glissante ou tumbling**.
- **Fenêtre par défaut** : 60s → **7 échantillons**.
- La sortie retourne **état + puissance + confiance** par appareil (vote majoritaire sur la fenêtre).
- Le dataset de démo utilisé est `Processed_Data_CSV/House_3_demo.csv`.

Exemple :
```bash
cd Projet_NILM
python streaming_demo.py \
  --house Processed_Data_CSV/House_3_demo.csv \
  --models-dir models/9 \
  --appliances kettle microwave fridge tv \
  --window-seconds 60 \
  --delay-ms 8000
```

## 6) API Flask + interface web (`api.py` + `templates/index.html`)
L’API expose un **mode manuel** (push d’échantillons) et un **mode auto‑stream** (lecture CSV en arrière‑plan).

**Endpoints principaux** :
- `POST /push_sample` → un échantillon (mode manuel)
- `POST /start_stream` → démarre la lecture CSV (par défaut `House_3_demo.csv`)
- `POST /stop_stream`
- `GET /results` → derniers résultats de fenêtres
- `GET /events` → **SSE** pour streaming temps réel (app mobile)
- `GET /status`, `GET /stream_status`, `GET /houses`, `POST /reset`

L’interface web (`/`) affiche l’état des fenêtres et des appareils.

## 7) Visualisation à distance (ThingSpeak)
`api.py` envoie périodiquement :
- puissance agrégée moyenne
- index de fenêtre
- états ON/OFF des appareils

vers **ThingSpeak** pour une visualisation accessible via Internet.

## 8) Application mobile Flutter (`nilm_monitor/`)
Fonctions principales :
- Connexion à l’API (IP/port configurables).
- Démarrage/arrêt du **streaming**.
- Réception en temps réel via **SSE** (`/events`).
- Affichage live des états + puissances.
- **Historique local** (SQLite) + graphiques (énergie, durée d’activation, etc.).

Fichiers clés :
- `lib/providers/nilm_provider.dart` : logique principale + SSE.
- `lib/services/nilm_api.dart` : client HTTP + SSE.
- `lib/services/database_service.dart` : stockage local.
- `lib/screens/*` : UI (Home, Settings, History).

## 9) Exécution rapide
### 9.1 Installation Python
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 9.2 Entraînement + désagrégation (batch)
```bash
cd Projet_NILM
python run_nilm.py \
  --train-house ../Processed_Data_CSV/House_9.csv \
  --test-house  ../Processed_Data_CSV/House_3.csv
```

### 9.3 Démarrer l’API
```bash
cd Projet_NILM
python api.py
# Serveur sur http://0.0.0.0:8080
```

### 9.4 Lancer l’app mobile
```bash
cd nilm_monitor
flutter pub get
flutter run
```
Configurer ensuite l’IP/port de la machine qui exécute `api.py`.

## 10) Tests
```bash
cd Projet_NILM
python -m pytest tests/
```

---

### Résumé
Ce projet propose une solution **NILM complète** : entraînement HMM, désagrégation, **simulation streaming**, **API Flask**, **app Flutter**, et **visualisation ThingSpeak**. Il sert de base solide pour le rapport final en détaillant la chaîne complète depuis les données REFIT jusqu’à la visualisation mobile et web.
