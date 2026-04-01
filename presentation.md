# Présentation de l'Avancement du Projet NILM  
## Désagrégation de la Consommation Électrique par Modèles de Markov Cachés

---

## Plan de la présentation

1. [Contexte et Objectif](#1-contexte-et-objectif)
2. [Architecture Globale du Projet](#2-architecture-globale-du-projet)
3. [Description des Étapes et des Scripts](#3-description-des-étapes-et-des-scripts)
4. [Résultats et Évaluation](#4-résultats-et-évaluation)
5. [Contenu des Slides](#5-contenu-des-slides)

---

## 1. Contexte et Objectif

### Problème adressé : NILM (Non-Intrusive Load Monitoring)

Le **NILM** (ou surveillance non-intrusive de la charge) est un problème de **désagrégation de l'énergie** :
- **Entrée** : Un seul signal de puissance agrégée (compteur principal du foyer)
- **Sortie** : Estimation de la consommation individuelle de chaque appareil (bouilloire, micro-ondes, réfrigérateur, télévision, etc.)
- **Enjeu** : Identifier quel appareil consomme combien, sans capteur sur chaque prise

### Jeu de données : REFIT

- Dataset public de 20 foyers britanniques instrumentés
- Fréquence d'échantillonnage : **toutes les 8 secondes**
- Colonnes disponibles : `Aggregate`, `Appliance1` à `Appliance9`
- Appareils présents selon les maisons (ex. bouilloire, frigo, lave-linge, TV, micro-ondes…)

---

## 2. Architecture Globale du Projet

### Structure du dépôt

```
Projet_adem/
│
├── README.md                          ← Description générale du projet
├── presentation.md                    ← Ce fichier
│
├── Processed_Data_CSV/                ← Données REFIT (CSV par maison)
│   ├── README.md                      ← Instructions de téléchargement
│   └── House_1.xlsx / House_X.csv    ← Fichiers de données
│
├── Projet_NILM/                       ← Code opérationnel principal
│   ├── run_nilm.py                   ← Point d'entrée : orchestre le pipeline
│   ├── preprocessing.py              ← Nettoyage et préparation des signaux
│   ├── train_hmm.py                  ← Entraînement des modèles HMM
│   ├── disaggregate.py               ← Décodage Viterbi + désagrégation
│   ├── refit_metadata.py             ← Correspondance appareils ↔ colonnes CSV
│   ├── plot_prf_metrics.py           ← Calcul et tracé des métriques PRF
│   ├── requirements.txt              ← Dépendances Python
│   └── tests/
│       ├── __init__.py
│       └── test_mapping.py           ← Tests unitaires du mapping
│
└── Adam BEN RHAIEM-HMM.../           ← Code de recherche et exploration
    ├── Clustering/                   ← Algorithmes de clustering d'appareils
    ├── Event Detection/              ← Algorithmes de détection d'événements
    ├── Preprocessing/                ← Variantes de filtrage / interpolation
    ├── HMM-based Load Disaggregation/← Variantes Viterbi (standard, sparse)
    └── Datasets/                     ← Références sur les jeux de données
```

### Pipeline NILM — Vue d'ensemble

```
Données brutes REFIT (CSV)
        │
        ▼
┌─────────────────────────────────┐
│  ÉTAPE 1 : Prétraitement        │  ← preprocessing.py
│  - Rééchantillonnage (8s)       │
│  - Filtre de Hampel             │
│  - Interpolation des NaN        │
│  - Écrêtage des valeurs < 0     │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  ÉTAPE 2 : Entraînement HMM     │  ← train_hmm.py
│  - 1 modèle HMM par appareil    │
│  - GaussianHMM (hmmlearn)       │
│  - Sauvegarde JSON des modèles  │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  ÉTAPE 3 : Désagrégation        │  ← disaggregate.py
│  - Algorithme de Viterbi        │
│  - Mode sous-comptage / NILM    │
│  - Correction sémantique        │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  ÉTAPE 4 : Évaluation & Plots   │  ← plot_prf_metrics.py
│  - Précision / Rappel / F1      │
│  - Visualisations temporelles   │
│  - Détection d'événements       │
└─────────────────────────────────┘
```

---

## 3. Description des Étapes et des Scripts

### Script 1 : `refit_metadata.py` — Métadonnées et Mapping

**Rôle :** Pont entre les noms canoniques d'appareils et les colonnes du CSV par maison.

**Problème résolu :** Dans REFIT, chaque maison a ses propres appareils dans des colonnes différentes. Par exemple :
- Maison 3 : `Appliance9` = Bouilloire (Kettle)
- Maison 9 : `Appliance7` = Bouilloire (Kettle)

**Fonctions clés :**

| Fonction | Description |
|---|---|
| `get_appliance_column(house, target)` | Retourne la colonne CSV d'un appareil dans une maison |
| `get_house_appliances(house)` | Retourne tous les appareils d'une maison |
| `parse_house_number(filepath)` | Extrait le numéro de maison depuis le nom de fichier |

**Mécanisme de correspondance :**
- Dictionnaire d'alias : `"fridge"` → `["fridge", "fridge-freezer", "fridgefreezer"]`
- Matching en deux passes : exact d'abord, puis sous-chaîne
- Prévient les faux positifs (ex. "freezer" ne correspond pas à "Fridge-Freezer")

---

### Script 2 : `preprocessing.py` — Nettoyage des Signaux

**Rôle :** Transformer les données brutes REFIT en séries temporelles propres, exploitables par le HMM.

**Étapes détaillées :**

#### Étape 2.1 — Chargement (`load_refit_csv`)
- Lecture du CSV avec `pandas`
- Gestion des deux formats de timestamp (`Time` ou `Unix`)
- Suppression des doublons, tri chronologique
- Écrêtage des puissances négatives à 0

#### Étape 2.2 — Rééchantillonnage
- Rééchantillonnage à une grille régulière de **8 secondes** (fréquence native REFIT)
- Garantit des séries homogènes même en cas de lacunes ou d'irrégularités

#### Étape 2.3 — Filtre de Hampel (`hampel_filter`)
- **But :** Suppression des valeurs aberrantes (pics de mesure, erreurs capteur)
- **Méthode :** Médiane glissante + MAD (Median Absolute Deviation)
- Si une valeur dépasse 3σ de la médiane locale → remplacée par la médiane
- Fenêtre par défaut : 15 points (~2 minutes)

```
Signal brut :  ... 150W  148W  2500W  151W ...   ← pic aberrant détecté
Après Hampel : ... 150W  148W   150W  151W ...   ← remplacé par médiane
```

#### Étape 2.4 — Interpolation (`interpolate_missing`)
- Comble les trous (NaN) après rééchantillonnage
- Méthode linéaire par défaut
- Limitation : max 5 points consécutifs interpolés (évite les artefacts)

#### Étape 2.5 — Finalisation
- Écrêtage final à 0 (pas de puissance négative)
- Remplacement des NaN résiduels par 0

**Visualisation optionnelle :** Comparaison signal brut vs. signal traité (PNG)

---

### Script 3 : `train_hmm.py` — Entraînement des Modèles HMM

**Rôle :** Apprendre le comportement de chaque appareil sous forme de Modèle de Markov Caché Gaussien.

#### Pourquoi le HMM ?

Un appareil électrique peut être modélisé comme un système à **états cachés discrets** :
- **Bouilloire / Micro-ondes / TV :** 2 états → OFF, ON
- **Réfrigérateur :** 3 états → OFF, LOW (compresseur faible), HIGH (compresseur fort)

Le HMM capture :
- La **probabilité d'émission** : quelle puissance émet chaque état
- La **matrice de transition** : probabilité de passer d'un état à un autre
- La **distribution initiale** : probabilité de départ dans chaque état

#### Architecture du HMM par appareil

```
État caché :   OFF ──────────────────────► ON
                ◄──────────────────────
Émission :     ~0W                         ~2000W (bouilloire)
               ~0W                         ~800W  (micro-ondes)
               ~0W                         ~100W  (TV)
               OFF ──► LOW ──► HIGH ──► ...
Émission :     ~0W    ~60W    ~120W       (réfrigérateur)
```

#### Fonctions clés :

| Fonction | Description |
|---|---|
| `train_appliance_hmm(series, n_states)` | Entraîne un GaussianHMM sur une série de puissance |
| `save_models(models, house)` | Sauvegarde les modèles en JSON |
| `load_models(house, appliances)` | Recharge les modèles JSON pour inférence |
| `run_training(house_csv, appliances)` | Pipeline d'entraînement complet |

**Initialisation intelligente :** Les moyennes initiales sont fixées par quantiles (0%, 33%, 67%, 100%) pour guider la convergence de l'EM.

**Limitation des données :** Sous-échantillonnage à 50 000 points max pour accélérer l'entraînement.

---

### Script 4 : `disaggregate.py` — Désagrégation et Décodage

**Rôle :** Utiliser les modèles HMM entraînés pour estimer l'état de chaque appareil sur de nouvelles données.

#### Mode 1 : Sous-comptage (`disaggregate_submetering`)
- Utilise la **colonne réelle** de chaque appareil dans le CSV
- Applique Viterbi sur chaque signal d'appareil indépendamment
- Usage : **évaluation** (on connaît la vérité terrain)

#### Mode 2 : NILM pur (`disaggregate_nilm`)
- Utilise **uniquement le signal agrégé** (scénario réaliste)
- Pour chaque instant, cherche la combinaison d'états dont la somme est la plus proche de l'agrégat mesuré
- Usage : **déploiement réel** (pas de sous-capteurs)

#### Correction sémantique des états — Point clé

**Problème :** Le HMM est invariant par permutation — l'état numéroté "0" peut être OFF ou ON selon l'initialisation aléatoire.

**Solution :** Après entraînement, on trie les états par **moyenne d'émission croissante** :
- État avec la plus faible moyenne → "OFF"
- État suivant → "LOW" (si 3 états)
- État avec la plus forte moyenne → "ON" / "HIGH"

Cela garantit la cohérence des labels entre différentes maisons (crucial pour l'évaluation cross-house).

#### Détection d'événements (`detect_state_events`)
- Identifie les **transitions d'état** : OFF→ON, ON→OFF, OFF→LOW, etc.
- Enregistre : timestamp, état avant, état après, puissance mesurée
- Génère des visualisations de fenêtres autour de chaque événement

#### Sorties

| Colonne produite | Description |
|---|---|
| `<appareil>_power` | Puissance estimée (W) |
| `<appareil>_state` | Indice d'état HMM brut |
| `<appareil>_state_label` | Étiquette sémantique (OFF/ON/LOW/HIGH) |

---

### Script 5 : `plot_prf_metrics.py` — Métriques de Performance

**Rôle :** Évaluer quantitativement les résultats de désagrégation.

**Métriques calculées (sklearn) :**
- **Précision** : parmi les instants classés ON, combien sont vraiment ON ?
- **Rappel** : parmi les vrais instants ON, combien sont détectés ?
- **F1-score** : moyenne harmonique de la précision et du rappel

**Vérité terrain :** `puissance > 10W` → appareil allumé (seuil empirique)

**Visualisation :** Graphique en barres groupées (Précision | Rappel | F1) par appareil

---

### Script 6 : `run_nilm.py` — Point d'Entrée Principal

**Rôle :** Orchestrer l'ensemble du pipeline via la ligne de commande.

**Arguments CLI :**

| Argument | Description |
|---|---|
| `--train-house` | CSV de la maison d'entraînement |
| `--test-house` | CSV de la maison de test |
| `--appliances` | Liste des appareils (défaut: kettle microwave fridge tv) |
| `--mode` | `train`, `disaggregate`, ou `all` |
| `--nilm` | Mode NILM pur (signal agrégé uniquement) |
| `--fridge-states N` | Nombre d'états pour le réfrigérateur (2 ou 3) |
| `--plot-preprocessing` | Affiche les signaux avant/après prétraitement |
| `--detect-events` | Détecte et visualise les transitions d'état |
| `--limit N` | Limite les données pour tests rapides |

**Exemples de commandes :**

```bash
# Pipeline complet : entraînement sur Maison 9, test sur Maison 3
python run_nilm.py --train-house House_9.csv --test-house House_3.csv \
                   --fridge-states 2 --plot-preprocessing --detect-events

# Mode NILM pur (signal agrégé uniquement)
python run_nilm.py --train-house House_9.csv --test-house House_3.csv \
                   --mode disaggregate --nilm

# Entraînement seul
python run_nilm.py --train-house House_9.csv --mode train
```

---

### Script 7 : `tests/test_mapping.py` — Tests Unitaires

**Rôle :** Valider la correspondance appareils ↔ colonnes CSV pour les maisons 3 et 9.

**Classes de tests :**

| Classe | Tests couverts |
|---|---|
| `TestHouse3Mapping` | 9 appareils de la Maison 3 |
| `TestHouse9Mapping` | 9 appareils de la Maison 9 |
| `TestCrossHouseMapping` | Même appareil, colonnes différentes selon la maison |
| `TestAliasResolution` | Résolution des alias (ex. "Fridge-Freezer" → "fridge") |
| `TestParseHouseNumber` | Extraction du numéro depuis le nom de fichier |

---

## 4. Résultats et Évaluation

### Configuration testée
- **Maison d'entraînement :** Maison 9 (House_9.csv)
- **Maison de test :** Maison 3 (House_3.csv)
- **Appareils :** Bouilloire, Micro-ondes, Réfrigérateur, Télévision

### Mapping des colonnes utilisées

| Appareil | Maison 9 (entraînement) | Maison 3 (test) |
|---|---|---|
| Fridge | Appliance1 (Fridge-Freezer) | Appliance2 (Fridge-Freezer) |
| Washing Machine | Appliance3 | Appliance6 |
| Kettle | Appliance7 | Appliance9 |
| Microwave | Appliance6 | Appliance8 |
| TV | Appliance5 | Appliance7 |

### Types d'évaluation disponibles

1. **Précision / Rappel / F1** par appareil (via `plot_prf_metrics.py`)
2. **Précision ON/OFF** simple (via `disaggregate.py → evaluate_results`)
3. **Visualisation temporelle** : états décodés vs. signal réel
4. **Événements détectés** : qualité des transitions ON/OFF

---

## 5. Contenu des Slides

> Les slides suivants constituent un plan détaillé pour la présentation orale à l'encadrant.

---

### **Slide 1 — Titre**
**Titre :** Désagrégation Non-Intrusive de la Consommation Électrique par Modèles de Markov Cachés  
**Sous-titre :** Implémentation d'un pipeline NILM sur le dataset REFIT  
**Auteur :** Adam Ben Rhaiem  
**Encadrant :** [Nom de l'encadrant]  
**Date :** [Date de la présentation]

---

### **Slide 2 — Contexte et Motivation**
**Titre :** Pourquoi la désagrégation d'énergie ?

**Points clés :**
- 🏠 La facture d'électricité ne dit pas quel appareil consomme quoi
- 📊 Le NILM permet de dresser un **profil de consommation par appareil** sans capteur supplémentaire
- 💡 Applications : économies d'énergie, détection d'anomalies, smart home

**Visuel suggéré :** Schéma montrant compteur principal → signal agrégé → estimation par appareil

---

### **Slide 3 — Le Jeu de Données REFIT**
**Titre :** Dataset REFIT — 20 foyers instrumentés au Royaume-Uni

**Points clés :**
- 20 maisons, mesures toutes les **8 secondes**
- Colonnes : `Aggregate`, `Appliance1` à `Appliance9`
- Appareils différents selon les maisons → nécessite un mapping

**Tableau :**

| Maison | Appareils disponibles |
|---|---|
| House 3 | Toaster, Frigo, Congélateur, Sèche-linge, Lave-vaisselle, Lave-linge, TV, Micro-ondes, Bouilloire |
| House 9 | Frigo-Congélateur, Sèche-linge, Lave-linge, Lave-vaisselle, TV, Micro-ondes, Bouilloire, Hi-Fi, Chauffage |

---

### **Slide 4 — Architecture du Pipeline**
**Titre :** Pipeline NILM — Vue d'ensemble

**Visuel :** Diagramme en blocs (cf. section 2)

```
CSV brut → Prétraitement → Entraînement HMM → Viterbi → Évaluation
```

**Étapes :**
1. Prétraitement (nettoyage des signaux)
2. Entraînement d'un HMM Gaussien par appareil
3. Décodage par algorithme de Viterbi
4. Évaluation cross-house (Précision / Rappel / F1)

---

### **Slide 5 — Étape 1 : Prétraitement**
**Titre :** Nettoyage et préparation des signaux

**Points clés :**
- **Problèmes des données brutes :** valeurs aberrantes, lacunes, irrégularités temporelles
- **Filtre de Hampel :** détection et remplacement des pics (fenêtre 15 pts, seuil 3σ)
- **Interpolation :** combler les trous (max 5 points consécutifs)
- **Rééchantillonnage :** grille régulière de 8 secondes

**Visuel suggéré :** Graphique avant/après du filtre de Hampel sur un signal réel

---

### **Slide 6 — Étape 2 : Modèle de Markov Caché**
**Titre :** Modélisation par HMM Gaussien

**Points clés :**
- Chaque appareil = automate à états cachés discrets
- **2 états** : OFF / ON (bouilloire, micro-ondes, TV)
- **3 états** : OFF / LOW / HIGH (réfrigérateur)
- Émissions **gaussiennes** : chaque état émet une puissance selon une loi normale

**Schéma HMM :**
```
     p(ON→OFF)              p(OFF→ON)
OFF ─────────────────────► ON
 ◄─────────────────────────
Émission : N(μ_OFF, σ²_OFF)    Émission : N(μ_ON, σ²_ON)
```

**Entraînement :** Algorithme EM (Baum-Welch) sur les séries de puissance par appareil

---

### **Slide 7 — Étape 3 : Algorithme de Viterbi**
**Titre :** Décodage des séquences d'états cachés

**Points clés :**
- Problème : trouver la **séquence d'états la plus probable** étant donné les observations
- Algorithme de Viterbi : programmation dynamique sur le treillis HMM
- **Correction sémantique :** tri des états par moyenne de puissance → garantit OFF < LOW < HIGH

**Visuel suggéré :** Graphique temporel montrant signal de puissance + états décodés (OFF/ON)

---

### **Slide 8 — Évaluation Cross-House**
**Titre :** Transfert entre maisons — Entraîner sur House 9, tester sur House 3

**Principe :**
- Entraîner les modèles HMM sur la Maison 9
- Appliquer ces modèles sur la Maison 3 (jamais vue)
- Évaluer la capacité de généralisation

**Métriques :**
- **Précision** : instants ON correctement détectés / total détectés ON
- **Rappel** : instants ON correctement détectés / total vrais ON
- **F1-score** : équilibre précision/rappel

**Visuel suggéré :** Graphique barres groupées PRF par appareil

---

### **Slide 9 — Modes de Désagrégation**
**Titre :** Deux modes de fonctionnement

| Mode | Source du signal | Usage |
|---|---|---|
| **Sous-comptage** | Colonne réelle de l'appareil | Évaluation (vérité terrain disponible) |
| **NILM pur** | Signal agrégé uniquement | Déploiement réel (aucun capteur supplémentaire) |

**Mode NILM :** Pour chaque instant, cherche la combinaison d'états dont la somme ≈ agrégat mesuré

---

### **Slide 10 — Détection d'Événements**
**Titre :** Identification des transitions d'état

**Points clés :**
- Détection automatique des transitions : OFF→ON, ON→OFF, OFF→LOW, etc.
- Enregistrement : timestamp, état avant, état après, puissance
- Visualisation : fenêtre temporelle autour de chaque événement

**Visuel suggéré :** Exemple de détection d'allumage de bouilloire (pic rapide à ~2000W)

---

### **Slide 11 — Structure du Code**
**Titre :** Organisation du projet

```
run_nilm.py          ← Point d'entrée CLI
    │
    ├── preprocessing.py     ← Nettoyage des données
    ├── train_hmm.py         ← Entraînement HMM
    ├── disaggregate.py      ← Viterbi + désagrégation
    ├── refit_metadata.py    ← Mapping maison ↔ appareils
    └── plot_prf_metrics.py  ← Métriques PRF
```

**Tests :** `tests/test_mapping.py` — 5 classes de tests unitaires

**Dépendances :** `numpy`, `pandas`, `matplotlib`, `hmmlearn`, `scikit-learn`, `scipy`, `ruptures`

---

### **Slide 12 — Travaux de Recherche Complémentaires**
**Titre :** Algorithmes explorés — Code de recherche

| Domaine | Algorithmes testés |
|---|---|
| Détection d'événements | CUMSUM, CA-CFAR, segmentation binaire, seuil local, états stables |
| Prétraitement | Variantes du filtre de Hampel, interpolation |
| Désagrégation | Viterbi standard, Viterbi sparse |
| Clustering | Regroupement d'appareils par profil de consommation |

---

### **Slide 13 — Bilan et Perspectives**
**Titre :** Bilan de l'avancement et prochaines étapes

**Réalisé ✅ :**
- Pipeline complet de prétraitement (Hampel, interpolation, rééchantillonnage)
- Entraînement HMM Gaussien par appareil
- Décodage Viterbi avec correction sémantique
- Évaluation cross-house (Précision / Rappel / F1)
- Deux modes : sous-comptage et NILM pur
- Détection et visualisation des événements
- Tests unitaires du mapping REFIT

**En cours / Perspectives 🔄 :**
- Amélioration du mode NILM pur (Viterbi factoriel sur signal agrégé)
- Extension à plus d'appareils et de maisons
- Évaluation en énergie (MAE, RMSE) en plus du ON/OFF
- Interface de visualisation interactive

---

### **Slide 14 — Conclusion**
**Titre :** Conclusion

- Le projet implémente un **système NILM fonctionnel** basé sur les HMM Gaussiens
- La correction sémantique des états résout le problème d'**invariance par permutation** du HMM
- L'architecture modulaire permet un passage facile entre **évaluation** (sous-comptage) et **déploiement réel** (NILM pur)
- La généralisation cross-house valide la robustesse de l'approche

**Questions ?**

---

## Annexes

### A. Commandes d'utilisation

```bash
# Installation des dépendances
pip install -r Projet_NILM/requirements.txt

# Pipeline complet (train sur House_9, test sur House_3)
python Projet_NILM/run_nilm.py \
    --train-house House_9.csv \
    --test-house House_3.csv \
    --fridge-states 2 \
    --plot-preprocessing \
    --detect-events

# Calcul des métriques PRF
python Projet_NILM/plot_prf_metrics.py \
    --train-house House_9.csv \
    --test-house House_3.csv

# Tests unitaires
python -m pytest Projet_NILM/tests/
```

### B. Dépendances principales

| Bibliothèque | Version min | Rôle |
|---|---|---|
| `numpy` | ≥ 1.21 | Calcul numérique |
| `pandas` | ≥ 1.3 | Manipulation des séries temporelles |
| `matplotlib` | ≥ 3.4 | Visualisations |
| `hmmlearn` | ≥ 0.2.7 | Modèles HMM Gaussiens |
| `scikit-learn` | ≥ 0.24 | Métriques de classification |
| `scipy` | ≥ 1.7 | Calcul scientifique |
| `ruptures` | ≥ 1.1 | Détection de ruptures (optionnel) |

### C. Références

- **Dataset REFIT :** Murray, D. et al., "A publicly available dataset for energy disaggregation research," *ASHRAE Transactions*, 2015. [https://pureportal.strath.ac.uk](https://pureportal.strath.ac.uk)
- **NILM Survey :** Nalmpantis, C. & Vrakas, D., "Machine learning approaches for non-intrusive load monitoring," *Artificial Intelligence Review*, 2019.
- **HMM pour NILM :** Kolter, J.Z. & Jaakkola, T., "Approximate inference in additive factorial HMMs with application to energy disaggregation," *AISTATS*, 2012.
