# PRISMA — Documentation d'Architecture & Résumé du Dépôt

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)
[![ELN 2022](https://img.shields.io/badge/Standard-ELN%202022-critical)](https://www.hematology.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Dispositif Médical Logiciel (SaMD) de Grade Clinique**  
> Détection de la Maladie Résiduelle Minimale (MRD) dans les Leucémies Aiguës Myéloïdes (LAM) par approche *Different from Normal* (DfN — EuroFlow/ELN 2022) avec supervision humaine.

---

## Table des Matières

1. [Résumé Exécutif](#1-résumé-exécutif)
2. [Architecture Technique & Stack](#2-architecture-technique--stack)
3. [Le Pipeline Algorithmique](#3-le-pipeline-algorithmique--le-cœur-scientifique)
4. [L'Interface Utilisateur](#4-linterface-utilisateur--uxui)
5. [Cartographie du Dépôt](#5-cartographie-du-dépôt--directory-map)
6. [Installation & Lancement](#6-installation--lancement)
7. [Configuration](#7-configuration)
8. [Rétrospective Technique & Post-Mortem](#8-rétrospective-technique--post-mortem)

---

## 1. Résumé Exécutif

### Mission Principale

**PRISMA** est une application de bureau de grade clinique conçue pour la **détection et la quantification de la Maladie Résiduelle Minimale (MRD)** dans les Leucémies Aiguës Myéloïdes (LAM) à partir de fichiers `.FCS` issus de cytométrie en flux multiparamétrique.

### Problème Clinique Résolu

La MRD post-thérapeutique dans la LAM est l'un des facteurs pronostiques les plus puissants à ce jour. Une MRD ≥ 0.1% après consolidation prédit un risque de rechute significativement accru (ELN 2022). Cependant, sa détection de haute sensibilité est techniquement ardue pour plusieurs raisons :

- **Hétérogénéité phénotypique** : chaque leucémie possède un Immunophénotype Associé à la Leucémie (LAIP) unique.
- **Fonds de moelle normale variable** : les cellules souches hématopoïétiques saines post-chimiothérapie partagent parfois les mêmes marqueurs que les blastes résiduels.
- **Variabilité inter-instrument** : les effets "batch" entre acquisitions dégradent la reproductibilité.

**PRISMA** répond à ce triple défi en implémentant la stratégie clinique de référence **EuroFlow/ELN 2022** : la **Détection d'Anomalie (DfN — Different from Normal)**, couplée à un moteur de clustering non supervisé (FlowSOM) et une interface de **curation humaine** ("Human-in-the-loop") donnant le contrôle final au clinicien.

> **Objectif de performance** : Sensibilité ≥ 0.01% avec un seuil de positivité clinique à 0.1%, conformément aux standards ELN 2022.

---

## 2. Architecture Technique & Stack

### Technologies Clés

| Couche | Technologie | Rôle |
|--------|-------------|------|
| **Langage** | Python 3.10+ | Core, analyses, GUI |
| **Interface Graphique** | PyQt5 | Application bureau (Wizard 5 étapes) |
| **Clustering** | `flowsom` (Python) | Auto-organisation SOM + métaclustering |
| **Accélération GPU** | FlowSomGpu (module custom) | Fallback GPU → CPU automatique |
| **Correction de Batch** | `harmonypy` | Alignement inter-fichier (scatter uniquement) |
| **Optimisation** | `optuna` | Optimisation bayésienne des hyperparamètres MRD |
| **Données** | `pandas`, `numpy`, `anndata` | Manipulation & stockage matriciel |
| **Statistiques** | `scikit-learn` | GMM, RANSAC, métriques clustering |
| **Visualisation** | `plotly`, `matplotlib` | Radar interactif, heatmaps, grilles SOM |
| **Export** | `kaleido`, `ReportLab` | PNG haute résolution, PDF clinique |
| **Configuration** | YAML + dataclasses | Cascade defaults → YAML → CLI args |

### Architecture Globale

L'application suit une architecture **layered** stricte séparant les responsabilités :

```
┌──────────────────────────────────────────────────────────────────┐
│                     Points d'Entrée                              │
│   CLI (cli/main.py)  ·  GUI (launch_gui.py)  ·  Dev (run_pipeline.py) │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│               Gestion de Configuration                           │
│   PipelineConfig (pipeline_config.py)                            │
│   Cascade : default_config.yaml → mrd_config.yaml → CLI args     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│            Orchestrateur Principal (pipeline_executor.py)        │
│  FlowSOMPipeline · BatchPipeline                                 │
│                                                                  │
│  Phase 1 ─ CHARGEMENT     FCS → FlowSample                      │
│  Phase 2 ─ PRÉPROCESSING  PreprocessingService                   │
│  Phase 3 ─ CLUSTERING     ClusteringService + FlowSOM            │
│  Phase 4 ─ MRD            MRDCalculator (JF · Flo · ELN)        │
│  Phase 5 ─ VISUALISATION  Plots + Radar + Rapports               │
│  Phase 6 ─ EXPORT         FCS · CSV · JSON · HTML · PDF          │
│  Phase 7 ─ MAPPING        PopulationMappingService (optionnel)   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PipelineResult                                 │
│  data · metacluster_stats · mfi_matrix · gating_report          │
│  clustering_metrics · output_files · curated_mrd_*              │
└──────────────────────────────────────────────────────────────────┘
```

### Séparation des Couches

| Dossier | Responsabilité |
|---------|----------------|
| `config/` | Configuration, constantes ELN, dataclasses |
| `src/core/` | Algorithmes de bas niveau (gating, clustering, normalisation) |
| `src/analysis/` | Logique métier MRD (calcul, scoring, mapping) |
| `src/services/` | Orchestration de haut niveau (preprocessing, clustering, export) |
| `src/pipeline/` | Exécuteur principal + cache NBM |
| `src/io/` | Lecture/écriture FCS, CSV, JSON |
| `src/visualization/` | Plots, radar, rapports HTML/PDF |
| `src/models/` | Modèles de données (`FlowSample`, `PipelineResult`) |
| `gui/` | Interface PyQt5 (wizard, widgets, workers) |
| `cli/` | Point d'entrée CLI + parsers |
| `tests/` | Tests unitaires |

---

## 3. Le Pipeline Algorithmique — Le Cœur Scientifique

### 3.1 Préprocessing & Portes Biologiques

Avant tout clustering, chaque fichier `.FCS` passe par une chaîne de 4 portes séquentielles implémentées dans `src/core/auto_gating.py` et `src/core/gating.py` :

| Porte | Méthode | Paramètres |
|-------|---------|------------|
| **Cellules viables** | GMM (2-3 composantes) sur FSC-A × SSC-A | Déchets exclus automatiquement |
| **Singulets** | RANSAC sur FSC-A vs FSC-H + filtre MAD | Agrégats éliminés |
| **CD45** | KDE — détection du "pied de pic" adaptatif | Seuil auto-ajusté par densité |
| **CD34/Précurseurs** | Percentile configurable | Optionnel selon le panel |

La transformation spectrale est configurable dans `default_config.yaml` :
- **Logicle** (ELN standard, `T=2^18, M=4.5, W=0.5, A=0`) — par défaut
- **ArcSinh** (cofacteur=5.0)
- **Log10**

### 3.2 L'Alignement — Harmony Partiel

**Problème résolu** : L'application d'Harmony sur l'ensemble des marqueurs fusionnait mathématiquement les blastes tumoraux avec les cellules souches saines (cf. section 8 - Rétrospective).

**Solution implémentée** dans `src/services/clustering_service.py` et `config/default_config.yaml` :

```yaml
data_integration:
  harmony:
    enabled: false   # Désactivé par défaut
    markers_to_correct: ["FSC-A", "SSC-A", "CD45-A"]  # Scatter uniquement
```

> **Principe** : Harmony ne corrige que la "structure géométrique" (diffusion laser, taille cellulaire) mais **préserve intégralement les marqueurs de lignée** (CD34, CD117, HLA-DR, etc.) qui portent la signature tumorale. C'est le "Verrouillage Biologique".

### 3.3 Le Clustering FlowSOM

**`FlowSOMClusterer`** (`src/core/clustering.py`) entraîne une grille SOM auto-organisée :

- Taille de grille : 10×10 par défaut (configurable)
- `rlen` auto-calculé : `√N × 0.1`, borné entre [10, 100]
- Fallback automatique GPU → CPU

**Sélection du k optimal** (`src/core/metaclustering.py`) en 3 phases :
1. **Phase 1** : Score silhouette sur le codebook SOM (screening rapide)
2. **Phase 2** : Stabilité bootstrap (ARI entre k tirages aléatoires)
3. **Phase 3** : Score composite : `stabilité × 0.65 + silhouette × 0.35`

### 3.4 Le Moteur de Scoring MRD — Trois Méthodes

`src/analysis/mrd_calculator.py` implémente trois méthodes indépendantes, toutes activables simultanément (`method: "all"`) :

#### Méthode JF (Jabbour-Faderl) — Conservative
```
SI (% patho intra-nœud > 10%) ET (% sain du nœud / total_sain < 0.1%)
→ Nœud classé MRD positif
```
*Caractéristiques* : Exige une pureté locale forte. Excellente spécificité. Idéale pour les MRD faibles.

#### Méthode Flo — Ratio-Tolérant
```
SI (% patho intra-nœud / % sain intra-nœud > normal_marrow_multiplier)
→ Nœud classé MRD positif
```
*Caractéristiques* : Tolère les nœuds mixtes. Le `normal_marrow_multiplier` (défaut : 5.0) est un hyperparamètre optimisable par Optuna.

#### Méthode ELN 2022 (DfN) — Standard Clinique de Référence
```
SI (% patho > % sain dans le nœud)
ET (n_événements_dans_nœud >= 50)   ← Porte Clinique ELN (Hard-Stop)
ET (MRD_globale >= 0.1%)             ← Seuil de positivité clinique
→ Nœud classé MRD positif
```
*Caractéristiques* : Implémentation directe du standard EuroFlow/ELN 2022. Méthode recommandée en usage clinique.

### 3.5 Le Bouclier Topologique (Filtre JF / `normal_marrow_multiplier`)

Ce paramètre est le **gardien géométrique** du système : il empêche les nœuds SOM appartenant massivement à la moelle normale d'être étiquetés MRD, même si quelques cellules pathologiques s'y retrouvent par attraction topologique.

La valeur 5.0 signifie : un nœud n'est MRD que si les cellules pathologiques sont **5 fois plus représentées** que les cellules normales attendues dans ce nœud. Ce multiplicateur est optimisé par Optuna sur la cohorte d'entraînement avec pour objectif de **minimiser les Faux Positifs sur les échantillons NBM (Moelle Osseuse Normale)**.

### 3.6 La Porte Clinique ELN — Hard-Stop des 50 Cellules

Règle d'or implémentée dans `config/mrd_config.yaml` :

```yaml
eln_standards:
  min_cluster_events: 50    # Limite de Quantification (LoQ)
  clinical_positivity_pct: 0.1  # Seuil clinique ELN 2022 (%)
```

**Tout nœud contenant moins de 50 événements est forcé à MRD = 0**, quelle que soit la proportion pathologique mesurée. Cette règle élimine instantanément les micro-faux positifs techniques issus du bruit de fond de l'instrument. Impact immédiat et décisif sur la spécificité de la cohorte lors de son implémentation.

### 3.7 Le Filtre Phénotypique Biologique (Optionnel)

`src/analysis/blast_detection.py` implémente un score composite /10 inspiré du score d'Ogata, normalisé sur la moelle de référence via Z-scores :

| Marqueur | Contribution | Condition |
|----------|-------------|-----------|
| CD34 | +1.4 | Z > 1.0 (bright) |
| CD117 | +1.0 | Z > 0.5 |
| HLA-DR | +0.5 | Z > 0.3 |
| CD13 | +0.8 | Z > 0.3 |
| CD33 | +0.15 | Z > 0.2 |
| CD45 | -0.1 | Z < 0.0 (dim) |
| SSC | -3.0 | Z < 0.0 (low complexity) |

**Seuils de classification** :
- `BLAST_HIGH` ≥ 6.0 → Phénotype blastique fort (forte présomption)
- `BLAST_MODERATE` ≥ 3.0 → Atypique, surveillance clinique recommandée
- `BLAST_WEAK` > 0.0 → Signal subtil nécessitant curation experte
- `NON_BLAST` = 0.0 → Aucune signature blastique détectée

Une **distance de Mahalanobis** complémentaire (`score_nodes_mahalanobis()`) positionne chaque nœud dans l'espace multivarié de la moelle de référence, détectant les populations "globalement différentes du normal" au-delà de l'analyse marqueur par marqueur.

### 3.8 Sous-Echantillonnage Stratifié Optimal

Le ratio de génération de cellules NBM/Patho dans la grille SOM est configurable dans `default_config.yaml` :

```yaml
stratified_downsampling:
  enabled: true
  balance_ratio: 3.0   # 3x plus de cellules NBM que pathologique
```

Ce ratio est biologiquement justifié : dans un prélèvement post-traitement, les cellules normales dominent très largement. Surreprésenter la moelle normale dans la grille SOM assure que la **topologie du "normal" est bien apprise**, rendant les blastes résiduels véritablement "différents" dans l'espace SOM et renforçant l'approche DfN.

### 3.9 L'Optimisation Bayésienne (Optuna)

`optimize_optuna_mrd.py` définit un espace de recherche réduit aux **paramètres mathématiques uniquement** (après abandon du paradigme supervisé) :

- `normal_marrow_multiplier` : [1.0, 20.0] — force du bouclier topologique
- `d2_min_normal_threshold` : seuil de distance Mahalanobis minimale
- `min_cluster_events` : LoQ de quantification (centré autour de 50)

**Ce que l'IA ne touche plus** : les poids biologiques des marqueurs (w_CD34, w_CD117, etc.) qui sont désormais dictés par les standards ELN dans `mrd_config.yaml`. L'IA règle des "serrures mathématiques", non des critères biologiques.

---

## 4. L'Interface Utilisateur — UX/UI

### 4.1 Philosophie de Design

L'interface suit le système de design **"The Deep Medical Clarity"** :

- **Dark Mode systématique** (Catppuccin Mocha) : réduction de la fatigue visuelle en contexte clinique prolongé
- **Hiérarchie de l'information médicale** : les valeurs critiques (% MRD) sont toujours en premier plan
- **Progressivité** : architecture en wizard 5 étapes guidant l'utilisateur sans surcharge cognitive
- **Feedback temps réel** : chaque action déclenche une mise à jour visible immédiate (jauge, tableau)

**Palette** :
- Fond : `#1e1e2e` (Catppuccin Base)
- Accent : `#6366f1` (Indigo)
- Texte : `#e2e8f0`
- Succès / MRD Positif : `#a6e3a1` (Green)
- Danger / Alerte : `#f38ba8` (Red)

### 4.2 Architecture de la Fenêtre Principale

**PRISMA** (`FlowSomAnalyzerPro` dans `gui/main_window.py`) — QMainWindow avec wizard 5 étapes :

```
PRISMA / FlowSomAnalyzerPro (QMainWindow)
├── Sidebar (navigation, indicateurs d'état par étape)
│   ├── Etape 1 : Accueil
│   ├── Etape 2 : Import FCS
│   ├── Etape 3 : Parametrage
│   ├── Etape 4 : Execution
│   └── Etape 5 : Resultats
│
└── QStackedWidget (contenu actif)
    ├── HomeTab             — landing page, résumé de la méthode, CTA
    ├── ImportStep          — drag & drop FCS, sélection dossier NBM/Patho
    ├── SettingsStep        — paramètres SOM, MRD, gating, export
    ├── ExecutionStep       — console de logs + barre de progression
    └── ResultsStep         — onglets : Jauge MRD · Plots · Clusters · Radar
```

### 4.3 Widgets Clés

#### `MRDGauge` (`gui/widgets/mrd_gauge.py`)
Widget circulaire affichant le pourcentage MRD global. Couleur dynamique selon le seuil ELN (vert < 0.1%, rouge ≥ 0.1%). Mise à jour en temps réel lors de la curation.

#### `MRDNodeTable` (`gui/widgets/mrd_node_table.py`)
Tableau interactif listant chaque nœud SOM suspect avec :
- Pourcentage patho/sain intra-nœud
- Score phénotypique blastique
- Nombre d'événements
- Contrôles de curation (GARDER / ECARTER)

#### `ExpertFocusDialog` (`gui/dialogs/expert_focus_dialog.py`)
Dialogue de curation expert permettant :
- Visualisation du radar morphologique de chaque nœud suspect (Z-scores sur [-3.5, +4.5])
- Sélection exclusive GARDER (MRD) / ECARTER (Bruit Normal)
- Recalcul en temps réel de la jauge MRD globale

#### `MRDRadarChart` (`src/visualization/mrd_radar.py`)
Radar interactif Plotly normalisé sur la moelle de référence :
- Marqueurs ordonnés selon ELN 2022 (CD34 → CD117 → HLA-DR → ... → SSC)
- Chaque nœud suspect affiché en surimpression du profil NBM de référence
- L'anomalie "saute aux yeux" : le blaste résiduel sort visuellement de l'enveloppe normale

### 4.4 La Mécanique "Human-in-the-Loop"

C'est le composant le plus critique du point de vue clinique :

1. **Affichage des cartes décisionnelles** : chaque nœud SOM classé MRD positif est présenté avec son radar plot individuel.

2. **Sélecteur Exclusif** : Boutons `GARDER (MRD)` / `ECARTER (Bruit)` — un seul état possible par nœud.

3. **Recalcul Temps Réel** : chaque décision recalcule instantanément le % MRD global affiché sur la jauge.

4. **Injection dans `PipelineResult`** : la méthode `get_human_curated_results()` injecte les décisions humaines dans l'objet `PipelineResult` via les champs `curated_mrd_*`.

5. **Rapport PDF Curé** : le rapport final (`pdf_report.py`) affiche explicitement **"MRD Validée par l'Expert"** vs **"MRD Brute IA"**, assurant la traçabilité de la décision médicale pour l'audit clinique.

```
Algorithme DfN → N noeuds MRD positifs
                    ↓
          ExpertFocusDialog
          [Radar Plot par noeud]
                    ↓
    Médecin : GARDER (MRD) ou ECARTER (Bruit Normal)
                    ↓
    Recalcul temps réel : % MRD curé
                    ↓
    get_human_curated_results() → PipelineResult.curated_mrd_*
                    ↓
    PDF Report : "MRD Validée par l'Expert : X.XX%"
```

### 4.5 La Console de Logs & l'Audit Trail

**`LogConsole`** (`gui/widgets/log_console.py`) — QPlainTextEdit coloré avec niveaux de sévérité (DEBUG/INFO/WARNING/ERROR) et horodatage.

La **`GatingLogger`** dans `src/utils/logger.py` constitue un audit trail complet de toutes les décisions de gating automatique (nombre de cellules retirées par porte, seuils appliqués), exporté en JSON pour la traçabilité réglementaire.

---

## 5. Cartographie du Dépôt — Directory Map

```
flowsom_pipeline_pro/
│
├── launch_gui.py              ← Point d'entrée PyInstaller (production)
├── run_pipeline.py            ← Lanceur développement (sans pip install)
├── optimize_optuna_mrd.py     ← Optimisation bayésienne des hyperparamètres MRD
├── analyze_results.py         ← Utilitaire post-analyse de cohorte
├── setup.py                   ← Métadonnées package Python
│
├── config/                    ← CONFIGURATION — modifier ici en priorité
│   ├── default_config.yaml    ← ★ Config maître (toutes sections, commentée)
│   ├── mrd_config.yaml        ← ★ Paramètres MRD (seuils ELN, JF, Flo, BPF)
│   ├── pipeline_config.py     ← Dataclasses + chargeur en cascade
│   ├── constants.py           ← Constantes globales, palettes, seuils ELN 2022
│   └── panels/
│       └── aml_panel.yaml     ← Définitions de panel AML (marqueurs, canaux)
│
├── cli/                       ← INTERFACE LIGNE DE COMMANDE
│   ├── main.py                ← Orchestrateur CLI principal
│   ├── parsers.py             ← Définitions argparse
│   └── __init__.py
│
├── src/                       ← BIBLIOTHÈQUE CORE (logique métier)
│   │
│   ├── core/                  ← Algorithmes de bas niveau
│   │   ├── clustering.py      ← FlowSOM (GPU/CPU fallback)
│   │   ├── auto_gating.py     ← Gating adaptatif (GMM, KDE, RANSAC)
│   │   ├── gating.py          ← Gating manuel (percentile)
│   │   ├── metaclustering.py  ← Sélection k optimal (silhouette + stabilité)
│   │   ├── normalizers.py     ← Zscore, MinMax
│   │   └── transformers.py    ← Logicle, ArcSinh, Log10
│   │
│   ├── analysis/              ← ★ LOGIQUE MÉTIER MRD
│   │   ├── mrd_calculator.py  ← ★ Trois méthodes MRD (JF, Flo, ELN DfN)
│   │   ├── blast_detection.py ← Score phénotypique blastique (Ogata + Mahalanobis)
│   │   ├── population_mapping.py ← Mapping référence LAIP
│   │   └── statistics.py      ← Utilitaires statistiques
│   │
│   ├── services/              ← Orchestration de haut niveau
│   │   ├── clustering_service.py  ← ★ Markers → Stack → Harmony → FlowSOM
│   │   ├── preprocessing_service.py ← QC → Gating → Transform → Normalize
│   │   ├── export_service.py  ← Orchestration tous exports
│   │   └── population_mapping_service.py ← Mapping populations référence
│   │
│   ├── pipeline/              ← Exécuteur pipeline principal
│   │   ├── pipeline_executor.py ← ★ FlowSOMPipeline (7 phases)
│   │   ├── batch_pipeline.py  ← Mode batch (fichier par fichier)
│   │   ├── nbm_cache_manager.py ← Cache moelle normale (optimisation)
│   │   └── plotting_worker.py ← Tâches de plotting asynchrones
│   │
│   ├── io/                    ← Entrées/Sorties
│   │   ├── fcs_reader.py      ← Découverte et chargement FCS (via flowsom)
│   │   ├── fcs_writer.py      ← Export FCS compatible Kaluza
│   │   ├── csv_exporter.py    ← Export CSV (cellules, stats, MFI)
│   │   ├── json_exporter.py   ← Export JSON (métadonnées, logs gating)
│   │   ├── cluster_distribution_exporter.py ← Distribution TXT/CSV
│   │   └── patho_fcs_exporter.py ← FCS pathologique avec colonne Is_MRD
│   │
│   ├── models/                ← Modèles de données
│   │   ├── sample.py          ← FlowSample (fichier + métadonnées + données)
│   │   ├── pipeline_result.py ← ★ PipelineResult + ClusteringMetrics
│   │   └── gate_result.py     ← Audit trail gating
│   │
│   ├── visualization/         ← Visualisation & Rapports
│   │   ├── flowsom_plots.py   ← Grille SOM, heatmaps, MST
│   │   ├── gating_plots.py    ← Scatter plots avec portes (Matplotlib)
│   │   ├── mrd_radar.py       ← ★ Radar interactif Z-score (Plotly)
│   │   ├── html_report.py     ← Rapport HTML auto-contenu (Plotly.js embarqué)
│   │   ├── pdf_report.py      ← Rapport PDF clinique (ReportLab + Matplotlib)
│   │   ├── population_viz.py  ← Distribution populations
│   │   └── plot_helpers.py    ← Utilitaires communs plotting
│   │
│   ├── monitoring/
│   │   └── performance_monitor.py ← Surveillance CPU/RAM/GPU
│   │
│   ├── utils/
│   │   ├── logger.py          ← Logging + GatingLogger (audit trail)
│   │   ├── validators.py      ← Validation données cytométriques
│   │   ├── marker_harmonizer.py ← Harmonisation noms marqueurs inter-fichiers
│   │   ├── class_balancer.py  ← Sous-échantillonnage stratifié
│   │   └── kaleido_scope.py   ← Export PNG via Kaleido
│   │
│   └── exceptions.py          ← Exceptions custom (ClinicalMathError, etc.)
│
├── gui/                       ← INTERFACE GRAPHIQUE PyQt5
│   ├── main_window.py         ← ★ PRISMA UI (classe FlowSomAnalyzerPro)
│   ├── styles.py              ← Stylesheet Catppuccin Mocha
│   ├── workers.py             ← QThread workers (pipeline, plots)
│   ├── adapters/
│   │   └── mrd_adapter.py     ← Adaptateur résultats MRD → GUI
│   ├── dialogs/
│   │   ├── pipeline_dashboard.py ← Progress dialog + logs temps réel
│   │   └── expert_focus_dialog.py ← ★ Dialogue curation expert
│   ├── tabs/
│   │   └── home_tab.py        ← Onglet accueil
│   └── widgets/
│       ├── log_console.py     ← Console logs colorée
│       ├── mrd_gauge.py       ← ★ Jauge MRD circulaire dynamique
│       ├── mrd_node_table.py  ← ★ Tableau noeuds interactif (curation)
│       ├── settings_card.py   ← Conteneur settings groupés
│       └── toggle_switch.py   ← Toggle switch personnalisé
│
└── tests/                     ← Tests unitaires
    ├── test_mrd_calculator.py
    ├── test_clustering_utils.py
    └── test_gating.py
```

**Légende** : `★` = Fichier critique, lire en priorité pour comprendre le système.

---

## 6. Installation & Lancement

### Prérequis

- Python 3.10+
- Conda ou venv recommandé
- CUDA 11.x+ (optionnel, pour accélération GPU)

### Installation

```bash
# Cloner le dépôt
git clone <repo_url>
cd flowsom_pipeline_pro

# Créer l'environnement
conda create -n flowsom python=3.10
conda activate flowsom

# Installer les dépendances
pip install -e .
```

### Lancement GUI (Mode Production)

```bash
python launch_gui.py
```

### Lancement CLI (Mode Développement)

```bash
python run_pipeline.py \
  --patho_dir /chemin/vers/fichiers/patho \
  --sain_dir /chemin/vers/fichiers/nbm \
  --config config/default_config.yaml
```

### Optimisation des Hyperparamètres MRD

```bash
python optimize_optuna_mrd.py \
  --n_trials 200 \
  --cohort_dir /chemin/vers/cohorte
```

---

## 7. Configuration

### `config/default_config.yaml` — Configuration Maître

```yaml
# Transformation spectrale (ELN standard)
transform:
  method: "logicle"
  cofactor: 5.0

# Sous-échantillonnage stratifié (ratio NBM/Patho)
stratified_downsampling:
  enabled: true
  balance_ratio: 3.0  # 3x plus de NBM que de pathologique

# Correction de batch Harmony (scatter uniquement — Verrouillage Biologique)
data_integration:
  harmony:
    enabled: false
    markers_to_correct: ["FSC-A", "SSC-A", "CD45-A"]

# Grille SOM
flowsom:
  grid_size: 10
  n_metaclusters: 8
  learning_rate: 0.05
```

### `config/mrd_config.yaml` — Paramètres MRD

```yaml
mrd:
  enabled: true
  method: "all"  # Calculer les 3 méthodes simultanément

  # Bouclier Topologique (Méthode Flo)
  method_flo:
    normal_marrow_multiplier: 5.0

  # Filtre JF
  method_jf:
    max_normal_marrow_pct: 0.001   # 0.1%
    min_patho_cells_pct: 0.10      # 10%

  # Standards ELN 2022 (Hard-Stops)
  eln_standards:
    min_cluster_events: 50         # Limite de Quantification
    clinical_positivity_pct: 0.001 # 0.1%
```

---

## 8. Rétrospective Technique & Post-Mortem

*Ce chapitre retrace l'évolution architecturale du projet, de l'esquisse Proof-of-Concept jusqu'au SaMD de grade clinique. Il constitue une ressource essentielle pour tout développeur reprenant le projet afin de comprendre les **décisions d'architecture** et les **erreurs évitées à grand coût**.*

---

### 8.1 Phase 1 — Le Mur du Machine Learning Supervisé

La première architecture reposait sur un pipeline d'apprentissage supervisé classique :

> **Idée initiale** : Optuna trouve les poids optimaux de chaque marqueur (w_CD34, w_CD117, w_SSC…) sur la cohorte, de façon à maximiser la séparation blastes/normaux.

Cette approche s'est heurtée à trois obstacles fondamentaux et successifs.

#### A. Le Paradoxe d'Harmony

**Le problème** : Harmony, appliqué à l'ensemble des marqueurs, identifiait la signature blastique (CD34-bright, CD117+, CD45-dim) comme un "effet batch" à corriger. Il fusionnait mathématiquement les blastes massifs (ex : patient Lacroix, 25% MRD) avec les cellules souches hématopoïétiques normales des NBM de référence.

**Conséquence directe** : La leucémie disparaissait mathématiquement après correction. Le pipeline devenait aveugle à son propre signal cible.

**Solution architecturale** : Harmony Partiel — ne corriger que les paramètres physiques de l'instrument (FSC-A, SSC-A, CD45-A), jamais les marqueurs de lignée. Implémenté dans `clustering_service.py`, configurable dans `default_config.yaml`.

#### B. Le Mythe du LAIP Universel

**Le problème** : La LAM est biologiquement hétérogène. Il n'existe pas de "portrait-robot du blaste LAM" applicable à toute la cohorte. Optuna, en cherchant un tel portrait-robot, convergeait vers des stratégies absurdes :

> Exemple réel observé sur la cohorte : l'algorithme mettait le poids du SSC à -11 pour bloquer toutes les leucémies *sauf* celle du patient Lacroix (qui avait le phénotype dominant statistiquement la cohorte).

**Conséquence directe** : Taux de Faux Positifs bloqué à 4-11 patients, impossible à faire descendre. En captant un LAIP spécifique, l'algorithme "flashait" inévitablement sur les cellules souches saines en régénération post-chimiothérapie des patients en rémission.

**Solution architecturale** : Abandon du paradigme supervisé. Les poids biologiques sont désormais **figés** selon les standards ELN (CD34 : fort, CD45 : négatif, SSC : bas) dans `mrd_config.yaml`. Optuna ne touche plus qu'aux paramètres mathématiques.

#### C. Le Dilemme des Méthodes : Fortes MRD vs Faibles MRD

C'est la limite fondamentale observée lors de la synthèse de cohorte :

| Méthode | Fortes MRD (>5%) | Faibles MRD (<2%) | Faux Positifs NBM |
|---------|-----------------|-------------------|-------------------|
| **Mahalanobis Global (Hybride)** | Excellente — Lacroix 25% détectée | Insuffisante | ~15% FP |
| **Linéaire Topologique (JF/Flo)** | Effondrement — Lacroix sous-estimée à 9% | Excellente — Day Laurent 2.12% précis | Faible |

**Analyse** :
- La méthode Mahalanobis "balaie large" dans l'espace multivarié — efficace quand la moelle est massivement envahie, mais le bruit de fond génère des faux positifs sur les petites variations normales.
- La méthode topologique regarde le voisinage immédiat SOM — précise pour les faibles MRD, mais quand la moelle est envahie à 30%, la topologie du normal est tellement déformée que les filtres de proximité deviennent aveugles.

**Solution architecturale** : Implémentation des **trois méthodes en parallèle** avec rapport détaillé par méthode, laissant le clinicien choisir la valeur la plus pertinente selon le contexte (rechute massive vs MRD de surveillance).

---

### 8.2 Phase 2 — Le Pivot DfN (EuroFlow/ELN 2022)

L'abandon du supervisé naïf au profit de la **Détection d'Anomalie (DfN)** constitue le pivot architectural central du projet.

**Les quatre changements clés** :

1. **Harmony Partiel** — préservation de la signature tumorale (cf. ci-dessus).

2. **Verrouillage Biologique** — suppression de l'optimisation des poids biologiques dans `optimize_optuna_mrd.py`. Les poids sont dictés par la clinique, pas par l'IA.

3. **Nouvel espace Optuna** — l'optimisation bayésienne est recentrée sur les seuls paramètres mathématiques (`normal_marrow_multiplier`, `d2_min_normal_threshold`). L'IA règle des "serrures mathématiques", non des critères biologiques.

4. **Hard-Stop ELN** — la règle des 50 cellules minimum (`min_cluster_events: 50`) a éliminé instantanément les micro-faux positifs techniques. Impact immédiat et décisif sur la spécificité de la cohorte.

---

### 8.3 Phase 3 — Le Sous-Echantillonnage Stratifié Optimal

**Découverte** : Le ratio NBM/Patho dans la grille SOM est un hyperparamètre critique souvent négligé.

Un ratio de **3.0 (3× plus de cellules NBM que pathologique)** s'est révélé optimal pour :
- Assurer que la topologie SOM représente fidèlement l'hématopoïèse normale
- Rendre les blastes résiduels véritablement "différents" dans l'espace SOM
- Eviter la distorsion topologique en cas de forte MRD (qui sinon "écraserait" le normal dans la grille)

Ce ratio est biologiquement cohérent : dans un prélèvement post-traitement en rémission, les cellules normales dominent très largement. Le SOM doit apprendre cette normalité pour que l'anomalie soit détectable.

---

### 8.4 Phase 4 — Le "Human-in-the-Loop" : L'Interface de Curation

Même avec le meilleur DfN algorithmique, la biologie reste la biologie. Certains cas "borderlines" requièrent l'œil d'un expert pour distinguer :
- Une petite population de cellules souches CD34+ normales en régénération
- Un vrai résidu blastique CD34+ leucémique

**Solution architecturale** : Redonner le contrôle final au médecin via l'interface de curation.

| Composant | Fichier | Rôle |
|-----------|---------|------|
| `ExpertFocusDialog` | `gui/dialogs/expert_focus_dialog.py` | Dialogue de curation interactif |
| `MRDNodeTable` | `gui/widgets/mrd_node_table.py` | Vue tabulaire des noeuds suspects |
| `MRDGauge` | `gui/widgets/mrd_gauge.py` | Jauge MRD mise à jour en temps réel |
| `MRDRadarChart` | `src/visualization/mrd_radar.py` | Radar Z-score par noeud |
| `get_human_curated_results()` | `src/models/pipeline_result.py` | Injection décision humaine → PipelineResult |

---

### 8.5 Phase 5 — L'Oeil Expert : Suppression Manuelle de Clusters

Dernier niveau de raffinement : la possibilité pour l'expert de **supprimer manuellement des clusters d'aspect non-blastique** détectés par l'algorithme.

Cette fonctionnalité "brise" le semi-supervisé pur mais permet des gains majeurs de sensibilité et spécificité pour les cas difficiles :
- Populations granuleuses CD34+ pathologiques atypiques
- Clusters de monocytes activés post-chimiothérapie mimant un phénotype blastique
- Artefacts d'agrégation cellulaire résiduels

**Philosophie** : Le système est semi-supervisé par choix délibéré. L'algorithme propose, l'expert dispose. La traçabilité de la décision est assurée par `GatingLogger` et intégrée au rapport clinique final.

---

### 8.6 Bilan : Evolution Architecturale

```
Version 1.0 (PoC ML)                    Version Actuelle (SaMD DfN)
─────────────────────                    ───────────────────────────
Harmony global (tous marqueurs)    →     Harmony partiel (scatter only)
Poids marqueurs optimisés par IA   →     Poids dictés par ELN 2022
LAIP universel recherché           →     DfN : "Different from Normal"
Optuna = hématologue IA            →     Optuna = réglage serrures math
Pas de hard-stop cellulaire        →     Hard-stop 50 cellules (ELN LoQ)
Résultat brut IA                   →     Human-in-the-Loop + curation
1 méthode de scoring               →     3 méthodes parallèles (JF/Flo/ELN)
Rapport PDF basique                →     Rapport "MRD Validée Expert"
```

---

### 8.7 Limites Connues et Axes d'Amélioration

1. **Dépendance à la qualité du panel** : Le système requiert un panel multiparamétrique complet (CD34, CD117, HLA-DR, CD45, CD33, CD13, FSC, SSC). Les panels réduits dégradent la performance du filtre phénotypique.

2. **Homogénéité de la cohorte NBM** : La qualité de la moelle normale de référence détermine directement la baseline DfN. Des NBM hétérogènes (donneurs âgés, régénération post-chimio) peuvent élargir la zone normale et masquer de faibles MRD.

3. **Calibration inter-centre** : Les paramètres Logicle et les seuils de gating sont calibrés pour un instrument spécifique. Un déploiement multi-centre nécessiterait une recalibration instrumentale ou une extension du module Harmony.

4. **Validation clinique prospective** : Ce logiciel est un outil d'aide à la décision. Toute valeur MRD doit être interprétée en contexte clinique complet. La validation sur cohorte prospective multicentrique reste nécessaire avant toute utilisation décisionnelle autonome.

5. **Performance sur les rechutes massives** : La méthode topologique (JF/Flo) reste limitée pour les rechutes > 20-30% (déformation topologique du SOM). La méthode Mahalanobis ou un avis expert direct est alors préférable.

6. **Absence de normalisation absolue des comptages** : Le % MRD est calculé sur les cellules CD45+ analysées, non sur le nombre absolu de cellules par mL, ce qui peut varier selon la qualité du prélèvement.

---

## Auteurs & Contexte

Développé dans le cadre d'une initiative de recherche en hématologie clinique, ce projet illustre la convergence entre les méthodes de bioinformatique computationnelle (clustering non supervisé, optimisation bayésienne) et les standards cliniques de pointe (ELN 2022, EuroFlow).

**Stack** : Python · PyQt5 · FlowSOM · Optuna · Harmony · Plotly · Pandas · Scikit-learn

---

*Document généré — Architecture correspondant à la branche `main` courante.*
