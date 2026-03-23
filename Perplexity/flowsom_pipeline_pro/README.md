# FlowSOM Analysis Pipeline Pro

Pipeline d'analyse de cytométrie en flux pour la **Maladie Résiduelle Détectable (MRD)** en hématologie — conforme aux recommandations **ELN 2022**.

Architecture modulaire production-ready. **Version 2.0.**

---

## Table des matières

- [FlowSOM Analysis Pipeline Pro](#flowsom-analysis-pipeline-pro)
  - [Table des matières](#table-des-matières)
  - [Présentation](#présentation)
  - [Fonctionnalités](#fonctionnalités)
    - [Pré-traitement](#pré-traitement)
    - [Pré-gating](#pré-gating)
    - [Clustering FlowSOM](#clustering-flowsom)
    - [Analyse MRD](#analyse-mrd)
    - [Export](#export)
  - [Architecture](#architecture)
  - [Installation](#installation)
    - [Prérequis](#prérequis)
    - [Installation de base](#installation-de-base)
    - [Installation avec toutes les dépendances](#installation-avec-toutes-les-dépendances)
    - [Options d'installation](#options-dinstallation)
  - [Démarrage rapide](#démarrage-rapide)
    - [1. Préparer les données](#1-préparer-les-données)
    - [2. Lancer l'analyse](#2-lancer-lanalyse)
    - [3. Consulter les résultats](#3-consulter-les-résultats)
  - [Utilisation](#utilisation)
    - [Via la ligne de commande (CLI)](#via-la-ligne-de-commande-cli)
    - [Via l'API Python](#via-lapi-python)
    - [Configuration programmatique](#configuration-programmatique)
    - [Via un fichier de configuration YAML](#via-un-fichier-de-configuration-yaml)
  - [Référence CLI](#référence-cli)
  - [Fichier de configuration YAML](#fichier-de-configuration-yaml)
  - [Sorties produites](#sorties-produites)
    - [Distribution Sain/Patho par cluster](#distribution-sainpatho-par-cluster)
    - [Colonnes ajoutées aux fichiers FCS](#colonnes-ajoutées-aux-fichiers-fcs)
  - [Seuils cliniques ELN 2022](#seuils-cliniques-eln-2022)
    - [Poids de détection des blastes](#poids-de-détection-des-blastes)
  - [Dépendances](#dépendances)
    - [Obligatoires](#obligatoires)
    - [Optionnelles](#optionnelles)
  - [Structure du projet](#structure-du-projet)
  - [Notes importantes](#notes-importantes)
    - [Reproductibilité](#reproductibilité)
    - [Vérifications automatiques](#vérifications-automatiques)
    - [Transformation recommandée](#transformation-recommandée)

---

## Présentation

**FlowSOM Analysis Pipeline Pro** automatise l'analyse cytométrique de moelle osseuse selon le protocole **LAIP MRD**, en intégrant :

- La construction d'un **MST (Minimum Spanning Tree) FlowSOM** sur des moelles normales (NBM) de référence
- Le **pré-gating automatique** ou manuel des populations cellulaires (débris, doublets, CD45+, CD34+)
- La **détection de MRD** par comparaison du profil patient à la référence NBM
- Le calcul de **scores de blastes** selon les marqueurs ELN 2022
- L'export complet des résultats (FCS, CSV, JSON, graphiques PNG/PDF)

> **Contexte clinique** : La MRD en LAM (Leucémie Aiguë Myéloïde) est mesurée par la fréquence des cellules aberrantes (LAIP) rapportée aux cellules leucocytaires totales. Le seuil de positivité ELN 2022 est un fold-change ≥ 1.9× par rapport à la fréquence observée dans ≥ 15 moelles normales poolées.

---

## Fonctionnalités

### Pré-traitement

| Étape | Description |
|---|---|
| **Lecture FCS** | Support natif via `flowio`, compensation automatique si `$SPILL` présent |
| **Transformation** | Logicle (défaut), arcsinh, log10 — appliquée uniquement aux canaux fluorescence |
| **Normalisation** | Z-score (défaut) ou Min-Max |
| **Downsampling** | 50 000 cell./fichier, 1 M cell. total (configurable) |

### Pré-gating

Deux modes disponibles :

- **`auto` (défaut)** : Modèles de mélange gaussien (GMM) + filtre RANSAC pour détection robuste des seuils
- **`manual`** : Seuils fixes basés sur des percentiles configurables

| Gate | Description |
|---|---|
| **Viables** | Exclusion des débris hors fenêtre FSC/SSC |
| **Singlets** | Sélection des singlets via ratio FSC-A / FSC-H |
| **CD45+** | Sélection des leucocytes (CD45 positif) |
| **CD34+** | Sélection des progéniteurs (optionnel) |

**Paramètres GMM/RANSAC configurables via YAML** (`pregate_advanced`) :

| Paramètre | Défaut | Description |
|---|---|---|
| `gmm_max_samples` | `200 000` | Plafond de sous-échantillonnage avant fitting GMM. Réduire pour accélérer sur grands datasets ; augmenter pour plus de précision. |
| `ransac_r2_threshold` | `0.85` | R² minimum sur les inliers RANSAC. En dessous → fallback automatique vers gating ratio FSC-A/FSC-H. |
| `ransac_mad_factor` | `3.0` | Facteur multiplicatif appliqué à la MAD (écart absolu médian) pour définir le seuil d'exclusion des doublets (`médiane + N × MAD`). |

### Clustering FlowSOM

- Grille SOM 2D (défaut 10×10 = 100 nodes) - réglable
- Métaclustering par consensus (hiérarchique agglomératif)
- **Auto-clustering** : recherche du nombre optimal de clusters (5–35) par score composite :
  - Phase 1 — Silhouette sur le codebook SOM (élimination rapide)
  - Phase 2 — Stabilité bootstrap (ARI sur 10 ré-échantillonnages)
  - Phase 3 — Score composite : `stabilité × 0.65 + silhouette × 0.35`
- Accélération **GPU optionnelle** (CuPy / RAPIDS / FlowSomGpu)
- Reproductibilité garantie : `seed=42` par défaut

### Analyse MRD

- **Détection de blastes** : score par node basé sur les poids ELN (CD34=+3, CD117=+2.5, CD45=−2, HLA-DR=+1.5…)
- **Évaluation MRD** : comparaison fold-change patient vs NBM de référence
- **Mapping de populations** : assignation automatique de chaque metacluster à une population connue (granulo, B, T/NK, plasmocytes, hématogones)
- **Tests statistiques** : Mann-Whitney U, Kolmogorov-Smirnov, fold-change par cluster

### Export

- Fichiers **FCS** avec colonnes de clustering ajoutées (compatible Kaluza)
- **CSV** : cellules complètes, statistiques par cluster, MFI par marqueur/metacluster
- **Distribution Sain/Patho** : tableau CSV + rapport texte ASCII de la représentation cellulaire par nœud SOM et par métacluster, triée par % Patho décroissant (voir [section dédiée](#distribution-sainpatho-par-cluster))
- **JSON** : log de gating, métadonnées du run, configuration
- **Graphiques** : heatmap MFI, distribution des metaclusters, UMAP, plots de gating

---

## Architecture

```
flowsom_pipeline_pro/
├── config/                   # Configuration centralisée
│   ├── constants.py          # Seuils cliniques ELN 2022, paramètres par défaut
│   ├── pipeline_config.py    # PipelineConfig (dataclass + from_yaml / from_args)
│   └── default_config.yaml   # Valeurs par défaut du pipeline
│
├── src/
│   ├── core/                 # Logique algorithmique pure (sans état)
│   │   ├── transformers.py   # arcsinh, logicle, log10
│   │   ├── normalizers.py    # z-score, min-max
│   │   ├── gating.py         # Pré-gating manuel (PreGating)
│   │   ├── auto_gating.py    # Pré-gating GMM/RANSAC (AutoGating)
│   │   ├── clustering.py     # FlowSOMClusterer (GPU→CPU fallback)
│   │   └── metaclustering.py # Recherche automatique du k optimal
│   │
│   ├── models/               # Structures de données (dataclasses)
│   │   ├── sample.py         # FlowSample
│   │   ├── gate_result.py    # GateResult
│   │   └── pipeline_result.py# PipelineResult + ClusteringMetrics
│   │
│   ├── utils/
│   │   ├── logger.py         # GatingLogger (export JSON)
│   │   └── validators.py     # Vérifications NaN, compensation, équilibre…
│   │
│   ├── io/                   # Entrées/Sorties fichiers
│   │   ├── fcs_reader.py     # Lecture FCS → FlowSample
│   │   ├── fcs_writer.py     # Export FCS + colonnes clustering
│   │   ├── csv_exporter.py   # Export statistiques CSV
│   │   ├── cluster_distribution_exporter.py  # Distribution Sain/Patho par cluster (TXT + CSV)
│   │   └── json_exporter.py  # Export métadonnées JSON
│   │
│   ├── visualization/        # Graphiques matplotlib (thème sombre)
│   │   ├── plot_helpers.py   # Utilitaires de mise en forme
│   │   ├── gating_plots.py   # Plots de pré-gating
│   │   └── flowsom_plots.py  # Heatmap MFI, UMAP, tailles metaclusters
│   │
│   ├── analysis/             # Analyses biologiques
│   │   ├── population_mapping.py  # Assignation populations → nodes
│   │   ├── blast_detection.py     # Score de blastes ELN 2022
│   │   └── statistics.py          # Tests stat + évaluation MRD
│   │
│   ├── services/             # Orchestration des couches core + io
│   │   ├── preprocessing_service.py  # Chaîne QC → gating → transform → normalize
│   │   ├── clustering_service.py     # Sélection marqueurs → FlowSOM → DataFrame
│   │   └── export_service.py         # ExportService (FCS, CSV, plots, JSON)
│   │
│   └── pipeline/
│       └── pipeline_executor.py  # FlowSOMPipeline.execute() — 6 étapes
│
├── cli/                      # Interface ligne de commande
│   ├── parsers.py            # build_argument_parser()
│   └── main.py               # Point d'entrée main()
│
└── setup.py                  # Packaging + entry_points CLI
```

Le pipeline suit une **architecture en couches** : chaque couche ne dépend que des couches inférieures, garantissant testabilité et maintenabilité.

---

## Installation

### Prérequis

- Python ≥ 3.10
- pip

### Installation de base

```bash
# Cloner ou copier le dossier flowsom_pipeline_pro
cd /chemin/vers/flowsom_pipeline_pro

# Installation en mode développement (recommandé)
pip install -e .
```

### Installation avec toutes les dépendances

```bash
# Toutes les dépendances (sans GPU)
pip install -e ".[full]"

# + accélération GPU (nécessite CUDA + CuPy)
pip install -e ".[full,gpu]"
```

### Options d'installation

| Extra | Contenu | Commande |
|---|---|---|
| `flowsom` | Algorithme FlowSOM (saeyslab) | `pip install -e ".[flowsom]"` |
| `fcs` | Lecture FCS via flowio | `pip install -e ".[fcs]"` |
| `fcs_export` | Export FCS avec fcswrite | `pip install -e ".[fcs_export]"` |
| `logicle` | Transformation logicle via FlowKit | `pip install -e ".[logicle]"` |
| `pytometry` | Transformations cytométriques avancées | `pip install -e ".[pytometry]"` |
| `gpu` | Accélération CuPy/RAPIDS | `pip install -e ".[gpu]"` |
| `reports` | Export PDF via ReportLab | `pip install -e ".[reports]"` |
| `full` | Tout sauf GPU | `pip install -e ".[full]"` |

---

## Démarrage rapide

### 1. Préparer les données

```
Data/
├── Moelle normale/    ← ≥ 15 fichiers FCS de moelles normales (NBM)
└── Patho/             ← Fichiers FCS patients (optionnel)
```

### 2. Lancer l'analyse

```bash
# Analyse NBM seul
flowsom-analyze --healthy-folder "Data/Moelle normale" --output Results

# Analyse comparée NBM vs Patient
flowsom-analyze \
    --healthy-folder "Data/Moelle normale" \
    --patho-folder "Data/Patho" \
    --compare-mode \
    --output Results_MRD
```

### 3. Consulter les résultats

```
Results/
├── fcs/        ← Fichiers FCS avec clusters ajoutés
├── csv/        ← Statistiques, MFI, données cellules
├── plots/      ← Graphiques PNG (gating, heatmap, UMAP)
└── other/      ← Log de gating JSON, métadonnées du run
```

---

## Utilisation

### Via la ligne de commande (CLI)

Après installation, la commande `flowsom-analyze` est disponible globalement :

```bash
# Auto-détection du fichier de configuration
flowsom-analyze

# Fichier de configuration explicite
flowsom-analyze --config config_flowsom.yaml

# Dossier NBM uniquement
flowsom-analyze --healthy-folder "Data/NBM" --output Results

# Mode comparaison complet
flowsom-analyze \
    --healthy-folder "Data/NBM" \
    --patho-folder "Data/Patho" \
    --compare-mode \
    --output Results

# FlowSOM personnalisé — grille 15×15, 20 métaclusters
flowsom-analyze \
    --healthy-folder "Data/NBM" \
    --xdim 15 --ydim 15 \
    --n-metaclusters 20 \
    --n-iterations 20 \
    --seed 42

# Auto-clustering (k optimal automatique entre 5 et 35)
flowsom-analyze \
    --healthy-folder "Data/NBM" \
    --auto-clustering \
    --min-clusters 5 \
    --max-clusters 35

# Transformation arcsinh avec cofacteur 150 (données non-compensées)
flowsom-analyze \
    --healthy-folder "Data/NBM" \
    --transform arcsinh \
    --cofactor 150

# Sans GPU, sortie silencieuse
flowsom-analyze \
    --healthy-folder "Data/NBM" \
    --no-gpu \
    --quiet
```

### Via l'API Python

```python
from flowsom_pipeline_pro import FlowSOMPipeline, PipelineConfig

# Depuis un fichier YAML
config = PipelineConfig.from_yaml("config_flowsom.yaml")
result = FlowSOMPipeline(config).execute()

# Consultation des résultats
print(result.summary())
print(f"Cellules analysées : {result.n_cells:,}")
print(f"Métaclusters       : {result.n_metaclusters}")
print(f"Silhouette         : {result.clustering_metrics.silhouette_score:.3f}")

# Export CSV
result.export_csv("mes_resultats.csv")
result.export_metadata("metadata_run.json")
```

### Configuration programmatique

```python
from flowsom_pipeline_pro import PipelineConfig, FlowSOMPipeline

config = PipelineConfig()

# Chemins
config.paths.healthy_folder = "Data/Moelle normale"
config.paths.patho_folder   = "Data/Patho"
config.paths.output_dir     = "Results"

# FlowSOM
config.flowsom.xdim          = 12
config.flowsom.ydim          = 12
config.flowsom.n_metaclusters = 20
config.flowsom.seed           = 42

# Pré-gating automatique (GMM)
config.pregate.mode    = "auto"
config.pregate.viable  = True
config.pregate.singlets = True
config.pregate.cd45    = True
config.pregate.cd34    = False

# Transformation logicle (recommandée)
config.transform.method          = "logicle"
config.transform.apply_to_scatter = False

# Désactiver le GPU
config.gpu.enabled = False

result = FlowSOMPipeline(config).execute()
```

### Via un fichier de configuration YAML

```bash
# Auto-détection : place le fichier dans le répertoire courant
cp default_config.yaml config_flowsom.yaml
# Éditer config_flowsom.yaml...
flowsom-analyze
```

---

## Référence CLI

```
usage: flowsom-analyze [-h] [--config PATH] [--healthy-folder DIR]
                       [--patho-folder DIR] [--output DIR]
                       [--compare-mode | --no-compare-mode]
                       [--no-pregate-viable] [--no-pregate-singlets]
                       [--no-pregate-cd45] [--pregate-cd34]
                       [--pregate-mode {auto,manual}]
                       [--xdim N] [--ydim N] [--n-metaclusters N]
                       [--learning-rate F] [--sigma F] [--n-iterations N]
                       [--seed N] [--auto-clustering]
                       [--min-clusters N] [--max-clusters N]
                       [--transform {arcsinh,logicle,log10,none}]
                       [--cofactor F] [--normalize {zscore,minmax,none}]
                       [--no-downsample] [--max-cells-per-file N]
                       [--max-cells-total N]
                       [--no-save-plots] [--plot-format {png,pdf,svg}]
                       [--dpi N] [--no-gpu] [-v | -q]
```

| Argument | Défaut | Description |
|---|---|---|
| `--config`, `-c` | auto-détecté | Fichier YAML de configuration |
| `--healthy-folder` | — | **Obligatoire** — Dossier FCS sains (NBM) |
| `--patho-folder` | — | Dossier FCS pathologiques (mode comparaison) |
| `--output`, `-o` | `Results` | Dossier de sortie |
| `--compare-mode` | `true` | Mode comparaison Sain vs Pathologique |
| `--no-pregate-viable` | — | Désactiver le gate viables |
| `--no-pregate-singlets` | — | Désactiver le gate singlets |
| `--no-pregate-cd45` | — | Désactiver le gate CD45+ |
| `--pregate-cd34` | désactivé | Activer le gate CD34+ |
| `--pregate-mode` | `auto` | Mode gating : `auto` (GMM) ou `manual` |
| `--xdim` | `10` | Dimension X de la grille SOM |
| `--ydim` | `10` | Dimension Y de la grille SOM |
| `--n-metaclusters` | `8` | Nombre de métaclusters |
| `--n-iterations` | `10` | Itérations d'entraînement SOM |
| `--seed` | `42` | Graine aléatoire (reproductibilité) |
| `--auto-clustering` | désactivé | Recherche automatique du k optimal |
| `--min-clusters` | `5` | Minimum pour l'auto-clustering |
| `--max-clusters` | `35` | Maximum pour l'auto-clustering |
| `--transform` | `logicle` | Transformation : `arcsinh`, `logicle`, `log10`, `none` |
| `--cofactor` | `5.0` | Cofacteur arcsinh |
| `--normalize` | `zscore` | Normalisation : `zscore`, `minmax`, `none` |
| `--no-downsample` | — | Désactiver le downsampling |
| `--max-cells-per-file` | `50000` | Cellules max par fichier |
| `--max-cells-total` | `1000000` | Cellules max total |
| `--plot-format` | `png` | Format graphiques : `png`, `pdf`, `svg` |
| `--dpi` | `300` | Résolution graphiques |
| `--no-gpu` | — | Désactiver l'accélération GPU |
| `--verbose`, `-v` | — | Mode DEBUG |
| `--quiet`, `-q` | — | Mode WARNING uniquement |

---

## Fichier de configuration YAML

Le fichier `config_flowsom.yaml` est auto-détecté dans le répertoire courant ou dans le répertoire du script. Les arguments CLI ont **priorité absolue** sur le YAML.

```yaml
# config_flowsom.yaml — Configuration complète

paths:
  healthy_folder: "Data/Moelle normale"
  patho_folder:   "Data/Patho"
  output_dir:     "Results"

analysis:
  compare_mode: true

pregate:
  apply:                true
  mode:                 "auto"       # auto (GMM) | manual (percentiles)
  mode_blastes_vs_normal: true
  viable:               true
  singlets:             true
  cd45:                 true
  cd34:                 false

pregate_advanced:
  debris_min_percentile:    1.0
  debris_max_percentile:    99.0
  doublets_ratio_min:       0.6
  doublets_ratio_max:       1.4
  cd45_threshold_percentile: 5.0
  cd34_threshold_percentile: 85.0
  cd34_use_ssc_filter:      true
  cd34_ssc_max_percentile:  60.0
  # Paramètres GMM/RANSAC (mode auto uniquement)
  gmm_max_samples:          200000  # Plafond avant sous-échantillonnage GMM
  ransac_r2_threshold:      0.85    # R² min RANSAC avant fallback ratio
  ransac_mad_factor:        3.0     # médiane + N×MAD pour seuil doublets

flowsom:
  xdim:          10
  ydim:          10
  rlen:          "auto"     # "auto" ou entier
  n_metaclusters: 8
  learning_rate: 0.05
  sigma:         1.5
  n_iterations:  10
  seed:          42

auto_clustering:
  enabled:                  false
  min_clusters:             5
  max_clusters:             35
  n_bootstrap:              10
  sample_size_bootstrap:    20000
  min_stability_threshold:  0.75
  weight_stability:         0.65
  weight_silhouette:        0.35

transform:
  method:           "logicle"   # arcsinh | logicle | log10 | none
  cofactor:         5.0
  apply_to_scatter: false

normalize:
  method: "zscore"              # zscore | minmax | none

markers:
  exclude_scatter:    true
  exclude_additional: []        # ex: ["ViaDye", "Dump"]

downsampling:
  enabled:            true
  max_cells_per_file: 50000
  max_cells_total:    1000000

visualization:
  save_plots:   true
  plot_format:  "png"           # png | pdf | svg
  dpi:          300

gpu:
  enabled: true

logging:
  level: "INFO"                 # DEBUG | INFO | WARNING | ERROR

# Distribution Sain/Patho par cluster — export automatique
export_cluster_distribution:
  enabled: true
  level: "both"                 # "node" (noeuds SOM) | "metacluster" | "both"
  sort_by: "pct_patho_in_cluster"  # Colonne de tri (toute colonne numérique)
  ascending: false              # false = décroissant (patho enrichi en premier)
  txt_enabled: true             # Rapport texte ASCII
  csv_enabled: true             # CSV prêt Excel / R
  txt_decimal_places: 1         # Précision dans le rapport texte (ex: 92.7%)
  csv_decimal_places: 3         # Précision dans le CSV (ex: 92.747%)
  sain_labels: ["Sain", "Normal", "NBM", "Healthy", "Moelle normale"]
  patho_labels: ["Pathologique", "Patho", "AML", "Disease"]
```

---

## Sorties produites

```
Results/
├── fcs/
│   ├── cells_clustered.fcs          # Toutes les cellules + colonnes FlowSOM
│   └── cells_clustered_kaluza.fcs   # Variante compatible Kaluza
│
├── csv/
│   ├── cells_complete.csv                         # DataFrame complet (1 ligne = 1 cellule)
│   ├── cluster_statistics.csv                     # Statistiques par metacluster (n, %, MFI)
│   ├── mfi_matrix.csv                             # Matrice MFI (marqueurs × metaclusters)
│   ├── per_file_summary.csv                       # Résumé par fichier FCS source
│   ├── cluster_distribution_nodes_<ts>.csv        # Distribution Sain/Patho par nœud SOM
│   └── cluster_distribution_metaclusters_<ts>.csv # Distribution Sain/Patho par métacluster
│
├── plots/
│   ├── gating_overview.png          # Vue d'ensemble du gating (FSC/SSC)
│   ├── gating_debris.png            # Gate débris
│   ├── gating_singlets.png          # Gate singlets
│   ├── gating_cd45.png              # Gate CD45+
│   ├── gating_cd34.png              # Gate CD34+ (si activé)
│   ├── flowsom_mfi_heatmap.png      # Heatmap MFI par metacluster
│   ├── flowsom_metacluster_sizes.png# Distribution des tailles de metaclusters
│   └── flowsom_umap.png             # Projection UMAP (si disponible)
│
└── other/
    ├── cluster_distribution_<ts>.txt        # Rapport texte ASCII (nœuds SOM + métaclusters)
    ├── gating_log_<timestamp>.json          # Log détaillé de chaque étape de gating
    └── analysis_metadata.json              # Métadonnées complètes du run (config, métriques, durée)
```

### Distribution Sain/Patho par cluster

Les fichiers `cluster_distribution_*` récapitulent pour chaque nœud SOM (grille fine, ex. 100 nœuds pour 10×10) et chaque métacluster (regroupement) :

| Colonne | Description |
|---|---|
| `cluster_id` | Identifiant du nœud SOM ou du métacluster |
| `metacluster` | Métacluster majoritaire du nœud (vue nœud uniquement) |
| `n_total` | Nombre total de cellules dans ce cluster |
| `pct_total` | % de l'ensemble des cellules |
| `n_sain` | Nombre de cellules Sain dans le cluster |
| `pct_sain_in_cluster` | % des cellules du cluster qui sont Sain |
| `pct_sain_of_sain` | % des cellules Sain totales qui se trouvent dans ce cluster |
| `n_patho` | Nombre de cellules Pathologiques dans le cluster |
| `pct_patho_in_cluster` | **% des cellules du cluster qui sont Patho** (colonne de tri principale) |
| `pct_patho_of_patho` | % des cellules Patho totales qui se trouvent dans ce cluster |

**Tri par défaut** : `pct_patho_in_cluster` décroissant — les clusters les plus enrichis en cellules pathologiques apparaissent en premier.

Le rapport texte `.txt` contient les deux tableaux (métacluster + nœud) avec alignement ASCII, lisible directement dans un terminal ou un éditeur.

### Colonnes ajoutées aux fichiers FCS

| Colonne | Type | Description |
|---|---|---|
| `FlowSOM_node` | int | Index du node SOM (0 – xdim×ydim−1) |
| `FlowSOM_metacluster` | int | Index du métacluster |
| `Condition` | str | `"Healthy"` ou `"Pathological"` |
| `Source_file` | str | Nom du fichier FCS source |

---

## Seuils cliniques ELN 2022

Ces valeurs sont figées dans [config/constants.py](config/constants.py) et ne doivent pas être modifiées sans validation biomédicale.

| Constante | Valeur | Signification |
|---|---|---|
| `MRD_LOD` | `9 × 10⁻⁵` (0.009 %) | Limite de détection MRD |
| `MRD_LOQ` | `5 × 10⁻⁵` (0.005 %) | Limite de quantification MRD |
| `NBM_FREQ_MAX` | `1.1 %` | Fréquence CD34+ maximale en moelle normale |
| `FOLD_CHANGE_MRD` | `1.9×` | Seuil fold-change FU/NBM pour positivité MRD |
| `MIN_EVENTS_PER_NODE` | `17` | Minimum d'événements par node FlowSOM (ELN) |

### Poids de détection des blastes

| Marqueur | Poids | Rationnel |
|---|---|---|
| CD34 | +3.0 | Marqueur majeur de progéniteurs immatures |
| CD117 (c-kit) | +2.5 | Récepteur tyrosine kinase des précurseurs myéloïdes |
| CD45 | −2.0 | Faible en blastes, élevé en lymphocytes matures |
| HLA-DR | +1.5 | Exprimé sur précurseurs myéloïdes et blast |
| CD33 | +1.0 | Antigène myéloïde précoce |
| CD13 | +0.5 | Antigène myéloïde modérément précoce |
| CD19 / CD3 | −1.5 | Lignées B et T (exclusion) |
| SSC | −1.0 | Granularité faible dans les blastes |

---

## Dépendances

### Obligatoires

| Package | Version min | Usage |
|---|---|---|
| `numpy` | ≥ 1.24 | Calcul matriciel |
| `pandas` | ≥ 2.0 | Manipulation de données tabulaires |
| `scikit-learn` | ≥ 1.3 | Clustering, métriques de qualité |
| `scipy` | ≥ 1.11 | Tests statistiques |
| `matplotlib` | ≥ 3.7 | Visualisations |
| `seaborn` | ≥ 0.12 | Heatmaps |
| `anndata` | ≥ 0.10 | Structure de données cellules × marqueurs |
| `pyyaml` | ≥ 6.0 | Lecture de configuration YAML |
| `tqdm` | ≥ 4.65 | Barres de progression |

### Optionnelles

| Package | Installation | Usage |
|---|---|---|
| `flowsom` | `pip install flowsom` | Algorithme FlowSOM (saeyslab) |
| `flowio` | `pip install flowio` | Lecture/écriture FCS natif |
| `fcswrite` | `pip install fcswrite` | Export FCS avec colonnes ajoutées |
| `flowkit` | `pip install flowkit` | Transformation logicle précise |
| `pytometry` | `pip install pytometry` | Transformations cytométriques avancées |
| `umap-learn` | `pip install umap-learn` | Réduction dimensionnelle UMAP |
| `cupy` | voir [cupy.dev](https://cupy.dev) | Accélération GPU (CUDA requis) |
| `reportlab` | `pip install reportlab` | Export rapports PDF |

---

## Structure du projet

```
flowsom_pipeline_pro/
├── __init__.py              # Entrée publique : FlowSOMPipeline, PipelineConfig
├── setup.py                 # Packaging pip
├── README.md
│
├── config/
│   ├── constants.py         # Toutes les constantes (NE PAS MODIFIER les seuils ELN)
│   ├── pipeline_config.py   # PipelineConfig dataclass
│   └── default_config.yaml  # Valeurs par défaut
│
├── cli/
│   ├── parsers.py           # build_argument_parser(), detect_config_file()
│   └── main.py              # main() — point d'entrée flowsom-analyze
│
├── src/
│   ├── core/                # Algorithmes purs (sans effets de bord)
│   ├── models/              # Dataclasses (FlowSample, PipelineResult…)
│   ├── utils/               # Logger, validators
│   ├── io/                  # Lecture/écriture fichiers
│   │   ├── csv_exporter.py                    # Stats, MFI, données cellulaires
│   │   ├── cluster_distribution_exporter.py   # Distribution Sain/Patho (TXT + CSV)
│   │   ├── fcs_reader.py / fcs_writer.py      # FCS natif (flowio / fcswrite)
│   │   └── json_exporter.py                   # Métadonnées de run
│   ├── visualization/       # Graphiques matplotlib
│   ├── analysis/            # Analyses biologiques (MRD, blastes, stats)
│   ├── services/            # Orchestration (preprocessing, clustering, export)
│   └── pipeline/
│       └── pipeline_executor.py  # FlowSOMPipeline — 7 étapes
│
└── tests/                   # Tests unitaires (à compléter)
```

---

## Notes importantes

### Reproductibilité

Toujours fixer `seed=42` (comportement par défaut). Le SOM FlowSOM est stochastique ; des résultats différents entre deux runs sont le signe d'une graine non fixée.

### Vérifications automatiques

À chaque exécution, le pipeline vérifie :
- Absence de valeurs `NaN` dans la matrice d'expression
- Présence d'une matrice de compensation `$SPILL` dans les métadonnées FCS
- Équilibre du nombre de cellules entre conditions (avertissement si ratio > 10×)
- Exclusion des canaux FSC/SSC/Time du clustering
- Nombre minimum d'événements par node (ELN : ≥ 17)

### Transformation recommandée

La transformation **logicle** est recommandée pour les données de cytométrie multiparamétrique. Elle gère correctement les valeurs négatives et les distributions bimodales, contrairement au log10 ou à l'arcsinh avec cofacteur inadapté.

> ⚠️ Si les données contiennent beaucoup de valeurs négatives, c'est souvent le signe que la compensation n'a pas été appliquée. Vérifier la présence de `$SPILL` dans les métadonnées FCS.
