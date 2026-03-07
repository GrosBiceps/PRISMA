# FlowSOM Analysis Pipeline Pro

Pipeline modulaire d'analyse FlowSOM pour données de cytométrie en flux, conforme aux recommandations **ELN 2022** pour la détection de la **Maladie Résiduelle Détectable (MRD)** en hématologie.

## 📋 Description

`flowsom_pipeline_pro` est un package Python production-ready, issu de la refactorisation complète d'un script monolithique.  
La configuration passe par un fichier **YAML** ; les paramètres peuvent être surchargés ponctuellement via la ligne de commande.

### Fonctionnalités principales

- **Chargement FCS multi-fichiers** — compensation automatique (`$SPILL`), panel commun par intersection
- **Pré-gating combiné automatique** (GMM + RANSAC) :
  - Gate 1 — Débris (FSC/SSC, GMM adaptatif)
  - Gate 2 — Singlets (FSC-H vs FSC-A, régression RANSAC)
  - Gate 3 — CD45+ leucocytes (GMM asymétrique NBM/patho)
  - Gate 4 — CD34+ blastes (optionnel)
- **Transformation cytométrique** : Logicle (défaut), arcsinh, log10
- **Normalisation** : z-score (défaut), min-max
- **Clustering FlowSOM** GPU (NVIDIA CUDA) avec fallback CPU transparent
- **Métaclustering** par consensus hiérarchique ou k optimal auto-détecté
- **Visualisations complètes** (voir section dédiée)
- **Rapport HTML auto-contenu** (~10 MB, figures Plotly + Matplotlib embarquées)
- **Exports professionnels** : FCS réannoté, CSV complet/par fichier, statistiques, JSON
- **Mode comparaison** Sain (NBM) vs Pathologique

## 🚀 Installation

### Prérequis

- Python 3.10+
- CUDA 11.0+ (optionnel, pour accélération GPU)

### Dépendances

```bash
pip install numpy pandas scipy scikit-learn
pip install matplotlib seaborn plotly
pip install flowio fcswrite pytometry anndata
pip install umap-learn
pip install PyYAML reportlab
# Module GPU local (présent dans le workspace) :
# FlowSomGpu/  →  détecté automatiquement si CUDA disponible
```

## 📖 Utilisation

### Lancement standard

```bash
python -m flowsom_pipeline_pro --config config_flowsom.yaml --verbose
```

### Aide

```bash
python -m flowsom_pipeline_pro --help
```

### Surcharger les chemins sans modifier le YAML

```bash
python -m flowsom_pipeline_pro \
    --config config_flowsom.yaml \
    --healthy-folder "Data/NBM" \
    --patho-folder "Data/LAM" \
    --output Results_LAM
```

### Désactiver le GPU pour un run spécifique

```bash
python -m flowsom_pipeline_pro --config config_flowsom.yaml --no-gpu
```

### Mode silencieux

```bash
python -m flowsom_pipeline_pro --config config_flowsom.yaml --quiet
```

## ⚙️ Options CLI

| Option | Description |
|---|---|
| `--config PATH` | Fichier de configuration YAML (défaut: `config_flowsom.yaml`) |
| `--healthy-folder PATH` | Surcharge `paths.healthy_folder` |
| `--patho-folder PATH` | Surcharge `paths.patho_folder` |
| `--output PATH` | Surcharge `paths.output_dir` |
| `--no-gpu` | Force le mode CPU |
| `--verbose / -v` | Logging DEBUG |
| `--quiet / -q` | Logging WARNING uniquement |

> Tous les autres paramètres (FlowSOM, gating, transformation, visualisation…) sont définis dans le fichier YAML.

## 📄 Fichier de configuration YAML

Structure des sections principales de `config_flowsom.yaml` :

```yaml
paths:
  healthy_folder: "Data/Moelle normale"
  patho_folder: "Data/Patho"          # optionnel
  output_dir: "Results"

analysis:
  compare_mode: true

pregate:
  apply: true
  mode: "auto"                         # auto (GMM+RANSAC) | manual (percentiles)
  mode_blastes_vs_normal: true         # gating CD45 asymétrique patho/sain
  viable: true                         # Gate 1 — débris
  singlets: true                       # Gate 2 — doublets
  cd45: true                           # Gate 3 — leucocytes
  cd34: false                          # Gate 4 — blastes CD34+

flowsom:
  xdim: 10
  ydim: 10
  rlen: "auto"                         # auto = sqrt(N)×0.1, borné 10–100
  n_metaclusters: 8
  seed: 42

auto_clustering:
  enabled: false                       # true = recherche k optimal (bootstrap)
  min_clusters: 5
  max_clusters: 35

transform:
  method: "logicle"                    # arcsinh | logicle | log10 | none
  cofactor: 5

normalize:
  method: "zscore"                     # zscore | minmax | none

markers:
  exclude_scatter: true
  exclude_additional: ["CD45"]

downsampling:
  enabled: true
  max_cells_per_file: 50000
  max_cells_total: 1000000

visualization:
  save_plots: true
  umap_enabled: true
  plot_format: "png"
  dpi: 300

gpu:
  enabled: true

logging:
  level: "INFO"
```

## 📊 Visualisations produites

### Pré-gating QC (Matplotlib, PNG)

| Fichier | Contenu |
|---|---|
| `combined_01_overview.png` | Vue d'ensemble FSC/SSC tous gates |
| `combined_02_debris.png` | Gate 1 — exclusion débris (GMM) |
| `combined_03_singlets.png` | Gate 2 — singlets par fichier (RANSAC) |
| `combined_04_cd45.png` | Gate 3 — CD45 seuil adaptatif |

### Clustering FlowSOM

| Fichier | Type | Contenu |
|---|---|---|
| `mst_static_TIMESTAMP.png` | Matplotlib | MST statique — noeuds colorés par métacluster dominant, taille proportionnelle au nombre de cellules |
| `mst_interactive_TIMESTAMP.html` | Plotly | MST interactif — hover par node (ID, métacluster, n cellules, top marqueurs) |
| `som_grid_TIMESTAMP.html` | Plotly | Grille SOM — cellules positionnées avec jitter, colorées par métacluster (jusqu'à 50 000 cellules) |
| `umap_TIMESTAMP.png` | Matplotlib | UMAP 2D coloré par métacluster |
| `mfi_heatmap_TIMESTAMP.png` | Matplotlib | Heatmap MFI normalisée par métacluster |
| `metacluster_distribution_TIMESTAMP.png` | Matplotlib | Distribution % cellules par métacluster et condition |

### Rapport Sankey + HTML

| Fichier | Contenu |
|---|---|
| `sankey_global_TIMESTAMP.html` | Flux de cellules à travers tous les gates |
| `flowsom_report_TIMESTAMP.html` | Rapport auto-contenu (~10 MB) — toutes figures + tables conditions + résumé pipeline |

## 📁 Structure des sorties

```
Results/
├── fcs/
│   └── flowsom_results_TIMESTAMP.fcs       # FCS réannoté (FlowSOM_cluster, FlowSOM_metacluster, xGrid, yGrid…)
├── csv/
│   ├── flowsom_complete_TIMESTAMP.csv      # Toutes cellules avec assignations
│   ├── per_file/                           # Un CSV par fichier FCS source
│   │   ├── NomFichier1_TIMESTAMP.csv
│   │   └── NomFichier2_TIMESTAMP.csv
│   ├── flowsom_statistics_TIMESTAMP.csv    # Statistiques par métacluster
│   └── flowsom_mfi_TIMESTAMP.csv          # Matrice MFI (n_metaclusters × n_marqueurs)
├── plots/
│   ├── gating/
│   │   ├── combined_01_overview.png
│   │   ├── combined_02_debris.png
│   │   ├── combined_03_singlets.png
│   │   └── combined_04_cd45.png
│   ├── umap_TIMESTAMP.png
│   ├── mst_static_TIMESTAMP.png
│   ├── mfi_heatmap_TIMESTAMP.png
│   └── metacluster_distribution_TIMESTAMP.png
└── other/
    ├── flowsom_report_TIMESTAMP.html       # Rapport HTML complet
    ├── sankey_global_TIMESTAMP.html
    ├── mst_interactive_TIMESTAMP.html
    ├── som_grid_TIMESTAMP.html
    ├── gating_log_TIMESTAMP.json           # Log structuré des gates
    └── flowsom_metadata_TIMESTAMP.json     # Métadonnées complètes du run
```

### Colonnes ajoutées dans le FCS exporté

| Colonne | Description |
|---|---|
| `FlowSOMcluster` | Assignation node SOM (1–100 pour grille 10×10) |
| `FlowSOMmetacluster` | Assignation métacluster (0-indexé) |
| `xGrid` / `yGrid` | Coordonnées cellule sur la grille SOM |
| `xNodes` / `yNodes` | Coordonnées node SOM dans l'espace MST |
| `size` | Nombre de cellules dans le node |
| `ConditionNum` | 0 = Sain, 1 = Pathologique |

## 🔧 Paramètres recommandés

### Suivi MRD LAM (Leucémie Aiguë Myéloïde)

```yaml
pregate:
  viable: true
  singlets: true
  cd45: true
  cd34: false          # blastes CD45-dim inclus de ce fait
  mode_blastes_vs_normal: true

flowsom:
  xdim: 10
  ydim: 10
  n_metaclusters: 8

transform:
  method: "logicle"
  cofactor: 5
```

### Analyse moelle normale / immunophénotypage standard

```yaml
pregate:
  cd34: false
  mode_blastes_vs_normal: false

flowsom:
  xdim: 12
  ydim: 12
  n_metaclusters: 15

transform:
  method: "arcsinh"
  cofactor: 150
```

### Exploration haute résolution

```yaml
flowsom:
  xdim: 15
  ydim: 15
  rlen: "auto"

auto_clustering:
  enabled: true
  min_clusters: 10
  max_clusters: 35
```

## 🐛 Dépannage

### Erreur GPU / CUDA

```bash
python -m flowsom_pipeline_pro --config config_flowsom.yaml --no-gpu
```
Ou dans le YAML : `gpu: { enabled: false }`

### Mémoire insuffisante (gros jeux de données)

```yaml
downsampling:
  enabled: true
  max_cells_per_file: 20000
  max_cells_total: 400000
```

### Encoding Windows (PowerShell)

Si des caractères Unicode causent des erreurs lors de la redirection :

```powershell
$env:PYTHONUTF8="1"
python -m flowsom_pipeline_pro --config config_flowsom.yaml --verbose
```

### Marqueurs non trouvés / panel partiel

Le pipeline construit automatiquement le **panel commun** par intersection des marqueurs présents dans tous les fichiers FCS. Les marqueurs exclus sont loggés en WARNING. Vérifier en mode verbose :

```bash
python -m flowsom_pipeline_pro --config config_flowsom.yaml --verbose
```

## 🏗️ Architecture du package

```
flowsom_pipeline_pro/
├── __main__.py                   # Point d'entrée (python -m)
├── cli/
│   └── main.py                   # Parseur CLI (argparse)
├── config/
│   ├── pipeline_config.py        # Dataclasses de configuration typées
│   └── default_config.yaml
└── src/
    ├── pipeline/
    │   └── pipeline_executor.py  # Orchestrateur des 7 étapes
    ├── services/
    │   ├── preprocessing_service.py   # Chargement + pré-gating combiné
    │   ├── clustering_service.py      # FlowSOM GPU/CPU
    │   ├── export_service.py          # Export FCS/CSV/JSON
    │   └── population_mapping_service.py  # Mapping populations via MFI
    ├── core/
    │   ├── auto_gating.py        # GMM + RANSAC adaptatif
    │   ├── clustering.py         # FlowSOMClusterer (GPU/CPU)
    │   ├── metaclustering.py     # Consensus hiérarchique
    │   └── normalizers.py
    ├── visualization/
    │   ├── flowsom_plots.py      # UMAP, MST statique, MST Plotly, SOM Grid
    │   ├── gating_plots.py       # 4 plots QC pré-gating + Sankey
    │   ├── html_report.py        # Rapport HTML auto-contenu
    │   └── population_viz.py     # Barplots, radar, heatmap clinique
    ├── io/
    │   ├── fcs_writer.py
    │   ├── csv_exporter.py
    │   └── json_exporter.py
    └── utils/
        └── logger.py             # Log structuré + GateResult
```

## 👤 Auteur

**Florian Magne**
- Pharmacien / Interne en médecine biologique
- Spécialisation : Bioinformatique et cytométrie en flux
- Version : 3.0 (Mars 2026)

## 📄 Licence

MIT License — Utilisation libre pour recherche et clinique

## 🔗 Références

- FlowSOM : Van Gassen et al., *Cytometry Part A*, 2015
- ELN 2022 MRD recommendations : Schuurhuis et al., *Blood*, 2022
- FlowSomGpu : Implémentation accélérée GPU (module local) - Magne Florian
