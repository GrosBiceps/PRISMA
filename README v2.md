# 🔬 FlowSOM Analysis Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FlowSOM](https://img.shields.io/badge/FlowSOM-Latest-green.svg)](https://github.com/saeyslab/FlowSOM_Python)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-success.svg)]()

> **Pipeline complet d'analyse automatisée de cytométrie en flux** — Du fichier FCS brut au rapport PDF diagnostique via clustering FlowSOM et gating adaptatif par intelligence artificielle.

---

## 📋 Table des Matières

- [Vue d'Ensemble](#-vue-densemble)
- [Fonctionnalités Majeures](#-fonctionnalités-majeures)
- [Architecture du Pipeline](#-architecture-du-pipeline)
- [Installation](#-installation)
- [Configuration et Utilisation](#-configuration-et-utilisation)
- [Documentation Technique](#-documentation-technique)
- [Visualisations](#-visualisations)
- [Contrôle Qualité](#-contrôle-qualité)
- [Structure du Projet](#-structure-du-projet)
- [Licence](#-licence)

---

## 🎯 Vue d'Ensemble

**FlowSOM Analysis Pipeline** est un notebook Python scientifique conçu pour l'analyse avancée de données de cytométrie en flux. Il constitue le miroir headless de l'application FlowSOM Analyzer, permettant :

- 🐛 **Debug & Introspection** : Visualiser les données à chaque étape du traitement
- ⚡ **Tuning Rapide** : Tester différents paramètres sans relancer l'application complète
- 🧩 **Séparation des Responsabilités** : Logique métier pure, sans dépendance à l'interface utilisateur

Le pipeline transforme des fichiers FCS bruts en rapports diagnostiques structurés via une chaîne de traitement automatisée intégrant :
- Gating adaptatif par modèles statistiques (GMM/RANSAC)
- Transformations standards de cytométrie (Arcsinh/Logicle)
- Clustering FlowSOM avec visualisations interactives
- Export de rapports PDF professionnels

**Auteur** : Florian Magne  
**Version** : 1.0  
**Date** : Janvier 2026

---

## ✨ Fonctionnalités Majeures

### 🤖 Gating Adaptatif Intelligent

Le pipeline implémente une stratégie de gating avancée basée sur l'intelligence artificielle et les statistiques, remplaçant les seuils fixes par des modèles adaptatifs :

#### **1. Détection de Débris (GMM 2D)**
- **Algorithme** : Modèle de Mélange Gaussien (*Gaussian Mixture Model*, GMM) à 2-3 composantes sur l'espace FSC-A × SSC-A
- **Principe** : Identification automatique des clusters naturels (débris, cellules viables, événements saturés)
- **Sélection automatique** : Choix du nombre de composantes par critère BIC (*Bayesian Information Criterion*)
- **Avantage** : S'adapte à la proportion réelle de débris (10 % ou 1 %) au lieu de couper à un percentile fixe

```python
# Citation ligne 1435
maskdebris = AutoGating.autogatedebris(Xraw, varnames, ncomponents=3)
```

#### **2. Exclusion de Doublets (RANSAC Robuste)**
- **Algorithme** : Régression linéaire robuste RANSAC sur FSC-A vs FSC-H
- **Principe** : Modélise la diagonale des singlets (cellules individuelles), détecte les doublets au-dessus
- **Contrôle qualité** : Vérification du coefficient R² sur les inliers (seuil : R² ≥ 0.85)
- **Fallback automatique** : Si R² < 0.85, bascule vers gating par ratio FSC-A/FSC-H classique
- **Gating par fichier** : Application séparée pour chaque fichier FCS avec stockage des scatter plots

```python
# Citation ligne 1485
masksinglets = AutoGating.autogatesinglets(
    Xraw, varnames, 
    fileorigin=fileorigins, 
    perfile=True,
    r2_threshold=0.85
)
```

#### **3. Sélection CD45+ (GMM 1D Bimodal)**
- **Algorithme** : GMM à 2 composantes sur l'intensité CD45
- **Principe** : Trouve le creux naturel entre populations CD45⁻ et CD45⁺
- **Seuil adaptatif** : Calculé à l'intersection des distributions gaussiennes
- **Mode uniforme** : Option de seuil par percentile pour compatibilité

```python
# Citation ligne 1563
maskcd45 = AutoGating.autogatecd45(Xraw, varnames, ncomponents=2)
```

### 🎭 Gating Asymétrique Unique

Innovation majeure du pipeline : **stratégie de gating différentiel** pour comparaison blastes pathologiques vs moelle saine.

#### **Principe de la Stratégie Asymétrique**

| Condition | Gates Appliqués | Population Résultante |
|-----------|-----------------|----------------------|
| **Pathologique** | Débris → Doublets → CD45+ → (CD34+ optionnel) | Leucocytes CD45+ stricts (ou blastes CD34+) |
| **Saine (NBM)** | Débris → Doublets **UNIQUEMENT** | **Toutes les cellules viables** (CD45+, CD45⁻, progéniteurs) |

```python
# Citation ligne 1718
if MODE_BLASTES_VS_NORMAL:
    # Patho : Gate CD45 STRICT appliqué
    mask_cd45[mask_patho] = mask_cd45_full[mask_patho]
    # Sain : Gate CD45 IGNORÉ → toutes cellules conservées
    mask_cd45[mask_sain] = True
```

**Objectif** : Comparer une population leucocytaire pathologique sélectionnée à la **diversité complète** de la moelle normale (incluant progéniteurs CD45⁻/low), révélant les écarts phénotypiques subtils.

### 🔄 Transformations Standards

Support des transformations canoniques de cytométrie avec implémentation optimisée :

- **Arcsinh** : Transformation hyperbolique inverse (cofacteur 5 ou 150)
- **Logicle** : Transformation biexponentielle précise via FlowKit (T=262144, M=4.5, W=0.5)
- **Log10** : Transformation logarithmique classique
- **Z-score / Min-Max** : Normalisations statistiques

```python
# Citation lignes 1387, 1390, 1393
transformed = DataTransformer.arcsinh_transform(data, cofactor=5.0)
transformed = DataTransformer.logicle_transform(data, T=262144.0, M=4.5, W=0.5)
transformed = DataTransformer.log_transform(data, base=10.0)
```

### 📊 Visualisations Interactives

#### **Dashboard de Gating Style CytoPy**
Graphiques séparés pour chaque étape du gating avec overlay conservés/exclus :
- Gate 1 : Vue d'ensemble FSC-A × SSC-A
- Gate 2 : Débris avec zones rectangulaires adaptatifs
- Gate 3 : Singlets avec droites RANSAC par fichier
- Gate 4 : CD45+ avec seuil bimodal
- Gate 5 : CD34+ blastes (optionnel)

```python
# Citation ligne 1758
plot_gating(ax, x, y, mask, title, xlabel, ylabel, label_in, label_out)
```

#### **Diagramme de Sankey (Flux d'Événements)**
Visualisation interactive du flux cellulaire à travers les gates successifs :
- Événements initiaux → Débris → Doublets → CD45⁻ → CD34⁻
- Largeur proportionnelle au nombre d'événements
- Couleurs distinctes par condition (Sain/Pathologique)

```python
# Citation ligne 1858
fig_sankey = go.Figure(data=[go.Sankey(node=..., link=...)])
```

#### **Scatter Plots Haute Qualité**
- Densité 2D avec colormap personnalisée (BlueYellowRed)
- Sous-échantillonnage intelligent (max 100k points)
- Overlay de statistiques (R², slope, intercept pour RANSAC)
- Export PNG haute résolution (150 DPI, fond sombre)

```python
# Citation ligne 1872
plot_density(ax, x, y, title, xlabel, ylabel, nbins=120)
```

---

## 🏗️ Architecture du Pipeline

Le pipeline s'articule en **7 étapes séquentielles** :

```
┌─────────────────────────────────────────────────────────────┐
│  1️⃣  IMPORTS & CONFIGURATION                                │
│     ├─ Librairies (flowsom, anndata, scanpy, plotly)       │
│     └─ Classes utilitaires (DataTransformer, PreGating,    │
│        AutoGating, GateResult)                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2️⃣  CHARGEMENT FCS                                         │
│     ├─ HEALTHY_FOLDER → fichiers sains (NBM)               │
│     ├─ PATHOLOGICAL_FOLDER → fichiers patients             │
│     └─ Conversion en AnnData avec métadonnées              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3️⃣  EXPLORATION & QC                                       │
│     ├─ Dimensions (cellules × marqueurs)                   │
│     ├─ Statistiques descriptives (min/max/médiane)         │
│     └─ Détection de valeurs aberrantes                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4️⃣  CONTRÔLE QUALITÉ                                       │
│     ├─ Vérification des marqueurs requis (FSC, SSC, CD45)  │
│     ├─ Filtrage NaN/Inf                                     │
│     └─ Validation des plages d'intensité                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  5️⃣  PRE-GATING SÉQUENTIEL                                  │
│     ├─ G1 : Débris (GMM FSC-A×SSC-A)                        │
│     ├─ G2 : Doublets (RANSAC FSC-A vs FSC-H)               │
│     ├─ G3 : CD45+ (GMM CD45 1D)                             │
│     └─ G4 : CD34+ blastes (GMM CD34 + SSC low)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  6️⃣  TRANSFORMATION                                         │
│     ├─ Arcsinh (cofacteur 5 ou 150)                        │
│     ├─ Logicle via FlowKit (T=262144, M=4.5, W=0.5)        │
│     └─ Application sur marqueurs sélectionnés              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  7️⃣  ANALYSE FLOWSOM & RAPPORT                              │
│     ├─ Clustering FlowSOM (xdim×ydim)                      │
│     ├─ Métaclustering (ConsensusCluster/Hierarchical)      │
│     ├─ Visualisations (UMAP, t-SNE, heatmaps)              │
│     └─ Export rapport PDF avec diagnostics                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip ou conda pour la gestion des dépendances

### Dépendances Critiques

Le pipeline nécessite les librairies suivantes :

```bash
# Cytométrie et analyse
pip install flowsom              # Citation ligne 1369
pip install anndata              # Citation ligne 1373
pip install scanpy               # Citation ligne 1374

# Visualisation
pip install plotly               # Citation ligne 1377
pip install matplotlib seaborn   # (lignes 1370-1371)

# Transformations avancées
pip install flowkit              # Citation ligne 1378 (Logicle précis)

# Machine Learning
pip install scikit-learn         # GMM, RANSAC, clustering
pip install umap-learn           # UMAP dimensionality reduction

# Utilitaires
pip install numpy pandas scipy
pip install fcswrite              # Export FCS (optionnel)
```

### Installation Complète

```bash
# Cloner le repository
git clone <URL_DU_REPO>
cd FlowSOM_Analysis_Pipeline

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer toutes les dépendances
pip install -r requirements.txt

# Lancer Jupyter Notebook
jupyter notebook FlowSOM_Analysis_Pipeline.ipynb
```

---

## ⚙️ Configuration et Utilisation

### 1. Configuration des Chemins de Données

**Localisation** : Cellule **"2. Chargement des Fichiers FCS"**

```python
# Citation ligne 1610 (approximatif)
# CONFIGURATION DES CHEMINS
HEALTHY_FOLDER = Path(r"C:/Travail/normale")      # Fichiers sains (NBM)
PATHOLOGICAL_FOLDER = Path(r"/Data/Patho")       # Fichiers pathologiques

# Mode d'analyse
COMPARE_MODE = True  # True : Sain vs Patho | False : Patho seul
```

**Instructions** :
1. Remplacer `HEALTHY_FOLDER` par le chemin absolu vers vos fichiers FCS de référence (moelle normale)
2. Remplacer `PATHOLOGICAL_FOLDER` par le chemin vers vos fichiers patients
3. Mettre `COMPARE_MODE = False` si vous analysez uniquement des données pathologiques sans référence

### 2. Paramètres de Gating

**Localisation** : Cellule **"5. Pre-Gating - Configuration"**

#### **Mode de Gating**
```python
GATING_MODE = "auto"  # "auto" (GMM adaptatif) ou "manual" (percentiles fixes)
```

#### **Gates Activés/Désactivés**
```python
APPLY_PREGATING = True     # Activer le pre-gating complet
GATE_DEBRIS = True         # Gate 1 : Débris
GATE_DOUBLETS = True       # Gate 2 : Doublets
GATE_CD45 = True           # Gate 3 : Leucocytes CD45+
FILTER_BLASTS = False      # Gate 4 : Blastes CD34+ (True pour sous-filtrage)
```

#### **Mode Asymétrique Blastes vs Normal**
```python
MODE_BLASTES_VS_NORMAL = True   # GATING DIFFÉRENTIEL : Activer pour comparer blastes patients vs moelle complète
```

**Explication** :
- `True` : Gate CD45 appliqué uniquement aux cellules pathologiques ; moelle saine conservée intégralement
- `False` : Gate CD45 appliqué uniformément aux deux conditions

#### **Paramètres Manual (si `GATING_MODE = "manual"`)**
```python
# Gate Débris (percentiles FSC-A/SSC-A)
DEBRIS_MIN_PERCENTILE = 1.0   # Exclure les 1 % les plus bas
DEBRIS_MAX_PERCENTILE = 99.0  # Exclure les 1 % les plus hauts

# Gate Doublets (ratio FSC-A/FSC-H)
RATIO_MIN = 0.6
RATIO_MAX = 1.5

# Gate CD45 (percentile CD45)
CD45_THRESHOLD_PERCENTILE = 10.0  # Top 90 % → CD45+

# Gate CD34 (percentile CD34 + SSC low)
CD34_THRESHOLD_PERCENTILE = 85    # Top 15 % → CD34 bright
USE_SSC_FILTER_FOR_BLASTS = True
SSC_MAX_PERCENTILE_BLASTS = 70    # Blastes = faible granularité
```

### 3. Transformation des Données

**Localisation** : Cellule **"6. Transformation"**

```python
TRANSFORM_METHOD = "arcsinh"  # "arcsinh", "logicle", "log", ou "none"
TRANSFORM_COFACTOR = 5.0      # Cofacteur pour Arcsinh (typique : 5 ou 150)
```

### 4. Lancement de l'Analyse

Une fois configuré, exécuter toutes les cellules du notebook :
- **Jupyter Notebook** : `Cell > Run All`
- **JupyterLab** : `Run > Run All Cells`

Le pipeline génère automatiquement :
- Graphiques de gating (PNG haute résolution)
- Diagramme de Sankey interactif
- Scatter plots RANSAC par fichier
- Rapport de gating structuré (JSON)
- Rapport final PDF (si activé)

---

## 🔧 Documentation Technique

### Classe `AutoGating`

**Rôle** : Implémentation du gating adaptatif par modèles statistiques (GMM/RANSAC)

#### **Méthodes Principales**

##### `safe_fit_gmm(data, n_components=2, ...)`
Wrapper robuste pour le fitting GMM avec gestion d'erreurs avancée :
- **Retry automatique** : Jusqu'à 5 tentatives avec différentes initialisations
- **Fallback unimodal** : Si échec, régression vers 1 composante
- **Sous-échantillonnage** : Max 200k points avant fitting pour convergence rapide
- **Vérification convergence** : Warnings si `gmm.converged_ == False`

```python
# Citation ligne 1437
gmm = AutoGating.safe_fit_gmm(
    data, 
    n_components=2, 
    n_init=3, 
    max_retries=5, 
    subsample=True
)
```

##### `autogate_debris(X, varnames, ...)`
Gate débris adaptatif par GMM 2D sur FSC-A × SSC-A.
- **Sélection BIC** : Teste 2 et 3 composantes, choisit le meilleur modèle
- **Exclusion intelligente** : Rejette les clusters < 2 % d'événements ou FSC très bas
- **Retour** : `GateResult` structuré avec masque, statistiques, et warnings

##### `autogate_singlets(X, varnames, fileorigin=None, perfile=True, r2_threshold=0.85)`
Gate singlets robuste par RANSAC avec contrôle qualité R².
- **Pré-filtre** : Exclusion des outliers extrêmes (percentiles 1-99) avant régression
- **RANSAC** : Régression linéaire robuste FSC-A vs FSC-H
- **Contrôle R²** : Si R² < 0.85 sur inliers → fallback vers ratio FSC-A/FSC-H simple
- **Gating par fichier** : Application séparée pour chaque FCS avec stockage scatter data
- **Stockage** : Dictionnaire global `ransac_scatter_data` pour visualisation dans rapport

```python
# Citation ligne 1503
masksinglets = AutoGating.autogate_singlets(
    Xraw, varnames,
    fileorigin=file_origins,
    perfile=True,
    r2_threshold=0.85
)
```

**Structure de Retour** : `GateResult`
```python
# Citation ligne 1535
@dataclass
class GateResult:
    mask: np.ndarray          # Masque booléen (True = conservé)
    n_kept: int               # Nombre d'événements conservés
    n_total: int              # Nombre total d'événements
    method: str               # Méthode utilisée (ex: "ransac_singlets")
    gate_name: str            # Nom du gate (ex: "G2_singlets")
    details: Dict[str, Any]   # Détails spécifiques (R², BIC, etc.)
    warnings: List[str]       # Warnings éventuels
    
    @property
    def pct_kept(self) -> float:
        return (self.n_kept / max(self.n_total, 1)) * 100
```

### Contrôle Qualité du Gating

#### **Métriques de Performance**

##### **1. R² RANSAC (Doublets)**
- **Seuil** : R² ≥ 0.85 sur les inliers
- **Calcul** : `r2_score(y_true, y_pred)` sur les points alignés sur la diagonale
- **Interprétation** :
  - R² > 0.90 : Excellent ajustement, singlets bien définis
  - 0.85 ≤ R² < 0.90 : Ajustement acceptable
  - R² < 0.85 : **Fallback automatique vers gating ratio** (FSC-A/FSC-H entre 0.6 et 1.5)

```python
# Citation ligne 1485 (approximatif)
if r2_val < r2_threshold:
    warnings.warn(f"R² faible ({r2_val:.2f}) → fallback gating ratio")
    singlets_mask = fallback_ratio_gating(fsca, fsch)
```

##### **2. BIC (Débris GMM)**
- **Critère** : *Bayesian Information Criterion* pour sélection du nombre de composantes
- **Principe** : BIC = -2 * log-likelihood + k * log(n), favorise la parcimonie
- **Usage** : Teste GMM à 2 et 3 composantes, sélectionne le modèle avec BIC minimal

##### **3. Convergence GMM**
- **Flag** : `gmm.converged_` doit être `True`
- **Warnings automatiques** : Si non convergence après `max_iter` itérations (défaut : 200)
- **Action** : Retry avec différentes initialisations via `safe_fit_gmm`

#### **Log Structuré JSON**

Toutes les opérations de gating sont tracées dans `gating_log_entries` pour audit :

```python
# Citation ligne 1535 (approximatif)
gating_log_entries = []

def log_gating_event(gate_name: str, method: str, status: str, 
                     details: Dict = None, warning_msg: str = None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "gate_name": gate_name,
        "method": method,
        "status": status,  # "success", "fallback", "warning", "error"
        "details": details or {}
    }
    if warning_msg:
        entry["warning"] = warning_msg
    gating_log_entries.append(entry)
```

**Export** : `json.dump(gating_log_entries, open("gating_log.json", "w"))`

---

## 📈 Visualisations

### Graphiques de Gating (Style CytoPy)

Chaque gate produit un graphique professionnel avec :
- **Scatter plot avec densité 2D** : Colormap BlueYellowRed (120 bins)
- **Overlay conservés/exclus** : Vert pastel (conservés) / Rouge pastel (exclus)
- **Statistiques intégrées** : N conservés, % total, paramètres (seuils, R², etc.)
- **Formatage intelligent des axes** : K pour milliers, M pour millions

**Exports** :
- `gating_01_overview.png` : Vue d'ensemble FSC-A × SSC-A
- `gating_02_debris.png` : Gate débris avec zones rectangulaires
- `gating_03_singlets.png` : Gate doublets avec droites RANSAC par fichier
- `gating_04_cd45.png` : Gate CD45 avec seuil bimodal
- `gating_05_cd34.png` : Gate CD34 blastes (si activé)

### Diagramme de Sankey Interactif

Visualisation du flux cellulaire à travers les gates :
- **Nœuds** : Événements initiaux, populations intermédiaires, populations finales
- **Liens** : Largeur proportionnelle au nombre d'événements
- **Couleurs** : Bleu (Sain), Orange (Pathologique), Rouge (exclus)
- **Interactivité** : Hover pour statistiques détaillées

```python
# Citation ligne 1858 (approximatif)
fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        label=["Initial", "Post-Débris", "Singlets", "CD45+", "Final"],
        color=["lightblue", "lightgreen", "yellow", "orange", "green"]
    ),
    link=dict(
        source=[0, 1, 2, 3],
        target=[1, 2, 3, 4],
        value=[n_debris, n_singlets, n_cd45, n_final],
        color=["rgba(255,0,0,0.3)", ...]
    )
)])
fig_sankey.show()
```

### Scatter Plots RANSAC (Fichier par Fichier)

Pour chaque fichier FCS, génération d'un scatter plot FSC-A vs FSC-H :
- **Points** : Échantillonnage aléatoire (max 2000 points)
- **Droite RANSAC** : Régression robuste y = slope × x + intercept
- **Annotation** : R², équation, % singlets
- **Stockage** : Dictionnaire `ransac_scatter_data[filename]` pour rapport HTML

```python
# Citation ligne 1872 (approximatif)
ransac_scatter_data[filename] = {
    "fsch": fsch[sample_idx].ravel().tolist(),
    "fsca": fsca[sample_idx].ravel().tolist(),
    "pred": ransac.predict(fsch[sample_idx]).tolist(),
    "r2": float(r2_val),
    "slope": float(slope),
    "intercept": float(intercept)
}
```

---

## 🛡️ Contrôle Qualité

### Validation Pré-Gating

Avant application du gating, vérifications automatiques :

1. **Marqueurs requis** : FSC-A, FSC-H, SSC-A, CD45
2. **Valeurs valides** : Filtrage NaN/Inf via `np.isfinite()`
3. **Plages raisonnables** : FSC/SSC > 0, CD45 ∈ [0, 10^6]
4. **Nombre d'événements** : Minimum 200 cellules par fichier

### Diagnostics RANSAC (Doublets)

- **R² ≥ 0.85** : Ajustement excellent, singlets bien définis
- **0.80 ≤ R² < 0.85** : Warning émis, inspection manuelle recommandée
- **R² < 0.80** : **Fallback automatique** vers gating ratio (0.6 ≤ FSC-A/FSC-H ≤ 1.5)

### Tableau Récapitulatif Singlets

Affichage après gating par fichier :

```
Fichier                       Méthode            R²     Singlets (%)
────────────────────────────────────────────────────────────────────
Patient_001.fcs               ransac           0.923     87.3
Patient_002.fcs               ratio_fallback   0.782     84.1
NBM_control.fcs               ransac           0.945     91.8
```

### Warnings Automatiques

- **GMM non-convergent** : "GMM n_components=3 non-convergé après 200 itérations"
- **R² faible RANSAC** : "R² = 0.78 < 0.85 → fallback ratio FSC-A/FSC-H"
- **Marqueur absent** : "CD34 non trouvé → gate blastes ignoré"
- **Fichier trop petit** : "Fichier Patient_X.fcs < 200 cellules → skip gating"

---

## 📁 Structure du Projet

```
FlowSOM_Analysis_Pipeline/
│
├── FlowSOM_Analysis_Pipeline.ipynb    # Notebook principal
│
├── requirements.txt                   # Dépendances Python
│
├── README.md                          # Ce fichier
│
├── data/                              # Données (non versionné)
│   ├── normale/                       # Fichiers FCS sains (NBM)
│   └── Patho/                         # Fichiers FCS pathologiques
│
├── outputs/                           # Résultats générés
│   ├── gating_*.png                   # Graphiques de gating
│   ├── gating_log.json                # Log structuré JSON
│   ├── sankey_flow.html               # Diagramme interactif
│   └── report_final.pdf               # Rapport d'analyse
│
└── utils/                             # Modules utilitaires (optionnel)
    ├── transformations.py             # Classe DataTransformer
    ├── pregating.py                   # Classe PreGating
    └── autogating.py                  # Classe AutoGating
```

---

## 📚 Références et Inspirations

Ce pipeline s'inspire des meilleures pratiques en cytométrie computationnelle :

- **FlowSOM** : Van Gassen et al. (2015), *Cytometry Part A*  
  Self-Organizing Maps pour clustering de cytométrie en flux
  
- **CytoPy** : Burgoyne et al. (2020), *bioRxiv*  
  Framework d'analyse automatisée avec gating autonome (GMM/KDE)
  
- **RANSAC** : Fischler & Bolles (1981), *Communications of the ACM*  
  Régression robuste aux outliers pour singlets
  
- **GMM** : Scikit-learn Documentation  
  Gaussian Mixture Models pour détection de populations

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour proposer des améliorations :

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

## 📄 Licence

Ce projet est distribué sous licence MIT. Voir `LICENSE` pour plus d'informations.

---

## 📧 Contact

**Florian Magne**  
CHU de Limoges — Service de Biologie Médicale  
📧 Email : [florian.magne@chu-limoges.fr](mailto:florian.magne@chu-limoges.fr)  
🔗 GitHub : [FlorianMgs](https://github.com/FlorianMgs)

---

## 🙏 Remerciements

- **Équipe CRIBL Laboratory** pour la collaboration scientifique
- **FlowSOM Python Team** pour l'implémentation du package
- **Communauté Scanpy/AnnData** pour l'écosystème d'analyse single-cell
- **Plotly Team** pour les visualisations interactives de haute qualité

---

<div align="center">

**⭐ Si ce projet vous est utile, n'hésitez pas à lui donner une étoile ! ⭐**

</div>
