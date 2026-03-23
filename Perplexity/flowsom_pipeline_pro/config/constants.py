"""
constants.py — Constantes globales du pipeline FlowSOM.

Toutes les constantes métier, seuils cliniques ELN 2022 et palettes
sont centralisées ici. Ne jamais hardcoder ces valeurs dans un autre module.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------
PIPELINE_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Patterns scatter / time — jamais intégrés dans le clustering FlowSOM
# ---------------------------------------------------------------------------
SCATTER_PATTERNS: tuple[str, ...] = ("FSC", "SSC", "TIME", "EVENT", "WIDTH", "HEIGHT")

# ---------------------------------------------------------------------------
# Seuils cliniques MRD — ELN 2022
# NE PAS MODIFIER sans validation biomédicale explicite
# ---------------------------------------------------------------------------
MRD_LOD: float = 9e-5  # Limite de détection  (0.009%)
MRD_LOQ: float = 5e-5  # Limite de quantification (0.005%)
NBM_FREQ_MAX: float = 0.011  # Fréquence max CD34+ dans moelle normale (1.1%)
FOLD_CHANGE_MRD: float = 1.9  # Fold-change FU/NBM pour positivité MRD
MRD_FOLD_CHANGE_THRESHOLD: float = FOLD_CHANGE_MRD  # Alias (backward compat)
MIN_EVENTS_PER_NODE: int = 17  # ELN : minimum d'événements par node FlowSOM

# ---------------------------------------------------------------------------
# Paramètres FlowSOM par défaut
# ---------------------------------------------------------------------------
DEFAULT_XDIM: int = 10
DEFAULT_YDIM: int = 10
DEFAULT_N_METACLUSTERS: int = 8
DEFAULT_SEED: int = 42
DEFAULT_RLEN: str = "auto"
DEFAULT_LEARNING_RATE: float = 0.05
DEFAULT_SIGMA: float = 1.5
DEFAULT_N_ITERATIONS: int = 10

# ---------------------------------------------------------------------------
# Paramètres de transformation
# ---------------------------------------------------------------------------
DEFAULT_TRANSFORM_METHOD: str = "logicle"  # arcsinh | logicle | log10 | none
DEFAULT_ARCSINH_COFACTOR: float = 5.0  # Standard flow cytometry
DEFAULT_LOGICLE_T: float = 262144.0  # 2^18
DEFAULT_LOGICLE_M: float = 4.5
DEFAULT_LOGICLE_W: float = 0.5
DEFAULT_LOGICLE_A: float = 0.0

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
DEFAULT_NORMALIZE_METHOD: str = "zscore"  # zscore | minmax | none

# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------
DEFAULT_MAX_CELLS_PER_FILE: int = 50_000
DEFAULT_MAX_CELLS_TOTAL: int = 1_000_000

# ---------------------------------------------------------------------------
# GMM / AutoGating
# ---------------------------------------------------------------------------
GMM_MAX_SAMPLES: int = 200_000
RANSAC_R2_THRESHOLD: float = 0.85
RANSAC_MAD_FACTOR: float = 3.0

# ---------------------------------------------------------------------------
# Pré-gating (percentiles par défaut)
# ---------------------------------------------------------------------------
DEFAULT_DEBRIS_MIN_PCT: float = 1.0
DEFAULT_DEBRIS_MAX_PCT: float = 99.0
DEFAULT_SINGLETS_RATIO_MIN: float = 0.6
DEFAULT_SINGLETS_RATIO_MAX: float = 1.4
DEFAULT_CD45_THRESHOLD_PCT: float = 5.0
DEFAULT_CD34_THRESHOLD_PCT: float = 85.0
DEFAULT_CD34_SSC_MAX_PCT: float = 60.0

# ---------------------------------------------------------------------------
# Auto-clustering (stabilité AMJI/ARI + Silhouette)
# ---------------------------------------------------------------------------
DEFAULT_MIN_CLUSTERS: int = 5
DEFAULT_MAX_CLUSTERS: int = 35
DEFAULT_N_BOOTSTRAP: int = 10
DEFAULT_SAMPLE_SIZE_BOOTSTRAP: int = 20_000
DEFAULT_MIN_STABILITY_THRESHOLD: float = 0.75
DEFAULT_W_STABILITY: float = 0.65
DEFAULT_W_SILHOUETTE: float = 0.35

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
DEFAULT_PLOT_FORMAT: str = "png"
DEFAULT_DPI: int = 300
PLOT_DARK_BG: str = "#1e1e2e"
PLOT_ACCENT: str = "#6366f1"
PLOT_GRID: str = "#45475a"
PLOT_TEXT: str = "#e2e8f0"

# Palettes de densité (style FlowJo)
DENSITY_CMAP_COLORS: list[str] = [
    "#0d0d0d",
    "#1a1a2e",
    "#0077b6",
    "#00b4d8",
    "#90e0ef",
    "#f9e2af",
    "#ffffff",
]
COLOR_KEPT: str = "#a6e3a1"  # Vert pastel — cellules conservées
COLOR_EXCLUDED: str = "#f38ba8"  # Rouge pastel — cellules exclues

# Couleurs populations (pour mapping de référence)
COLORS_POPULATIONS: dict[str, str] = {
    "Granulocytes": "#4cc9f0",
    "Monocytes": "#f72585",
    "B_cells": "#7209b7",
    "T_NK_cells": "#3a0ca3",
    "Plasmocytes": "#f4a261",
    "Hemato_I": "#2dc653",
    "Blasts_CD34": "#ef233c",
    "Unknown": "#6c757d",
}

# ---------------------------------------------------------------------------
# Marqueurs LSC (Leukemic Stem Cells)
# ---------------------------------------------------------------------------
LSC_CORE_MARKERS: tuple[str, ...] = ("CD34", "CD38", "CD123")
LSC_EXTENDED_MARKERS: tuple[str, ...] = (
    "CD45RA",
    "CD90",
    "TIM3",
    "CLL-1",
    "CD97",
    "GPR56",
)

# ---------------------------------------------------------------------------
# Noms de colonnes d'export standard
# ---------------------------------------------------------------------------
COL_FLOWSOM_CLUSTER: str = "FlowSOM_cluster"
COL_FLOWSOM_METACLUSTER: str = "FlowSOM_metacluster"
COL_CONDITION: str = "condition"
COL_FILE_ORIGIN: str = "file_origin"
COL_CONDITION_NUM: str = "Condition_Num"
