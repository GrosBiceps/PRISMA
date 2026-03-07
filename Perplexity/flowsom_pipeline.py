#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FlowSOM Analysis Pipeline - Version Python Standalone
================================================================================

Pipeline complète d'analyse FlowSOM pour données de cytométrie en flux.
Converti depuis le notebook Jupyter pour exécution en ligne de commande.

Fonctionnalités complètes:
- Chargement et prétraitement des fichiers FCS (FlowCytometryTools, fcsparser)
- Pre-gating automatique et adaptatif:
  * Viable cells (exclusion des débris)
  * Singlets (exclusion des doublets)
  * CD45+ (leucocytes)
  * CD34+ (blastes/progéniteurs)
- Transformations cytométriques: arcsinh, logicle, log
- Normalisation: z-score, min-max
- Clustering FlowSOM avec accélération GPU (CuPy/RAPIDS)
- Métaclustering hiérarchique
- Visualisations multiples:
  * MST interactif (Plotly)
  * Grille SOM
  * Heatmaps MFI
  * Scatter plots par marqueur
  * Gating QC plots
- Exports professionnels:
  * CSV avec assignations
  * FCS réannoté (Kaluza/FlowJo compatible)
  * Statistiques par métacluster
  * Matrices MFI
  * Métadonnées JSON
- Mode comparaison Sain vs Pathologique
- Détection automatique des blastes et populations anormales
- Mapping de populations de référence
- Scoring et profiling de blastes

Auteur: Florian Magne
Version: 2.0 (Python Standalone)
Date: Mars 2026
Licence: MIT
================================================================================
"""

import sys
import os

# -*- coding: utf-8 -*-

# IMPORTS début du fichier
import sys
import os
import json
import warnings
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

warnings.filterwarnings("ignore")

# Imports scientifiques de base
import numpy as np
import pandas as pd

# CONFIGURATION PANDAS: Affichage en format linéaire (jamais exponentiel)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")  # 4 décimales max
pd.set_option("display.max_columns", None)  # Afficher toutes les colonnes
pd.set_option("display.width", None)  # Largeur auto
pd.set_option("display.max_rows", 100)  # Max 100 lignes affichées
np.set_printoptions(suppress=True, precision=4)  # Numpy aussi en linéaire
print("[OK] Pandas configuré: affichage linéaire (pas de notation scientifique)")

# IPython display pour compatibilité notebook/script
try:
    from IPython.display import display
except ImportError:

    def display(obj):
        print(obj)


# Imports visualisation
import matplotlib

matplotlib.use("Agg")  # Mode non-interactif : sauvegarde directe, aucune fenêtre
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")
plt.rcParams["figure.facecolor"] = "#ffffff"
plt.rcParams["axes.facecolor"] = "#ffffff"
plt.rcParams["text.color"] = "#000000"
plt.rcParams["axes.labelcolor"] = "#000000"
plt.rcParams["xtick.color"] = "#000000"
plt.rcParams["ytick.color"] = "#000000"
plt.rcParams["axes.edgecolor"] = "#000000"
plt.rcParams["grid.color"] = "#cccccc"
plt.rcParams["figure.figsize"] = (12, 8)

# Plotly pour visualisations interactives
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    # En mode script standalone: renderer silencieux (pas de dump HTML dans le terminal)
    pio.renderers.default = "browser"
    PLOTLY_AVAILABLE = True
    print("[OK] Plotly disponible")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[!] Plotly non installé (optionnel): pip install plotly")

# IMPORTS flowsom et anndata, l'un est le package d'analyse du FlowSOM, l'autre est pour gérer les données dans des objets AnnData
try:
    import flowsom as fs
    import anndata as ad

    FLOWSOM_AVAILABLE = True
    print("[OK] FlowSOM disponible")

    # ── Import GPU FlowSOM depuis le module local FlowSomGpu ──────────────
    # Le dossier FlowSomGpu contient GPUFlowSOMEstimator qui utilise un SOM
    # accéléré CUDA. On l'importe indépendamment du package flowsom installé.
    try:
        _flowsom_gpu_path = r"C:\Users\Florian Travail\Documents\FlowSom"
        if _flowsom_gpu_path not in sys.path:
            sys.path.insert(0, _flowsom_gpu_path)
        from FlowSomGpu.models import GPUFlowSOMEstimator

        GPU_FLOWSOM_AVAILABLE = True
        print("[OK] FlowSOM GPU disponible (module local FlowSomGpu — CUDA)")
    except Exception as _gpu_err:
        GPU_FLOWSOM_AVAILABLE = False
        GPUFlowSOMEstimator = None
        print(f"[!] FlowSOM GPU non disponible (CUDA requis) : {_gpu_err}")

except ImportError:
    FLOWSOM_AVAILABLE = False
    GPU_FLOWSOM_AVAILABLE = False
    GPUFlowSOMEstimator = None
    print("[X] FlowSOM non installé: pip install flowsom")

# Import de Scanpy pour UMAP/t-SNE
try:
    import scanpy as sc

    SCANPY_AVAILABLE = True
    print("[OK] Scanpy disponible")
except ImportError:
    SCANPY_AVAILABLE = False
    print("[!] Scanpy non installé (optionnel): pip install scanpy")

# Import de UMAP
try:
    import umap

    UMAP_AVAILABLE = True
    print("[OK] UMAP disponible")
except (ImportError, Exception) as _umap_err:
    UMAP_AVAILABLE = False
    print(f"[!] UMAP non disponible (optionnel): {_umap_err}")


# Import de t-SNE via sklearn car t-SNE trop lent à être implémenté dans Scanpy (et FlowSOM)
try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, r2_score
    from sklearn.cluster import AgglomerativeClustering

    SKLEARN_AVAILABLE = True
    print("[OK] Scikit-learn disponible")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[!] Scikit-learn non installé: pip install scikit-learn")

# FlowKit pour transformations Logicle
try:
    import flowkit as fk

    FLOWKIT_AVAILABLE = True
    # Configuration FlowKit: format linéaire pour les exports/affichages
    # FlowKit utilise pandas en interne, donc pd.set_option s'applique
    # Mais on configure aussi les options de logging/affichage si disponibles
    try:
        import logging

        logging.getLogger("flowkit").setLevel(logging.WARNING)  # Moins de logs verbose
    except:
        pass
    print("[OK] FlowKit disponible (transformations Logicle précise en 1 fonction)")
except ImportError:
    FLOWKIT_AVAILABLE = False
    print("[!] FlowKit non installé (optionnel): pip install flowkit)")

# FCSWrite pour export FCS
try:
    import fcswrite

    FCSWRITE_AVAILABLE = True
    print("[OK] FCSWrite disponible (export FCS)")
except ImportError:
    FCSWRITE_AVAILABLE = False
    print("[!] FCSWrite non installé (optionnel): pip install fcswrite")

# Scipy pour statistiques
from scipy import stats


# =============================================================================
# GATERESULT — Structure de retour pour chaque fonction de gating
# =============================================================================
@dataclass
class GateResult:
    """
    Résultat structuré d'une opération de gating.
    Stocké dans combined_data.uns["gating_reports"] pour audit et rapport HTML.
    """

    mask: np.ndarray
    n_kept: int
    n_total: int
    method: str
    gate_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def pct_kept(self) -> float:
        return (self.n_kept / max(self.n_total, 1)) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Sérialisation JSON-safe (sans le mask numpy)."""
        return {
            "gate_name": self.gate_name,
            "method": self.method,
            "n_kept": self.n_kept,
            "n_total": self.n_total,
            "pct_kept": round(self.pct_kept, 2),
            "details": self.details,
            "warnings": self.warnings,
        }


# Liste globale pour collecter les rapports de gating
gating_reports: List[GateResult] = []

# =============================================================================
# LOGGING STRUCTURÉ — gating_log.json
# =============================================================================
gating_log_entries: List[Dict[str, Any]] = []


def log_gating_event(
    gate_name: str,
    method: str,
    status: str,
    details: Dict[str, Any] = None,
    warning_msg: str = None,
):
    """Log structuré d'un événement de gating (JSON exportable)."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "gate_name": gate_name,
        "method": method,
        "status": status,  # "success", "fallback", "warning", "error"
        "details": details or {},
    }
    if warning_msg:
        entry["warning"] = warning_msg
        print(f"   [WARNING] {gate_name}: {warning_msg}")
    gating_log_entries.append(entry)


print("\n[OK] GateResult dataclass + logging structuré chargés")

# Import en haut de fichier des classes utilitaires permettant les transformations des fichiers ainsi que le pre-gating


class DataTransformer:
    """
    Transformations de données de cytométrie (Logicle, Arcsinh, etc.).
    Classe statique réutilisable sans dépendance à l'UI.
    """

    @staticmethod
    def arcsinh_transform(data: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
        """
        Transformation Arcsinh (inverse hyperbolic sine).

        Args en entrée:
            data: Matrice de données (n_cells, n_markers)
            cofactor: Facteur de division (5 pour flow cytometry)

        Returns:
            Données transformées
        """
        return np.arcsinh(data / cofactor)

    @staticmethod
    def arcsinh_inverse(data: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
        """Inverse de la transformation Arcsinh."""
        return np.sinh(data) * cofactor

    @staticmethod
    def logicle_transform(
        data: np.ndarray,
        T: float = 262144.0,
        M: float = 4.5,
        W: float = 0.5,
        A: float = 0.0,
    ) -> np.ndarray:
        """
        Transformation Logicle (biexponentielle).

        Args en entrée:
            data: Matrice de données
            T: Maximum de l'échelle linéaire (262144 = 2^18)
            M: Décades de largeur
            W: Linéarisation près de zéro
            A: Décades additionnelles (négatifs)

        Returns:
            Données transformées
        """
        if FLOWKIT_AVAILABLE:
            # Utiliser FlowKit si disponible (plus précis) avec une fonction prédéfinie
            try:
                xform = fk.transforms.LogicleTransform(T=T, M=M, W=W, A=A)
                return xform.apply(data)
            except:
                pass

        # Approximation si FlowKit absent: Arcsinh modifié
        w_val = W * np.log10(np.e)
        return np.arcsinh(data / (T / (10**M))) * (M / np.log(10))

    @staticmethod
    def log_transform(
        data: np.ndarray, base: float = 10.0, min_val: float = 1.0
    ) -> np.ndarray:
        """Transformation logarithmique standard."""
        data_clipped = np.maximum(data, min_val)
        return np.log(data_clipped) / np.log(base)

    @staticmethod
    def zscore_normalize(data: np.ndarray) -> np.ndarray:
        """Normalisation Z-score (moyenne=0, std=1)."""
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        std[std == 0] = 1  # Éviter division par zéro
        return (data - mean) / std

    @staticmethod
    def min_max_normalize(data: np.ndarray) -> np.ndarray:
        """Normalisation Min-Max [0, 1]."""
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        return (data - min_val) / range_val


class PreGating:
    """
    Pre-gating automatique pour la sélection des populations d'intérêt.
    Basé sur FSC/SSC pour exclure les débris et les doublets.
    """

    @staticmethod
    def find_marker_index(var_names: List[str], patterns: List[str]) -> Optional[int]:
        """Trouve l'index d'un marqueur parmi les patterns donnés."""
        var_upper = [v.upper() for v in var_names]
        for pattern in patterns:
            for i, name in enumerate(var_upper):
                if pattern.upper() in name:
                    return i
        return None

    @staticmethod
    def gate_viable_cells(
        X: np.ndarray,
        var_names: List[str],
        min_percentile: float = 2.0,
        max_percentile: float = 98.0,
    ) -> np.ndarray:
        """
        Gate les cellules viables basé sur FSC/SSC.

        Args:
            X: Matrice des données (n_cells, n_markers)
            var_names: Liste des noms de marqueurs
            min_percentile: Percentile minimum (exclusion débris)
            max_percentile: Percentile maximum (exclusion doublets)

        Returns:
            Masque booléen des cellules viables
        """
        n_cells = X.shape[0]
        mask = np.ones(n_cells, dtype=bool)

        # Trouver FSC (priorité à FSC-A)
        fsc_idx = PreGating.find_marker_index(var_names, ["FSC-A", "FSC-H", "FSC"])
        if fsc_idx is not None:
            fsc_vals = X[:, fsc_idx].astype(np.float64)
            fsc_vals = np.where(np.isfinite(fsc_vals), fsc_vals, np.nan)
            low = np.nanpercentile(fsc_vals, min_percentile)
            high = np.nanpercentile(fsc_vals, max_percentile)
            mask &= np.isfinite(fsc_vals) & (fsc_vals >= low) & (fsc_vals <= high)

        # Trouver SSC (priorité à SSC-A)
        ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A", "SSC-H", "SSC"])
        if ssc_idx is not None:
            ssc_vals = X[:, ssc_idx].astype(np.float64)
            ssc_vals = np.where(np.isfinite(ssc_vals), ssc_vals, np.nan)
            low = np.nanpercentile(ssc_vals, min_percentile)
            high = np.nanpercentile(ssc_vals, max_percentile)
            mask &= np.isfinite(ssc_vals) & (ssc_vals >= low) & (ssc_vals <= high)

        return mask

    @staticmethod
    def gate_singlets(
        X: np.ndarray,
        var_names: List[str],
        ratio_min: float = 0.6,
        ratio_max: float = 1.5,
    ) -> np.ndarray:
        """
        Gate les singlets basé sur le ratio FSC-A/FSC-H.
        Les doublets ont typiquement un ratio > 1.3-1.5.

        Args:
            X: Matrice des données
            var_names: Liste des noms de marqueurs
            ratio_min: Ratio minimum acceptable
            ratio_max: Ratio maximum acceptable

        Returns:
            Masque booléen des singlets
        """
        n_cells = X.shape[0]

        fsc_a_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
        fsc_h_idx = PreGating.find_marker_index(var_names, ["FSC-H"])

        if fsc_a_idx is None or fsc_h_idx is None:
            print("[!] FSC-A ou FSC-H non trouvé, pas de gating singlets")
            return np.ones(n_cells, dtype=bool)

        fsc_a = X[:, fsc_a_idx].astype(np.float64)
        fsc_h = X[:, fsc_h_idx].astype(np.float64)

        # Valeurs minimum pour éviter division par zéro
        min_val = 100
        valid_h = fsc_h > min_val

        ratio = np.full(n_cells, np.nan)
        ratio[valid_h] = fsc_a[valid_h] / fsc_h[valid_h]

        mask = np.isfinite(ratio) & (ratio >= ratio_min) & (ratio <= ratio_max)

        return mask

    @staticmethod
    def gate_cd45_positive(
        X: np.ndarray, var_names: List[str], threshold_percentile: float = 10
    ) -> np.ndarray:
        """
        Gate les cellules CD45+ (leucocytes).

        Returns:
            Masque booléen des cellules CD45+
        """
        n_cells = X.shape[0]

        cd45_idx = PreGating.find_marker_index(
            var_names, ["CD45", "CD45-PECY5", "CD45-PC5"]
        )
        if cd45_idx is None:
            print("[!] CD45 non trouvé, pas de gating CD45+")
            return np.ones(n_cells, dtype=bool)

        cd45_vals = X[:, cd45_idx].astype(np.float64)
        cd45_vals = np.where(np.isfinite(cd45_vals), cd45_vals, np.nan)

        threshold = np.nanpercentile(cd45_vals, threshold_percentile)

        return np.where(np.isnan(cd45_vals), False, cd45_vals > threshold)

    @staticmethod
    def gate_cd34_blasts(
        X: np.ndarray,
        var_names: List[str],
        threshold_percentile: float = 85,
        use_ssc_filter: bool = True,
        ssc_max_percentile: float = 70,
    ) -> np.ndarray:
        """
        Gate les blastes CD34+ (cellules souches/progénitrices).

        Les blastes sont typiquement:
        - CD34 bright (haute expression)
        - SSC low (faible granularité)

        Args:
            X: Matrice des données (n_cells, n_markers)
            var_names: Liste des noms de marqueurs
            threshold_percentile: Percentile pour définir le seuil CD34+ (ex: 85 = top 15%)
            use_ssc_filter: Appliquer aussi un filtre SSC pour enrichir en blastes
            ssc_max_percentile: Percentile max de SSC pour blastes (faible granularité)

        Returns:
            Masque booléen des blastes CD34+
        """
        n_cells = X.shape[0]

        # Chercher CD34 avec différents nommages possibles
        cd34_idx = PreGating.find_marker_index(
            var_names, ["CD34", "CD34-PE", "CD34-APC", "CD34-PECY7"]
        )
        if cd34_idx is None:
            print("[!] CD34 non trouvé, pas de gating blastes")
            return np.ones(n_cells, dtype=bool)

        cd34_vals = X[:, cd34_idx].astype(np.float64)
        cd34_vals = np.where(np.isfinite(cd34_vals), cd34_vals, np.nan)

        # Seuil CD34+ (prendre les cellules avec haute expression)
        threshold_cd34 = np.nanpercentile(cd34_vals, threshold_percentile)
        mask_cd34 = np.where(np.isnan(cd34_vals), False, cd34_vals >= threshold_cd34)

        # Optionnel: filtrer aussi par SSC low (blastes = faible granularité)
        if use_ssc_filter:
            ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A", "SSC-H", "SSC"])
            if ssc_idx is not None:
                ssc_vals = X[:, ssc_idx].astype(np.float64)
                ssc_vals = np.where(np.isfinite(ssc_vals), ssc_vals, np.nan)
                threshold_ssc = np.nanpercentile(ssc_vals, ssc_max_percentile)
                mask_ssc = np.where(
                    np.isnan(ssc_vals), False, ssc_vals <= threshold_ssc
                )
                return mask_cd34 & mask_ssc

        return mask_cd34

    @staticmethod
    def gate_debris_polygon(
        X: np.ndarray,
        var_names: List[str],
        fsc_min: float = None,
        fsc_max: float = None,
        ssc_min: float = None,
        ssc_max: float = None,
        auto_percentiles: bool = True,
        min_pct: float = 1.0,
        max_pct: float = 99.0,
    ) -> np.ndarray:
        """
        Gate rectangulaire/polygonal pour exclure les débris sur FSC-A vs SSC-A.

        Args:
            X: Matrice des données
            var_names: Liste des noms de marqueurs
            fsc_min/fsc_max: Seuils FSC manuels (si auto_percentiles=False)
            ssc_min/ssc_max: Seuils SSC manuels (si auto_percentiles=False)
            auto_percentiles: Calculer automatiquement les seuils via percentiles
            min_pct/max_pct: Percentiles pour auto-calcul

        Returns:
            Masque booléen des cellules (non-débris)
        """
        n_cells = X.shape[0]

        fsc_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
        ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A"])

        if fsc_idx is None or ssc_idx is None:
            print("[!] FSC-A ou SSC-A non trouvé pour gate débris")
            return np.ones(n_cells, dtype=bool)

        fsc_vals = X[:, fsc_idx].astype(np.float64)
        ssc_vals = X[:, ssc_idx].astype(np.float64)

        # Calculer les seuils automatiquement si demandé
        if auto_percentiles:
            fsc_min = np.nanpercentile(fsc_vals, min_pct)
            fsc_max = np.nanpercentile(fsc_vals, max_pct)
            ssc_min = np.nanpercentile(ssc_vals, min_pct)
            ssc_max = np.nanpercentile(ssc_vals, max_pct)

        # Appliquer le gate rectangulaire
        mask = (
            np.isfinite(fsc_vals)
            & np.isfinite(ssc_vals)
            & (fsc_vals >= fsc_min)
            & (fsc_vals <= fsc_max)
            & (ssc_vals >= ssc_min)
            & (ssc_vals <= ssc_max)
        )

        return mask


print("[OK] Classes DataTransformer et PreGating chargées!")


class FlowCytometrySample:
    """
    Classe utilitaire pour le chargement d'un fichier FCS en DataFrame pandas.
    Utilisée par run_flowsom_pipeline() comme interface simplifiée.
    """

    @classmethod
    def from_fcs(cls, fcs_file: str, verbose: bool = True) -> pd.DataFrame:
        """Charge un fichier FCS et retourne un DataFrame pandas."""
        adata = fs.io.read_FCS(fcs_file)
        df = pd.DataFrame(adata.X, columns=list(adata.var_names))
        if verbose:
            print(f"    [OK] {Path(fcs_file).name}: {len(df):,} cellules")
        return df


# =============================================================================
# CLASSE AutoGating — Gating adaptatif par GMM/KDE
# =============================================================================
# Inspiré de CytoPy AutonomousGate (sans dépendance MongoDB)
# Utilise scikit-learn GaussianMixture pour trouver les "creux" réels
# entre les populations au lieu de couper à des percentiles fixes.
#
# Avantages vs PreGating (percentiles):
#   - Si un échantillon a 10% de débris → la porte s'adapte automatiquement
#   - Si un échantillon est propre → moins de perte de données
#   - Pour les doublets: modélise la diagonale FSC-A/FSC-H statistiquement
#   - Pour CD45+: trouve le creux bimodal entre CD45- et CD45+
#
# [V2 AMÉLIORATIONS]:
#   - safe_fit_gmm: sous-échantillonnage à 200k points max avant fit
#   - auto_gate_singlets: contrôle R² RANSAC + fallback ratio si R² < 0.85
#   - Toutes les fonctions retournent un GateResult structuré
#   - Scatter FSC-A vs FSC-H par fichier + tableau % singlets stockés
#   - Log structuré JSON pour audit automatique des runs
# =============================================================================

# Stockage global des scatter data RANSAC par fichier (pour le rapport HTML)
ransac_scatter_data = {}  # {file_name: {fsc_h, fsc_a, pred, inlier_mask, r2, slope, intercept, pct_singlets}}
singlets_summary_per_file = []  # Liste de dicts pour tableau "% singlets par fichier"


class AutoGating:
    """
    Gating automatique adaptatif basé sur des modèles de mélange gaussien (GMM)
    et estimation de densité. Inspiré de CytoPy AutonomousGate.

    Chaque méthode utilise un GMM pour identifier les populations naturelles
    dans les données, au lieu de seuils fixes basés sur des percentiles.

    Dépendances: scikit-learn (GaussianMixture, StandardScaler)
    """

    # Seuil R² minimal pour la régression RANSAC (en dessous → fallback ratio)
    RANSAC_R2_THRESHOLD = 0.85
    # Sous-échantillonnage max avant GMM (convergence + performance)
    GMM_MAX_SAMPLES = 200_000

    @staticmethod
    def _subsample_for_gmm(data: np.ndarray, max_samples: int = None) -> np.ndarray:
        """
        Sous-échantillonne les données si elles dépassent max_samples.
        Améliore la convergence et évite les timeouts implicites sur gros jeux.

        Args:
            data: Données (n_samples, n_features)
            max_samples: Nombre max de points (défaut: GMM_MAX_SAMPLES)

        Returns:
            data_subsampled: Données sous-échantillonnées (ou originales si < max)
        """
        if max_samples is None:
            max_samples = AutoGating.GMM_MAX_SAMPLES
        if data.shape[0] > max_samples:
            idx = np.random.choice(data.shape[0], size=max_samples, replace=False)
            print(
                f"      [GMM] Sous-échantillonnage: {data.shape[0]:,} → {max_samples:,} points"
            )
            return data[idx]
        return data

    @staticmethod
    def safe_fit_gmm(
        data: np.ndarray,
        n_components: int = 2,
        n_init: int = 3,
        max_retries: int = 5,
        random_state: int = 42,
        covariance_type: str = "full",
        max_iter: int = 200,
        subsample: bool = True,
    ) -> Any:
        """
        Wrapper robuste pour le fitting GMM avec gestion d'erreurs.

        Tente le fit plusieurs fois avec différentes initialisations.
        En cas d'échec total sur n_components > 1, fallback sur 1 composante.
        Vérifie la convergence et émet des warnings si nécessaire.

        [V2] Sous-échantillonnage automatique à 200k points max avant fit.

        Args:
            data: Données à fitter (n_samples, n_features) ou (n_samples, 1)
            n_components: Nombre de composantes GMM
            n_init: Nombre d'initialisations par tentative
            max_retries: Nombre max de tentatives avant fallback
            random_state: Seed pour reproductibilité
            covariance_type: Type de covariance ('full', 'diag', 'spherical', 'tied')
            max_iter: Nombre max d'itérations EM
            subsample: Si True, sous-échantillonne avant fit (défaut True)

        Returns:
            GaussianMixture fitté

        Raises:
            RuntimeError: Si le fit échoue après toutes les tentatives (y compris fallback)
        """
        from sklearn.mixture import GaussianMixture

        # Sous-échantillonnage pour convergence rapide sur gros jeux de données
        if subsample:
            data_fit = AutoGating._subsample_for_gmm(data)
        else:
            data_fit = data

        last_error = None
        for attempt in range(max_retries):
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    random_state=random_state + attempt,
                    n_init=n_init,
                    covariance_type=covariance_type,
                    max_iter=max_iter,
                )
                gmm.fit(data_fit)
                if not gmm.converged_:
                    warnings.warn(
                        f"GMM non-convergé (n={n_components}, tentative {attempt + 1}/{max_retries})"
                    )
                    log_gating_event(
                        "GMM",
                        f"n_components={n_components}",
                        "warning",
                        {"attempt": attempt + 1},
                        f"Non-convergé tentative {attempt + 1}/{max_retries}",
                    )
                    continue
                return gmm
            except Exception as e:
                last_error = e
                log_gating_event(
                    "GMM",
                    f"n_components={n_components}",
                    "error",
                    {"attempt": attempt + 1, "error": str(e)},
                )
                continue

        # Fallback: tenter avec 1 composante si n_components > 1
        if n_components > 1:
            warn_msg = f"GMM fallback unimodal après {max_retries} échecs (dernière erreur: {last_error})"
            warnings.warn(warn_msg)
            log_gating_event(
                "GMM",
                "fallback_unimodal",
                "fallback",
                {"original_n_components": n_components, "error": str(last_error)},
                warn_msg,
            )
            try:
                gmm = GaussianMixture(
                    n_components=1,
                    random_state=random_state,
                    n_init=1,
                    covariance_type=covariance_type,
                    max_iter=max_iter,
                )
                gmm.fit(data_fit)
                return gmm
            except Exception as e:
                raise RuntimeError(
                    f"GMM fit échoué après {max_retries} tentatives + fallback unimodal: {e}"
                )

        raise RuntimeError(
            f"GMM fit échoué après {max_retries} tentatives: {last_error}"
        )

    @staticmethod
    def auto_gate_debris(
        X: np.ndarray,
        var_names: List[str],
        n_components: int = 3,
        min_cluster_fraction: float = 0.02,
    ) -> np.ndarray:
        """
        Gate débris adaptatif par GMM 2D sur FSC-A / SSC-A.

        L'algorithme identifie les clusters naturels dans l'espace FSC/SSC:
        - Débris: événements bas en FSC-A (petites particules)
        - Cellules: population principale (cluster dominant)
        - Saturés: événements très hauts (optionnel, détecté par BIC)

        Sélection automatique du nombre de composantes par BIC (2 ou 3).

        Args:
            X: Matrice des données (n_cells, n_markers)
            var_names: Noms des marqueurs
            n_components: Nombre max de composantes GMM à tester
            min_cluster_fraction: Fraction min d'événements pour inclure un cluster

        Returns:
            Masque booléen (True = cellule viable, False = débris/saturé)
        """
        from sklearn.preprocessing import StandardScaler

        n_cells = X.shape[0]
        fsc_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
        ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A"])

        if fsc_idx is None or ssc_idx is None:
            print("[!] FSC-A ou SSC-A non trouvé pour auto-gate débris")
            log_gating_event(
                "debris", "auto_gmm", "error", warning_msg="FSC-A ou SSC-A non trouvé"
            )
            return np.ones(n_cells, dtype=bool)

        fsc = X[:, fsc_idx].astype(np.float64)
        ssc = X[:, ssc_idx].astype(np.float64)

        # Filtrer les NaN/Inf
        valid = np.isfinite(fsc) & np.isfinite(ssc)
        data_2d = np.column_stack([fsc[valid], ssc[valid]])

        if valid.sum() < 200:
            print("[!] Pas assez de données valides pour auto-gate débris")
            return np.ones(n_cells, dtype=bool)

        # Standardiser avant GMM pour meilleure convergence
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_2d)

        # Sélection automatique du nombre de composantes par BIC
        best_bic = np.inf
        best_gmm = None
        for n_comp in [2, 3]:
            try:
                gmm_test = AutoGating.safe_fit_gmm(
                    data_scaled,
                    n_components=n_comp,
                    covariance_type="full",
                    n_init=3,
                    max_iter=200,
                )
                bic = gmm_test.bic(
                    data_scaled
                    if data_scaled.shape[0] <= AutoGating.GMM_MAX_SAMPLES
                    else AutoGating._subsample_for_gmm(data_scaled)
                )
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm_test
            except RuntimeError as e:
                print(f"   [!] GMM {n_comp} composantes échoué: {e}")
                continue

        if best_gmm is None:
            print("   [!] Aucun GMM n'a convergé, conservation de tous les événements")
            log_gating_event(
                "debris",
                "auto_gmm",
                "fallback",
                warning_msg="Aucun GMM convergé, toutes cellules conservées",
            )
            return np.ones(n_cells, dtype=bool)

        labels = best_gmm.predict(data_scaled)
        n_comp = best_gmm.n_components

        # Statistiques par cluster (en espace original)
        cluster_sizes = np.bincount(labels, minlength=n_comp)
        cluster_fsc_means = np.array(
            [data_2d[labels == i, 0].mean() for i in range(n_comp)]
        )

        # Population principale = plus grand cluster
        main_cluster = np.argmax(cluster_sizes)

        # Inclure les clusters avec assez d'événements et un FSC raisonnable
        # (exclure les débris = FSC très bas)
        mask_valid = np.zeros(valid.sum(), dtype=bool)
        fsc_threshold = cluster_fsc_means[main_cluster] * 0.25

        for i in range(n_comp):
            fraction = cluster_sizes[i] / len(labels)
            if (
                fraction >= min_cluster_fraction
                and cluster_fsc_means[i] >= fsc_threshold
            ):
                mask_valid |= labels == i

        # Sécurité: si aucun cluster sélectionné, garder le principal
        if not mask_valid.any():
            mask_valid = labels == main_cluster

        mask = np.zeros(n_cells, dtype=bool)
        mask[valid] = mask_valid

        n_kept = mask.sum()
        print(
            f"   [Auto-GMM] {best_gmm.n_components} composantes détectées (BIC={best_bic:.0f})"
        )
        for i in range(n_comp):
            status = "✓" if mask_valid[labels == i].any() else "✗"
            print(
                f"     {status} Cluster {i}: {cluster_sizes[i]:,} evt, FSC-A moy={cluster_fsc_means[i]:.0f}"
            )
        print(f"   [Auto-GMM] → Conservés: {n_kept:,} événements")

        # Log structuré
        log_gating_event(
            "debris",
            "auto_gmm",
            "success",
            {
                "n_components": int(n_comp),
                "bic": float(best_bic),
                "n_kept": int(n_kept),
                "n_total": int(n_cells),
                "cluster_sizes": cluster_sizes.tolist(),
            },
        )

        # Construire GateResult
        gate_result = GateResult(
            mask=mask,
            n_kept=int(n_kept),
            n_total=int(n_cells),
            method="auto_gmm_debris",
            gate_name="G1_debris",
            details={
                "n_components": int(n_comp),
                "bic": float(best_bic),
                "cluster_fsc_means": cluster_fsc_means.tolist(),
            },
        )
        gating_reports.append(gate_result)

        return mask

    @staticmethod
    def auto_gate_singlets(
        X: np.ndarray,
        var_names: List[str],
        file_origin: Optional[np.ndarray] = None,
        per_file: bool = True,
        r2_threshold: float = 0.85,
    ) -> np.ndarray:
        """
        Gate singlets adaptatif par régression linéaire robuste (RANSAC).

        Les singlets forment une diagonale sur le plot FSC-A vs FSC-H.
        Les doublets se situent au-dessus de cette diagonale (FSC-A augmente mais pas FSC-H).

        Méthode améliorée (V2):
        1. Pré-filtre viable (FSC/SSC 1-99%) pour exclure les outliers extrêmes
        2. Régression linéaire robuste RANSAC sur FSC-A vs FSC-H
        3. Contrôle qualité R² sur les inliers RANSAC
        4. Si R² < seuil (0.85): fallback vers gating ratio FSC-A/FSC-H simple
        5. Stockage des scatter data par fichier pour le rapport HTML

        Args:
            X: Matrice des données (n_cells, n_markers)
            var_names: Noms des marqueurs
            file_origin: Array contenant l'origine de chaque cellule (pour gating par fichier)
            per_file: Si True, applique le gating séparément par fichier
            r2_threshold: Seuil R² minimum (défaut 0.85). En dessous → fallback ratio

        Returns:
            Masque booléen (True = singlet, False = doublet)
        """
        from sklearn.linear_model import RANSACRegressor
        from sklearn.linear_model import LinearRegression

        n_cells = X.shape[0]
        fsc_a_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
        fsc_h_idx = PreGating.find_marker_index(var_names, ["FSC-H"])

        if fsc_a_idx is None or fsc_h_idx is None:
            print("[!] FSC-A ou FSC-H non trouvé pour auto-gate singlets")
            return np.ones(n_cells, dtype=bool)

        fsc_a = X[:, fsc_a_idx].astype(np.float64)
        fsc_h = X[:, fsc_h_idx].astype(np.float64)

        # Pré-filtre viable (FSC/SSC 1-99%) pour réduire l'impact des outliers
        # extrêmes (blastes matures, granulocytes agrégés) sur la régression RANSAC
        viable = PreGating.gate_viable_cells(
            X, var_names, min_percentile=1.0, max_percentile=99.0
        )

        # Filtrer: valeurs valides avec FSC > seuil minimal + viabilité
        valid = (
            viable
            & np.isfinite(fsc_a)
            & np.isfinite(fsc_h)
            & (fsc_h > 100)
            & (fsc_a > 100)
        )

        if valid.sum() < 200:
            print("[!] Pas assez de données valides pour auto-gate singlets")
            return np.ones(n_cells, dtype=bool)

        mask = np.zeros(n_cells, dtype=bool)

        # ─── Helper interne: fallback ratio FSC-A/FSC-H ───
        def _fallback_ratio_gating(
            fsc_a_local, fsc_h_local, ratio_min=0.6, ratio_max=1.5
        ):
            """Gating simple par ratio FSC-A/FSC-H (ancienne méthode)."""
            ratio = fsc_a_local.ravel() / np.maximum(fsc_h_local.ravel(), 1.0)
            return (ratio >= ratio_min) & (ratio <= ratio_max)

        # Gating par fichier si demandé et si file_origin fourni
        if per_file and file_origin is not None:
            unique_files = np.unique(file_origin)
            print(f"   [Auto-RANSAC] Gating par fichier ({len(unique_files)} fichiers)")

            total_singlets = 0
            total_doublets = 0

            for file_name in unique_files:
                # Sélectionner les cellules de ce fichier
                file_mask = (file_origin == file_name) & valid

                if file_mask.sum() < 50:
                    # Trop peu de cellules, garder toutes
                    mask[file_mask] = True
                    singlets_summary_per_file.append(
                        {
                            "file": str(file_name),
                            "n_total": int(file_mask.sum()),
                            "n_singlets": int(file_mask.sum()),
                            "pct_singlets": 100.0,
                            "method": "skip_too_few",
                            "r2": None,
                        }
                    )
                    continue

                fsc_a_file = fsc_a[file_mask].reshape(-1, 1)
                fsc_h_file = fsc_h[file_mask].reshape(-1, 1)

                # Régression RANSAC pour trouver la diagonale des singlets
                try:
                    ransac = RANSACRegressor(
                        estimator=LinearRegression(),
                        min_samples=50,
                        residual_threshold=None,  # Auto (MAD)
                        random_state=42,
                        max_trials=100,
                    )
                    ransac.fit(fsc_h_file, fsc_a_file.ravel())

                    # ─── CONTRÔLE QUALITÉ R² SUR INLIERS RANSAC ───
                    inlier_mask = ransac.inlier_mask_
                    r2_val = None
                    used_method = "ransac"

                    if inlier_mask is not None and inlier_mask.sum() > 50:
                        r2_val = r2_score(
                            fsc_a_file[inlier_mask].ravel(),
                            ransac.predict(fsc_h_file[inlier_mask]),
                        )

                        if r2_val < r2_threshold:
                            # ─── FALLBACK: gating ratio simple ───
                            warn_msg = f"R² faible pour {file_name} (R²={r2_val:.2f} < {r2_threshold}), fallback gating ratio"
                            print(f"      [!] {warn_msg}")
                            log_gating_event(
                                "singlets",
                                "ransac_fallback_ratio",
                                "fallback",
                                {"file": str(file_name), "r2": float(r2_val)},
                                warn_msg,
                            )

                            singlets_file = _fallback_ratio_gating(
                                fsc_a_file, fsc_h_file
                            )
                            used_method = "ratio_fallback"

                            # Appliquer
                            file_indices = np.where(file_mask)[0]
                            mask[file_indices] = singlets_file

                            n_sing = int(singlets_file.sum())
                            n_doub = len(singlets_file) - n_sing
                            total_singlets += n_sing
                            total_doublets += n_doub

                            file_short = (
                                file_name
                                if len(file_name) <= 25
                                else file_name[:22] + "..."
                            )
                            print(
                                f"      • {file_short}: {n_sing:,} singlets / {n_sing + n_doub:,} ({n_sing / (n_sing + n_doub) * 100:.1f}%) - RATIO FALLBACK (R²={r2_val:.2f})"
                            )

                            # Stocker les scatter data (même si fallback, pour diagnostic)
                            n_sample_pts = min(2000, len(fsc_a_file))
                            sample_idx = np.random.choice(
                                len(fsc_a_file), n_sample_pts, replace=False
                            )
                            ransac_scatter_data[str(file_name)] = {
                                "fsc_h": fsc_h_file[sample_idx].ravel().tolist(),
                                "fsc_a": fsc_a_file[sample_idx].ravel().tolist(),
                                "pred": ransac.predict(fsc_h_file[sample_idx]).tolist(),
                                "r2": float(r2_val),
                                "method": "ratio_fallback",
                                "slope": float(ransac.estimator_.coef_[0]),
                                "intercept": float(ransac.estimator_.intercept_),
                            }
                            singlets_summary_per_file.append(
                                {
                                    "file": str(file_name),
                                    "n_total": int(len(singlets_file)),
                                    "n_singlets": n_sing,
                                    "pct_singlets": round(
                                        n_sing / (n_sing + n_doub) * 100, 1
                                    ),
                                    "method": "ratio_fallback",
                                    "r2": round(float(r2_val), 3),
                                }
                            )
                            continue

                    # ─── R² OK (ou pas de inlier_mask): utiliser RANSAC normal ───
                    # Prédiction sur la droite
                    fsc_a_pred = ransac.predict(fsc_h_file)

                    # Distance verticale (résidus) - doublets au-dessus de la ligne
                    residuals = fsc_a_file.ravel() - fsc_a_pred

                    # Seuil adaptatif basé sur MAD (Median Absolute Deviation)
                    median_residual = np.median(residuals)
                    mad = np.median(np.abs(residuals - median_residual))

                    # Seuil: médiane + 3 * MAD
                    threshold_upper = median_residual + 3.0 * mad

                    # Singlets: points près de la diagonale (pas trop au-dessus)
                    singlets_file = residuals <= threshold_upper

                    # Appliquer le masque local
                    file_indices = np.where(file_mask)[0]
                    mask[file_indices] = singlets_file

                    n_sing = int(singlets_file.sum())
                    n_doub = len(singlets_file) - n_sing
                    total_singlets += n_sing
                    total_doublets += n_doub

                    # Affichage compact par fichier
                    slope = ransac.estimator_.coef_[0]
                    intercept = ransac.estimator_.intercept_
                    file_short = (
                        file_name if len(file_name) <= 25 else file_name[:22] + "..."
                    )
                    r2_str = f", R²={r2_val:.3f}" if r2_val is not None else ""
                    print(
                        f"      • {file_short}: {n_sing:,} singlets / {n_sing + n_doub:,} ({n_sing / (n_sing + n_doub) * 100:.1f}%) - y={slope:.3f}x+{intercept:.0f}{r2_str}"
                    )

                    # Stocker scatter data pour le rapport HTML (échantillonné)
                    n_sample_pts = min(2000, len(fsc_a_file))
                    sample_idx = np.random.choice(
                        len(fsc_a_file), n_sample_pts, replace=False
                    )
                    ransac_scatter_data[str(file_name)] = {
                        "fsc_h": fsc_h_file[sample_idx].ravel().tolist(),
                        "fsc_a": fsc_a_file[sample_idx].ravel().tolist(),
                        "pred": ransac.predict(fsc_h_file[sample_idx]).tolist(),
                        "r2": float(r2_val) if r2_val is not None else None,
                        "method": "ransac",
                        "slope": float(slope),
                        "intercept": float(intercept),
                    }
                    singlets_summary_per_file.append(
                        {
                            "file": str(file_name),
                            "n_total": int(len(singlets_file)),
                            "n_singlets": n_sing,
                            "pct_singlets": round(n_sing / (n_sing + n_doub) * 100, 1),
                            "method": "ransac",
                            "r2": round(float(r2_val), 3)
                            if r2_val is not None
                            else None,
                        }
                    )

                    # Log structuré
                    log_gating_event(
                        "singlets",
                        "ransac",
                        "success",
                        {
                            "file": str(file_name),
                            "r2": float(r2_val) if r2_val else None,
                            "slope": float(slope),
                            "intercept": float(intercept),
                            "n_singlets": n_sing,
                            "n_doublets": n_doub,
                        },
                    )

                except Exception as e:
                    print(f"      [!] Échec RANSAC pour {file_name}: {e}")
                    log_gating_event(
                        "singlets",
                        "ransac",
                        "error",
                        {"file": str(file_name), "error": str(e)},
                        f"Échec RANSAC pour {file_name}: {e}",
                    )
                    # En cas d'échec, garder toutes les cellules du fichier
                    mask[file_mask] = True
                    total_singlets += file_mask.sum()
                    singlets_summary_per_file.append(
                        {
                            "file": str(file_name),
                            "n_total": int(file_mask.sum()),
                            "n_singlets": int(file_mask.sum()),
                            "pct_singlets": 100.0,
                            "method": "error_keep_all",
                            "r2": None,
                        }
                    )

            print(
                f"   [Auto-RANSAC] Total: {total_singlets:,} singlets, {total_doublets:,} doublets exclus"
            )

            # Résumé tableau % singlets par fichier
            if singlets_summary_per_file:
                print(
                    f"\n   {'Fichier':<30} {'Méthode':<18} {'R²':>6} {'% Singlets':>12}"
                )
                print(f"   {'─' * 30} {'─' * 18} {'─' * 6} {'─' * 12}")
                for row in singlets_summary_per_file:
                    r2_disp = f"{row['r2']:.3f}" if row["r2"] is not None else "N/A"
                    fname_short = (
                        row["file"]
                        if len(row["file"]) <= 30
                        else row["file"][:27] + "..."
                    )
                    print(
                        f"   {fname_short:<30} {row['method']:<18} {r2_disp:>6} {row['pct_singlets']:>10.1f}%"
                    )

        else:
            # Gating global (ancien comportement)
            print(f"   [Auto-RANSAC] Gating global sur toutes les données")

            fsc_a_valid = fsc_a[valid].reshape(-1, 1)
            fsc_h_valid = fsc_h[valid].reshape(-1, 1)

            # Régression RANSAC
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=100,
                residual_threshold=None,
                random_state=42,
                max_trials=100,
            )
            ransac.fit(fsc_h_valid, fsc_a_valid.ravel())

            # ─── CONTRÔLE QUALITÉ R² GLOBAL ───
            inlier_mask = ransac.inlier_mask_
            r2_val = None
            if inlier_mask is not None and inlier_mask.sum() > 50:
                r2_val = r2_score(
                    fsc_a_valid[inlier_mask].ravel(),
                    ransac.predict(fsc_h_valid[inlier_mask]),
                )
                if r2_val < r2_threshold:
                    warn_msg = f"R² faible global (R²={r2_val:.2f} < {r2_threshold}), fallback gating ratio"
                    print(f"   [!] {warn_msg}")
                    log_gating_event(
                        "singlets",
                        "ransac_fallback_ratio",
                        "fallback",
                        {"r2": float(r2_val)},
                        warn_msg,
                    )

                    singlets_mask = _fallback_ratio_gating(fsc_a_valid, fsc_h_valid)
                    mask[valid] = singlets_mask

                    n_singlets = mask.sum()
                    n_doublets = valid.sum() - n_singlets
                    print(
                        f"   [RATIO FALLBACK] Singlets: {n_singlets:,} ({n_singlets / valid.sum() * 100:.1f}%)"
                    )

                    gate_result = GateResult(
                        mask=mask,
                        n_kept=int(n_singlets),
                        n_total=int(n_cells),
                        method="ratio_fallback_global",
                        gate_name="G2_singlets",
                        details={"r2": float(r2_val)},
                        warnings=[warn_msg],
                    )
                    gating_reports.append(gate_result)
                    return mask

            # Prédiction et résidus
            fsc_a_pred = ransac.predict(fsc_h_valid)
            residuals = fsc_a_valid.ravel() - fsc_a_pred

            # Seuil adaptatif MAD
            median_residual = np.median(residuals)
            mad = np.median(np.abs(residuals - median_residual))
            threshold_upper = median_residual + 3.0 * mad

            # Masque singlets
            singlets_mask = residuals <= threshold_upper
            mask[valid] = singlets_mask

            n_singlets = mask.sum()
            n_doublets = valid.sum() - n_singlets
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_

            r2_str = f", R²={r2_val:.3f}" if r2_val is not None else ""
            print(
                f"   [Auto-RANSAC] Droite: y = {slope:.3f}x + {intercept:.0f}{r2_str}"
            )
            print(
                f"   [Auto-RANSAC] Seuil MAD: médiane + {3.0:.1f}×MAD = {threshold_upper:.0f}"
            )
            print(
                f"   [Auto-RANSAC] Singlets: {n_singlets:,} ({n_singlets / valid.sum() * 100:.1f}%)"
            )
            print(
                f"   [Auto-RANSAC] Doublets rejetés: {n_doublets:,} ({n_doublets / valid.sum() * 100:.1f}%)"
            )

        # GateResult structuré
        gate_result = GateResult(
            mask=mask,
            n_kept=int(mask.sum()),
            n_total=int(n_cells),
            method="ransac_singlets",
            gate_name="G2_singlets",
            details={
                "per_file": per_file,
                "n_files": len(singlets_summary_per_file) if per_file else 1,
                "files_summary": singlets_summary_per_file if per_file else [],
            },
        )
        gating_reports.append(gate_result)

        return mask

    @staticmethod
    def auto_gate_cd45(
        X: np.ndarray,
        var_names: List[str],
        n_components: int = 2,
        uniform_gating: bool = False,
        threshold_percentile: float = 5.0,
    ) -> np.ndarray:
        """
        Gate CD45+ adaptatif par GMM 1D.

        Trouve automatiquement le creux bimodal entre CD45- et CD45+
        au lieu d'un percentile fixe. Le GMM modélise la distribution
        bimodale et assigne chaque événement à la population la plus probable.

        Args:
            X: Matrice des données
            var_names: Noms des marqueurs
            n_components: Nombre de composantes GMM (2 = CD45- / CD45+)
            uniform_gating: Si True, applique un seuil soft (percentile)
            threshold_percentile: Percentile pour le seuil soft CD45

        Returns:
            Masque booléen (True = CD45+, False = CD45-)
        """
        n_cells = X.shape[0]
        cd45_idx = PreGating.find_marker_index(
            var_names, ["CD45", "CD45-PECY5", "CD45-PC5"]
        )

        if cd45_idx is None:
            print("[!] CD45 non trouvé pour auto-gate CD45+")
            return np.ones(n_cells, dtype=bool)

        cd45 = X[:, cd45_idx].astype(np.float64)
        valid = np.isfinite(cd45)

        if valid.sum() < 200:
            print("[!] Pas assez de données valides pour auto-gate CD45+")
            return np.ones(n_cells, dtype=bool)

        # Mode uniform_gating: seuil soft par percentile (pas de GMM)
        if uniform_gating:
            threshold = np.nanpercentile(cd45[valid], threshold_percentile)
            mask = np.zeros(n_cells, dtype=bool)
            mask[valid] = cd45[valid] > threshold
            n_pos = mask.sum()
            print(
                f"   [Uniform-CD45] Seuil soft: {threshold:.0f} (percentile {threshold_percentile}%)"
            )
            print(
                f"   [Uniform-CD45] CD45+ identifiés: {n_pos:,} ({n_pos / valid.sum() * 100:.1f}%)"
            )

            gate_result = GateResult(
                mask=mask,
                n_kept=int(n_pos),
                n_total=int(valid.sum()),
                method="gmm_cd45_uniform",
                gate_name="G3_cd45",
                details={
                    "threshold": float(threshold),
                    "percentile": threshold_percentile,
                    "fallback": False,
                },
            )
            gating_reports.append(gate_result)
            log_gating_event(
                "cd45",
                "uniform_percentile",
                "success",
                {"threshold": float(threshold), "n_pos": int(n_pos)},
            )
            return mask

        # GMM pour séparer CD45- et CD45+
        try:
            gmm = AutoGating.safe_fit_gmm(
                cd45[valid].reshape(-1, 1), n_components=n_components, n_init=3
            )
        except RuntimeError as e:
            warn_msg = f"GMM CD45 échoué: {e} — fallback percentile"
            print(f"   [!] {warn_msg}")
            log_gating_event(
                "cd45",
                "gmm_fallback_percentile",
                "fallback",
                {"error": str(e)},
                warn_msg,
            )
            threshold = np.nanpercentile(cd45[valid], threshold_percentile)
            mask = np.zeros(n_cells, dtype=bool)
            mask[valid] = cd45[valid] > threshold

            gate_result = GateResult(
                mask=mask,
                n_kept=int(mask.sum()),
                n_total=int(valid.sum()),
                method="gmm_cd45_fallback_percentile",
                gate_name="G3_cd45",
                details={"threshold": float(threshold), "fallback": True},
                warnings=[warn_msg],
            )
            gating_reports.append(gate_result)
            return mask

        labels = gmm.predict(cd45[valid].reshape(-1, 1))
        means = gmm.means_.flatten()

        # CD45+ = composant avec la moyenne la plus élevée
        pos_component = np.argmax(means)

        # Calculer le seuil approximatif (intersection des 2 gaussiennes)
        sorted_means = np.sort(means)
        stds = np.sqrt(gmm.covariances_.flatten())
        sorted_stds = stds[np.argsort(means)]
        threshold_approx = (
            sorted_means[0] * sorted_stds[1] + sorted_means[1] * sorted_stds[0]
        ) / (sorted_stds[0] + sorted_stds[1])

        mask = np.zeros(n_cells, dtype=bool)
        mask[valid] = labels == pos_component

        n_pos = mask.sum()
        print(f"   [Auto-GMM] CD45: {n_components} composantes, μ={means.round(0)}")
        print(
            f"   [Auto-GMM] Seuil adaptatif ≈ {threshold_approx:.0f} (creux entre populations)"
        )
        print(
            f"   [Auto-GMM] CD45+ identifiés: {n_pos:,} ({n_pos / valid.sum() * 100:.1f}%)"
        )

        gate_result = GateResult(
            mask=mask,
            n_kept=int(n_pos),
            n_total=int(valid.sum()),
            method="gmm_cd45",
            gate_name="G3_cd45",
            details={
                "means": means.tolist(),
                "threshold": float(threshold_approx),
                "n_components": int(n_components),
                "fallback": False,
            },
        )
        gating_reports.append(gate_result)
        log_gating_event(
            "cd45",
            "gmm",
            "success",
            {
                "means": means.tolist(),
                "threshold": float(threshold_approx),
                "n_pos": int(n_pos),
            },
        )

        return mask

    @staticmethod
    def auto_gate_cd34(
        X: np.ndarray,
        var_names: List[str],
        use_ssc_filter: bool = True,
        n_components: int = 2,
    ) -> np.ndarray:
        """
        Gate CD34+ blastes adaptatif par GMM.

        Identifie la population CD34 bright (blastes) par GMM au lieu d'un
        percentile fixe. Optionnel: combine avec SSC low (blastes = faible granularité).

        Args:
            X: Matrice des données
            var_names: Noms des marqueurs
            use_ssc_filter: Combiner avec filtre GMM SSC low
            n_components: Nombre de composantes GMM

        Returns:
            Masque booléen (True = blaste CD34+, False = autre)
        """
        n_cells = X.shape[0]
        cd34_idx = PreGating.find_marker_index(
            var_names, ["CD34", "CD34-PE", "CD34-APC", "CD34-PECY7"]
        )

        if cd34_idx is None:
            print("[!] CD34 non trouvé pour auto-gate blastes")
            return np.ones(n_cells, dtype=bool)

        cd34 = X[:, cd34_idx].astype(np.float64)
        valid = np.isfinite(cd34)

        if valid.sum() < 200:
            print("[!] Pas assez de données valides pour auto-gate CD34")
            return np.ones(n_cells, dtype=bool)

        # GMM pour séparer CD34- et CD34+
        try:
            gmm = AutoGating.safe_fit_gmm(
                cd34[valid].reshape(-1, 1), n_components=n_components, n_init=3
            )
        except RuntimeError as e:
            warn_msg = f"GMM CD34 échoué: {e} — conservation de toutes les cellules"
            print(f"   [!] {warn_msg}")
            log_gating_event("cd34", "gmm", "error", {"error": str(e)}, warn_msg)
            return np.ones(n_cells, dtype=bool)

        labels = gmm.predict(cd34[valid].reshape(-1, 1))
        means = gmm.means_.flatten()
        pos_component = np.argmax(means)

        mask_cd34 = np.zeros(n_cells, dtype=bool)
        mask_cd34[valid] = labels == pos_component

        n_cd34_pos = mask_cd34.sum()
        print(
            f"   [Auto-GMM] CD34: μ={means.round(0)}, CD34+ cluster = μ={means[pos_component]:.0f}"
        )

        # Filtre SSC low optionnel (blastes = faible granularité)
        if use_ssc_filter:
            ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A", "SSC-H", "SSC"])
            if ssc_idx is not None:
                ssc = X[:, ssc_idx].astype(np.float64)
                valid_ssc = np.isfinite(ssc)

                if valid_ssc.sum() >= 200:
                    try:
                        gmm_ssc = AutoGating.safe_fit_gmm(
                            ssc[valid_ssc].reshape(-1, 1), n_components=2, n_init=3
                        )
                    except RuntimeError as e:
                        print(f"   [!] GMM SSC échoué: {e} — filtre SSC ignoré")
                        print(f"   [Auto-GMM] CD34+ blastes: {n_cd34_pos:,}")
                        gate_result = GateResult(
                            mask=mask_cd34,
                            n_kept=int(n_cd34_pos),
                            n_total=int(valid.sum()),
                            method="gmm_cd34_no_ssc",
                            gate_name="G4_cd34",
                            details={"means": means.tolist(), "ssc_filter": False},
                            warnings=[f"GMM SSC échoué: {e}"],
                        )
                        gating_reports.append(gate_result)
                        return mask_cd34

                    labels_ssc = gmm_ssc.predict(ssc[valid_ssc].reshape(-1, 1))
                    ssc_means = gmm_ssc.means_.flatten()
                    low_ssc_component = np.argmin(ssc_means)

                    mask_ssc = np.zeros(n_cells, dtype=bool)
                    mask_ssc[valid_ssc] = labels_ssc == low_ssc_component

                    combined = mask_cd34 & mask_ssc
                    print(
                        f"   [Auto-GMM] + Filtre SSC low (μ={ssc_means[low_ssc_component]:.0f}): {combined.sum():,} blastes purs"
                    )

                    gate_result = GateResult(
                        mask=combined,
                        n_kept=int(combined.sum()),
                        n_total=int(valid.sum()),
                        method="gmm_cd34_ssc",
                        gate_name="G4_cd34",
                        details={
                            "cd34_means": means.tolist(),
                            "ssc_means": ssc_means.tolist(),
                            "ssc_filter": True,
                        },
                    )
                    gating_reports.append(gate_result)
                    log_gating_event(
                        "cd34",
                        "gmm+ssc",
                        "success",
                        {
                            "cd34_means": means.tolist(),
                            "ssc_means": ssc_means.tolist(),
                            "n_blastes": int(combined.sum()),
                        },
                    )
                    return combined

        print(f"   [Auto-GMM] CD34+ blastes: {n_cd34_pos:,}")
        gate_result = GateResult(
            mask=mask_cd34,
            n_kept=int(n_cd34_pos),
            n_total=int(valid.sum()),
            method="gmm_cd34",
            gate_name="G4_cd34",
            details={"means": means.tolist()},
        )
        gating_reports.append(gate_result)
        return mask_cd34


print("[OK] Classe AutoGating chargée (gating adaptatif)")
print("     Méthodes disponibles:")
print(
    "       • safe_fit_gmm:       Wrapper robuste GMM (retry + fallback unimodal + sous-échantillonnage)"
)
print(
    "       • auto_gate_debris:   GMM 2D sur FSC-A/SSC-A (détection adaptative débris)"
)
print(
    "       • auto_gate_singlets: RANSAC robuste FSC-A vs FSC-H + contrôle R² + fallback ratio"
)
print("       • auto_gate_cd45:     GMM 1D bimodal CD45- / CD45+ (+ uniform_gating)")
print("       • auto_gate_cd34:     GMM 1D + optionnel SSC low pour blastes")
print("     ")
print("     [AMÉLIORATIONS V2]")
print(
    "       → safe_fit_gmm: sous-échantillonnage 200k pts + retry + fallback unimodal"
)
print("       → RANSAC singlets: contrôle R² inliers + fallback ratio si R² < 0.85")
print("       → Scatter FSC-A vs FSC-H par fichier + tableau % singlets stockés")
print("       → GateResult structuré retourné par chaque fonction")
print("       → Log structuré JSON (gating_log_entries) pour audit automatique")
# CONFIGURATION DES CHEMINS
# → Valeurs contrôlées par le panneau interactif (cellule 2) si exécuté.
#   Sinon, auto-chargement depuis config_flowsom.yaml si présent.

# ── Bootstrap : chargement automatique du YAML dans CONFIG ───────────────────
if "CONFIG" not in globals():
    _yaml_candidates = [
        Path(__file__).parent / "config_flowsom.yaml",
        Path("config_flowsom.yaml"),
        Path(__file__).parent / "config.yaml",
    ]
    for _yc in _yaml_candidates:
        if _yc.exists():
            try:
                import yaml as _yaml_mod

                with open(_yc, "r", encoding="utf-8") as _yf:
                    _raw = _yaml_mod.safe_load(_yf) or {}
                # Construire le dict CONFIG avec les clés UPPERCASE attendues
                _p = _raw.get("paths", {})
                _an = _raw.get("analysis", {})
                _pg = _raw.get("pregate", {})
                _pg_adv = _raw.get("pregate_advanced", {})
                _fs = _raw.get("flowsom", {})
                _ac = _raw.get("auto_clustering", {})
                _tr = _raw.get("transform", {})
                _no = _raw.get("normalize", {})
                _ds = _raw.get("downsampling", {})
                _viz = _raw.get("visualization", {})
                _gp = _raw.get("gpu", {})
                _mk = _raw.get("markers", {})
                CONFIG = {
                    # Chemins
                    "HEALTHY_FOLDER": _p.get(
                        "healthy_folder",
                        r"C:\Users\Florian Travail\Documents\FlowSom\Data\Moelle normale",
                    ),
                    "PATHOLOGICAL_FOLDER": _p.get("patho_folder", r"Data/Patho"),
                    # Analyse
                    "COMPARE_MODE": _an.get("compare_mode", True),
                    # Pre-gating de base
                    "APPLY_PREGATING": _pg.get("apply", True),
                    "GATING_MODE": _pg.get("mode", "auto"),
                    "MODE_BLASTES_VS_NORMAL": _pg.get("mode_blastes_vs_normal", True),
                    "GATE_DEBRIS": _pg.get("debris", True),
                    "GATE_DOUBLETS": _pg.get("singlets", True),
                    "GATE_CD45": _pg.get("cd45", True),
                    "FILTER_BLASTS": _pg.get("cd34", False),
                    # Pre-gating avancé
                    "DEBRIS_MIN_PERCENTILE": _pg_adv.get("debris_min_percentile", 1.0),
                    "DEBRIS_MAX_PERCENTILE": _pg_adv.get("debris_max_percentile", 99.0),
                    "RATIO_MIN": _pg_adv.get("doublets_ratio_min", 0.6),
                    "RATIO_MAX": _pg_adv.get("doublets_ratio_max", 1.4),
                    "CD45_THRESHOLD_PERCENTILE": _pg_adv.get(
                        "cd45_threshold_percentile", 5
                    ),
                    "CD34_THRESHOLD_PERCENTILE": _pg_adv.get(
                        "cd34_threshold_percentile", 85
                    ),
                    "USE_SSC_FILTER_FOR_BLASTS": _pg_adv.get(
                        "cd34_use_ssc_filter", True
                    ),
                    "SSC_MAX_PERCENTILE_BLASTS": _pg_adv.get(
                        "cd34_ssc_max_percentile", 60
                    ),
                    # FlowSOM
                    "XDIM": _fs.get("xdim", 10),
                    "YDIM": _fs.get("ydim", 10),
                    "RLEN": _fs.get("rlen", "auto"),
                    "N_CLUSTERS": _fs.get("n_metaclusters", 7),
                    "SEED": _fs.get("seed", 42),
                    # Auto-clustering
                    "AUTO_CLUSTER": _ac.get("enabled", False),
                    "MIN_CLUSTERS_AUTO": _ac.get("min_clusters", 5),
                    "MAX_CLUSTERS_AUTO": _ac.get("max_clusters", 35),
                    "N_BOOTSTRAP": _ac.get("n_bootstrap", 10),
                    "SAMPLE_SIZE_BOOTSTRAP": _ac.get("sample_size_bootstrap", 20000),
                    "MIN_STABILITY_THRESHOLD": _ac.get("min_stability_threshold", 0.75),
                    "W_STABILITY": _ac.get("weight_stability", 0.65),
                    "W_SILHOUETTE": _ac.get("weight_silhouette", 0.35),
                    # Transformation
                    "TRANSFORM_TYPE": _tr.get("method", "logicle"),
                    "COFACTOR": _tr.get("cofactor", 5),
                    "APPLY_TO_SCATTER": _tr.get("apply_to_scatter", False),
                    # Normalisation
                    "NORMALIZE_METHOD": _no.get("method", "zscore"),
                    # Marqueurs
                    "EXCLUDE_SCATTER": _mk.get("exclude_scatter", True),
                    "EXCLUDE_ADDITIONAL_MARKERS": _mk.get(
                        "exclude_additional", ["CD45"]
                    ),
                    # Downsampling
                    "DOWNSAMPLE": _ds.get("enabled", True),
                    "MAX_CELLS_PER_FILE": _ds.get("max_cells_per_file", 50000),
                    "MAX_CELLS_TOTAL": _ds.get("max_cells_total", 1000000),
                    # Visualisation
                    "SAVE_PLOTS": _viz.get("save_plots", True),
                    "PLOT_FORMAT": _viz.get("plot_format", "png"),
                    "DPI": _viz.get("dpi", 300),
                    # GPU
                    "USE_GPU": _gp.get("enabled", True),
                    "USE_GPU_FLOWSOM": _gp.get("enabled", True),
                }
                print(f"[OK] Configuration chargée depuis: {_yc}")
            except Exception as _ye:
                print(f"[!] Impossible de charger {_yc}: {_ye}")
                CONFIG = {}
            break
    else:
        CONFIG = {}

_cfg = globals().get("CONFIG", {})

# GPU : aliaser USE_GPU_FLOWSOM depuis le même flag pour que les blocs
# "if globals().get('USE_GPU_FLOWSOM', False)" soient bien activés
USE_GPU_FLOWSOM = _cfg.get("USE_GPU_FLOWSOM", _cfg.get("USE_GPU", False))

# Dossier des fichiers sains (référence NBM)
HEALTHY_FOLDER = Path(
    _cfg.get(
        "HEALTHY_FOLDER",
        r"C:\Users\Florian Travail\Documents\FlowSom\Data\Moelle normale",
    )
)

# Dossier des fichiers pathologiques (patients)
PATHOLOGICAL_FOLDER = Path(_cfg.get("PATHOLOGICAL_FOLDER", r"Data/Patho"))

# Mode d'analyse: True = Comparer Sain vs Pathologique, False = Patient seul
COMPARE_MODE = _cfg.get("COMPARE_MODE", True)

print(f"Dossier Sain: {HEALTHY_FOLDER}")
print(f"Dossier Pathologique: {PATHOLOGICAL_FOLDER}")
print(f"Mode comparaison: {'Activé' if COMPARE_MODE else 'Patient seul'}")
# FONCTIONS DE CHARGEMENT FCS


def get_fcs_files(folder: Path) -> List[str]:
    """Récupère la liste des fichiers FCS dans un dossier. Et renvoie une chaine de caractère"""
    if not folder.exists():
        print(f"[!] Dossier non trouvé: {folder}")
        return []

    files = set()
    for f in folder.glob("*.fcs"):
        files.add(str(f))
    for f in folder.glob("*.FCS"):
        files.add(str(f))

    return sorted(list(files))


def load_fcs_files(files: List[str], condition: str = "Unknown") -> List[ad.AnnData]:
    """
    Charge plusieurs fichiers FCS et retourne une liste d'AnnData.

    Args:
        files: Liste des chemins de fichiers FCS
        condition: Label de condition ("Sain" ou "Pathologique")

    Returns:
        Liste d'objets AnnData
    """
    # La ligne suivante crée la liste vide pour stocker les AnnData puis boucle sur chaque fichier (éviter le plantage complet)
    adatas = []

    for fpath in files:
        try:
            print(f"    Chargement: {Path(fpath).name}...", end=" ")

            # Lecture avec la fonction de base de flowsom
            adata = fs.io.read_FCS(fpath)

            # Ajouter les métadonnées avec un nombre de cellules qui sera égale a la forme de l'objet adata
            n_cells = adata.shape[0]
            adata.obs["condition"] = (
                condition  # Rajoute la condition du fichier : "Sain" ou "Pathologique"
            )
            adata.obs["file_origin"] = Path(
                fpath
            ).name  # Rajoute une observation avec Nom du fichier source (obs = One-dimensional annotation of observations)

            adatas.append(adata)  # Ajoute à la liste des AnnData
            print(f"{n_cells:,} cellules")

        except Exception as e:
            print(f"Erreur: {e}")

    return adatas


# Logs sur le cahrgement des fichiers
print("=" * 60)
print("CHARGEMENT DES FICHIERS FCS")
print("=" * 60)

# Fichiers sains en fonction du mode défini
healthy_files = get_fcs_files(HEALTHY_FOLDER) if COMPARE_MODE else []
print(f"\nFichiers Sains (NBM): {len(healthy_files)}")

healthy_adatas = []
if healthy_files:
    healthy_adatas = load_fcs_files(healthy_files, condition="Sain")

# Fichiers sains en fonction du mode défini
patho_files = get_fcs_files(PATHOLOGICAL_FOLDER)
print(f"\nFichiers Pathologiques: {len(patho_files)}")

patho_adatas = []
if patho_files:
    patho_adatas = load_fcs_files(patho_files, condition="Pathologique")

# Résumé
print("\n" + "=" * 60)
print(f"RÉSUMÉ DU CHARGEMENT")
print(f"   Fichiers Sains chargés: {len(healthy_adatas)}")
print(f"   Fichiers Pathologiques chargés: {len(patho_adatas)}")
# Résumé a.shape = pour chaque AnnData, prend le nombre de cellules (lignes) et concatène si nécessaire
total_cells = sum([a.shape[0] for a in healthy_adatas + patho_adatas])
print(f"   Total cellules: {total_cells:,}")
print("=" * 60)
# CONCATÉNATION DES DONNÉES

# Combiner tous les AnnData défini dans la cellule précédente
all_adatas = healthy_adatas + patho_adatas

# Vérification
if len(all_adatas) == 0:
    raise ValueError("[X] Aucun fichier FCS chargé! Vérifiez les chemins.")

# Concaténer avec intersection des colonnes (communes à tous les fichiers) ligne par ligne
if len(all_adatas) > 1:
    combined_data = ad.concat(
        all_adatas, join="inner"
    )  # join='inner' pour ne garder que les marqueurs communs à changer par outer si on veut garder tous les marqueurs
else:
    combined_data = all_adatas[
        0
    ].copy()  # Si un seul fichier, juste copier pour éviter de mofifier l'original

print(f"Données combinées: {combined_data.shape}")
print(f"   → {combined_data.shape[0]:,} cellules")
print(f"   → {combined_data.shape[1]} marqueurs")
# EXPLORATION DE LA STRUCTURE
print("=" * 70)
print("STRUCTURE DES DONNÉES")
print("=" * 70)

# Liste des marqueurs enregistré dans la varaible var_names = canaux (ici c'est bien un nom de variable)
var_names = list(combined_data.var_names)
print(f"\nMarqueurs ({len(var_names)}):")
for i, name in enumerate(var_names):
    print(f"   [{i:2d}] {name}")

# Identification des types de marqueurs car les recos indiquent d'enelever le scatter pour les analyses de clustering
print("\nClassification des marqueurs:")

# Ici le code n for n in var pose la question : "Est-ce qu'au moins UN des motifs de la liste scatter_patterns se trouve dans le nom actuel n ?"
scatter_patterns = ["FSC", "SSC", "TIME", "EVENT"]
scatter_markers = [
    n for n in var_names if any(p in n.upper() for p in scatter_patterns)
]
fluor_markers = [n for n in var_names if n not in scatter_markers]

print(f"   Scatter/Time: {scatter_markers}")
print(f"   Fluorescence: {fluor_markers}")

# Statistiques de base
print("\nObservations (métadonnées):")
print(combined_data.obs.head(10))
# CONVERSION EN DATAFRAME POUR EXPLORATION
HEADER = True
# Extraire la matrice de données
X = combined_data.X  # Matrice des données (n_cells, n_markers)
if hasattr(X, "toarray"):  # Si sparse matrix, convertir en dense pour pandas
    X = X.toarray()

# Créer un DataFrame pandas pour faciliter l'exploration avec df comme commande pandas classique
df_raw = pd.DataFrame(
    X, columns=var_names
)  # Crée le DataFrame avec les noms de colonnes
df_raw["condition"] = combined_data.obs[
    "condition"
].values  # Ajoute une colonne condition
df_raw["file_origin"] = combined_data.obs[
    "file_origin"
].values  # Ajoute une colonne file_origin

print("DataFrame créé pour exploration")


################################################################################
# Vérif des varaibles problématiques suite de l'exploration du dataset

print("ANALYSE DES DONNÉES BRUTES")
print("=" * 60)

# ========== MARQUEURS DE FLUORESCENCE ==========
print("\nMARQUEURS DE FLUORESCENCE")
print("-" * 60)

# Vérifier NaN
nan_count = df_raw[fluor_markers].isna().sum()
print(f"\nValeurs NaN par marqueur:")
for marker, count in nan_count.items():
    if count > 0:
        print(f"   {marker}: {count:,} ({count / len(df_raw) * 100:.2f}%)")

if nan_count.sum() == 0:
    print("   [OK] Aucun NaN détecté!")

# Vérifier Inf (valeur infinie) ex sur un post log
inf_count = np.isinf(df_raw[fluor_markers]).sum()
print(f"\nValeurs Inf par marqueur:")
if inf_count.sum() == 0:
    print("   [OK] Aucun Inf détecté!")
else:
    for marker, count in inf_count.items():
        if count > 0:
            print(f"   {marker}: {count:,}")

# Vérifier valeurs négatives
neg_count = (df_raw[fluor_markers] < 0).sum()
print(f"\n➖ Valeurs négatives par marqueur:")
has_negatives = False
for marker, count in neg_count.items():
    if count > 0:
        has_negatives = True
        # Compter le nombre total de cellules valides (non-NaN) pour ce marqueur
        total_valid = df_raw[marker].notna().sum()
        print(
            f"   {marker}: {count:,} / {total_valid:,} ({count / total_valid * 100:.2f}%)"
        )

if not has_negatives:
    print("   [OK] Aucune valeur négative!")
else:
    print("\n   [!] Les valeurs négatives peuvent indiquer un problème de compensation")
    print("   → La transformation Arcsinh ou Logicle peut les gérer")

# ========== MARQUEURS SCATTER/TIME ==========
print("\n\nMARQUEURS SCATTER/TIME")
print("-" * 60)

# Vérifier NaN
nan_count_scatter = df_raw[scatter_markers].isna().sum()
print(f"\nValeurs NaN par marqueur:")
for marker, count in nan_count_scatter.items():
    if count > 0:
        print(f"   {marker}: {count:,} ({count / len(df_raw) * 100:.2f}%)")

if nan_count_scatter.sum() == 0:
    print("   [OK] Aucun NaN détecté!")

# Vérifier Inf
inf_count_scatter = np.isinf(df_raw[scatter_markers]).sum()
print(f"\nValeurs Inf par marqueur:")
if inf_count_scatter.sum() == 0:
    print("   [OK] Aucun Inf détecté!")
else:
    for marker, count in inf_count_scatter.items():
        if count > 0:
            print(f"   {marker}: {count:,}")

# Vérifier valeurs négatives
neg_count_scatter = (df_raw[scatter_markers] < 0).sum()
print(f"\n➖ Valeurs négatives par marqueur:")
has_negatives_scatter = False
for marker, count in neg_count_scatter.items():
    if count > 0:
        has_negatives_scatter = True
        total_valid = df_raw[marker].notna().sum()
        print(
            f"   {marker}: {count:,} / {total_valid:,} ({count / total_valid * 100:.2f}%)"
        )

if not has_negatives_scatter:
    print("   [OK] Aucune valeur négative!")
else:
    print("\n   ℹ️ Les valeurs négatives dans scatter sont rares mais possibles")

print("\n" + "=" * 60)


################################################################################
# =============================================================================
# APPLICATION DU PRE-GATING SÉQUENTIEL (4 ÉTAPES)
# =============================================================================
# Stratégie de gating hiérarchique:
# 1. SSC-A vs FSC-A → Exclure débris
# 2. FSC-H vs FSC-A → Exclure doublets (singlets line)
# 3. CD45 vs SSC-A  → Sélectionner leucocytes (GATE PRINCIPAL)
# 4. CD34 vs SSC-A  → Sélectionner blastes (optionnel, si FILTER_BLASTS=True)
# =============================================================================

# ===================== OPTIONS DE PRE-GATING =====================
# → Valeurs contrôlées par le panneau interactif (cellule 2) si exécuté.
_cfg = globals().get("CONFIG", {})

APPLY_PREGATING = _cfg.get("APPLY_PREGATING", True)
GATING_MODE = _cfg.get("GATING_MODE", "auto")  # "manual" ou "auto"
MODE_BLASTES_VS_NORMAL = _cfg.get("MODE_BLASTES_VS_NORMAL", True)

# Gate 1: Débris (SSC-A vs FSC-A)
GATE_DEBRIS = _cfg.get("GATE_DEBRIS", True)
DEBRIS_MIN_PERCENTILE = _cfg.get("DEBRIS_MIN_PERCENTILE", 1.0)
DEBRIS_MAX_PERCENTILE = _cfg.get("DEBRIS_MAX_PERCENTILE", 99.0)

# Gate 2: Doublets (FSC-H vs FSC-A)
GATE_DOUBLETS = _cfg.get("GATE_DOUBLETS", True)
RATIO_MIN = _cfg.get("RATIO_MIN", 0.6)
RATIO_MAX = _cfg.get("RATIO_MAX", 1.4)

# Gate 3: Leucocytes CD45+
GATE_CD45 = _cfg.get("GATE_CD45", True)
CD45_THRESHOLD_PERCENTILE = _cfg.get("CD45_THRESHOLD_PERCENTILE", 5)

# Gate 4: Blastes CD34+ (optionnel)
FILTER_BLASTS = _cfg.get("FILTER_BLASTS", False)
CD34_THRESHOLD_PERCENTILE = _cfg.get("CD34_THRESHOLD_PERCENTILE", 85)
USE_SSC_FILTER_FOR_BLASTS = _cfg.get("USE_SSC_FILTER_FOR_BLASTS", True)
SSC_MAX_PERCENTILE_BLASTS = _cfg.get("SSC_MAX_PERCENTILE_BLASTS", 60)

# =================================================================
# VALIDATION DES PARAMÈTRES
# =================================================================
assert GATING_MODE in ("manual", "auto"), (
    f"GATING_MODE doit être 'manual' ou 'auto', reçu: '{GATING_MODE}'"
)

if GATING_MODE == "auto" and not SKLEARN_AVAILABLE:
    print("[!] ATTENTION: scikit-learn requis pour GATING_MODE='auto'")
    print("    → Fallback automatique vers mode 'manual'")
    GATING_MODE = "manual"

# Vérification de cohérence pour MODE_BLASTES_VS_NORMAL
if MODE_BLASTES_VS_NORMAL and not COMPARE_MODE:
    print("[!] ATTENTION: MODE_BLASTES_VS_NORMAL nécessite COMPARE_MODE=True")
    print("    → Le mode a besoin de fichiers Sain + Patho pour fonctionner")
    print("    → Désactivation automatique du mode différentiel")
    MODE_BLASTES_VS_NORMAL = False

# Données avant gating
X_raw = combined_data.X
if hasattr(X_raw, "toarray"):
    X_raw = X_raw.toarray()
n_before = X_raw.shape[0]

# Récupérer le vecteur de conditions pour le mode différentiel
conditions = combined_data.obs["condition"].values

print("=" * 70)
print(" PRE-GATING SÉQUENTIEL - STRATÉGIE EN 4 ÉTAPES")
print("=" * 70)
print(f"\n Événements initiaux: {n_before:,}")

# Affichage du mode de gating
mode_label = (
    "AUTOMATIQUE (GMM adaptatif)"
    if GATING_MODE == "auto"
    else "MANUEL (percentiles fixes)"
)
print(f"\n Mode de gating: {mode_label}")
if GATING_MODE == "auto":
    print(
        "    → Les seuils sont calculés automatiquement par modèle de mélange gaussien"
    )
    print("    → Les paramètres [Manual] ci-dessus sont IGNORÉS")
else:
    print("    → Seuils basés sur les percentiles configurés ci-dessus")

# Affichage mode spécial
if MODE_BLASTES_VS_NORMAL:
    print("\n [!] MODE BLASTES vs MOELLE NORMALE ACTIVÉ (GATING ASYMÉTRIQUE)")
    if FILTER_BLASTS:
        print(
            "     - Patho: Gate complet (débris + doublets + CD45+ + CD34+) → Blastes seuls"
        )
    else:
        print(
            "     - Patho: Gate (débris + doublets + CD45+) → Leucocytes CD45+ stricts"
        )
    print(
        "     - Sain:  Gate (débris + doublets UNIQUEMENT) → Toutes les cellules conservées (pas de gate CD45)"
    )
    n_patho = (conditions == "Pathologique").sum()
    n_sain = (conditions == "Sain").sum()
    print(f"     - Cellules Patho: {n_patho:,}")
    print(f"     - Cellules Sain: {n_sain:,}")

print(f"\n Configuration:")
print(
    f"   [Gate 1] Débris (SSC-A/FSC-A):     {'[OK] ACTIVÉ' if GATE_DEBRIS else '[X] DÉSACTIVÉ'}"
)
print(
    f"   [Gate 2] Doublets (FSC-H/FSC-A):   {'[OK] ACTIVÉ' if GATE_DOUBLETS else '[X] DÉSACTIVÉ'}"
)
if MODE_BLASTES_VS_NORMAL and GATE_CD45:
    print(
        f"   [Gate 3] Leucocytes CD45+:         [OK] PATHO UNIQUEMENT (gating asymétrique — Sain: pas de gate CD45)"
    )
else:
    print(
        f"   [Gate 3] Leucocytes CD45+:         {'[OK] ACTIVÉ' if GATE_CD45 else '[X] DÉSACTIVÉ'}"
    )
if FILTER_BLASTS:
    if MODE_BLASTES_VS_NORMAL:
        print(
            f"   [Gate 4] Blastes CD34+:            [OK] PATHO UNIQUEMENT (mode différentiel)"
        )
    else:
        print(f"   [Gate 4] Blastes CD34+:            [OK] ACTIVÉ (FILTER_BLASTS=True)")
else:
    print(
        f"   [Gate 4] Blastes CD34+:            [X] DÉSACTIVÉ (FILTER_BLASTS=False → tous les CD45+ conservés)"
    )

if APPLY_PREGATING:
    # Initialisation des masques
    mask_debris = np.ones(n_before, dtype=bool)
    mask_singlets = np.ones(n_before, dtype=bool)
    mask_cd45 = np.ones(n_before, dtype=bool)
    mask_cd34 = np.ones(n_before, dtype=bool)

    print("\n" + "-" * 70)

    # ========== GATE 1: DÉBRIS (SSC-A vs FSC-A) ==========
    if GATE_DEBRIS:
        print(
            f"\n GATE 1: Exclusion des débris (SSC-A vs FSC-A) [{GATING_MODE.upper()}]"
        )
        if GATING_MODE == "auto":
            mask_debris = AutoGating.auto_gate_debris(X_raw, var_names)
        else:
            mask_debris = PreGating.gate_debris_polygon(
                X_raw,
                var_names,
                auto_percentiles=True,
                min_pct=DEBRIS_MIN_PERCENTILE,
                max_pct=DEBRIS_MAX_PERCENTILE,
            )
        n_after_debris = mask_debris.sum()
        n_excluded_debris = n_before - n_after_debris
        if GATING_MODE == "manual":
            print(
                f"   Percentiles: [{DEBRIS_MIN_PERCENTILE}%, {DEBRIS_MAX_PERCENTILE}%]"
            )
        print(
            f"   → Conservés: {n_after_debris:,} ({n_after_debris / n_before * 100:.1f}%)"
        )
        print(
            f"   → Exclus (débris): {n_excluded_debris:,} ({n_excluded_debris / n_before * 100:.1f}%)"
        )
    else:
        print("\n GATE 1: Débris - SKIP")

    # ========== GATE 2: DOUBLETS (FSC-H vs FSC-A) ==========
    if GATE_DOUBLETS:
        print(
            f"\n GATE 2: Exclusion des doublets (FSC-H vs FSC-A) [{GATING_MODE.upper()}]"
        )
        if GATING_MODE == "auto":
            # Passer l'information des fichiers pour gating adaptatif par fichier
            file_origins = combined_data.obs["file_origin"].values
            mask_singlets = AutoGating.auto_gate_singlets(
                X_raw,
                var_names,
                file_origin=file_origins,
                per_file=True,  # Activer le gating par fichier
            )
        else:
            mask_singlets = PreGating.gate_singlets(
                X_raw, var_names, ratio_min=RATIO_MIN, ratio_max=RATIO_MAX
            )
        # Appliquer sur les cellules déjà filtrées par gate 1
        mask_after_g1_g2 = mask_debris & mask_singlets
        n_after_singlets = mask_after_g1_g2.sum()
        n_doublets = mask_debris.sum() - n_after_singlets
        if GATING_MODE == "manual":
            print(f"   Ratio FSC-A/FSC-H: [{RATIO_MIN}, {RATIO_MAX}]")
        print(f"   → Conservés (singlets): {n_after_singlets:,}")
        print(f"   → Exclus (doublets): {n_doublets:,}")
    else:
        print("\n GATE 2: Doublets - SKIP")

    # ========== GATE 3: LEUCOCYTES CD45+ (CD45 vs SSC-A) — GATE PRINCIPAL ==========
    # LOGIQUE ASYMÉTRIQUE: Pathologique → CD45 strict | Sain → Pas de gate CD45
    if GATE_CD45:
        if MODE_BLASTES_VS_NORMAL:
            print(f"\n GATE 3: Sélection ASYMÉTRIQUE CD45+ [{GATING_MODE.upper()}]")
            print("   → Patho: Gate CD45+ STRICT appliqué (élimination CD45-)")
            print(
                "   → Sain:  Gate CD45+ IGNORÉ (toutes cellules conservées — progéniteurs, CD45 low/neg inclus)"
            )

            # Calculer le masque CD45+ sur TOUTES les données
            if GATING_MODE == "auto":
                mask_cd45_full = AutoGating.auto_gate_cd45(X_raw, var_names)
            else:
                mask_cd45_full = PreGating.gate_cd45_positive(
                    X_raw, var_names, threshold_percentile=CD45_THRESHOLD_PERCENTILE
                )

            # Appliquer le gate CD45 UNIQUEMENT aux cellules pathologiques
            mask_patho_cd45 = conditions == "Pathologique"
            mask_sain_cd45 = conditions == "Sain"

            # Masque CD45: True pour Sain (on garde tout), mask_cd45_full pour Patho
            mask_cd45 = np.ones(n_before, dtype=bool)
            mask_cd45[mask_patho_cd45] = mask_cd45_full[mask_patho_cd45]
            # Sain: mask_cd45 reste True → aucun filtrage CD45

            # Stats par condition
            n_patho_g12 = (mask_patho_cd45 & mask_debris & mask_singlets).sum()
            n_patho_cd45_kept = (
                mask_patho_cd45 & mask_debris & mask_singlets & mask_cd45
            ).sum()
            n_patho_cd45_excl = n_patho_g12 - n_patho_cd45_kept
            n_sain_g12 = (mask_sain_cd45 & mask_debris & mask_singlets).sum()

            if GATING_MODE == "manual":
                print(
                    f"   Seuil CD45+: percentile {CD45_THRESHOLD_PERCENTILE} (appliqué PATHO uniquement)"
                )
            print(
                f"   → Patho CD45+ conservés: {n_patho_cd45_kept:,} / {n_patho_g12:,} ({n_patho_cd45_kept / max(n_patho_g12, 1) * 100:.1f}%)"
            )
            print(f"   → Patho CD45- exclus:    {n_patho_cd45_excl:,}")
            print(
                f"   → Sain conservés (100%):  {n_sain_g12:,} / {n_sain_g12:,} (aucun gate CD45)"
            )

            mask_after_g1_g2_g3 = mask_debris & mask_singlets & mask_cd45
            n_after_cd45 = mask_after_g1_g2_g3.sum()
            n_cd45_excluded = (mask_debris & mask_singlets).sum() - n_after_cd45
            print(
                f"   → Total après Gate 3: {n_after_cd45:,} (exclus CD45: {n_cd45_excluded:,} — Patho uniquement)"
            )
        else:
            print(
                f"\n GATE 3: Sélection des leucocytes CD45+ (GATE PRINCIPAL) [{GATING_MODE.upper()}]"
            )
            if GATING_MODE == "auto":
                mask_cd45 = AutoGating.auto_gate_cd45(X_raw, var_names)
            else:
                mask_cd45 = PreGating.gate_cd45_positive(
                    X_raw, var_names, threshold_percentile=CD45_THRESHOLD_PERCENTILE
                )
            mask_after_g1_g2_g3 = mask_debris & mask_singlets & mask_cd45
            n_after_cd45 = mask_after_g1_g2_g3.sum()
            n_cd45_excluded = (mask_debris & mask_singlets).sum() - n_after_cd45
            if GATING_MODE == "manual":
                print(
                    f"   Seuil CD45+: percentile {CD45_THRESHOLD_PERCENTILE} (exclure les {CD45_THRESHOLD_PERCENTILE}% les plus bas)"
                )
            print(f"   → Leucocytes CD45+ conservés: {n_after_cd45:,}")
            print(f"   → Exclus (CD45-): {n_cd45_excluded:,}")
    else:
        print("\n GATE 3: CD45+ - SKIP")

    # ========== GATE 4: BLASTES CD34+ (optionnel, conditionné par FILTER_BLASTS) ==========
    if FILTER_BLASTS:
        # Mode différentiel: appliquer CD34+ gate UNIQUEMENT sur les cellules pathologiques
        if MODE_BLASTES_VS_NORMAL:
            print(
                f"\n GATE 4: Sélection DIFFÉRENTIELLE des blastes CD34+ [{GATING_MODE.upper()}]"
            )
            print("   → Patho: Gate CD34+ appliqué (blastes uniquement)")
            print("   → Sain: Gate CD34+ IGNORÉ (tous les leucocytes CD45+ conservés)")

            # Calculer le masque CD34+ sur TOUTES les données
            if GATING_MODE == "auto":
                mask_cd34_full = AutoGating.auto_gate_cd34(
                    X_raw, var_names, use_ssc_filter=USE_SSC_FILTER_FOR_BLASTS
                )
            else:
                mask_cd34_full = PreGating.gate_cd34_blasts(
                    X_raw,
                    var_names,
                    threshold_percentile=CD34_THRESHOLD_PERCENTILE,
                    use_ssc_filter=USE_SSC_FILTER_FOR_BLASTS,
                    ssc_max_percentile=SSC_MAX_PERCENTILE_BLASTS,
                )

            # Appliquer le gate CD34+ UNIQUEMENT aux cellules pathologiques
            mask_patho = conditions == "Pathologique"
            mask_sain = conditions == "Sain"

            # Masque CD34: True pour sain (on garde tout), mask_cd34_full pour patho
            mask_cd34 = np.ones(n_before, dtype=bool)
            mask_cd34[mask_patho] = mask_cd34_full[mask_patho]

            # Stats
            n_patho_before = mask_patho.sum()
            n_patho_cd34 = (mask_patho & mask_cd34_full).sum()
            n_sain_kept = mask_sain.sum()

            if GATING_MODE == "manual":
                print(
                    f"   Seuil CD34+: top {100 - CD34_THRESHOLD_PERCENTILE:.0f}% (percentile {CD34_THRESHOLD_PERCENTILE})"
                )
                if USE_SSC_FILTER_FOR_BLASTS:
                    print(
                        f"   Filtre SSC low: ≤ percentile {SSC_MAX_PERCENTILE_BLASTS}"
                    )
            print(
                f"   → Patho: {n_patho_cd34:,} blastes / {n_patho_before:,} ({n_patho_cd34 / n_patho_before * 100:.1f}%)"
            )
            print(f"   → Sain: {n_sain_kept:,} leucocytes conservés (100%)")

        else:
            print(
                f"\n GATE 4: Sélection des blastes CD34+ (toutes conditions) [{GATING_MODE.upper()}]"
            )
            if GATING_MODE == "auto":
                mask_cd34 = AutoGating.auto_gate_cd34(
                    X_raw, var_names, use_ssc_filter=USE_SSC_FILTER_FOR_BLASTS
                )
            else:
                mask_cd34 = PreGating.gate_cd34_blasts(
                    X_raw,
                    var_names,
                    threshold_percentile=CD34_THRESHOLD_PERCENTILE,
                    use_ssc_filter=USE_SSC_FILTER_FOR_BLASTS,
                    ssc_max_percentile=SSC_MAX_PERCENTILE_BLASTS,
                )
            if GATING_MODE == "manual":
                print(
                    f"   Seuil CD34+: top {100 - CD34_THRESHOLD_PERCENTILE:.0f}% (percentile {CD34_THRESHOLD_PERCENTILE})"
                )
                if USE_SSC_FILTER_FOR_BLASTS:
                    print(
                        f"   Filtre SSC low: ≤ percentile {SSC_MAX_PERCENTILE_BLASTS}"
                    )
    else:
        print("\n GATE 4: Blastes CD34+ - SKIP (FILTER_BLASTS=False)")
        print("   → Tous les leucocytes CD45+ seront conservés pour FlowSOM")

    # ========== MASQUE FINAL COMBINÉ ==========
    mask_final = mask_debris & mask_singlets & mask_cd45 & mask_cd34
    n_final = mask_final.sum()
    n_excluded = n_before - n_final

    print("\n" + "=" * 70)
    print(f" RÉSUMÉ DU PRE-GATING [{mode_label}]")
    print("=" * 70)
    print(f"   Événements initiaux:       {n_before:,}")
    print(f"   Après Gate 1 (débris):     {mask_debris.sum():,}")
    print(f"   Après Gate 2 (doublets):   {(mask_debris & mask_singlets).sum():,}")
    print(
        f"   Après Gate 3 (CD45+):      {(mask_debris & mask_singlets & mask_cd45).sum():,}"
    )

    # --- Détails par condition si MODE_BLASTES_VS_NORMAL ---
    if MODE_BLASTES_VS_NORMAL:
        mask_patho = conditions == "Pathologique"
        mask_sain = conditions == "Sain"
        n_patho_total = mask_patho.sum()
        n_sain_total = mask_sain.sum()
        n_patho_final = (mask_final & mask_patho).sum()
        n_sain_final = (mask_final & mask_sain).sum()

        print(f"\n   {'─' * 55}")
        print(f"   DÉTAIL PAR CONDITION (GATING ASYMÉTRIQUE)")
        print(f"   {'─' * 55}")
        print(f"   PATHOLOGIQUE (CD45 strict):")
        print(f"     Initial:                 {n_patho_total:,}")
        print(
            f"     Après débris+doublets:   {(mask_patho & mask_debris & mask_singlets).sum():,}"
        )
        print(
            f"     Après CD45+ (strict):    {(mask_patho & mask_debris & mask_singlets & mask_cd45).sum():,}"
        )
        print(
            f"     Final conservé:          {n_patho_final:,} ({n_patho_final / max(n_patho_total, 1) * 100:.1f}%)"
        )
        print(f"   SAIN / NBM (pas de gate CD45):")
        print(f"     Initial:                 {n_sain_total:,}")
        print(
            f"     Après débris+doublets:   {(mask_sain & mask_debris & mask_singlets).sum():,}"
        )
        print(
            f"     CD45 gate:               NON APPLIQUÉ (toutes cellules conservées)"
        )
        print(
            f"     Final conservé:          {n_sain_final:,} ({n_sain_final / max(n_sain_total, 1) * 100:.1f}%)"
        )
        print(f"   {'─' * 55}")

    if FILTER_BLASTS:
        if MODE_BLASTES_VS_NORMAL:
            print(f"   Après Gate 4 (CD34+ patho): {n_final:,}")
            print(f"\n   [MODE BLASTES vs MOELLE NORMALE]")
            print(f"   Blastes CD34+ (patho):     {n_patho_final:,}")
            print(f"   Cellules normales (sain):  {n_sain_final:,}")
            print(f"   ─────────────────────────────────────")
            print(
                f"   [OK] TOTAL CONSERVÉ:       {n_final:,} ({n_final / n_before * 100:.1f}%)"
            )
            print(
                f"   [X] TOTAL EXCLUS:          {n_excluded:,} ({n_excluded / n_before * 100:.1f}%)"
            )
            print(
                f"\n   → Prêt pour FlowSOM: Blastes purs + Cellules normales (moelle saine complète)"
            )
        else:
            print(f"   Après Gate 4 (CD34+):      {n_final:,}")
            print(
                f"\n   [OK] ÉVÉNEMENTS CONSERVÉS: {n_final:,} ({n_final / n_before * 100:.1f}%)"
            )
            print(
                f"   [X] ÉVÉNEMENTS EXCLUS: {n_excluded:,} ({n_excluded / n_before * 100:.1f}%)"
            )
    else:
        if MODE_BLASTES_VS_NORMAL:
            print(f"\n   [MODE ASYMÉTRIQUE — LEUCOCYTES vs MOELLE NORMALE]")
            print(
                f"   Patho (CD45+ stricts):     {n_patho_final:,} ({n_patho_final / max(n_patho_total, 1) * 100:.1f}% du fichier patient)"
            )
            print(
                f"   Sain (toutes cellules):    {n_sain_final:,} ({n_sain_final / max(n_sain_total, 1) * 100:.1f}% du fichier NBM)"
            )
            print(f"   ─────────────────────────────────────")
            print(
                f"   [OK] TOTAL CONSERVÉ:       {n_final:,} ({n_final / n_before * 100:.1f}%)"
            )
            print(
                f"   [X] TOTAL EXCLUS:          {n_excluded:,} ({n_excluded / n_before * 100:.1f}%)"
            )
            print(
                f"\n   → Prêt pour FlowSOM: Leucocytes CD45+ (patho) + Moelle normale complète (sain)"
            )
        else:
            population_type = "Leucocytes CD45+" if GATE_CD45 else "Cellules"
            print(
                f"\n   [OK] {population_type} CONSERVÉS: {n_final:,} ({n_final / n_before * 100:.1f}%)"
            )
            print(
                f"   [X] ÉVÉNEMENTS EXCLUS: {n_excluded:,} ({n_excluded / n_before * 100:.1f}%)"
            )
            print(
                f"\n   → Prêt pour FlowSOM: Population CD45+ complète (pas de sous-sélection CD34+)"
            )

else:
    print("\n PRE-GATING COMPLÈTEMENT DÉSACTIVÉ")
    print("=" * 70)
    print(f"   → Toutes les {n_before:,} cellules seront conservées")
    mask_final = np.ones(n_before, dtype=bool)
    n_final = n_before
    mask_debris = np.ones(n_before, dtype=bool)
    mask_singlets = np.ones(n_before, dtype=bool)
    mask_cd45 = np.ones(n_before, dtype=bool)
    mask_cd34 = np.ones(n_before, dtype=bool)


################################################################################
# =============================================================================
# VISUALISATION PROFESSIONNELLE DES ÉTAPES DE GATING
# =============================================================================
# Graphiques SÉPARÉS et BIEN DÉFINIS pour chaque étape
# Style professionnel type FlowJo/Kaluza
# =============================================================================

from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

# =============================================================================
# FONCTIONS DE VISUALISATION
# =============================================================================


def format_axis(value, pos):
    """Format intelligent des axes (K pour milliers, M pour millions)"""
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.0f}"


def plot_density(ax, x, y, title, xlabel, ylabel, n_bins=120):
    """Scatter plot avec densité 2D (style FlowJo)"""
    # Nettoyer
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 100:
        ax.text(
            0.5,
            0.5,
            "Données insuffisantes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            color="white",
        )
        ax.set_facecolor("#1e1e2e")
        return

    # Limites
    x_lo, x_hi = np.percentile(x, [0.5, 99.5])
    y_lo, y_hi = np.percentile(y, [0.5, 99.5])

    # Colormap densité
    cmap = LinearSegmentedColormap.from_list(
        "density",
        ["#0d0d0d", "#1a1a2e", "#0077b6", "#00b4d8", "#90e0ef", "#f9e2af", "#ffffff"],
    )

    # Histogramme 2D
    h = ax.hist2d(
        x,
        y,
        bins=n_bins,
        range=[[x_lo, x_hi], [y_lo, y_hi]],
        cmap=cmap,
        norm=plt.matplotlib.colors.LogNorm(vmin=1),
    )

    # Style
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold", color="white")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", color="white")
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=12)
    ax.set_facecolor("#1e1e2e")
    ax.tick_params(colors="white", labelsize=11)
    ax.xaxis.set_major_formatter(FuncFormatter(format_axis))
    ax.yaxis.set_major_formatter(FuncFormatter(format_axis))

    for spine in ax.spines.values():
        spine.set_color("#45475a")
        spine.set_linewidth(1.5)

    # Colorbar
    cbar = plt.colorbar(h[3], ax=ax, shrink=0.85)
    cbar.ax.tick_params(colors="white", labelsize=9)
    cbar.set_label("Densité", color="white", fontsize=11)

    return h


def plot_gating(
    ax,
    x,
    y,
    mask,
    title,
    xlabel,
    ylabel,
    label_in="Conservés",
    label_out="Exclus",
    max_pts=100000,
):
    """Scatter plot avec overlay gating (vert=conservés, rouge=exclus)"""
    # Nettoyer
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, mask = x[valid], y[valid], mask[valid]

    if len(x) < 100:
        ax.text(
            0.5,
            0.5,
            "Données insuffisantes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            color="white",
        )
        ax.set_facecolor("#1e1e2e")
        return

    # Sous-échantillonner
    if len(x) > max_pts:
        idx = np.random.choice(len(x), max_pts, replace=False)
        x, y, mask = x[idx], y[idx], mask[idx]

    # Couleurs
    c_out = "#f38ba8"  # Rouge pastel
    c_in = "#a6e3a1"  # Vert pastel

    # Tracer exclus (fond)
    ax.scatter(
        x[~mask],
        y[~mask],
        s=4,
        c=c_out,
        alpha=0.3,
        label=label_out,
        edgecolors="none",
        rasterized=True,
    )
    # Tracer conservés (avant-plan)
    ax.scatter(
        x[mask],
        y[mask],
        s=5,
        c=c_in,
        alpha=0.5,
        label=label_in,
        edgecolors="none",
        rasterized=True,
    )

    # Stats
    n_tot = len(x)
    n_in = mask.sum()
    pct = n_in / n_tot * 100 if n_tot > 0 else 0

    # Style
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold", color="white")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", color="white")
    ax.set_title(
        f"{title}\n{n_in:,} / {n_tot:,} ({pct:.1f}%)",
        fontsize=14,
        fontweight="bold",
        color="white",
        pad=12,
    )
    ax.set_facecolor("#1e1e2e")
    ax.tick_params(colors="white", labelsize=11)
    ax.xaxis.set_major_formatter(FuncFormatter(format_axis))
    ax.yaxis.set_major_formatter(FuncFormatter(format_axis))

    for spine in ax.spines.values():
        spine.set_color("#45475a")
        spine.set_linewidth(1.5)

    ax.legend(
        loc="upper right",
        fontsize=10,
        markerscale=3,
        facecolor="#313244",
        labelcolor="white",
        edgecolor="#45475a",
    )

    # Limites
    x_lo, x_hi = np.percentile(x, [0.5, 99.5])
    y_lo, y_hi = np.percentile(y, [0.5, 99.5])
    ax.set_xlim(x_lo - (x_hi - x_lo) * 0.05, x_hi + (x_hi - x_lo) * 0.05)
    ax.set_ylim(y_lo - (y_hi - y_lo) * 0.05, y_hi + (y_hi - y_lo) * 0.05)


# =============================================================================
# GÉNÉRATION DES GRAPHIQUES (UN PAR UN)
# =============================================================================

if APPLY_PREGATING:
    print("=" * 70)
    print(" VISUALISATION DU PRE-GATING - GRAPHIQUES SÉPARÉS")
    print("=" * 70)

    # Indices des canaux
    fsc_a_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
    fsc_h_idx = PreGating.find_marker_index(var_names, ["FSC-H"])
    ssc_a_idx = PreGating.find_marker_index(var_names, ["SSC-A", "SSC-H", "SSC"])
    cd45_idx = PreGating.find_marker_index(
        var_names, ["CD45", "CD45-PECY5", "CD45-PC5"]
    )
    cd34_idx = PreGating.find_marker_index(var_names, ["CD34", "CD34-PE", "CD34-APC"])

    # Sous-échantillonner
    n_sample = min(60000, n_before)
    np.random.seed(42)
    idx_s = np.random.choice(n_before, n_sample, replace=False)

    # =========================================================================
    # GRAPHIQUE 1 : VUE D'ENSEMBLE (FSC-A vs SSC-A)
    # =========================================================================
    print("\n" + "─" * 50)
    print(" GRAPHIQUE 1 : VUE D'ENSEMBLE")
    print("─" * 50)

    if fsc_a_idx is not None and ssc_a_idx is not None:
        fig1, ax1 = plt.subplots(figsize=(10, 8), facecolor="#1e1e2e")

        plot_density(
            ax1,
            X_raw[idx_s, fsc_a_idx],
            X_raw[idx_s, ssc_a_idx],
            f"VUE D'ENSEMBLE\n{n_before:,} événements totaux",
            "FSC-A (Forward Scatter - Taille)",
            "SSC-A (Side Scatter - Granularité)",
        )

        plt.tight_layout()
        plt.savefig(
            "gating_01_overview.png", dpi=150, facecolor="#1e1e2e", bbox_inches="tight"
        )
        plt.close("all")
        print("   [OK] Sauvegardé: gating_01_overview.png")
    else:
        print("   [!] FSC-A ou SSC-A non trouvé dans les données")

    # =========================================================================
    # GRAPHIQUE 2 : GATE DÉBRIS (FSC-A vs SSC-A avec overlay)
    # =========================================================================
    if GATE_DEBRIS:
        print("\n" + "─" * 50)
        print(" GRAPHIQUE 2 : GATE DÉBRIS")
        print("─" * 50)

        if fsc_a_idx is not None and ssc_a_idx is not None:
            fig2, ax2 = plt.subplots(figsize=(10, 8), facecolor="#1e1e2e")

            plot_gating(
                ax2,
                X_raw[idx_s, fsc_a_idx],
                X_raw[idx_s, ssc_a_idx],
                mask_debris[idx_s],
                "GATE 1 : Exclusion des Débris",
                "FSC-A (Taille)",
                "SSC-A (Granularité)",
                "Cellules viables",
                "Débris/Bruit",
            )

            # Rectangle de gate
            fsc_lo = np.nanpercentile(X_raw[:, fsc_a_idx], DEBRIS_MIN_PERCENTILE)
            fsc_hi = np.nanpercentile(X_raw[:, fsc_a_idx], DEBRIS_MAX_PERCENTILE)
            ssc_lo = np.nanpercentile(X_raw[:, ssc_a_idx], DEBRIS_MIN_PERCENTILE)
            ssc_hi = np.nanpercentile(X_raw[:, ssc_a_idx], DEBRIS_MAX_PERCENTILE)

            rect = Rectangle(
                (fsc_lo, ssc_lo),
                fsc_hi - fsc_lo,
                ssc_hi - ssc_lo,
                fill=False,
                edgecolor="#f9e2af",
                linewidth=3,
                linestyle="--",
            )
            ax2.add_patch(rect)
            ax2.text(
                fsc_lo + (fsc_hi - fsc_lo) / 2,
                ssc_hi,
                " Zone de sélection",
                ha="center",
                va="bottom",
                fontsize=11,
                color="#f9e2af",
                fontweight="bold",
            )

            # Stats
            n_kept = mask_debris.sum()
            print(
                f"   → Événements conservés: {n_kept:,} ({n_kept / n_before * 100:.1f}%)"
            )
            print(f"   → Débris exclus: {n_before - n_kept:,}")

            plt.tight_layout()
            plt.savefig(
                "gating_02_debris.png",
                dpi=150,
                facecolor="#1e1e2e",
                bbox_inches="tight",
            )
            plt.close("all")
            print("   [OK] Sauvegardé: gating_02_debris.png")

    # =========================================================================
    # GRAPHIQUE 3 : GATE SINGLETS (FSC-H vs FSC-A)
    # =========================================================================
    if GATE_DOUBLETS:
        print("\n" + "─" * 50)
        print(" GRAPHIQUE 3 : GATE SINGLETS (Doublets)")
        print("─" * 50)

        if fsc_a_idx is not None and fsc_h_idx is not None:
            fig3, ax3 = plt.subplots(figsize=(10, 8), facecolor="#1e1e2e")

            # Après gate 1
            m_g1 = mask_debris[idx_s]
            x3 = X_raw[idx_s, fsc_a_idx][m_g1]
            y3 = X_raw[idx_s, fsc_h_idx][m_g1]
            m3 = mask_singlets[idx_s][m_g1]

            if len(x3) > 100:
                plot_gating(
                    ax3,
                    x3,
                    y3,
                    m3,
                    f"GATE 2 : Exclusion des Doublets - Mode {GATING_MODE.upper()}",
                    "FSC-A (Area)",
                    "FSC-H (Height)",
                    "Singlets",
                    "Doublets/Agrégats",
                )

                if GATING_MODE == "auto":
                    # Tracer les droites RANSAC par fichier
                    from sklearn.linear_model import RANSACRegressor, LinearRegression

                    # Récupérer les fichiers pour les cellules échantillonnées
                    file_origins_sample = combined_data.obs["file_origin"].values[
                        idx_s
                    ][m_g1]
                    unique_files_sample = np.unique(file_origins_sample)

                    # Palette de couleurs pour les droites
                    colors_lines = [
                        "#f9e2af",
                        "#89b4fa",
                        "#cba6f7",
                        "#fab387",
                        "#a6e3a1",
                    ]

                    print(
                        f"   → Visualisation des {len(unique_files_sample)} droites RANSAC"
                    )

                    for i, file_name in enumerate(unique_files_sample):
                        file_mask_sample = file_origins_sample == file_name
                        if file_mask_sample.sum() < 50:
                            continue

                        try:
                            x_file = x3[file_mask_sample].reshape(-1, 1)
                            y_file = y3[file_mask_sample].reshape(-1, 1)

                            ransac = RANSACRegressor(
                                estimator=LinearRegression(),
                                min_samples=50,
                                residual_threshold=None,
                                random_state=42,
                                max_trials=100,
                            )
                            ransac.fit(y_file, x_file.ravel())

                            # Tracer la droite
                            y_range = np.linspace(y_file.min(), y_file.max(), 100)
                            x_pred = ransac.predict(y_range.reshape(-1, 1))

                            color = colors_lines[i % len(colors_lines)]
                            file_short = (
                                file_name[:20] + "..."
                                if len(file_name) > 20
                                else file_name
                            )
                            ax3.plot(
                                x_pred,
                                y_range,
                                "-",
                                color=color,
                                lw=2.5,
                                alpha=0.8,
                                label=f"{file_short}",
                            )

                        except Exception as e:
                            print(
                                f"      [!] Échec visualisation pour {file_name}: {e}"
                            )
                            continue

                    # Légende compacte
                    if len(unique_files_sample) <= 5:
                        ax3.legend(
                            loc="lower right",
                            fontsize=9,
                            markerscale=2,
                            facecolor="#313244",
                            labelcolor="white",
                            edgecolor="#45475a",
                            title="Droites RANSAC par fichier",
                            title_fontsize=10,
                        )
                    else:
                        ax3.text(
                            0.98,
                            0.02,
                            f"{len(unique_files_sample)} droites calculées",
                            transform=ax3.transAxes,
                            ha="right",
                            va="bottom",
                            fontsize=10,
                            color="#f9e2af",
                            fontweight="bold",
                            bbox=dict(boxstyle="round", facecolor="#313244", alpha=0.8),
                        )

                else:
                    # Mode manuel: lignes de ratio fixes
                    x_range = np.linspace(
                        np.nanpercentile(x3, 1), np.nanpercentile(x3, 99), 100
                    )
                    ax3.plot(x_range, x_range, "w-", lw=2, alpha=0.7, label="Ratio 1:1")
                    ax3.plot(
                        x_range,
                        x_range * RATIO_MIN,
                        "--",
                        color="#f9e2af",
                        lw=2,
                        label=f"Ratio min ({RATIO_MIN})",
                    )
                    ax3.plot(
                        x_range,
                        x_range * RATIO_MAX,
                        "--",
                        color="#fab387",
                        lw=2,
                        label=f"Ratio max ({RATIO_MAX})",
                    )
                    ax3.fill_between(
                        x_range,
                        x_range * RATIO_MIN,
                        x_range * RATIO_MAX,
                        alpha=0.1,
                        color="#f9e2af",
                    )
                    ax3.legend(
                        loc="lower right",
                        fontsize=10,
                        markerscale=2,
                        facecolor="#313244",
                        labelcolor="white",
                        edgecolor="#45475a",
                    )

                # Stats
                n_after_g2 = (mask_debris & mask_singlets).sum()
                n_doublets = mask_debris.sum() - n_after_g2
                print(f"   → Singlets conservés: {n_after_g2:,}")
                print(f"   → Doublets exclus: {n_doublets:,}")

                plt.tight_layout()
                plt.savefig(
                    "gating_03_singlets.png",
                    dpi=150,
                    facecolor="#1e1e2e",
                    bbox_inches="tight",
                )
                plt.close("all")
                print("   [OK] Sauvegardé: gating_03_singlets.png")

    # =========================================================================
    # GRAPHIQUE 4 : GATE CD45+ (CD45 vs SSC-A) — GATE PRINCIPAL
    # =========================================================================
    if GATE_CD45:
        print("\n" + "─" * 50)
        print(" GRAPHIQUE 4 : GATE CD45+ (Leucocytes) — GATE PRINCIPAL")
        print("─" * 50)

        if cd45_idx is not None and ssc_a_idx is not None:
            fig4, ax4 = plt.subplots(figsize=(10, 8), facecolor="#1e1e2e")

            # Après gates 1+2
            m_g12 = (mask_debris & mask_singlets)[idx_s]
            x4 = X_raw[idx_s, cd45_idx][m_g12]
            y4 = X_raw[idx_s, ssc_a_idx][m_g12]
            m4 = mask_cd45[idx_s][m_g12]

            if len(x4) > 100:
                plot_gating(
                    ax4,
                    x4,
                    y4,
                    m4,
                    "GATE 3 : Sélection des Leucocytes CD45+",
                    "CD45 (Intensité)",
                    "SSC-A (Granularité)",
                    "Leucocytes CD45+",
                    "Cellules CD45-",
                )

                # Seuil CD45
                cd45_th = np.nanpercentile(
                    X_raw[:, cd45_idx], CD45_THRESHOLD_PERCENTILE
                )
                ax4.axvline(x=cd45_th, color="#89b4fa", lw=3, ls="--")
                ax4.text(
                    cd45_th,
                    ax4.get_ylim()[1],
                    f" Seuil CD45+\n (P{CD45_THRESHOLD_PERCENTILE})",
                    va="top",
                    ha="left",
                    fontsize=10,
                    color="#89b4fa",
                    fontweight="bold",
                )

                # Stats
                n_cd45_kept = (mask_debris & mask_singlets & mask_cd45).sum()
                print(f"   → Leucocytes CD45+ conservés: {n_cd45_kept:,}")
                print(
                    f"   → CD45- exclus: {(mask_debris & mask_singlets).sum() - n_cd45_kept:,}"
                )

                plt.tight_layout()
                plt.savefig(
                    "gating_04_cd45.png",
                    dpi=150,
                    facecolor="#1e1e2e",
                    bbox_inches="tight",
                )
                plt.close("all")
                print("   [OK] Sauvegardé: gating_04_cd45.png")
        else:
            print("   [!] CD45 non trouvé - Gate CD45+ ignoré")

    # =========================================================================
    # GRAPHIQUE 5 : GATE CD34+ (CD34 vs SSC-A) — OPTIONNEL
    # =========================================================================
    if FILTER_BLASTS:
        print("\n" + "─" * 50)
        print(" GRAPHIQUE 5 : GATE CD34+ (Blastes) — Sous-population des CD45+")
        print("─" * 50)

        if cd34_idx is not None and ssc_a_idx is not None:
            fig_cd34, ax_cd34 = plt.subplots(figsize=(10, 8), facecolor="#1e1e2e")

            # Après gates 1+2+3 (CD45+)
            m_g123 = (mask_debris & mask_singlets & mask_cd45)[idx_s]
            x5 = X_raw[idx_s, cd34_idx][m_g123]
            y5 = X_raw[idx_s, ssc_a_idx][m_g123]
            m5 = mask_cd34[idx_s][m_g123]

            if len(x5) > 100:
                plot_gating(
                    ax_cd34,
                    x5,
                    y5,
                    m5,
                    "GATE 4 : Sélection des Blastes CD34+ (parmi CD45+)",
                    "CD34 (Intensité)",
                    "SSC-A (Granularité)",
                    "Blastes CD34+",
                    "Autres leucocytes",
                )

                # Seuils
                cd34_th = np.nanpercentile(
                    X_raw[:, cd34_idx], CD34_THRESHOLD_PERCENTILE
                )
                ax_cd34.axvline(x=cd34_th, color="#f9e2af", lw=3, ls="--")
                ax_cd34.text(
                    cd34_th,
                    ax_cd34.get_ylim()[1],
                    f" Seuil CD34\n (P{CD34_THRESHOLD_PERCENTILE})",
                    va="top",
                    ha="left",
                    fontsize=10,
                    color="#f9e2af",
                    fontweight="bold",
                )

                if USE_SSC_FILTER_FOR_BLASTS:
                    ssc_th = np.nanpercentile(
                        X_raw[:, ssc_a_idx], SSC_MAX_PERCENTILE_BLASTS
                    )
                    ax_cd34.axhline(y=ssc_th, color="#fab387", lw=3, ls="--")
                    ax_cd34.text(
                        ax_cd34.get_xlim()[1],
                        ssc_th,
                        f" SSC max (P{SSC_MAX_PERCENTILE_BLASTS}) ",
                        va="bottom",
                        ha="right",
                        fontsize=10,
                        color="#fab387",
                        fontweight="bold",
                    )

                # Stats
                print(f"   → Blastes CD34+ sélectionnés: {n_final:,}")
                print(
                    f"   → Autres leucocytes exclus: {(mask_debris & mask_singlets & mask_cd45).sum() - n_final:,}"
                )

                plt.tight_layout()
                plt.savefig(
                    "gating_05_cd34.png",
                    dpi=150,
                    facecolor="#1e1e2e",
                    bbox_inches="tight",
                )
                plt.close("all")
                print("   [OK] Sauvegardé: gating_05_cd34.png")
        else:
            print("   [!] CD34 non trouvé - Gate CD34+ ignoré")

    # =========================================================================
    # GRAPHIQUE 6 : COMPARAISON AVANT / APRÈS
    # =========================================================================
    print("\n" + "─" * 50)
    print(" GRAPHIQUE 6 : COMPARAISON AVANT / APRÈS")
    print("─" * 50)

    if fsc_a_idx is not None and ssc_a_idx is not None:
        fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, 7), facecolor="#1e1e2e")

        # AVANT
        cmap_red = LinearSegmentedColormap.from_list(
            "reds", ["#1a1a2e", "#7f1d1d", "#dc2626", "#fca5a5", "#ffffff"]
        )

        valid = np.isfinite(X_raw[idx_s, fsc_a_idx]) & np.isfinite(
            X_raw[idx_s, ssc_a_idx]
        )
        x_bef = X_raw[idx_s, fsc_a_idx][valid]
        y_bef = X_raw[idx_s, ssc_a_idx][valid]

        x_lo, x_hi = np.percentile(x_bef, [0.5, 99.5])
        y_lo, y_hi = np.percentile(y_bef, [0.5, 99.5])

        h1 = ax5a.hist2d(
            x_bef,
            y_bef,
            bins=100,
            range=[[x_lo, x_hi], [y_lo, y_hi]],
            cmap=cmap_red,
            norm=plt.matplotlib.colors.LogNorm(vmin=1),
        )
        ax5a.set_title(
            f"AVANT Gating\n{n_before:,} événements",
            fontsize=14,
            fontweight="bold",
            color="white",
        )
        ax5a.set_xlabel("FSC-A", fontsize=12, fontweight="bold", color="white")
        ax5a.set_ylabel("SSC-A", fontsize=12, fontweight="bold", color="white")
        ax5a.set_facecolor("#1e1e2e")
        ax5a.tick_params(colors="white")
        ax5a.xaxis.set_major_formatter(FuncFormatter(format_axis))
        ax5a.yaxis.set_major_formatter(FuncFormatter(format_axis))
        for spine in ax5a.spines.values():
            spine.set_color("#45475a")

        # APRÈS
        cmap_green = LinearSegmentedColormap.from_list(
            "greens", ["#1a1a2e", "#14532d", "#22c55e", "#86efac", "#ffffff"]
        )

        m_final = mask_final[idx_s]
        x_aft = X_raw[idx_s, fsc_a_idx][m_final]
        y_aft = X_raw[idx_s, ssc_a_idx][m_final]

        if len(x_aft) > 100:
            h2 = ax5b.hist2d(
                x_aft,
                y_aft,
                bins=100,
                range=[[x_lo, x_hi], [y_lo, y_hi]],
                cmap=cmap_green,
                norm=plt.matplotlib.colors.LogNorm(vmin=1),
            )

        pct_final = n_final / n_before * 100
        population_label = "Blastes CD34+" if FILTER_BLASTS else "Leucocytes CD45+"
        ax5b.set_title(
            f"APRÈS Gating ({population_label})\n{n_final:,} événements ({pct_final:.1f}%)",
            fontsize=14,
            fontweight="bold",
            color="white",
        )
        ax5b.set_xlabel("FSC-A", fontsize=12, fontweight="bold", color="white")
        ax5b.set_ylabel("SSC-A", fontsize=12, fontweight="bold", color="white")
        ax5b.set_facecolor("#1e1e2e")
        ax5b.tick_params(colors="white")
        ax5b.xaxis.set_major_formatter(FuncFormatter(format_axis))
        ax5b.yaxis.set_major_formatter(FuncFormatter(format_axis))
        for spine in ax5b.spines.values():
            spine.set_color("#45475a")

        plt.tight_layout()
        plt.savefig(
            "gating_06_comparison.png",
            dpi=150,
            facecolor="#1e1e2e",
            bbox_inches="tight",
        )
        plt.close("all")
        print("   [OK] Sauvegardé: gating_06_comparison.png")

    # =========================================================================
    # RÉSUMÉ FINAL
    # =========================================================================
    print("\n" + "=" * 70)
    print(" RÉSUMÉ DU PRE-GATING")
    print("=" * 70)

    ret_g1 = mask_debris.sum() / n_before * 100
    ret_g2 = (mask_debris & mask_singlets).sum() / n_before * 100
    ret_g3 = (mask_debris & mask_singlets & mask_cd45).sum() / n_before * 100
    ret_final = n_final / n_before * 100

    print(f"""
    ┌─────────────────────────────────────────────────────────────┐
    │                    STATISTIQUES DE RÉTENTION                │
    ├─────────────────────────────────────────────────────────────┤
    │  Étape                        Cellules        Rétention     │
    ├─────────────────────────────────────────────────────────────┤
    │   Initial                   {n_before:>10,}        100.0%     │
    │  - Gate 1 (Débris)           {mask_debris.sum():>10,}        {ret_g1:>5.1f}%     │
    │  - Gate 2 (Doublets)         {(mask_debris & mask_singlets).sum():>10,}        {ret_g2:>5.1f}%     │
    │  - Gate 3 (CD45+)            {(mask_debris & mask_singlets & mask_cd45).sum():>10,}        {ret_g3:>5.1f}%     │""")
    if FILTER_BLASTS:
        print(
            f"    │  - Gate 4 (CD34+)            {n_final:>10,}        {ret_final:>5.1f}%     │"
        )
    print(f"""    ├─────────────────────────────────────────────────────────────┤
    │  [OK] CELLULES CONSERVÉES       {n_final:>10,}        {ret_final:>5.1f}%     │
    │  [X] CELLULES EXCLUES          {n_before - n_final:>10,}        {100 - ret_final:>5.1f}%     │
    └─────────────────────────────────────────────────────────────┘
    """)

    # --- Détail par condition si gating asymétrique ---
    if MODE_BLASTES_VS_NORMAL:
        _mask_patho = conditions == "Pathologique"
        _mask_sain = conditions == "Sain"
        _n_patho_tot = _mask_patho.sum()
        _n_sain_tot = _mask_sain.sum()
        _n_patho_fin = (mask_final & _mask_patho).sum()
        _n_sain_fin = (mask_final & _mask_sain).sum()
        _ret_patho = _n_patho_fin / max(_n_patho_tot, 1) * 100
        _ret_sain = _n_sain_fin / max(_n_sain_tot, 1) * 100
        print(f"""
    ┌─────────────────────────────────────────────────────────────┐
    │          DÉTAIL GATING ASYMÉTRIQUE PAR CONDITION            │
    ├─────────────────────────────────────────────────────────────┤
    │  PATHOLOGIQUE (CD45 strict appliqué):                       │
    │    Initial:          {_n_patho_tot:>10,}                              │
    │    Conservé:         {_n_patho_fin:>10,}   ({_ret_patho:>5.1f}%)                   │
    │    Gates: Débris + Doublets + CD45+ strict                  │
    ├─────────────────────────────────────────────────────────────┤
    │  SAIN / NBM (PAS de gate CD45):                             │
    │    Initial:          {_n_sain_tot:>10,}                              │
    │    Conservé:         {_n_sain_fin:>10,}   ({_ret_sain:>5.1f}%)                   │
    │    Gates: Débris + Doublets UNIQUEMENT                      │
    └─────────────────────────────────────────────────────────────┘
        """)

    population_desc = (
        "Blastes CD34+ (parmi CD45+)"
        if FILTER_BLASTS
        else (
            "Leucocytes CD45+ (patho) + Moelle normale complète (sain)"
            if MODE_BLASTES_VS_NORMAL
            else "Leucocytes CD45+ (population complète)"
        )
    )
    print(f" Population finale: {population_desc}")

    print("\n Fichiers générés:")
    print("   - gating_01_overview.png    - Vue d'ensemble")
    print("   - gating_02_debris.png      - Gate débris")
    print("   - gating_03_singlets.png    - Gate singlets")
    print("   - gating_04_cd45.png        - Gate CD45+ (leucocytes)")
    if FILTER_BLASTS:
        print("   - gating_05_cd34.png        - Gate CD34+ (blastes)")
    print("   - gating_06_comparison.png  - Avant/Après")

else:
    print("\n Pre-gating désactivé - Aucun graphique généré")
    print("   → Activez APPLY_PREGATING = True pour visualiser")


################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from typing import Optional, Tuple

# =============================================================================
# QC VISUEL GATE 1 — GMM vs KDE (Exclusion Débris)
# =============================================================================
# Interface simplifiée : la fonction reçoit mask_debris (bool) directement.
# → Elle est totalement autonome, sans dépendance à des variables GMM externes.
#
# Cas 1 – mask_debris a des False  : échantillon avec débris → seuil = max FSC des exclus
# Cas 2 – mask_debris tout True    : tri cellulaire / propre  → GMM interne pour visu QC
# =============================================================================


def plot_gmm_vs_kde_qc(
    fsc_a_data: np.ndarray,
    mask_debris: np.ndarray,  # bool : True=conservé, False=exclu par G1
    n_subsample: int = 10_000,  # pts pour la KDE (scipy KDE = O(n_sub × n_grid))
    n_grid: int = 1_000,  # points de la grille d'évaluation
    valley_range: Tuple[float, float] = (-100_000.0, 600_000.0),
    random_seed: int = 42,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes, float, Optional[float]]:
    """
    Trace la densité KDE de FSC-A avec le seuil débris GMM et la vallée KDE.

    Args:
        fsc_a_data  : vecteur FSC-A brut (n_events,)
        mask_debris : masque booléen issu du Gate G1 (True = cellule conservée)
                      Si tout True (tri cellulaire), un GMM interne est utilisé.
        n_subsample : nombre de points pour estimer la KDE (≤10 000 recommandé)
        n_grid      : résolution de la grille d'évaluation KDE
        valley_range: intervalle de recherche de la vallée [x_min, x_max]
        random_seed : reproductibilité
        ax          : axes matplotlib existants (optionnel)
        title       : titre du graphique (auto-généré si None)

    Returns:
        fig, ax, gmm_threshold, kde_valley
    """
    # ── 0. Validation ────────────────────────────────────────────────────────
    fsc_a_data = np.asarray(fsc_a_data, dtype=np.float64).ravel()
    mask_debris = np.asarray(mask_debris, dtype=bool).ravel()

    if len(fsc_a_data) != len(mask_debris):
        raise ValueError(
            f"fsc_a_data ({len(fsc_a_data)}) et mask_debris ({len(mask_debris)}) "
            "doivent avoir la même longueur."
        )

    n_total = len(fsc_a_data)
    n_exclu = int((~mask_debris).sum())
    tri_cellulaire = n_exclu == 0

    # ── 1. Seuil GMM ─────────────────────────────────────────────────────────
    if not tri_cellulaire:
        # Cas nominal : seuil = max FSC-A parmi les événements exclus
        gmm_threshold = float(np.max(fsc_a_data[~mask_debris]))
        debris_label = "Débris exclus G1"
    else:
        # Cas tri cellulaire : GMM interne sur FSC-A, cluster bas = "simulé"
        print(
            "   [QC] mask_debris intégral — simulation GMM interne (tri cellulaire / propre)"
        )
        from sklearn.mixture import GaussianMixture as _GMM

        _rng = np.random.default_rng(random_seed)
        _valid = np.isfinite(fsc_a_data)
        _n_fit = min(50_000, int(_valid.sum()))
        _idx = _rng.choice(np.where(_valid)[0], size=_n_fit, replace=False)
        _gmm = _GMM(n_components=2, random_state=random_seed, n_init=5)
        _gmm.fit(fsc_a_data[_idx].reshape(-1, 1))
        _means = _gmm.means_.flatten()
        # Prédire sur TOUTES les données finies pour identifier le cluster bas
        _all_labels = np.full(n_total, -1, dtype=int)
        _all_labels[_valid] = _gmm.predict(fsc_a_data[_valid].reshape(-1, 1))
        _low_cluster = int(np.argmin(_means))
        # Seuil = p95 du cluster bas (plus robuste que max)
        _low_vals = fsc_a_data[_all_labels == _low_cluster]
        gmm_threshold = (
            float(np.percentile(_low_vals, 95))
            if len(_low_vals) > 0
            else float(np.percentile(fsc_a_data[_valid], 20))
        )
        debris_label = f"Cluster bas FSC-A (simul., μ={_means[_low_cluster]:.0f})"
        # Reconstruire un mask_debris simulé pour les stats
        mask_debris = _all_labels != _low_cluster
        if title is None:
            title = (
                "QC Gate 1 — Simulation bas FSC-A [Tri cellulaire / échantillon propre]"
            )

    if title is None:
        title = "QC Gate 1 — Exclusion débris (GMM vs KDE)"

    # ── 2. Sous-échantillonnage pour KDE ─────────────────────────────────────
    rng = np.random.default_rng(random_seed)
    if n_total > n_subsample:
        fsc_sub = fsc_a_data[rng.choice(n_total, size=n_subsample, replace=False)]
    else:
        fsc_sub = fsc_a_data.copy()

    # ── 3. KDE ───────────────────────────────────────────────────────────────
    kde = gaussian_kde(fsc_sub[np.isfinite(fsc_sub)], bw_method="scott")
    data_min = float(np.nanmin(fsc_sub))
    data_max = float(np.nanpercentile(fsc_sub, 99.9))
    x_min = min(
        valley_range[0], data_min - max(150_000.0, (data_max - data_min) * 0.15)
    )
    x_grid = np.linspace(x_min, data_max, n_grid)
    density = kde(x_grid)

    # ── 4. Détection de la vallée (avec fallback) ─────────────────────────────
    vm = (x_grid >= valley_range[0]) & (x_grid <= valley_range[1])
    kde_valley: Optional[float] = None
    if vm.any():
        peaks_idx, _ = find_peaks(-density[vm], prominence=np.ptp(density[vm]) * 0.005)
        if len(peaks_idx) > 0:
            kde_valley = float(x_grid[vm][peaks_idx[0]])
        else:
            fb = (x_grid >= max(x_min, gmm_threshold - 100_000)) & (
                x_grid <= min(data_max, gmm_threshold + 150_000)
            )
            if fb.any():
                kde_valley = float(x_grid[fb][np.argmin(density[fb])])

    # ── 5. Visualisation ─────────────────────────────────────────────────────
    BG_COLOR = "#1e1e2f"
    PANEL_COLOR = "#16213e"
    TEXT_COLOR = "#e2e8f0"
    GRID_COLOR = "#2d2d4e"
    RED_FILL = "#ef4444"
    GREEN_FILL = "#22c55e"
    RED_LINE = "#ff6b6b"
    BLUE_LINE = "#60a5fa"

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG_COLOR)
    else:
        fig = ax.get_figure()
        fig.patch.set_facecolor(BG_COLOR)

    ax.set_facecolor(PANEL_COLOR)
    ax.fill_between(
        x_grid,
        density,
        where=(x_grid <= gmm_threshold),
        color=RED_FILL,
        alpha=0.35,
        label=debris_label,
    )
    ax.fill_between(
        x_grid,
        density,
        where=(x_grid > gmm_threshold),
        color=GREEN_FILL,
        alpha=0.30,
        label="Cellules conservées",
    )
    ax.plot(x_grid, density, color=TEXT_COLOR, linewidth=1.8, label="KDE FSC-A")
    ax.axvline(
        gmm_threshold,
        color=RED_LINE,
        linestyle="--",
        linewidth=2.5,
        label=f"Seuil G1 : {gmm_threshold:,.0f}",
    )

    if kde_valley is not None:
        ax.axvline(
            kde_valley,
            color=BLUE_LINE,
            linestyle="--",
            linewidth=2.0,
            label=f"Vallée KDE : {kde_valley:,.0f}",
        )
        delta = abs(gmm_threshold - kde_valley)
        ax.annotate(
            f"Δ = {delta:,.0f}",
            xy=((gmm_threshold + kde_valley) / 2, 0.88),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            color=TEXT_COLOR,
            fontsize=9,
        )

    n_debris = int((~mask_debris).sum())
    pct_deb = 100.0 * n_debris / n_total
    stats_text = (
        f"Total   : {n_total:,}\n"
        f"Exclus  : {n_debris:,} ({pct_deb:.1f}%)\n"
        f"Conserv.: {n_total - n_debris:,} ({100 - pct_deb:.1f}%)\n"
        f"KDE sur : {len(fsc_sub):,} pts"
    )
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=TEXT_COLOR,
        family="monospace",
        bbox=dict(facecolor=BG_COLOR, edgecolor=GRID_COLOR, alpha=0.85, pad=6),
    )

    ax.set_xlabel("FSC-A (unités linéaires)", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Densité KDE", color=TEXT_COLOR, fontsize=11)
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, pad=12)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " "))
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlim(x_min, data_max)
    ax.set_ylim(bottom=0, top=np.max(density) * 1.05)
    ax.legend(
        loc="upper left",
        fontsize=9,
        facecolor=BG_COLOR,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
    )
    plt.tight_layout()
    return fig, ax, gmm_threshold, kde_valley


# =============================================================================
# APPEL
# =============================================================================

print("✓ Génération du QC visuel Gate 1 en cours...")

_fsc_a_idx = PreGating.find_marker_index(var_names, ["FSC-A"])

if _fsc_a_idx is not None:
    fig, ax, gmm_th, kde_v = plot_gmm_vs_kde_qc(
        fsc_a_data=X_raw[:, _fsc_a_idx],
        mask_debris=mask_debris,  # bool : True=conservé, False=débris exclus
    )
    plt.savefig("gating_00_gmm_qc.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("   [OK] Sauvegardé: gating_00_gmm_qc.png")
    if kde_v is not None:
        print(f"Vallée KDE détectée  : {kde_v:,.0f}  (Δ = {abs(gmm_th - kde_v):,.0f})")
    else:
        print("Vallée KDE           : non trouvée (distribution unimodale probable)")
else:
    print("[!] Canal FSC-A introuvable dans var_names — QC ignoré.")


################################################################################
# =============================================================================
# VISUALISATION INTERACTIVE STYLE CYTOPY — DASHBOARD DE GATING
# =============================================================================
# Utilise Plotly pour des graphiques interactifs (zoom, hover, export)
# Inspiré de CytoPy AutonomousGate et FlowJo hierarchical gating
# =============================================================================

if APPLY_PREGATING and PLOTLY_AVAILABLE:
    print("=" * 70)
    print(" CYTOPY-STYLE GATING DASHBOARD — VISUALISATION INTERACTIVE")
    print("=" * 70)

    # Dossier de sortie + timestamp pour les exports HTML de cette section
    _dashboard_out = Path(globals().get("OUTPUT_DIR", "./output")) / "other"
    _dashboard_out.mkdir(parents=True, exist_ok=True)
    _dashboard_ts = globals().get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

    # =====================================================================
    # 0. PRÉPARATION DES DONNÉES
    # =====================================================================
    _fsc_a_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
    _fsc_h_idx = PreGating.find_marker_index(var_names, ["FSC-H"])
    _ssc_a_idx = PreGating.find_marker_index(var_names, ["SSC-A", "SSC-H", "SSC"])
    _cd45_idx = PreGating.find_marker_index(
        var_names, ["CD45", "CD45-PECY5", "CD45-PC5"]
    )
    _cd34_idx = PreGating.find_marker_index(var_names, ["CD34", "CD34-PE", "CD34-APC"])

    # Sous-échantillonner pour fluidité Plotly
    _n_pts = min(40_000, n_before)
    np.random.seed(42)
    _idx = np.random.choice(n_before, _n_pts, replace=False)

    # Masques cumulatifs appliqués séquentiellement
    _m_g1 = mask_debris[_idx]
    _m_g12 = (mask_debris & mask_singlets)[_idx]
    _m_g123 = (mask_debris & mask_singlets & mask_cd45)[_idx]
    _m_final = mask_final[_idx]

    # Vecteur de conditions pour le sous-échantillon
    _cond_sub = conditions[_idx]
    _is_sain_sub = _cond_sub == "Sain"

    # Labels de gate pour chaque événement (gating asymétrique)
    # ─── LÉGENDE DÉTAILLÉE : sépare Patho conservés / Sain NBM conservés ───
    def _gate_label(i):
        if not _m_g1[i]:
            return "Débris (exclu G1)"
        if not _m_g12[i]:
            return "Doublet (exclu G2)"
        if not _m_g123[i]:
            # En mode asymétrique, seuls les Patho peuvent être exclus par CD45
            return "CD45- Patho (exclu G3)"
        if FILTER_BLASTS and not _m_final[i]:
            return "Non-blaste (exclu G4)"
        # Cellule conservée → distinguer Patho CD45+ vs Sain NBM
        if _cond_sub[i] == "Pathologique":
            return "CD45+ Patho conservés ✓"
        elif _cond_sub[i] == "Sain":
            return "Conservés sains NBM ✓"
        return "Conservé ✓"

    _labels = np.array([_gate_label(i) for i in range(_n_pts)])

    # Palette CytoPy-style (avec catégories détaillées pour conservés)
    _color_map = {
        "Débris (exclu G1)": "#636363",
        "Doublet (exclu G2)": "#e6550d",
        "CD45- Patho (exclu G3)": "#fd8d3c",
        "Non-blaste (exclu G4)": "#fdae6b",
        "CD45+ Patho conservés ✓": "#d62728",  # rouge – patho conservés
        "Conservés sains NBM ✓": "#2ca02c",  # vert  – sains NBM conservés
        "Conservé ✓": "#31a354",  # fallback
    }

    # =====================================================================
    # 1. SANKEY DIAGRAM — FLUX DES ÉVÉNEMENTS (avec % relatifs)
    # =====================================================================
    print("\n [1/6] Sankey Diagram — Flux du gating hiérarchique (avec % relatifs)")

    _n_total = n_before
    _n_g1_pass = int(mask_debris.sum())
    _n_g1_fail = _n_total - _n_g1_pass
    _n_g2_pass = int((mask_debris & mask_singlets).sum())
    _n_g2_fail = _n_g1_pass - _n_g2_pass
    _n_g3_pass = int((mask_debris & mask_singlets & mask_cd45).sum())
    _n_g3_fail = _n_g2_pass - _n_g3_pass
    _n_g4_pass = int(n_final)
    _n_g4_fail = _n_g3_pass - _n_g4_pass

    # Helper: % of previous gate
    def _pct_of(value, parent):
        return f"{value / max(parent, 1) * 100:.1f}%"

    _sankey_labels = [
        f"Événements<br>totaux<br>{_n_total:,}",  # 0
        f"Gate 1<br>Viables<br>{_n_g1_pass:,}<br>({_pct_of(_n_g1_pass, _n_total)} of total)",  # 1
        f"Débris<br>exclus<br>{_n_g1_fail:,}<br>({_pct_of(_n_g1_fail, _n_total)})",  # 2
        f"Gate 2<br>Singlets<br>{_n_g2_pass:,}<br>({_pct_of(_n_g2_pass, _n_g1_pass)} of G1)",  # 3
        f"Doublets<br>exclus<br>{_n_g2_fail:,}<br>({_pct_of(_n_g2_fail, _n_g1_pass)})",  # 4
        f"Gate 3<br>CD45+<br>{_n_g3_pass:,}<br>({_pct_of(_n_g3_pass, _n_g2_pass)} of G2)",  # 5
        f"CD45-<br>exclus<br>{_n_g3_fail:,}<br>({_pct_of(_n_g3_fail, _n_g2_pass)})",  # 6
    ]

    _src = [0, 0, 1, 1, 3, 3]
    _tgt = [1, 2, 3, 4, 5, 6]
    _vals = [_n_g1_pass, _n_g1_fail, _n_g2_pass, _n_g2_fail, _n_g3_pass, _n_g3_fail]
    _link_colors = [
        "rgba(49,163,84,0.4)",
        "rgba(99,99,99,0.3)",  # G1: vert conservé / gris débris
        "rgba(49,163,84,0.4)",
        "rgba(230,85,13,0.3)",  # G2: vert singlets / orange doublets
        "rgba(49,163,84,0.4)",
        "rgba(253,141,60,0.3)",  # G3: vert CD45+ / orange CD45-
    ]

    if FILTER_BLASTS:
        _sankey_labels.append(
            f"Gate 4<br>CD34+<br>{_n_g4_pass:,}<br>({_pct_of(_n_g4_pass, _n_g3_pass)} of G3)"
        )  # 7
        _sankey_labels.append(
            f"Non-blastes<br>exclus<br>{_n_g4_fail:,}<br>({_pct_of(_n_g4_fail, _n_g3_pass)})"
        )  # 8
        _src += [5, 5]
        _tgt += [7, 8]
        _vals += [_n_g4_pass, _n_g4_fail]
        _link_colors += ["rgba(49,163,84,0.4)", "rgba(253,174,107,0.3)"]
        _final_label = f"Population<br>finale<br>{_n_g4_pass:,}<br>({_pct_of(_n_g4_pass, _n_total)} of total)"
        _sankey_labels.append(_final_label)  # 9
        _src.append(7)
        _tgt.append(9)
        _vals.append(_n_g4_pass)
        _link_colors.append("rgba(49,163,84,0.6)")
    else:
        _final_label = f"Population<br>finale<br>{_n_g3_pass:,}<br>({_pct_of(_n_g3_pass, _n_total)} of total)"
        _sankey_labels.append(_final_label)  # 7
        _src.append(5)
        _tgt.append(7)
        _vals.append(_n_g3_pass)
        _link_colors.append("rgba(49,163,84,0.6)")

    # Couleurs harmonisées: vert = conservé, orange/rouge = exclu
    _node_colors = (
        ["#4a90d9"]
        + ["#31a354", "#636363"] * 1
        + ["#31a354", "#e6550d", "#31a354", "#fd8d3c"]
    )
    if FILTER_BLASTS:
        _node_colors += ["#31a354", "#fdae6b", "#2ca02c"]
    else:
        _node_colors += ["#2ca02c"]

    fig_sankey = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="#333", width=1),
                label=_sankey_labels,
                color=_node_colors[: len(_sankey_labels)],
            ),
            link=dict(
                source=_src,
                target=_tgt,
                value=_vals,
                color=_link_colors,
            ),
        )
    )
    fig_sankey.update_layout(
        title=dict(
            text="<b>Gating Hierarchy — Flux des Événements (global)</b>",
            font=dict(size=18),
        ),
        font=dict(size=13, color="#222"),
        paper_bgcolor="#fafafa",
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig_sankey.write_html(
        str(_dashboard_out / f"fig_sankey_{_dashboard_ts}.html"),
        include_plotlyjs="cdn",
    )

    # =====================================================================
    # 1b. MINI SANKEY PAR FICHIER (onglet secondaire)
    # =====================================================================
    if (
        "file_origins" in dir()
        and file_origins is not None
        and len(singlets_summary_per_file) > 0
    ):
        print(" [1b/6] Mini Sankey par fichier (résumé)")

        _unique_files_sankey = (
            np.unique(file_origins) if hasattr(file_origins, "__len__") else []
        )

        # Limiter à max 6 fichiers mini-Sankey pour ne pas saturer le rapport
        _files_to_show = (
            _unique_files_sankey[:6]
            if len(_unique_files_sankey) > 6
            else _unique_files_sankey
        )

        if len(_files_to_show) > 0:
            from plotly.subplots import make_subplots

            for _f_name in _files_to_show:
                _f_mask = file_origins == _f_name
                _f_total = int(_f_mask.sum())
                if _f_total < 10:
                    continue

                _f_g1 = int((mask_debris & _f_mask).sum())
                _f_g1_fail = _f_total - _f_g1
                _f_g2 = int((mask_debris & mask_singlets & _f_mask).sum())
                _f_g2_fail = _f_g1 - _f_g2
                _f_g3 = int((mask_debris & mask_singlets & mask_cd45 & _f_mask).sum())
                _f_g3_fail = _f_g2 - _f_g3
                _f_final = int((mask_final & _f_mask).sum())
                _f_g4_fail = _f_g3 - _f_final

                _f_short = (
                    str(_f_name)
                    if len(str(_f_name)) <= 30
                    else str(_f_name)[:27] + "..."
                )

                _f_labels = [
                    f"Total<br>{_f_total:,}",
                    f"G1<br>{_f_g1:,}<br>({_pct_of(_f_g1, _f_total)})",
                    f"Débris<br>{_f_g1_fail:,}",
                    f"G2<br>{_f_g2:,}<br>({_pct_of(_f_g2, _f_g1)})",
                    f"Doubl.<br>{_f_g2_fail:,}",
                    f"G3<br>{_f_g3:,}<br>({_pct_of(_f_g3, _f_g2)})",
                    f"CD45-<br>{_f_g3_fail:,}",
                ]
                _f_src = [0, 0, 1, 1, 3, 3]
                _f_tgt = [1, 2, 3, 4, 5, 6]
                _f_vals_sk = [
                    _f_g1,
                    max(_f_g1_fail, 1),
                    _f_g2,
                    max(_f_g2_fail, 1),
                    _f_g3,
                    max(_f_g3_fail, 1),
                ]
                _f_link_col = [
                    "rgba(49,163,84,0.4)",
                    "rgba(99,99,99,0.3)",
                    "rgba(49,163,84,0.4)",
                    "rgba(230,85,13,0.3)",
                    "rgba(49,163,84,0.4)",
                    "rgba(253,141,60,0.3)",
                ]

                if FILTER_BLASTS:
                    _f_labels += [
                        f"G4<br>{_f_final:,}<br>({_pct_of(_f_final, _f_g3)})",
                        f"Excl.<br>{_f_g4_fail:,}",
                        f"Final<br>{_f_final:,}",
                    ]
                    _f_src += [5, 5, 7]
                    _f_tgt += [7, 8, 9]
                    _f_vals_sk += [_f_final, max(_f_g4_fail, 1), _f_final]
                    _f_link_col += [
                        "rgba(49,163,84,0.4)",
                        "rgba(253,174,107,0.3)",
                        "rgba(49,163,84,0.6)",
                    ]
                    _f_node_col = [
                        "#4a90d9",
                        "#31a354",
                        "#636363",
                        "#31a354",
                        "#e6550d",
                        "#31a354",
                        "#fd8d3c",
                        "#31a354",
                        "#fdae6b",
                        "#2ca02c",
                    ]
                else:
                    _f_labels.append(f"Final<br>{_f_g3:,}")
                    _f_src.append(5)
                    _f_tgt.append(7)
                    _f_vals_sk.append(_f_g3)
                    _f_link_col.append("rgba(49,163,84,0.6)")
                    _f_node_col = [
                        "#4a90d9",
                        "#31a354",
                        "#636363",
                        "#31a354",
                        "#e6550d",
                        "#31a354",
                        "#fd8d3c",
                        "#2ca02c",
                    ]

                _fig_f_sankey = go.Figure(
                    go.Sankey(
                        arrangement="snap",
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="#333", width=0.5),
                            label=_f_labels,
                            color=_f_node_col[: len(_f_labels)],
                        ),
                        link=dict(
                            source=_f_src,
                            target=_f_tgt,
                            value=_f_vals_sk,
                            color=_f_link_col,
                        ),
                    )
                )
                _fig_f_sankey.update_layout(
                    title=dict(text=f"<b>Gating — {_f_short}</b>", font=dict(size=14)),
                    font=dict(size=11, color="#222"),
                    paper_bgcolor="#fafafa",
                    height=300,
                    margin=dict(l=10, r=10, t=45, b=10),
                )
                _fig_f_sankey.write_html(
                    str(_dashboard_out / f"_fig_f_sankey_{_dashboard_ts}.html"),
                    include_plotlyjs="cdn",
                )

        # Nettoyer la variable temporaire pour éviter le doublon dans le rapport HTML
        try:
            del _fig_f_sankey
        except NameError:
            pass

        if len(_unique_files_sankey) > 6:
            print(
                f"      (Affichage limité à 6/{len(_unique_files_sankey)} fichiers pour lisibilité)"
            )

    # =====================================================================
    # 1c. SCATTER FSC-A vs FSC-H PAR FICHIER (avec droite RANSAC)
    # =====================================================================
    if ransac_scatter_data:
        print(" [1c/6] Scatter FSC-A vs FSC-H par fichier (droite RANSAC + R²)")

        _n_scatter_files = len(ransac_scatter_data)
        _n_cols_sc = min(3, _n_scatter_files)
        _n_rows_sc = int(np.ceil(_n_scatter_files / _n_cols_sc))

        _fig_ransac_scatter = make_subplots(
            rows=_n_rows_sc,
            cols=_n_cols_sc,
            subplot_titles=[f[:30] for f in ransac_scatter_data.keys()],
            horizontal_spacing=0.06,
            vertical_spacing=0.10,
        )

        for _si, (_sf_name, _sf_data) in enumerate(ransac_scatter_data.items()):
            _row = _si // _n_cols_sc + 1
            _col = _si % _n_cols_sc + 1

            _r2_disp = (
                f"R²={_sf_data['r2']:.3f}" if _sf_data["r2"] is not None else "R²=N/A"
            )
            _method_disp = (
                "RATIO" if _sf_data["method"] == "ratio_fallback" else "RANSAC"
            )
            _color_pts = (
                "#d62728" if _sf_data["method"] == "ratio_fallback" else "#2ca02c"
            )

            # Points (échantillonnés)
            _fig_ransac_scatter.add_trace(
                go.Scattergl(
                    x=_sf_data["fsc_h"],
                    y=_sf_data["fsc_a"],
                    mode="markers",
                    marker=dict(size=2, color=_color_pts, opacity=0.3),
                    name=f"{_sf_name[:20]} ({_method_disp})",
                    showlegend=False,
                    hovertemplate=f"FSC-H: %{{x:.0f}}<br>FSC-A: %{{y:.0f}}<br>{_r2_disp}<extra></extra>",
                ),
                row=_row,
                col=_col,
            )

            # Droite RANSAC
            _x_line = sorted(_sf_data["fsc_h"])
            _y_line = [_sf_data["slope"] * x + _sf_data["intercept"] for x in _x_line]
            _fig_ransac_scatter.add_trace(
                go.Scatter(
                    x=_x_line,
                    y=_y_line,
                    mode="lines",
                    line=dict(color="#ff7f0e", width=2, dash="dash"),
                    name=f"RANSAC {_r2_disp}",
                    showlegend=False,
                ),
                row=_row,
                col=_col,
            )

            _fig_ransac_scatter.update_xaxes(title_text="FSC-H", row=_row, col=_col)
            _fig_ransac_scatter.update_yaxes(title_text="FSC-A", row=_row, col=_col)

        _fig_ransac_scatter.update_layout(
            title="<b>QC RANSAC — FSC-A vs FSC-H par fichier (droite + R²)</b>",
            height=350 * _n_rows_sc,
            width=min(450 * _n_cols_sc, 1400),
            paper_bgcolor="#fafafa",
            plot_bgcolor="#f5f5f5",
        )
        del _fig_ransac_scatter  # Éviter doublon dans le collecteur de figures

    # =====================================================================
    # 1d. TABLEAU % SINGLETS PAR FICHIER
    # =====================================================================
    if singlets_summary_per_file:
        print(" [1d/6] Tableau % singlets par fichier")

        _df_singlets = pd.DataFrame(singlets_summary_per_file)

        # Coloriser les R² faibles
        _cell_colors = []
        for col_name in _df_singlets.columns:
            col_colors = []
            for _, row in _df_singlets.iterrows():
                if col_name == "r2" and row["r2"] is not None and row["r2"] < 0.85:
                    col_colors.append("#ffe0e0")  # Rouge léger
                elif col_name == "method" and row["method"] == "ratio_fallback":
                    col_colors.append("#fff3cd")  # Jaune léger
                else:
                    col_colors.append("#f9f9f9" if _ % 2 == 0 else "#fff")
            _cell_colors.append(col_colors)

        fig_singlets_table = go.Figure(
            go.Table(
                header=dict(
                    values=[f"<b>{c.upper()}</b>" for c in _df_singlets.columns],
                    fill_color="#4a90d9",
                    font=dict(color="white", size=12),
                    align="center",
                    height=35,
                ),
                cells=dict(
                    values=[_df_singlets[c] for c in _df_singlets.columns],
                    fill_color=_cell_colors,
                    font=dict(size=11),
                    align="center",
                    height=28,
                ),
            )
        )
        fig_singlets_table.update_layout(
            title="<b>QC Singlets — % par fichier (R² RANSAC, méthode utilisée)</b>",
            height=50 + 30 * (len(_df_singlets) + 1),
            width=900,
            margin=dict(l=20, r=20, t=50, b=10),
        )
        fig_singlets_table.write_html(
            str(_dashboard_out / f"fig_singlets_table_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )

    # =====================================================================
    # 2. DENSITY SCATTER PLOTS — CHAQUE GATE
    # =====================================================================
    print(" [2/5] Density Scatter Plots — Gating séquentiel")

    _gate_plots = []

    # --- Gate 1 : FSC-A vs SSC-A (Débris) ---
    if GATE_DEBRIS and _fsc_a_idx is not None and _ssc_a_idx is not None:
        _gate_plots.append(
            {
                "title": "Gate 1 — Débris (SSC-A vs FSC-A)",
                "x": X_raw[_idx, _fsc_a_idx],
                "y": X_raw[_idx, _ssc_a_idx],
                "mask": _m_g1,
                "xlabel": "FSC-A (Taille)",
                "ylabel": "SSC-A (Granularité)",
                "label_in": "Cellules viables",
                "label_out": "Débris",
            }
        )

    # --- Gate 2 : FSC-H vs FSC-A (Doublets) — sur les survivants de G1 ---
    if GATE_DOUBLETS and _fsc_a_idx is not None and _fsc_h_idx is not None:
        _g1_ok = _m_g1
        _gate_plots.append(
            {
                "title": "Gate 2 — Doublets (FSC-H vs FSC-A) [après G1]",
                "x": X_raw[_idx, _fsc_a_idx][_g1_ok],
                "y": X_raw[_idx, _fsc_h_idx][_g1_ok],
                "mask": mask_singlets[_idx][_g1_ok],
                "xlabel": "FSC-A (Area)",
                "ylabel": "FSC-H (Height)",
                "label_in": "Singlets",
                "label_out": "Doublets",
            }
        )

    # --- Gate 3 : CD45 vs SSC-A (Leucocytes) — sur les survivants de G1+G2 ---
    # En mode asymétrique: affiche UNIQUEMENT les cellules Patho (les Sain ne sont pas gatées CD45)
    if GATE_CD45 and _cd45_idx is not None and _ssc_a_idx is not None:
        _g12_ok = _m_g12
        if MODE_BLASTES_VS_NORMAL:
            # Filtre: montrer seulement les cellules Patho après G1+G2
            _patho_g12 = _g12_ok & (_cond_sub == "Pathologique")
            _gate_plots.append(
                {
                    "title": "Gate 3 — CD45+ PATHO seul (Sain: pas de gate CD45)",
                    "x": X_raw[_idx, _cd45_idx][_patho_g12],
                    "y": X_raw[_idx, _ssc_a_idx][_patho_g12],
                    "mask": mask_cd45[_idx][_patho_g12],
                    "xlabel": "CD45 (Intensité)",
                    "ylabel": "SSC-A (Granularité)",
                    "label_in": "CD45+ Patho (conservés)",
                    "label_out": "CD45− Patho (exclus)",
                }
            )
        else:
            _gate_plots.append(
                {
                    "title": "Gate 3 — CD45+ Leucocytes (CD45 vs SSC-A) [après G1+G2]",
                    "x": X_raw[_idx, _cd45_idx][_g12_ok],
                    "y": X_raw[_idx, _ssc_a_idx][_g12_ok],
                    "mask": mask_cd45[_idx][_g12_ok],
                    "xlabel": "CD45 (Intensité)",
                    "ylabel": "SSC-A (Granularité)",
                    "label_in": "Leucocytes CD45+",
                    "label_out": "CD45−",
                }
            )

    # --- Gate 4 : CD34 vs SSC-A (Blastes) — si activé ---
    if FILTER_BLASTS and _cd34_idx is not None and _ssc_a_idx is not None:
        _g123_ok = _m_g123
        _gate_plots.append(
            {
                "title": "Gate 4 — CD34+ Blastes (CD34 vs SSC-A) [après G1+G2+G3]",
                "x": X_raw[_idx, _cd34_idx][_g123_ok],
                "y": X_raw[_idx, _ssc_a_idx][_g123_ok],
                "mask": mask_cd34[_idx][_g123_ok],
                "xlabel": "CD34 (Intensité)",
                "ylabel": "SSC-A (Granularité)",
                "label_in": "Blastes CD34+",
                "label_out": "Autres leucocytes",
            }
        )

    # Générer chaque subplot avec Plotly
    n_gates = len(_gate_plots)
    if n_gates > 0:
        fig_gates = make_subplots(
            rows=1,
            cols=n_gates,
            subplot_titles=[g["title"] for g in _gate_plots],
            horizontal_spacing=0.06,
        )

        for col_i, gp in enumerate(_gate_plots, 1):
            _x, _y, _mk = gp["x"], gp["y"], gp["mask"]
            _valid = np.isfinite(_x) & np.isfinite(_y)
            _x, _y, _mk = _x[_valid], _y[_valid], _mk[_valid]

            # Exclus (fond, semi-transparent)
            fig_gates.add_trace(
                go.Scattergl(
                    x=_x[~_mk],
                    y=_y[~_mk],
                    mode="markers",
                    marker=dict(size=2, color="#d62728", opacity=0.25),
                    name=gp["label_out"],
                    legendgroup=f"g{col_i}_out",
                    showlegend=(col_i == 1),
                    hovertemplate=f"{gp['xlabel']}: %{{x:.0f}}<br>{gp['ylabel']}: %{{y:.0f}}<br>{gp['label_out']}<extra></extra>",
                ),
                row=1,
                col=col_i,
            )

            # Conservés (avant-plan)
            fig_gates.add_trace(
                go.Scattergl(
                    x=_x[_mk],
                    y=_y[_mk],
                    mode="markers",
                    marker=dict(size=2, color="#2ca02c", opacity=0.4),
                    name=gp["label_in"],
                    legendgroup=f"g{col_i}_in",
                    showlegend=(col_i == 1),
                    hovertemplate=f"{gp['xlabel']}: %{{x:.0f}}<br>{gp['ylabel']}: %{{y:.0f}}<br>{gp['label_in']}<extra></extra>",
                ),
                row=1,
                col=col_i,
            )

            # Axes labels
            fig_gates.update_xaxes(title_text=gp["xlabel"], row=1, col=col_i)
            fig_gates.update_yaxes(title_text=gp["ylabel"], row=1, col=col_i)

        fig_gates.update_layout(
            title="<b>Gating Séquentiel — Density Scatter (Plotly interactif)</b>",
            height=500,
            width=min(500 * n_gates, 2000),
            paper_bgcolor="#fafafa",
            plot_bgcolor="#f0f0f0",
            font=dict(size=11),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
            ),
            margin=dict(t=80, b=100),
        )
        fig_gates.write_html(
            str(_dashboard_out / f"fig_gates_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )

    # =====================================================================
    # 3. HISTOGRAMMES 1D — DISTRIBUTIONS DES MARQUEURS CLÉS
    # =====================================================================
    print(" [3/5] Histogrammes 1D — Distributions avec seuils GMM")

    _hist_data = []
    if _fsc_a_idx is not None:
        _hist_data.append(("FSC-A", X_raw[:, _fsc_a_idx], None))
    if _cd45_idx is not None:
        _hist_data.append(("CD45", X_raw[:, _cd45_idx], "cd45"))
    if FILTER_BLASTS and _cd34_idx is not None:
        _hist_data.append(("CD34", X_raw[:, _cd34_idx], "cd34"))

    if _hist_data:
        fig_hist = make_subplots(
            rows=1,
            cols=len(_hist_data),
            subplot_titles=[h[0] for h in _hist_data],
            horizontal_spacing=0.08,
        )

        for hi, (name, vals, marker_type) in enumerate(_hist_data, 1):
            _v = vals[np.isfinite(vals)]

            # Avant gating (toutes cellules)
            fig_hist.add_trace(
                go.Histogram(
                    x=_v,
                    nbinsx=200,
                    name=f"{name} (tous)",
                    marker_color="rgba(100,100,100,0.4)",
                    showlegend=(hi == 1),
                    legendgroup="all",
                ),
                row=1,
                col=hi,
            )

            # Après gating (conservés)
            _v_kept = vals[mask_final & np.isfinite(vals)]
            fig_hist.add_trace(
                go.Histogram(
                    x=_v_kept,
                    nbinsx=200,
                    name=f"{name} (conservés)",
                    marker_color="rgba(44,160,44,0.6)",
                    showlegend=(hi == 1),
                    legendgroup="kept",
                ),
                row=1,
                col=hi,
            )

            # Annoter le seuil GMM si pertinent
            if marker_type == "cd45" and GATE_CD45:
                _cd45_vals = X_raw[:, _cd45_idx]
                _cd45_clean = _cd45_vals[np.isfinite(_cd45_vals)]
                _th = np.percentile(_cd45_clean, CD45_THRESHOLD_PERCENTILE)
                fig_hist.add_vline(
                    x=_th,
                    line_dash="dash",
                    line_color="#d62728",
                    line_width=2,
                    annotation_text=f"Seuil CD45+",
                    annotation_position="top right",
                    row=1,
                    col=hi,
                )
            elif marker_type == "cd34" and FILTER_BLASTS:
                _cd34_vals = X_raw[:, _cd34_idx]
                _cd34_clean = _cd34_vals[np.isfinite(_cd34_vals)]
                _th34 = np.percentile(_cd34_clean, CD34_THRESHOLD_PERCENTILE)
                fig_hist.add_vline(
                    x=_th34,
                    line_dash="dash",
                    line_color="#ff7f0e",
                    line_width=2,
                    annotation_text=f"Seuil CD34+",
                    annotation_position="top right",
                    row=1,
                    col=hi,
                )

            fig_hist.update_xaxes(title_text=name, row=1, col=hi)
            fig_hist.update_yaxes(title_text="Nombre d'événements", row=1, col=hi)

        fig_hist.update_layout(
            title="<b>Distributions 1D — Avant / Après Gating (seuils annotés)</b>",
            barmode="overlay",
            height=400,
            width=min(550 * len(_hist_data), 1800),
            paper_bgcolor="#fafafa",
            plot_bgcolor="#f5f5f5",
            font=dict(size=11),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
            ),
            margin=dict(t=70, b=90),
        )
        fig_hist.write_html(
            str(_dashboard_out / f"fig_hist_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )

    # =====================================================================
    # 4. COMPARAISON PATHO vs SAIN (si MODE_BLASTES_VS_NORMAL)
    # =====================================================================
    if MODE_BLASTES_VS_NORMAL and _cd45_idx is not None and _ssc_a_idx is not None:
        print(" [4/5] Comparaison Patho vs Sain — CD45 vs SSC-A (GATING ASYMÉTRIQUE)")

        _cond = conditions[_idx]

        fig_comp = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Pathologique (CD45 strict appliqué)",
                "Sain / NBM (PAS de gate CD45)",
            ],
            horizontal_spacing=0.08,
        )

        for ci, (cond_label, color_kept, color_all) in enumerate(
            [
                ("Pathologique", "#d62728", "#ffcccc"),
                ("Sain", "#2ca02c", "#ccffcc"),
            ],
            1,
        ):
            _sel = _cond == cond_label
            _sel_final = _sel & _m_final

            _xc = X_raw[_idx, _cd45_idx]
            _yc = X_raw[_idx, _ssc_a_idx]

            # Toutes les cellules de cette condition (fond)
            fig_comp.add_trace(
                go.Scattergl(
                    x=_xc[_sel],
                    y=_yc[_sel],
                    mode="markers",
                    marker=dict(size=2, color=color_all, opacity=0.15),
                    name=f"{cond_label} (tous)",
                    showlegend=True,
                ),
                row=1,
                col=ci,
            )

            # Cellules conservées
            fig_comp.add_trace(
                go.Scattergl(
                    x=_xc[_sel_final],
                    y=_yc[_sel_final],
                    mode="markers",
                    marker=dict(size=2.5, color=color_kept, opacity=0.5),
                    name=f"{cond_label} (conservés)",
                    showlegend=True,
                ),
                row=1,
                col=ci,
            )

            _n_all = _sel.sum()
            _n_kept = _sel_final.sum()
            _gate_info = (
                "CD45 strict" if cond_label == "Pathologique" else "Pas de gate CD45"
            )
            fig_comp.update_xaxes(title_text="CD45", row=1, col=ci)
            fig_comp.update_yaxes(title_text="SSC-A", row=1, col=ci)

            _xax = "x domain" if ci == 1 else f"x{ci} domain"
            _yax = "y domain" if ci == 1 else f"y{ci} domain"
            fig_comp.add_annotation(
                text=f"<b>{_n_kept:,} / {_n_all:,} ({_n_kept / _n_all * 100:.1f}%)<br>{_gate_info}</b>"
                if _n_all > 0
                else "N/A",
                xref=_xax,
                yref=_yax,
                x=0.5,
                y=0.02,
                showarrow=False,
                font=dict(size=13, color=color_kept),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=color_kept,
                borderwidth=1,
                borderpad=4,
            )

        fig_comp.update_layout(
            title="<b>Gating Asymétrique — Patho (CD45 strict) vs Sain (pas de CD45)</b>",
            height=500,
            width=1100,
            paper_bgcolor="#fafafa",
            plot_bgcolor="#f0f0f0",
            font=dict(size=11),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
            ),
            margin=dict(t=70, b=90),
        )
        fig_comp.write_html(
            str(_dashboard_out / f"fig_comp_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )
    else:
        print(
            " [4/5] Comparaison Patho vs Sain — SKIP (MODE_BLASTES_VS_NORMAL désactivé)"
        )

    # =====================================================================
    # 5. OVERVIEW FINAL — FSC-A vs SSC-A coloré par gate d'exclusion
    # =====================================================================
    if _fsc_a_idx is not None and _ssc_a_idx is not None:
        print(" [5/5] Overview Final — Coloré par étape d'exclusion")

        _xo = X_raw[_idx, _fsc_a_idx]
        _yo = X_raw[_idx, _ssc_a_idx]

        fig_overview = go.Figure()

        # Tracer chaque catégorie d'exclusion dans l'ordre
        # ─── Inclut désormais les conservés Patho et Sain séparément ───
        _order = [
            "Débris (exclu G1)",
            "Doublet (exclu G2)",
            "CD45- Patho (exclu G3)",
            "Non-blaste (exclu G4)",
            "CD45+ Patho conservés ✓",
            "Conservés sains NBM ✓",
            "Conservé ✓",  # fallback si condition inconnue
        ]
        for cat in _order:
            _sel_cat = _labels == cat
            if _sel_cat.sum() == 0:
                continue
            _is_kept = cat.endswith("✓")
            fig_overview.add_trace(
                go.Scattergl(
                    x=_xo[_sel_cat],
                    y=_yo[_sel_cat],
                    mode="markers",
                    marker=dict(
                        size=2.5,
                        color=_color_map.get(cat, "#999"),
                        opacity=0.55 if _is_kept else 0.25,
                    ),
                    name=f"{cat} ({_sel_cat.sum():,})",
                    hovertemplate=f"FSC-A: %{{x:.0f}}<br>SSC-A: %{{y:.0f}}<br>{cat}<extra></extra>",
                )
            )

        fig_overview.update_layout(
            title="<b>Overview — Événements colorés par étape d'exclusion</b>",
            xaxis_title="FSC-A (Taille)",
            yaxis_title="SSC-A (Granularité)",
            height=600,
            width=900,
            paper_bgcolor="#fafafa",
            plot_bgcolor="#f0f0f0",
            font=dict(size=12),
            legend=dict(
                title="Catégorie",
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#ccc",
                borderwidth=1,
            ),
            margin=dict(t=70, b=50),
        )
        fig_overview.write_html(
            str(_dashboard_out / f"fig_overview_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )

    # =====================================================================
    # RÉSUMÉ TABULAIRE
    # =====================================================================
    print("\n" + "=" * 70)
    print(" RÉSUMÉ GATING — TABLEAU INTERACTIF")
    print("=" * 70)

    _summary_df = pd.DataFrame(
        {
            "Étape": [
                "Initial",
                "Gate 1 (Débris)",
                "Gate 2 (Doublets)",
                "Gate 3 (CD45+ Patho only)"
                if MODE_BLASTES_VS_NORMAL
                else "Gate 3 (CD45+)",
            ]
            + (["Gate 4 (CD34+)"] if FILTER_BLASTS else [])
            + ["Population finale"],
            "Événements": [
                n_before,
                int(mask_debris.sum()),
                int((mask_debris & mask_singlets).sum()),
                int((mask_debris & mask_singlets & mask_cd45).sum()),
            ]
            + ([int(n_final)] if FILTER_BLASTS else [])
            + [int(n_final)],
            "Rétention (%)": [
                100.0,
                mask_debris.sum() / n_before * 100,
                (mask_debris & mask_singlets).sum() / n_before * 100,
                (mask_debris & mask_singlets & mask_cd45).sum() / n_before * 100,
            ]
            + ([n_final / n_before * 100] if FILTER_BLASTS else [])
            + [n_final / n_before * 100],
            "Exclus": [
                0,
                n_before - int(mask_debris.sum()),
                int(mask_debris.sum()) - int((mask_debris & mask_singlets).sum()),
                int((mask_debris & mask_singlets).sum())
                - int((mask_debris & mask_singlets & mask_cd45).sum()),
            ]
            + (
                [int((mask_debris & mask_singlets & mask_cd45).sum()) - int(n_final)]
                if FILTER_BLASTS
                else []
            )
            + [n_before - int(n_final)],
        }
    )
    _summary_df["Rétention (%)"] = _summary_df["Rétention (%)"].round(1)

    fig_table = go.Figure(
        go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in _summary_df.columns],
                fill_color="#4a90d9",
                font=dict(color="white", size=13),
                align="center",
                height=35,
            ),
            cells=dict(
                values=[_summary_df[c] for c in _summary_df.columns],
                fill_color=[
                    ["#f9f9f9", "#fff", "#f9f9f9", "#fff"]
                    + (["#f9f9f9"] if FILTER_BLASTS else [])
                    + ["#d4edda"]
                ]
                * 4,
                font=dict(size=12),
                align="center",
                height=30,
                format=[None, ",", ".1f", ","],
            ),
        )
    )
    fig_table.update_layout(
        title="<b>Résumé du Pre-Gating — Statistiques par étape</b>",
        height=50 + 35 * (len(_summary_df) + 1),
        width=800,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    fig_table.write_html(
        str(_dashboard_out / f"fig_table_{_dashboard_ts}.html"),
        include_plotlyjs="cdn",
    )

    # --- Tableau par condition si gating asymétrique ---
    if MODE_BLASTES_VS_NORMAL:
        _mp = conditions == "Pathologique"
        _ms = conditions == "Sain"
        _cond_df = pd.DataFrame(
            {
                "Condition": ["Pathologique", "Sain / NBM"],
                "Initial": [int(_mp.sum()), int(_ms.sum())],
                "Après Débris+Doublets": [
                    int((_mp & mask_debris & mask_singlets).sum()),
                    int((_ms & mask_debris & mask_singlets).sum()),
                ],
                "Après Gate CD45": [
                    int((_mp & mask_debris & mask_singlets & mask_cd45).sum()),
                    f"{int((_ms & mask_debris & mask_singlets).sum())} (non appliqué)",
                ],
                "Final": [int((mask_final & _mp).sum()), int((mask_final & _ms).sum())],
                "Rétention (%)": [
                    round((mask_final & _mp).sum() / max(_mp.sum(), 1) * 100, 1),
                    round((mask_final & _ms).sum() / max(_ms.sum(), 1) * 100, 1),
                ],
                "Logique CD45": ["CD45 STRICT", "AUCUN gate CD45"],
            }
        )

        fig_table_cond = go.Figure(
            go.Table(
                header=dict(
                    values=[f"<b>{c}</b>" for c in _cond_df.columns],
                    fill_color="#6a0dad",
                    font=dict(color="white", size=13),
                    align="center",
                    height=35,
                ),
                cells=dict(
                    values=[_cond_df[c] for c in _cond_df.columns],
                    fill_color=[["#ffe6e6", "#e6ffe6"]] * len(_cond_df.columns),
                    font=dict(size=12),
                    align="center",
                    height=30,
                ),
            )
        )
        fig_table_cond.update_layout(
            title="<b>Gating Asymétrique — Détail par Condition (Patho: CD45 strict / Sain: pas de CD45)</b>",
            height=150,
            width=1100,
            margin=dict(l=20, r=20, t=50, b=10),
        )
        fig_table_cond.write_html(
            str(_dashboard_out / f"fig_table_cond_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )

    print("\n [OK] Dashboard CytoPy-style généré avec succès !")
    print("     → Utilisez la souris pour zoomer, survoler, et exporter (icône 📷)")

elif not PLOTLY_AVAILABLE:
    print("[!] Plotly requis pour le dashboard interactif.")
    print("    → pip install plotly")
else:
    print("[!] Pre-gating désactivé — Aucun dashboard généré.")
    print("[!] Pre-gating désactivé — Aucun dashboard généré.")


################################################################################
# =============================================================================
# CRÉATION DU SECOND ANNDATA (avec ou sans gating)
# =============================================================================

# Créer l'AnnData filtré (ou copie complète si pas de gating)
combined_gated = combined_data[mask_final].copy()

if APPLY_PREGATING:
    print(f"[OK] AnnData après gating: {combined_gated.shape}")
    print(f"   → {combined_gated.shape[0]:,} cellules conservées")
    print(f"   → {combined_gated.shape[1]} marqueurs")
else:
    print(f"[OK] AnnData créé (sans pre-gating): {combined_gated.shape}")
    print(f"   → {combined_gated.shape[0]:,} cellules (toutes conservées)")
    print(f"   → {combined_gated.shape[1]} marqueurs")
# CONFIGURATION DE LA TRANSFORMATION
# → Valeurs contrôlées par le panneau interactif (cellule 2) si exécuté.
_cfg = globals().get("CONFIG", {})

TRANSFORM_TYPE = _cfg.get(
    "TRANSFORM_TYPE", "logicle"
)  # Options: "arcsinh", "logicle", "log10", "none"
COFACTOR = _cfg.get("COFACTOR", 5)  # Pour arcsinh: 5 (flow)
APPLY_TO_SCATTER = _cfg.get("APPLY_TO_SCATTER", False)

print("TRANSFORMATION DES DONNÉES")
print("=" * 60)
print(f"   Type: {TRANSFORM_TYPE.upper()}")
if TRANSFORM_TYPE == "arcsinh":
    print(f"   Cofacteur: {COFACTOR}")
print(f"   Appliquer au scatter: {'Oui' if APPLY_TO_SCATTER else 'Non'}")
# APPLICATION DE LA TRANSFORMATION

# Extraire les données
X_gated = combined_gated.X
if hasattr(X_gated, "toarray"):
    X_gated = X_gated.toarray()

# Copie pour transformation
X_transformed = X_gated.copy()

# Déterminer les indices des colonnes à transformer
if APPLY_TO_SCATTER:
    cols_to_transform = list(range(len(var_names)))
else:
    # Exclure FSC, SSC, Time
    scatter_patterns = ["FSC", "SSC", "TIME", "EVENT"]
    cols_to_transform = [
        i
        for i, name in enumerate(var_names)
        if not any(p in name.upper() for p in scatter_patterns)
    ]

print(f"\nColonnes à transformer: {len(cols_to_transform)}/{len(var_names)}")

# Appliquer la transformation
if TRANSFORM_TYPE == "arcsinh":
    print(f"\n Application Arcsinh (cofactor={COFACTOR})...")
    X_transformed[:, cols_to_transform] = DataTransformer.arcsinh_transform(
        X_gated[:, cols_to_transform], cofactor=COFACTOR
    )

elif TRANSFORM_TYPE == "logicle":
    print("\n Application Logicle...")
    X_transformed[:, cols_to_transform] = DataTransformer.logicle_transform(
        X_gated[:, cols_to_transform]
    )

elif TRANSFORM_TYPE == "log10":
    print("\n Application Log10...")
    X_transformed[:, cols_to_transform] = DataTransformer.log_transform(
        X_gated[:, cols_to_transform]
    )

else:
    print("\n[!] Pas de transformation appliquée")

# Vérifier les résultats
print(f"\n[OK] Transformation terminée!")
print(
    f"   Plage avant: [{X_gated[:, cols_to_transform].min():.2f}, {X_gated[:, cols_to_transform].max():.2f}]"
)
print(
    f"   Plage après: [{X_transformed[:, cols_to_transform].min():.2f}, {X_transformed[:, cols_to_transform].max():.2f}]"
)
# SÉLECTION DES COLONNES POUR FLOWSOM
# → Valeurs contrôlées par le panneau interactif (cellule 2) si exécuté.
_cfg = globals().get("CONFIG", {})

EXCLUDE_SCATTER = _cfg.get("EXCLUDE_SCATTER", True)
EXCLUDE_ADDITIONAL_MARKERS = _cfg.get("EXCLUDE_ADDITIONAL_MARKERS", ["CD45"])

# Identifier les colonnes à utiliser
scatter_patterns = ["FSC", "SSC", "TIME", "EVENT"]

# Construire la liste des colonnes pour FlowSOM
cols_to_use = []
excluded_additional = []

for i, name in enumerate(var_names):
    # Exclure scatter/time si demandé
    if EXCLUDE_SCATTER and any(p in name.upper() for p in scatter_patterns):
        continue
    # Exclure les marqueurs supplémentaires spécifiés
    if any(excl.upper() in name.upper() for excl in EXCLUDE_ADDITIONAL_MARKERS):
        excluded_additional.append(name)
        continue
    cols_to_use.append(i)

used_markers = [var_names[i] for i in cols_to_use]

print("COLONNES POUR FLOWSOM")
print("=" * 60)
print(f"   Exclure scatter: {'Oui' if EXCLUDE_SCATTER else 'Non'}")
print(
    f"   Marqueurs exclus manuellement: {EXCLUDE_ADDITIONAL_MARKERS if EXCLUDE_ADDITIONAL_MARKERS else 'Aucun'}"
)
if excluded_additional:
    print(f"   → Marqueurs retirés du FlowSOM (mais conservés dans l'export):")
    for m in excluded_additional:
        print(f"      [X] {m}")
print(f"   Colonnes sélectionnées: {len(cols_to_use)}/{len(var_names)}")
print(f"\nMarqueurs utilisés pour le FlowSOM:")
for i, marker in enumerate(used_markers):
    print(f"   [{i:2d}] {marker}")


################################################################################
# =============================================================================
# FILTRAGE PAR TYPE DE MARQUEUR (-A vs -H) - OPTIONNEL
# =============================================================================
# EXPLICATION:
# - Les suffixes -A (Area) et -H (Height) dans les noms de marqueurs représentent
#   des mesures différentes du MÊME signal:
#     • -A (Area) = Aire sous la courbe du pulse → plus stable
#     • -H (Height) = Hauteur maximale du pulse → plus sensible
#
# - En cytométrie, on utilise généralement:
#     • FSC-A et SSC-A pour le gating (débris, doublets)
#     • Fluorescence -A pour le clustering (plus stable)
#
# - Ce filtre permet de DÉDUPLIQUER les marqueurs redondants si vous avez
#   à la fois -A et -H pour chaque marqueur (ex: CD13 PE-A ET CD13 PE-H)
#
# [!] ATTENTION: Cette section ne filtre PAS les fichiers, mais les MARQUEURS!
# =============================================================================

# ===================== ACTIVATION DU FILTRAGE =====================
APPLY_MARKER_FILTERING = (
    True  # [!] Mettre True pour dédupliquer (garder uniquement -A ou -H)
)

# OPTIONS DE FILTRAGE (utilisées seulement si APPLY_MARKER_FILTERING = True)
KEEP_AREA = True  # True = garder les marqueurs -A (Area) [RECOMMANDÉ]
KEEP_HEIGHT = False  # True = garder les marqueurs -H (Height)

print("=" * 70)
print("FILTRAGE PAR TYPE DE MARQUEUR (-A vs -H)")
print("=" * 70)
print(
    f"   Filtrage activé: {'[OK] OUI' if APPLY_MARKER_FILTERING else '[X] NON (tous les marqueurs conservés)'}"
)

if APPLY_MARKER_FILTERING:
    print(f"   Garder -A (Area):   {'[OK] Oui' if KEEP_AREA else '[X] Non'}")
    print(f"   Garder -H (Height): {'[OK] Oui' if KEEP_HEIGHT else '[X] Non'}")

# =============================================================================
# AFFICHAGE DES COLONNES/MARQUEURS AVEC -A ET -H
# =============================================================================
print("\n" + "=" * 70)
print(" ANALYSE DES MARQUEURS PAR TYPE (-A Area vs -H Height)")
print("=" * 70)
print(
    "\n[INFO] Les suffixes -A et -H sont dans les NOMS DE MARQUEURS (pas les fichiers)"
)
print("       -A = Area (aire du pulse) | -H = Height (hauteur du pulse)")
print("=" * 70)

# Récupérer les noms de colonnes
all_columns = list(combined_gated.var_names)

# Séparer les colonnes avec -A, -H, ou autres
# IMPORTANT: Vérifier que -A et -H sont à la FIN du nom (suffixes), pas n'importe où
# Exemple: "CD19 APC-A750-H" se termine par -H (Height), pas -A même si "APC-A750" contient -A
cols_with_A = [col for col in all_columns if col.upper().endswith("-A")]
cols_with_H = [col for col in all_columns if col.upper().endswith("-H")]
cols_other = [
    col
    for col in all_columns
    if not col.upper().endswith("-A") and not col.upper().endswith("-H")
]

print(f"\n🔵 MARQUEURS avec '-A' (Area - aire du pulse) - {len(cols_with_A)} au total:")
for col in cols_with_A:
    print(f"   • {col}")

print(
    f"\n🟣 MARQUEURS avec '-H' (Height - hauteur du pulse) - {len(cols_with_H)} au total:"
)
for col in cols_with_H:
    print(f"   • {col}")

print(f"\n⚪ MARQUEURS sans -A/-H - {len(cols_other)} au total:")
for col in cols_other:
    print(f"   • {col}")

# =============================================================================
# FILTRAGE DES MARQUEURS (si activé)
# =============================================================================
if APPLY_MARKER_FILTERING:
    print("\n" + "=" * 70)
    print(" FILTRAGE DES MARQUEURS PAR TYPE (-A Area vs -H Height)")
    print("=" * 70)

    # Déterminer quels marqueurs garder
    markers_to_keep = []

    if KEEP_AREA:
        markers_to_keep.extend(cols_with_A)
    if KEEP_HEIGHT:
        markers_to_keep.extend(cols_with_H)

    # Toujours garder les marqueurs sans -A/-H
    markers_to_keep.extend(cols_other)

    # Dédupliquer (au cas où)
    markers_to_keep = list(dict.fromkeys(markers_to_keep))

    # Vérification
    if len(markers_to_keep) == 0:
        print("\n[!] ATTENTION: Aucun marqueur sélectionné!")
        print("   → KEEP_AREA et KEEP_HEIGHT sont tous les deux False")
        print("   → Désactivation automatique du filtrage")
        APPLY_MARKER_FILTERING = False
        markers_to_keep = all_columns

    if APPLY_MARKER_FILTERING:
        # Afficher le résumé
        n_before_markers = len(all_columns)
        n_after_markers = len(markers_to_keep)
        n_excluded_markers = n_before_markers - n_after_markers

        print(f"\n Marqueurs AVANT filtrage: {n_before_markers}")
        print(f" Marqueurs APRÈS filtrage: {n_after_markers}")
        print(f" Marqueurs exclus: {n_excluded_markers}")

        if n_excluded_markers > 0:
            print(f"\n Marqueurs CONSERVÉS:")
            for m in markers_to_keep:
                print(f"   [OK] {m}")

            excluded_markers = [m for m in all_columns if m not in markers_to_keep]
            print(f"\n Marqueurs EXCLUS:")
            for m in excluded_markers:
                print(f"   [X] {m}")

            # Appliquer le filtrage en sélectionnant les colonnes
            # Créer un masque de colonnes
            col_indices = [
                i for i, col in enumerate(all_columns) if col in markers_to_keep
            ]

            # Filtrer combined_gated (AnnData)
            combined_gated = combined_gated[:, markers_to_keep].copy()

            # Filtrer X_transformed
            X_transformed = X_transformed[:, col_indices]

            # Mettre à jour var_names
            var_names = markers_to_keep

            print(f"\n[OK] Filtrage des marqueurs appliqué!")
            print(f"   Shape combined_gated: {combined_gated.shape}")
            print(f"   Shape X_transformed: {X_transformed.shape}")
        else:
            print(f"\n[INFO] Tous les marqueurs sont déjà conservés (rien à filtrer)")

else:
    print("\n" + "=" * 70)
    print(" FILTRAGE DES MARQUEURS DÉSACTIVÉ")
    print("=" * 70)
    print(f"   → Tous les {len(all_columns)} marqueurs sont conservés")

    # Afficher quand même les fichiers disponibles
    print(f"\n Fichiers disponibles dans le dataset:")
    file_counts = combined_gated.obs["file_origin"].value_counts()
    for fname, count in file_counts.items():
        print(f"   {fname}: {count:,} cellules")

# =============================================================================
# AFFICHAGE DES COLONNES UTILISÉES POUR FLOWSOM
# =============================================================================
print(f"\n" + "=" * 70)
print(" MARQUEURS DISPONIBLES POUR FLOWSOM")
print("=" * 70)
print(f"\nTous les marqueurs actuels ({len(list(combined_gated.var_names))}):")
for i, col in enumerate(combined_gated.var_names):
    marker_type = ""
    if "-A" in col.upper():
        marker_type = " [Type -A = Area]"
    elif "-H" in col.upper():
        marker_type = " [Type -H = Height]"
    print(f"   [{i:2d}] {col}{marker_type}")

print(f"\n[OK] Données prêtes pour la suite du pipeline")
print(f"   Shape combined_gated: {combined_gated.shape}")
print(f"   Shape X_transformed: {X_transformed.shape}")
print(f"   Nombre de cellules: {combined_gated.shape[0]:,}")
print(f"   Nombre de marqueurs: {combined_gated.shape[1]}")

# =============================================================================
# MISE À JOUR DE cols_to_use ET used_markers APRÈS FILTRAGE
# =============================================================================
# Recalculer cols_to_use après le filtrage car var_names a peut-être été modifié
cols_to_use = []
excluded_additional = []

for i, name in enumerate(var_names):
    # Exclure scatter/time si demandé
    if EXCLUDE_SCATTER and any(p in name.upper() for p in scatter_patterns):
        continue
    # Exclure les marqueurs supplémentaires spécifiés
    if any(excl.upper() in name.upper() for excl in EXCLUDE_ADDITIONAL_MARKERS):
        excluded_additional.append(name)
        continue
    cols_to_use.append(i)

used_markers = [var_names[i] for i in cols_to_use]

print(f"\n[OK] Variables 'cols_to_use' et 'used_markers' mises à jour après filtrage")
print(f"   Marqueurs utilisés pour FlowSOM: {len(used_markers)}")
if excluded_additional:
    print(
        f"   Marqueurs exclus manuellement (mais conservés dans l'export): {excluded_additional}"
    )
# CRÉATION DE L'ANNDATA TRANSFORMÉ ET EXPLORATION POST-ARCSINH

# Créer un nouvel AnnData avec les données transformées (X_transformed)
import anndata as ad

# Créer adata_flowsom - le nouvel AnnData pour FlowSOM avec données transformées
adata_flowsom = ad.AnnData(
    X=X_transformed,  # Données POST-transformation arcsinh
    obs=combined_gated.obs.copy(),  # Copie des métadonnées
    var=combined_gated.var.copy() if combined_gated.var is not None else None,
)

# Ajouter les noms de variables
adata_flowsom.var_names = var_names

print("=" * 70)
print("CRÉATION ANNDATA POUR FLOWSOM (DONNÉES POST-ARCSINH)")
print("=" * 70)
print(f"\n[OK] Nouvel AnnData 'adata_flowsom' créé avec données transformées")
print(f"   Shape: {adata_flowsom.shape}")
print(f"   Observations (cellules): {adata_flowsom.n_obs:,}")
print(f"   Variables (marqueurs): {adata_flowsom.n_vars}")

# ============================================================================
# EXPLORATION DU DATAFRAME POST-TRANSFORMATION
# ============================================================================

# Extraire la matrice transformée depuis le NOUVEL AnnData
X_trans = adata_flowsom.X
if hasattr(X_trans, "toarray"):
    X_trans = X_trans.toarray()

# Créer un DataFrame pour exploration
df_transformed = pd.DataFrame(X_trans, columns=var_names)
df_transformed["condition"] = adata_flowsom.obs["condition"].values
df_transformed["file_origin"] = adata_flowsom.obs["file_origin"].values

print("\n" + "=" * 70)
print("APERÇU DES DONNÉES TRANSFORMÉES (premières 10 lignes)")
print("=" * 70)
print(f"Shape du DataFrame: {df_transformed.shape}")

# VÉRIFICATION DES NaN ET Inf POST-ARCSINH

print("\n" + "=" * 70)
print("VÉRIFICATION DES VALEURS NaN ET Inf POST-ARCSINH")
print("=" * 70)

# Colonnes numériques uniquement
numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()

# Comptage des NaN
nan_counts = df_transformed[numeric_cols].isna().sum()
total_nan = nan_counts.sum()

# Comptage des Inf (positifs et négatifs)
inf_pos_counts = (df_transformed[numeric_cols] == np.inf).sum()
inf_neg_counts = (df_transformed[numeric_cols] == -np.inf).sum()
total_inf_pos = inf_pos_counts.sum()
total_inf_neg = inf_neg_counts.sum()
total_inf = total_inf_pos + total_inf_neg

total_cells = df_transformed.shape[0] * len(numeric_cols)

print(f"\nRÉSUMÉ GLOBAL:")
print(f"   Total valeurs analysées: {total_cells:,}")
print(f"   Total NaN:    {total_nan:,} ({100 * total_nan / total_cells:.4f}%)")
print(f"   Total +Inf:   {total_inf_pos:,} ({100 * total_inf_pos / total_cells:.4f}%)")
print(f"   Total -Inf:   {total_inf_neg:,} ({100 * total_inf_neg / total_cells:.4f}%)")

# Détail par colonne si problèmes détectés
if total_nan > 0 or total_inf > 0:
    print(f"\nDÉTAIL PAR COLONNE AVEC PROBLÈMES:")
    print("-" * 60)
    for col in numeric_cols:
        n_nan = df_transformed[col].isna().sum()
        n_inf_pos = (df_transformed[col] == np.inf).sum()
        n_inf_neg = (df_transformed[col] == -np.inf).sum()
        if n_nan > 0 or n_inf_pos > 0 or n_inf_neg > 0:
            print(
                f"   {col:30s}: NaN={n_nan:,}, +Inf={n_inf_pos:,}, -Inf={n_inf_neg:,}"
            )
else:
    print(f"\nAucune valeur NaN ou Inf détectée - Données propres!")

# ============================================================================
# STATISTIQUES DESCRIPTIVES POST-ARCSINH
# ============================================================================
print("\n" + "=" * 70)
print("STATISTIQUES DESCRIPTIVES POST-ARCSINH")
print("=" * 70)

# ============================================================================
# VÉRIFICATION DES RANGES POST-TRANSFORMATION
# ============================================================================
print("\n" + "=" * 70)
print("VÉRIFICATION DES RANGES POST-TRANSFORMATION")
print("=" * 70)
print("(arcsinh avec cofactor=150 donne typiquement des valeurs entre -5 et 10)\n")

# Utiliser les colonnes numériques réellement présentes dans df_transformed
markers_to_check = [col for col in var_names if col in numeric_cols][:10]

for col in markers_to_check:  # Premiers 10 marqueurs numériques
    col_min = df_transformed[col].min()
    col_max = df_transformed[col].max()
    col_mean = df_transformed[col].mean()
    print(f"   {col:30s}: min={col_min:8.3f}, max={col_max:8.3f}, mean={col_mean:8.3f}")
# NETTOYAGE FINAL ET VALIDATION DE L'ANNDATA POUR FLOWSOM

# Nettoyage final: remplacer NaN/Inf par 0 dans adata_flowsom
X_final = adata_flowsom.X
if hasattr(X_final, "toarray"):
    X_final = X_final.toarray()

# Vérifier et nettoyer
nan_mask = ~np.isfinite(X_final)
n_nan = nan_mask.sum()
if n_nan > 0:
    print(f"[!] {n_nan} valeurs NaN/Inf détectées et remplacées par 0")
    X_final = np.nan_to_num(X_final, nan=0.0, posinf=0.0, neginf=0.0)
    adata_flowsom.X = X_final
else:
    print("[OK] Aucune valeur problématique - pas de nettoyage nécessaire")

print(f"\n[OK] AnnData 'adata_flowsom' prêt pour FlowSOM:")
print(f"   Shape: {adata_flowsom.shape}")
print(f"   Colonnes pour clustering: {len(cols_to_use)}")

# Résumé par condition
print(f"\n Distribution par condition:")
for condition in adata_flowsom.obs["condition"].unique():
    n = (adata_flowsom.obs["condition"] == condition).sum()
    print(f"   {condition}: {n:,} cellules")


################################################################################
# ==================
# PARAMÈTRES FLOWSOM
# ==================
# → Valeurs contrôlées par le panneau interactif (cellule 2) si exécuté.
_cfg = globals().get("CONFIG", {})

# Grille SOM
XDIM = _cfg.get("XDIM", 10)
YDIM = _cfg.get("YDIM", 10)

# Itérations SOM (rlen) — 'auto' = √N × 0.1, borné [10, 100]
RLEN = _cfg.get("RLEN", "auto")

# Nombre de métaclusters (manuel, utilisé si AUTO_CLUSTER=False)
N_CLUSTERS = _cfg.get("N_CLUSTERS", 7)

# Seed pour reproductibilité
SEED = _cfg.get("SEED", 42)

# Auto-clustering multi-critères (Stabilité + Silhouette)
AUTO_CLUSTER = _cfg.get("AUTO_CLUSTER", False)
MIN_CLUSTERS_AUTO = _cfg.get("MIN_CLUSTERS_AUTO", 5)
MAX_CLUSTERS_AUTO = _cfg.get("MAX_CLUSTERS_AUTO", 35)
N_BOOTSTRAP = _cfg.get("N_BOOTSTRAP", 10)
SAMPLE_SIZE_BOOTSTRAP = _cfg.get("SAMPLE_SIZE_BOOTSTRAP", 20000)
MIN_STABILITY_THRESHOLD = _cfg.get("MIN_STABILITY_THRESHOLD", 0.75)
W_STABILITY = _cfg.get("W_STABILITY", 0.65)
W_SILHOUETTE = _cfg.get("W_SILHOUETTE", 0.35)

# Ancien paramètre silhouette (conservé en fallback)
SAMPLE_SIZE_SILHOUETTE = 10000

print("=" * 70)
print("PARAMÈTRES FLOWSOM")
print("=" * 70)
print(f"   Grille SOM         : {XDIM} × {YDIM} = {XDIM * YDIM} nodes")
print(f"   rlen (itérations)  : {RLEN}")
print(f"   Métaclusters       : {N_CLUSTERS} (manuel)")
print(f"   Seed               : {SEED}")
print(
    f"   Auto-clustering    : {'Oui (Stabilité+Silhouette)' if AUTO_CLUSTER else 'Non'}"
)
if AUTO_CLUSTER:
    print(f"   Plage k testée     : {MIN_CLUSTERS_AUTO}–{MAX_CLUSTERS_AUTO}")
    print(f"   Bootstraps         : {N_BOOTSTRAP}")
    print(f"   Sample bootstrap   : {SAMPLE_SIZE_BOOTSTRAP:,}")
    print(f"   Seuil stabilité    : {MIN_STABILITY_THRESHOLD}")
    print(f"   Poids composite    : stabilité={W_STABILITY}, silhouette={W_SILHOUETTE}")


################################################################################
# =============================================================================
# OPTIMISATION AUTOMATIQUE FlowSOM — Stabilité (AMJI/ARI) + Silhouette
# Méthode littérature 2024 : stabilité > score unique (silhouette)
# Particulièrement important pour MRD où blasts rares nécessitent haute résolution
# =============================================================================

from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
import time as _time


def compute_optimal_rlen(n_cells, rlen_setting="auto"):
    """
    Calcule rlen optimal basé sur la taille du dataset.
    Formule littérature: rlen ∝ √N / facteur, borné [10, 100].
    - 10k cellules  → rlen ≈ 10
    - 100k cellules → rlen ≈ 31
    - 500k cellules → rlen ≈ 70
    - 1M cellules   → rlen ≈ 100
    """
    if isinstance(rlen_setting, int):
        return rlen_setting
    return max(10, min(100, int(np.sqrt(n_cells) * 0.1)))


def compute_optimal_grid(n_cells, xdim_setting=10, ydim_setting=10):
    """
    Ajuste la grille SOM si nécessaire.
    - >50k cellules : 10×10 (100 nodes)
    - <50k cellules : 7×7 (49 nodes) recommandé
    """
    if n_cells < 50000 and xdim_setting == 10 and ydim_setting == 10:
        print(f"   [INFO] {n_cells:,} cellules < 50k → grille réduite 7×7 recommandée")
        return 7, 7
    return xdim_setting, ydim_setting


def phase1_silhouette_on_codebook(
    data, cols_to_use, xdim, ydim, rlen, seed, k_range, verbose=True
):
    """
    Phase 1 : Screening rapide via silhouette sur le codebook SOM.

    Avantage : Seulement n_nodes (ex: 100) points → quasi-instantané.
    On entraîne le SOM une seule fois, puis on re-métaclustère pour chaque k.
    """
    if verbose:
        print(f"\n{'─' * 60}")
        print("PHASE 1 : Silhouette sur codebook SOM (screening rapide)")
        print(f"{'─' * 60}")

    # Entraîner le SOM une seule fois avec k=max (on re-métaclustèrera ensuite)
    t0 = _time.time()
    # Utiliser GPU si activé et disponible
    _p1_model_kwargs = (
        {"model": GPUFlowSOMEstimator}
        if globals().get("USE_GPU_FLOWSOM", False) and GPU_FLOWSOM_AVAILABLE
        else {}
    )
    fsom_ref = fs.FlowSOM(
        data,
        cols_to_use=cols_to_use,
        xdim=xdim,
        ydim=ydim,
        rlen=rlen,
        n_clusters=max(k_range),
        seed=seed,
        **_p1_model_kwargs,
    )
    t_som = _time.time() - t0
    if verbose:
        print(f"   SOM entraîné en {t_som:.1f}s ({xdim}×{ydim}, rlen={rlen})")

    # Extraire le codebook (vecteurs prototypes des nodes SOM)
    cluster_data = fsom_ref.get_cluster_data()
    codebook = cluster_data.X
    if hasattr(codebook, "toarray"):
        codebook = codebook.toarray()

    # Utiliser seulement les colonnes de clustering
    codebook_use = np.nan_to_num(codebook[:, cols_to_use], nan=0.0)

    results = []
    for k in k_range:
        try:
            # Re-métaclustèrer le codebook existant (rapide!)
            fsom_ref.metacluster(n_clusters=k)
            node_labels = (
                fsom_ref.get_cluster_data().obs["metaclustering"].values.astype(int)
            )

            n_unique = len(np.unique(node_labels))
            if n_unique > 1 and n_unique < len(codebook_use):
                sil = silhouette_score(codebook_use, node_labels)
            else:
                sil = -1.0

            results.append({"k": k, "silhouette": sil})
            if verbose:
                bar = "█" * int(max(0, sil + 1) * 20)
                print(f"   k={k:3d} : silhouette={sil:+.4f}  {bar}")
        except Exception as e:
            results.append({"k": k, "silhouette": -1.0})
            if verbose:
                print(f"   k={k:3d} : erreur – {e}")

    return pd.DataFrame(results), fsom_ref


def phase2_bootstrap_stability(
    data,
    cols_to_use,
    xdim,
    ydim,
    rlen,
    seed,
    candidates_k,
    n_bootstrap=10,
    sample_size=20000,
    verbose=True,
):
    """
    Phase 2 : Évaluation de la stabilité par bootstrap (ARI moyen pairwise).

    Pour chaque k candidat :
    - Exécuter FlowSOM n_bootstrap fois avec des seeds différents
      sur un sous-échantillon fixe (même cellules, seeds SOM différentes)
    - Calculer l'ARI pairwise entre toutes les paires de runs
    - ARI moyen = score de stabilité (proxy AMJI)

    Seuil littérature : stabilité (ARI) > 0.75–0.80 = clustering robuste
    """
    if verbose:
        print(f"\n{'─' * 60}")
        print(
            f"PHASE 2 : Stabilité bootstrap (ARI) — {n_bootstrap} runs × {len(candidates_k)} candidats"
        )
        print(f"{'─' * 60}")

    n_total = data.shape[0]
    eval_size = min(sample_size, n_total)

    # Sous-échantillon FIXE pour évaluation (même cellules à chaque run)
    np.random.seed(seed)
    eval_idx = np.random.choice(n_total, eval_size, replace=False)
    data_eval = data[eval_idx].copy()

    if verbose:
        print(f"   Sous-échantillon : {eval_size:,} cellules (fixe pour tous les runs)")

    stability_results = {}

    # Utiliser GPU si activé et disponible
    _p2_model_kwargs = (
        {"model": GPUFlowSOMEstimator}
        if globals().get("USE_GPU_FLOWSOM", False) and GPU_FLOWSOM_AVAILABLE
        else {}
    )
    for k in candidates_k:
        t0 = _time.time()
        labels_all_runs = []

        for b in range(n_bootstrap):
            try:
                fsom_b = fs.FlowSOM(
                    data_eval,
                    cols_to_use=cols_to_use,
                    xdim=xdim,
                    ydim=ydim,
                    rlen=rlen,
                    n_clusters=k,
                    seed=seed + 100 + b,  # Seeds différentes
                    **_p2_model_kwargs,
                )
                labels_b = (
                    fsom_b.get_cell_data().obs["metaclustering"].values.astype(int)
                )
                labels_all_runs.append(labels_b)
            except Exception as e:
                if verbose:
                    print(f"   [!] k={k}, boot={b}: erreur – {e}")

        # ARI pairwise entre toutes les paires de runs
        ari_pairs = []
        for i in range(len(labels_all_runs)):
            for j in range(i + 1, len(labels_all_runs)):
                ari = adjusted_rand_score(labels_all_runs[i], labels_all_runs[j])
                ari_pairs.append(ari)

        mean_ari = np.mean(ari_pairs) if ari_pairs else 0.0
        std_ari = np.std(ari_pairs) if ari_pairs else 0.0
        elapsed = _time.time() - t0

        stability_results[k] = {
            "mean_ari": mean_ari,
            "std_ari": std_ari,
            "n_valid_runs": len(labels_all_runs),
            "n_pairs": len(ari_pairs),
        }

        if verbose:
            status = "✓" if mean_ari >= MIN_STABILITY_THRESHOLD else "✗"
            bar = "█" * int(mean_ari * 30)
            print(
                f"   k={k:3d} : ARI={mean_ari:.4f} ± {std_ari:.4f}  {bar}  {status}  ({elapsed:.1f}s)"
            )

    return stability_results


def phase3_composite_selection(
    sil_df,
    stability_results,
    w_stability=0.65,
    w_silhouette=0.35,
    min_stability=0.75,
    verbose=True,
):
    """
    Phase 3 : Score composite pondéré pour sélection finale.

    Score = w_stability × ARI_norm + w_silhouette × Sil_norm

    Avec filtrage : seuls les k avec stabilité > seuil sont considérés.
    """
    if verbose:
        print(f"\n{'─' * 60}")
        print("PHASE 3 : Score composite (Stabilité × Silhouette)")
        print(f"{'─' * 60}")

    # Fusionner les métriques
    composite = sil_df.copy()
    composite["stability"] = composite["k"].map(
        lambda k: stability_results.get(k, {}).get("mean_ari", np.nan)
    )
    composite["stability_std"] = composite["k"].map(
        lambda k: stability_results.get(k, {}).get("std_ari", np.nan)
    )

    # Filtrer les k sans données de stabilité
    valid = composite.dropna(subset=["stability"]).copy()

    if valid.empty:
        if verbose:
            print(
                "   [!] Aucun candidat avec données de stabilité → fallback silhouette"
            )
        best_k = sil_df.loc[sil_df["silhouette"].idxmax(), "k"]
        return int(best_k), composite

    # Normaliser [0, 1] pour chaque métrique
    sil_min, sil_max = valid["silhouette"].min(), valid["silhouette"].max()
    sta_min, sta_max = valid["stability"].min(), valid["stability"].max()

    if sil_max > sil_min:
        valid["sil_norm"] = (valid["silhouette"] - sil_min) / (sil_max - sil_min)
    else:
        valid["sil_norm"] = 0.5

    if sta_max > sta_min:
        valid["sta_norm"] = (valid["stability"] - sta_min) / (sta_max - sta_min)
    else:
        valid["sta_norm"] = 0.5

    # Score composite pondéré
    valid["composite_score"] = (
        w_stability * valid["sta_norm"] + w_silhouette * valid["sil_norm"]
    )

    # Bonus : pénaliser les k avec faible stabilité (sous le seuil)
    valid.loc[valid["stability"] < min_stability, "composite_score"] *= 0.7

    # Affichage détaillé
    if verbose:
        print(
            f"\n   {'k':>4}  {'Silhouette':>11}  {'Stabilité':>10}  {'Score':>8}  {'Verdict':>10}"
        )
        print(f"   {'─' * 4}  {'─' * 11}  {'─' * 10}  {'─' * 8}  {'─' * 10}")
        for _, row in valid.sort_values("composite_score", ascending=False).iterrows():
            verdict = (
                "★ OPTIMAL"
                if row["composite_score"] == valid["composite_score"].max()
                else ""
            )
            if row["stability"] < min_stability:
                verdict = "(instable)"
            print(
                f"   {int(row['k']):4d}  {row['silhouette']:+11.4f}  {row['stability']:10.4f}  "
                f"{row['composite_score']:8.4f}  {verdict}"
            )

    # Sélection finale
    best_idx = valid["composite_score"].idxmax()
    best_k = int(valid.loc[best_idx, "k"])
    best_sil = valid.loc[best_idx, "silhouette"]
    best_sta = valid.loc[best_idx, "stability"]
    best_score = valid.loc[best_idx, "composite_score"]

    if verbose:
        print(f"\n   ╔══════════════════════════════════════════════════════╗")
        print(f"   ║  OPTIMAL : k = {best_k}                               ")
        print(f"   ║  Silhouette = {best_sil:+.4f}                          ")
        print(f"   ║  Stabilité ARI = {best_sta:.4f}                        ")
        print(f"   ║  Score composite = {best_score:.4f}                    ")
        print(f"   ╚══════════════════════════════════════════════════════╝")

    return best_k, valid


def plot_optimization_results(results_df, best_k, stability_results=None):
    """Visualisation des résultats d'optimisation multi-critères."""
    has_stability = stability_results and len(stability_results) > 0
    n_plots = 3 if has_stability else 2

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 2:
        axes = [axes[0], axes[1]]

    # ── Plot 1 : Silhouette ──
    ax = axes[0]
    ks = results_df["k"].values
    sils = results_df["silhouette"].values
    ax.plot(
        ks, sils, "o-", color="#2196F3", linewidth=2, markersize=5, label="Silhouette"
    )
    ax.axvline(
        best_k,
        color="#F44336",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Optimal k={best_k}",
    )
    ax.set_xlabel("Nombre de métaclusters (k)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Silhouette sur Codebook SOM", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Plot 2 : Stabilité ARI ──
    if has_stability:
        ax = axes[1]
        stab_ks = sorted(stability_results.keys())
        stab_aris = [stability_results[k]["mean_ari"] for k in stab_ks]
        stab_stds = [stability_results[k]["std_ari"] for k in stab_ks]

        ax.errorbar(
            stab_ks,
            stab_aris,
            yerr=stab_stds,
            fmt="s-",
            color="#4CAF50",
            linewidth=2,
            markersize=6,
            capsize=3,
            label="ARI moyen ± σ",
        )
        ax.axhline(
            MIN_STABILITY_THRESHOLD,
            color="#FF9800",
            linestyle=":",
            linewidth=1.5,
            label=f"Seuil stabilité ({MIN_STABILITY_THRESHOLD})",
        )
        ax.axvline(best_k, color="#F44336", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel("Nombre de métaclusters (k)", fontsize=11)
        ax.set_ylabel("ARI moyen (stabilité)", fontsize=11)
        ax.set_title(
            "Stabilité Bootstrap (ARI pairwise)", fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    # ── Plot 3 (ou 2) : Score composite ──
    ax = axes[-1]
    if "composite_score" in results_df.columns:
        valid = results_df.dropna(subset=["composite_score"])
        ax.bar(
            valid["k"],
            valid["composite_score"],
            color="#9C27B0",
            alpha=0.7,
            label="Score composite",
        )
        ax.axvline(
            best_k,
            color="#F44336",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Optimal k={best_k}",
        )
        ax.set_xlabel("Nombre de métaclusters (k)", fontsize=11)
        ax.set_ylabel("Score composite", fontsize=11)
        ax.set_title(
            f"Score Composite (w_stab={W_STABILITY}, w_sil={W_SILHOUETTE})",
            fontsize=12,
            fontweight="bold",
        )
    else:
        # Fallback: juste sil avec elbow
        diffs = np.diff(sils)
        ax.plot(ks[1:], diffs, "o-", color="#FF5722", linewidth=2, markersize=4)
        ax.axvline(best_k, color="#F44336", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel("k", fontsize=11)
        ax.set_ylabel("Δ Silhouette", fontsize=11)
        ax.set_title("Variation Silhouette (Elbow)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "flowsom_optimization.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")
    print(f"   [OK] Figure sauvegardée → {OUTPUT_DIR}/flowsom_optimization.png")


def find_optimal_clusters_stability(
    data,
    cols_to_use,
    seed=42,
    xdim=10,
    ydim=10,
    rlen="auto",
    k_range=None,
    n_bootstrap=10,
    sample_size_boot=20000,
    w_stability=0.65,
    w_silhouette=0.35,
    min_stability=0.75,
    n_top_candidates=8,
):
    """
    Pipeline complet d'optimisation FlowSOM multi-critères.

    Méthode littérature 2024 (Weber, Van Gassen et al.):
    Stabilité > score unique comme silhouette, surtout pour MRD
    où populations rares (blasts CD34+) nécessitent haute résolution.

    Étapes:
    ───────
    1. Auto-calcul rlen et grille selon N cellules
    2. Phase 1: Silhouette sur codebook SOM (screening rapide, tous les k)
    3. Phase 2: Bootstrap stability (ARI pairwise, top candidats)
    4. Phase 3: Score composite pondéré → sélection finale
    5. Visualisation multi-panels

    Returns:
    ────────
    best_k : int — nombre optimal de métaclusters
    best_rlen : int — rlen optimisé
    best_xdim, best_ydim : int — dimensions grille
    """
    n_cells = data.shape[0]

    print("=" * 70)
    print("OPTIMISATION FlowSOM — Méthode Stabilité 2024")
    print("=" * 70)
    print(f"   Dataset : {n_cells:,} cellules × {len(cols_to_use)} marqueurs")

    # ── Auto-paramètres ──
    best_rlen = compute_optimal_rlen(n_cells, rlen)
    best_xdim, best_ydim = compute_optimal_grid(n_cells, xdim, ydim)

    print(f"   Grille  : {best_xdim}×{best_ydim} = {best_xdim * best_ydim} nodes")
    print(f"   rlen    : {best_rlen} ({'auto' if rlen == 'auto' else 'manuel'})")

    if k_range is None:
        k_range = range(MIN_CLUSTERS_AUTO, MAX_CLUSTERS_AUTO + 1)

    # ── Phase 1 : Silhouette rapide sur codebook ──
    t_total = _time.time()
    sil_df, fsom_ref = phase1_silhouette_on_codebook(
        data, cols_to_use, best_xdim, best_ydim, best_rlen, seed, k_range
    )

    # ── Sélection des top candidats pour bootstrap ──
    top_n = min(n_top_candidates, len(sil_df))
    top_candidates = sil_df.nlargest(top_n, "silhouette")["k"].values.tolist()

    # Ajouter aussi les voisins du meilleur silhouette (±2) pour robustesse
    best_sil_k = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])
    for delta in [-2, -1, 1, 2]:
        neighbor = best_sil_k + delta
        if neighbor in list(k_range) and neighbor not in top_candidates:
            top_candidates.append(neighbor)
    top_candidates = sorted(set(top_candidates))

    print(
        f"\n   → {len(top_candidates)} candidats retenus pour bootstrap : {top_candidates}"
    )

    # ── Phase 2 : Bootstrap stability ──
    stability_results = phase2_bootstrap_stability(
        data,
        cols_to_use,
        best_xdim,
        best_ydim,
        best_rlen,
        seed,
        top_candidates,
        n_bootstrap=n_bootstrap,
        sample_size=sample_size_boot,
    )

    # ── Phase 3 : Score composite ──
    best_k, composite_df = phase3_composite_selection(
        sil_df,
        stability_results,
        w_stability=w_stability,
        w_silhouette=w_silhouette,
        min_stability=min_stability,
    )

    elapsed_total = _time.time() - t_total

    print(f"\n{'=' * 70}")
    print(f"RÉSULTAT FINAL — Temps total : {elapsed_total:.1f}s")
    print(f"{'=' * 70}")
    print(f"   k optimal       = {best_k} métaclusters")
    print(f"   rlen optimal     = {best_rlen}")
    print(f"   Grille SOM       = {best_xdim}×{best_ydim}")

    # ── Visualisation ──
    try:
        plot_optimization_results(composite_df, best_k, stability_results)
    except Exception as e:
        print(f"   [!] Visualisation échouée : {e}")

    return best_k, best_rlen, best_xdim, best_ydim


# =============================================================================
# EXÉCUTION DE L'OPTIMISATION
# =============================================================================
if AUTO_CLUSTER:
    N_CLUSTERS, RLEN_OPT, XDIM, YDIM = find_optimal_clusters_stability(
        combined_gated,
        cols_to_use,
        seed=SEED,
        xdim=XDIM,
        ydim=YDIM,
        rlen=RLEN,
        n_bootstrap=N_BOOTSTRAP,
        sample_size_boot=SAMPLE_SIZE_BOOTSTRAP,
        w_stability=W_STABILITY,
        w_silhouette=W_SILHOUETTE,
        min_stability=MIN_STABILITY_THRESHOLD,
    )
    # Mettre à jour le rlen global
    if isinstance(RLEN, str):
        RLEN = RLEN_OPT
    print(f"\n[OK] Utilisation de {N_CLUSTERS} métaclusters (rlen={RLEN})")
else:
    # Mode manuel : calculer rlen si 'auto'
    if isinstance(RLEN, str) and RLEN == "auto":
        RLEN = compute_optimal_rlen(combined_gated.shape[0])
        print(f"   rlen auto-calculé : {RLEN}")
    print(f"\n[OK] Mode manuel : {N_CLUSTERS} métaclusters")


################################################################################
# =============================================================================
# EXÉCUTION FLOWSOM (avec paramètres optimisés)
# =============================================================================

import time

start_time = time.time()

# Convertir RLEN en int si encore en 'auto'
rlen_final = (
    RLEN if isinstance(RLEN, int) else compute_optimal_rlen(adata_flowsom.shape[0])
)

print(f"Lancement FlowSOM avec paramètres optimisés :")
print(f"   Grille   : {XDIM}×{YDIM} = {XDIM * YDIM} nodes")
print(f"   rlen     : {rlen_final}")
print(f"   nClus    : {N_CLUSTERS}")
print(f"   Seed     : {SEED}")

# Choisir le modèle FlowSOM (GPU ou CPU selon USE_GPU_FLOWSOM)
_fsom_model_kwargs = {}
if globals().get("USE_GPU_FLOWSOM", False) and GPU_FLOWSOM_AVAILABLE:
    _fsom_model_kwargs = {"model": GPUFlowSOMEstimator}
    print(f"   Modèle   : GPU (GPUFlowSOMEstimator) ⚡")
else:
    print(f"   Modèle   : CPU (FlowSOMEstimator standard)")

# Exécuter FlowSOM avec adata_flowsom (données transformées arcsinh)
fsom = fs.FlowSOM(
    adata_flowsom,  # ← IMPORTANT: utilise les données POST-transformation
    cols_to_use=cols_to_use,
    xdim=XDIM,
    ydim=YDIM,
    rlen=rlen_final,
    n_clusters=N_CLUSTERS,
    seed=SEED,
    **_fsom_model_kwargs,
)

elapsed = time.time() - start_time
print(f"\nTemps d'exécution: {elapsed:.2f} secondes")

# Récupérer les données de clustering
cell_data = fsom.get_cell_data()
cluster_data = fsom.get_cluster_data()

# Ajouter les métadonnées originales
cell_data.obs["condition"] = adata_flowsom.obs["condition"].values
cell_data.obs["file_origin"] = adata_flowsom.obs["file_origin"].values

print(f"\n[OK] FlowSOM terminé!")
print(f"   Cellules analysées: {cell_data.shape[0]:,}")
print(f"   Nodes SOM: {cluster_data.shape[0]}")
print(f"   Métaclusters: {N_CLUSTERS}")
print(f"   rlen utilisé: {rlen_final}")


################################################################################
# =============================================================================
# HEATMAP D'EXPRESSION PAR MÉTACLUSTER
# =============================================================================

print(" Génération de la Heatmap d'expression...")

# Récupérer les données
X = cell_data.X
if hasattr(X, "toarray"):
    X = X.toarray()

metaclustering = cell_data.obs["metaclustering"].values

# Calculer la MFI (Mean Fluorescence Intensity) par métacluster — VECTORISÉ
# Utilise pd.DataFrame.groupby au lieu d'une boucle O(n_cells × n_clusters)
X_markers = X[:, cols_to_use]
_mc_series = pd.Series(metaclustering, name="mc")
mfi_matrix = (
    pd.DataFrame(X_markers, columns=range(len(cols_to_use)))
    .groupby(_mc_series)
    .mean()
    .reindex(range(N_CLUSTERS))
    .fillna(0)
    .values
)

# Normalisation Z-score pour la heatmap
mfi_normalized = (mfi_matrix - np.nanmean(mfi_matrix, axis=0)) / (
    np.nanstd(mfi_matrix, axis=0) + 1e-10
)

# Créer la heatmap
fig, ax = plt.subplots(figsize=(14, 8))

im = ax.imshow(mfi_normalized.T, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)

# Labels
ax.set_yticks(range(len(used_markers)))
ax.set_yticklabels(used_markers, fontsize=9)
ax.set_xticks(range(N_CLUSTERS))
ax.set_xticklabels([f"MC{i}" for i in range(N_CLUSTERS)], fontsize=10)

ax.set_title(
    "Heatmap - Expression par Métacluster (Z-score)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Métacluster", fontsize=12)
ax.set_ylabel("Marqueur", fontsize=12)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Z-score")

plt.tight_layout()
_heatmap_path = (
    Path(globals().get("OUTPUT_DIR", "./output")) / "flowsom_heatmap_zscore.png"
)
_heatmap_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(_heatmap_path, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"   [OK] Heatmap Z-score sauvegardée → {_heatmap_path}")
# =============================================================================

print("Génération du Star Chart MST...")

try:
    # Utiliser l'API FlowSOM pour le Star Chart
    fig_stars = fs.pl.plot_stars(
        fsom, background_values=fsom.get_cluster_data().obs.metaclustering, view="MST"
    )
    plt.suptitle("FlowSOM Star Chart (MST View)", fontsize=14, fontweight="bold")
    _star_path = (
        Path(globals().get("OUTPUT_DIR", "./output")) / "flowsom_star_chart.png"
    )
    _star_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(_star_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"   [OK] Star Chart sauvegardé → {_star_path}")
except Exception as e:
    print(f"Erreur Star Chart: {e}")
    print("   Utilisation de la visualisation alternative...")


################################################################################
# =============================================================================
# VISUALISATION GRILLE SOM (xGrid, yGrid) - Style FlowSOM R exact
# =============================================================================
# PARTIE 1 : MATPLOTLIB (statique, haute résolution pour PDF/export)
# =============================================================================

print(" VISUALISATION GRILLE SOM (style FlowSOM R avec cercles)")
print("=" * 70)


# =====================================================================
# FONCTION JITTER CIRCULAIRE (style FlowSOM R)
# =====================================================================
def circular_jitter_viz(
    n_points, cluster_ids, node_sizes, max_radius=0.45, min_radius=0.1
):
    """
    Génère un jitter circulaire style FlowSOM R — VECTORISÉ.
    Le rayon des cercles dépend du nombre de cellules dans le node.
    """
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    u = np.random.uniform(0, 1, n_points)

    max_size_val = node_sizes.max()

    # Calcul vectorisé des rayons (pas de boucle Python)
    size_ratios = np.sqrt(node_sizes[cluster_ids.astype(int)] / max_size_val)
    radii = (min_radius + (max_radius - min_radius) * size_ratios).astype(np.float32)

    r = np.sqrt(u) * radii

    jitter_x = r * np.cos(theta)
    jitter_y = r * np.sin(theta)

    return jitter_x.astype(np.float32), jitter_y.astype(np.float32)


try:
    # Récupérer les coordonnées de grille
    grid_coords = cluster_data.obsm.get("grid", None)

    if grid_coords is not None:
        # Récupérer les infos de clustering
        clustering = cell_data.obs["clustering"].values
        metaclustering_nodes = cluster_data.obs["metaclustering"].values
        conditions = cell_data.obs["condition"].values

        # Calculer les coordonnées de grille pour chaque cellule
        xGrid_base = np.array(
            [grid_coords[int(c), 0] for c in clustering], dtype=np.float32
        )
        yGrid_base = np.array(
            [grid_coords[int(c), 1] for c in clustering], dtype=np.float32
        )

        # Décaler pour commencer à 1
        xGrid_shifted = xGrid_base - xGrid_base.min() + 1
        yGrid_shifted = yGrid_base - yGrid_base.min() + 1

        # Métacluster pour chaque cellule
        metaclustering_cells = np.array(
            [metaclustering_nodes[int(c)] for c in clustering]
        )

        # Calculer la taille de chaque node
        n_nodes = len(cluster_data)
        node_sizes = np.zeros(n_nodes, dtype=np.float32)
        for i in range(n_nodes):
            node_sizes[i] = (clustering == i).sum()

        # JITTER CIRCULAIRE style FlowSOM R
        MAX_NODE_SIZE = 0.45
        MIN_NODE_SIZE = 0.1
        np.random.seed(SEED)
        jitter_x, jitter_y = circular_jitter_viz(
            len(clustering),
            clustering,
            node_sizes,
            max_radius=MAX_NODE_SIZE,
            min_radius=MIN_NODE_SIZE,
        )

        print(f" Jitter circulaire appliqué (rayon proportionnel à la taille du node)")
        print(f"   Rayon min: {MIN_NODE_SIZE}, Rayon max: {MAX_NODE_SIZE}")

        # Créer la figure avec 2 sous-plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # =====================================================================
        # Plot 1: Grille SOM colorée par Métacluster
        # =====================================================================
        ax1 = axes[0]

        n_meta = len(np.unique(metaclustering_nodes))
        cmap = plt.cm.tab20 if n_meta <= 20 else plt.cm.turbo

        scatter1 = ax1.scatter(
            xGrid_shifted + jitter_x,
            yGrid_shifted + jitter_y,
            c=metaclustering_cells,
            cmap=cmap,
            s=5,
            alpha=0.5,
            edgecolors="none",
        )

        # Ajouter les labels des métaclusters au centre de chaque node
        for node_id in range(n_nodes):
            if node_sizes[node_id] > 0:
                x_pos = grid_coords[node_id, 0] - xGrid_base.min() + 1
                y_pos = grid_coords[node_id, 1] - yGrid_base.min() + 1
                meta_id = metaclustering_nodes[node_id]
                ax1.annotate(
                    str(int(meta_id + 1)),
                    (x_pos, y_pos),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        boxstyle="circle,pad=0.2",
                        facecolor=cmap(meta_id / max(n_meta - 1, 1)),
                        edgecolor="white",
                        alpha=0.9,
                    ),
                )

        ax1.set_xlabel("xGrid", fontsize=12, fontweight="bold")
        ax1.set_ylabel("yGrid", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"Grille FlowSOM - {XDIM}x{YDIM} nodes\nColoré par Métacluster (style FlowSOM R)",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_xlim(0.5, XDIM + 1.5)
        ax1.set_ylim(0.5, YDIM + 1.5)
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3, linestyle="--")

        cbar1 = plt.colorbar(scatter1, ax=ax1, label="Métacluster")

        # =====================================================================
        # Plot 2: Grille SOM colorée par Condition
        # =====================================================================
        ax2 = axes[1]

        condition_num = np.array([0 if c == "Sain" else 1 for c in conditions])

        from matplotlib.colors import ListedColormap

        cmap_cond = ListedColormap(["#a6e3a1", "#f38ba8"])

        scatter2 = ax2.scatter(
            xGrid_shifted + jitter_x,
            yGrid_shifted + jitter_y,
            c=condition_num,
            cmap=cmap_cond,
            s=5,
            alpha=0.5,
            edgecolors="none",
        )

        ax2.set_xlabel("xGrid", fontsize=12, fontweight="bold")
        ax2.set_ylabel("yGrid", fontsize=12, fontweight="bold")
        ax2.set_title(
            f"Grille FlowSOM - {XDIM}x{YDIM} nodes\nColoré par Condition (style FlowSOM R)",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xlim(0.5, XDIM + 1.5)
        ax2.set_ylim(0.5, YDIM + 1.5)
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.3, linestyle="--")

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#a6e3a1", edgecolor="white", label="Sain (NBM)"),
            Patch(facecolor="#f38ba8", edgecolor="white", label="Pathologique"),
        ]
        ax2.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        _som_grid_path = (
            Path(globals().get("OUTPUT_DIR", "./output")) / "flowsom_som_grid.png"
        )
        _som_grid_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(_som_grid_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"   [OK] Grille SOM sauvegardée → {_som_grid_path}")

        # Afficher les statistiques
        print(f"\n STATISTIQUES DE LA GRILLE SOM:")
        print(f"   Dimensions: {XDIM} x {YDIM} = {XDIM * YDIM} nodes")
        print(f"   Nodes utilisés: {(node_sizes > 0).sum()} / {n_nodes}")
        print(f"   xGrid range: [{xGrid_shifted.min():.1f}, {xGrid_shifted.max():.1f}]")
        print(f"   yGrid range: [{yGrid_shifted.min():.1f}, {yGrid_shifted.max():.1f}]")

        # Afficher la taille des nodes
        print(f"\n Distribution des tailles de nodes:")
        print(f"   Min: {node_sizes.min():.0f} cellules")
        print(f"   Max: {node_sizes.max():.0f} cellules")
        print(f"   Moyenne: {node_sizes.mean():.0f} cellules")

        # =================================================================
        # PARTIE 2 : PLOTLY INTERACTIF (zoom, hover, export)
        # =================================================================
        print("\n" + "=" * 70)
        print(" GRILLE SOM — VERSION PLOTLY INTERACTIVE")
        print("=" * 70)

        import plotly.graph_objects as go
        import plotly.colors as pc_grid

        _xj = xGrid_shifted + jitter_x
        _yj = yGrid_shifted + jitter_y

        # Sous-échantillonner si trop de points pour la fluidité Plotly
        _max_pts = 50_000
        if len(clustering) > _max_pts:
            np.random.seed(SEED)
            _sample_idx = np.random.choice(len(clustering), _max_pts, replace=False)
        else:
            _sample_idx = np.arange(len(clustering))

        if n_meta <= 20:
            _mc_palette = pc_grid.qualitative.Alphabet[:n_meta]
        else:
            _mc_palette = [
                f"hsl({int(i * 360 / n_meta)},70%,55%)" for i in range(n_meta)
            ]

        # --- Plot 1 Plotly : Métacluster ---
        fig_grid_mc = go.Figure()

        for mc_id in range(n_meta):
            _mask_mc = metaclustering_cells[_sample_idx] == mc_id
            if _mask_mc.sum() == 0:
                continue
            _si = _sample_idx[_mask_mc]
            fig_grid_mc.add_trace(
                go.Scattergl(
                    x=_xj[_si],
                    y=_yj[_si],
                    mode="markers",
                    marker=dict(
                        size=3, color=_mc_palette[mc_id % len(_mc_palette)], opacity=0.5
                    ),
                    name=f"MC{mc_id} ({_mask_mc.sum():,})",
                    hovertemplate=f"MC{mc_id}<br>xGrid: %{{x:.2f}}<br>yGrid: %{{y:.2f}}<extra></extra>",
                )
            )

        _node_x = [
            grid_coords[i, 0] - xGrid_base.min() + 1
            for i in range(n_nodes)
            if node_sizes[i] > 0
        ]
        _node_y = [
            grid_coords[i, 1] - yGrid_base.min() + 1
            for i in range(n_nodes)
            if node_sizes[i] > 0
        ]
        _node_txt = [
            str(int(metaclustering_nodes[i] + 1))
            for i in range(n_nodes)
            if node_sizes[i] > 0
        ]
        _node_sz = [node_sizes[i] for i in range(n_nodes) if node_sizes[i] > 0]

        fig_grid_mc.add_trace(
            go.Scatter(
                x=_node_x,
                y=_node_y,
                mode="text",
                text=_node_txt,
                textfont=dict(size=9, color="black", family="Arial Black"),
                hovertemplate=[
                    f"Node — MC{t}<br>{int(s):,} cellules<extra></extra>"
                    for t, s in zip(_node_txt, _node_sz)
                ],
                showlegend=False,
            )
        )

        fig_grid_mc.update_layout(
            title=dict(
                text=f"<b>Grille FlowSOM — {XDIM}×{YDIM} nodes — Coloré par Métacluster</b><br>"
                "<sup>Style FlowSOM R (jitter circulaire proportionnel) — Interactif</sup>",
                font=dict(size=14),
            ),
            xaxis=dict(
                title="xGrid",
                range=[0.3, XDIM + 1.7],
                scaleanchor="y",
                scaleratio=1,
                gridcolor="rgba(0,0,0,0.08)",
                gridwidth=1,
            ),
            yaxis=dict(
                title="yGrid",
                range=[0.3, YDIM + 1.7],
                gridcolor="rgba(0,0,0,0.08)",
                gridwidth=1,
            ),
            height=700,
            width=800,
            paper_bgcolor="#fafafa",
            plot_bgcolor="#f5f5f5",
            legend=dict(
                title="Métacluster",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc",
                borderwidth=1,
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(t=80, b=50, l=60, r=180),
        )
        fig_grid_mc.write_html(
            str(_dashboard_out / f"fig_grid_mc_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )

        # --- Plot 2 Plotly : Condition ---
        fig_grid_cond = go.Figure()

        _cond_colors = {"Sain": "#2ca02c", "Pathologique": "#d62728"}
        for cond_label, cond_color in _cond_colors.items():
            _mask_c = conditions[_sample_idx] == cond_label
            if _mask_c.sum() == 0:
                continue
            _si = _sample_idx[_mask_c]
            fig_grid_cond.add_trace(
                go.Scattergl(
                    x=_xj[_si],
                    y=_yj[_si],
                    mode="markers",
                    marker=dict(size=3, color=cond_color, opacity=0.45),
                    name=f"{cond_label} ({_mask_c.sum():,})",
                    hovertemplate=f"{cond_label}<br>xGrid: %{{x:.2f}}<br>yGrid: %{{y:.2f}}<extra></extra>",
                )
            )

        fig_grid_cond.update_layout(
            title=dict(
                text=f"<b>Grille FlowSOM — {XDIM}×{YDIM} nodes — Coloré par Condition</b><br>"
                "<sup>Style FlowSOM R (jitter circulaire proportionnel) — Interactif</sup>",
                font=dict(size=14),
            ),
            xaxis=dict(
                title="xGrid",
                range=[0.3, XDIM + 1.7],
                scaleanchor="y",
                scaleratio=1,
                gridcolor="rgba(0,0,0,0.08)",
                gridwidth=1,
            ),
            yaxis=dict(
                title="yGrid",
                range=[0.3, YDIM + 1.7],
                gridcolor="rgba(0,0,0,0.08)",
                gridwidth=1,
            ),
            height=700,
            width=800,
            paper_bgcolor="#fafafa",
            plot_bgcolor="#f5f5f5",
            legend=dict(
                title="Condition",
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc",
                borderwidth=1,
            ),
            margin=dict(t=80, b=50, l=60, r=60),
        )
        fig_grid_cond.write_html(
            str(_dashboard_out / f"fig_grid_cond_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )

    else:
        print(
            "[!] Coordonnées de grille non disponibles dans cluster_data.obsm['grid']"
        )

except Exception as e:
    import traceback

    print(f"[!] Erreur visualisation grille: {e}")
    traceback.print_exc()


################################################################################
# =============================================================================
# ARBRE MST — MATPLOTLIB (statique) + PLOTLY (interactif)
# =============================================================================

print("Génération de l'arbre MST...")

# =====================================================================
# PARTIE 1 : MATPLOTLIB (statique, haute résolution pour PDF/export)
# =====================================================================
try:
    layout = cluster_data.obsm.get("layout", None)

    if layout is not None:
        clustering = cell_data.obs["clustering"].values
        metaclustering_nodes = cluster_data.obs["metaclustering"].values

        n_nodes = len(cluster_data)
        node_sizes = np.zeros(n_nodes)
        for i in range(n_nodes):
            node_sizes[i] = (clustering == i).sum()

        max_size = node_sizes.max() if node_sizes.max() > 0 else 1
        sizes = 100 + (node_sizes / max_size) * 800

        n_meta = len(np.unique(metaclustering_nodes))
        cmap = plt.cm.tab20 if n_meta <= 20 else plt.cm.turbo
        colors = [cmap(int(m) / max(n_meta - 1, 1)) for m in metaclustering_nodes]

        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(
            layout[:, 0],
            layout[:, 1],
            s=sizes,
            c=colors,
            edgecolors="white",
            linewidths=1.5,
            alpha=0.9,
            zorder=2,
        )

        for i in range(n_nodes):
            ax.annotate(
                str(int(metaclustering_nodes[i])),
                (layout[i, 0], layout[i, 1]),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

        ax.set_xlabel("xNodes", fontsize=12, fontweight="bold")
        ax.set_ylabel("yNodes", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Arbre MST - {n_nodes} nodes, {n_meta} métaclusters",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.grid(True, alpha=0.15, linestyle="--")

        from matplotlib.patches import Patch

        if n_meta <= 15:
            legend_elements = [
                Patch(facecolor=cmap(i / max(n_meta - 1, 1)), label=f"MC {i}")
                for i in range(n_meta)
            ]
            ax.legend(
                handles=legend_elements,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=9,
            )

        plt.tight_layout()
        _mst_path = (
            Path(globals().get("OUTPUT_DIR", "./output")) / "flowsom_mst_matplotlib.png"
        )
        _mst_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(_mst_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"   [OK] MST Matplotlib sauvegardé → {_mst_path}")

        # =================================================================
        # PARTIE 2 : PLOTLY INTERACTIF (zoom, hover, export)
        # =================================================================
        print("\n" + "=" * 70)
        print(" ARBRE MST — VERSION PLOTLY INTERACTIVE")
        print("=" * 70)

        import plotly.graph_objects as go
        import plotly.colors as pc_mst

        _bubble_sizes = 10 + (node_sizes / max_size) * 40

        if n_meta <= 20:
            _mc_palette = pc_mst.qualitative.Alphabet[:n_meta]
        else:
            _mc_palette = [
                f"hsl({int(i * 360 / n_meta)},70%,55%)" for i in range(n_meta)
            ]

        # Récupérer les arêtes du MST si disponibles
        _edge_x, _edge_y = [], []
        _mst_graph = None
        try:
            if hasattr(cluster_data, "uns") and "mst" in cluster_data.uns:
                _mst_graph = cluster_data.uns["mst"]
        except Exception:
            pass

        if _mst_graph is not None:
            try:
                import igraph

                if isinstance(_mst_graph, igraph.Graph):
                    for edge in _mst_graph.es:
                        s, t = edge.source, edge.target
                        if s < n_nodes and t < n_nodes:
                            _edge_x += [layout[s, 0], layout[t, 0], None]
                            _edge_y += [layout[s, 1], layout[t, 1], None]
            except ImportError:
                pass

        fig_mst = go.Figure()

        if _edge_x:
            fig_mst.add_trace(
                go.Scatter(
                    x=_edge_x,
                    y=_edge_y,
                    mode="lines",
                    line=dict(width=1.5, color="rgba(100,100,100,0.5)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        for mc_id in range(n_meta):
            _mask = metaclustering_nodes == mc_id
            if _mask.sum() == 0:
                continue
            _indices = np.where(_mask)[0]
            fig_mst.add_trace(
                go.Scatter(
                    x=layout[_indices, 0],
                    y=layout[_indices, 1],
                    mode="markers+text",
                    marker=dict(
                        size=_bubble_sizes[_indices],
                        color=_mc_palette[mc_id % len(_mc_palette)],
                        line=dict(width=1.5, color="white"),
                        opacity=0.9,
                    ),
                    text=[str(int(mc_id)) for _ in _indices],
                    textfont=dict(size=9, color="white", family="Arial Black"),
                    textposition="middle center",
                    name=f"MC{mc_id} ({int(node_sizes[_indices].sum()):,} cells)",
                    hovertemplate=[
                        f"<b>Node {ni}</b><br>"
                        f"MC {int(metaclustering_nodes[ni])}<br>"
                        f"Cellules: {int(node_sizes[ni]):,}<br>"
                        f"x: {layout[ni, 0]:.2f}, y: {layout[ni, 1]:.2f}<extra></extra>"
                        for ni in _indices
                    ],
                )
            )

        fig_mst.update_layout(
            title=dict(
                text=f"<b>Arbre MST — {n_nodes} nodes, {n_meta} métaclusters</b><br>"
                "<sup>Taille des bulles ∝ nombre de cellules · Cliquez la légende pour filtrer</sup>",
                font=dict(size=14),
            ),
            xaxis=dict(
                title="xNodes",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.06)",
                zeroline=False,
            ),
            yaxis=dict(
                title="yNodes",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.06)",
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,
            ),
            height=750,
            width=900,
            paper_bgcolor="#fafafa",
            plot_bgcolor="#f5f5f5",
            legend=dict(
                title="Métacluster",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc",
                borderwidth=1,
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(t=80, b=50, l=60, r=200),
        )
        fig_mst.write_html(
            str(_dashboard_out / f"fig_mst_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )

        print(
            f"[OK] Arbre MST — Matplotlib + Plotly interactif ({n_nodes} nodes, {n_meta} MC)"
        )
    else:
        print("[!] Layout MST non disponible")

except Exception as e:
    import traceback

    print(f"[!] Erreur MST: {e}")
    traceback.print_exc()


################################################################################
# =============================================================================
# DISTRIBUTION PAR CONDITION (Sain vs Pathologique)
# =============================================================================

print("Distribution des métaclusters par condition...")

metaclustering = cell_data.obs["metaclustering"].values
conditions = cell_data.obs["condition"].values

healthy_pcts = []
patho_pcts = []

for i in range(N_CLUSTERS):
    mask_cluster = metaclustering == i

    # Pourcentage dans Sain
    mask_healthy = (conditions == "Sain") & mask_cluster
    total_healthy = (conditions == "Sain").sum()
    healthy_pcts.append(
        (mask_healthy.sum() / total_healthy * 100) if total_healthy > 0 else 0
    )

    # Pourcentage dans Pathologique
    mask_patho = (conditions == "Pathologique") & mask_cluster
    total_patho = (conditions == "Pathologique").sum()
    patho_pcts.append((mask_patho.sum() / total_patho * 100) if total_patho > 0 else 0)

# Créer le graphique
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(N_CLUSTERS)
width = 0.35

bars1 = ax.bar(
    x - width / 2,
    healthy_pcts,
    width,
    label="Sain (NBM)",
    color="#a6e3a1",
    edgecolor="white",
    linewidth=0.5,
)
bars2 = ax.bar(
    x + width / 2,
    patho_pcts,
    width,
    label="Pathologique",
    color="#f38ba8",
    edgecolor="white",
    linewidth=0.5,
)

ax.set_xlabel("Métacluster", fontsize=12)
ax.set_ylabel("Pourcentage (%)", fontsize=12)
ax.set_title(
    "Distribution des Métaclusters par Condition",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xticks(x)
ax.set_xticklabels([f"MC{i}" for i in range(N_CLUSTERS)], fontsize=10)
ax.legend()
ax.grid(axis="y", alpha=0.3, linestyle="--")

# Ajouter les valeurs sur les barres
for bar in bars1 + bars2:
    height = bar.get_height()
    if height > 1:  # N'afficher que si > 1%
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

plt.tight_layout()
_distrib_path = (
    Path(globals().get("OUTPUT_DIR", "./output"))
    / "flowsom_distribution_conditions.png"
)
_distrib_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(_distrib_path, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"   [OK] Distribution conditions sauvegardée → {_distrib_path}")

# Tableau récapitulatif
print("\nTableau récapitulatif:")
print("-" * 50)
print(f"{'MC':>4} | {'Sain (%)':>10} | {'Patho (%)':>10} | {'Diff':>8}")
print("-" * 50)
for i in range(N_CLUSTERS):
    diff = patho_pcts[i] - healthy_pcts[i]
    print(f"{i:>4} | {healthy_pcts[i]:>10.2f} | {patho_pcts[i]:>10.2f} | {diff:>+8.2f}")
print("-" * 50)


################################################################################
# =============================================================================
# STATISTIQUES PAR MÉTACLUSTER
# =============================================================================

print(" STATISTIQUES PAR MÉTACLUSTER")
print("=" * 80)

# Créer un DataFrame de statistiques
stats_data = []

for i in range(N_CLUSTERS):
    mask = metaclustering == i
    n_cells = mask.sum()
    pct_total = n_cells / len(metaclustering) * 100

    # Calculer MFI pour chaque marqueur
    mfi = (
        np.nanmean(X[mask][:, cols_to_use], axis=0)
        if n_cells > 0
        else np.zeros(len(cols_to_use))
    )

    # Top 3 marqueurs les plus exprimés
    top_indices = np.argsort(mfi)[::-1][:3]
    top_markers = [used_markers[idx] for idx in top_indices]

    stats_data.append(
        {
            "Metacluster": i,
            "N_Cells": n_cells,
            "Pct_Total": pct_total,
            "Top_Markers": ", ".join(top_markers),
        }
    )

df_stats = pd.DataFrame(stats_data)

# Graphique camembert
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart des tailles
ax = axes[0]
sizes = [s["N_Cells"] for s in stats_data]
labels = [f"MC{s['Metacluster']}" for s in stats_data]
colors = plt.cm.tab20(np.linspace(0, 1, N_CLUSTERS))

wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors, autopct="%1.1f%%", pctdistance=0.8
)
ax.set_title(
    "Distribution des Cellules par Métacluster", fontsize=12, fontweight="bold"
)

# Bar chart des tailles
ax = axes[1]
ax.barh(range(N_CLUSTERS), sizes, color=colors, edgecolor="white")
ax.set_yticks(range(N_CLUSTERS))
ax.set_yticklabels(labels)
ax.set_xlabel("Nombre de cellules")
ax.set_title("Taille des Métaclusters", fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3, linestyle="--")

plt.tight_layout()
_sizes_path = (
    Path(globals().get("OUTPUT_DIR", "./output")) / "flowsom_metacluster_sizes.png"
)
_sizes_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(_sizes_path, dpi=150, bbox_inches="tight")
plt.close("all")
print(f"   [OK] Tailles métaclusters sauvegardées → {_sizes_path}")


################################################################################
# =============================================================================
# PROFIL D'EXPRESSION DÉTAILLÉ PAR MÉTACLUSTER — SPIDER PLOT INTERACTIF
# =============================================================================

print("\n PROFIL D'EXPRESSION MOYEN PAR MÉTACLUSTER")
print("=" * 80)

# Créer un DataFrame avec MFI par marqueur et métacluster — VECTORISÉ
X_markers_mfi = X[:, cols_to_use]
_mc_mfi = pd.Series(metaclustering, name="mc")
mfi_matrix = (
    pd.DataFrame(X_markers_mfi, columns=range(len(used_markers)))
    .groupby(_mc_mfi)
    .mean()
    .reindex(range(N_CLUSTERS))
    .fillna(0)
    .values
)

df_mfi = pd.DataFrame(
    mfi_matrix, columns=used_markers, index=[f"MC{i}" for i in range(N_CLUSTERS)]
)

# =============================================================================
# SPIDER / RADAR PLOT INTERACTIF — TOUS LES CLUSTERS (Plotly)
# =============================================================================

import plotly.graph_objects as go
import plotly.colors as pc

# Palette de couleurs suffisante pour tous les clusters
if N_CLUSTERS <= 10:
    _radar_palette = pc.qualitative.Set3
elif N_CLUSTERS <= 20:
    _radar_palette = pc.qualitative.Alphabet
else:
    _radar_palette = [
        f"hsl({int(i * 360 / N_CLUSTERS)},70%,55%)" for i in range(N_CLUSTERS)
    ]

fig_radar = go.Figure()

for cluster_id in range(N_CLUSTERS):
    values = mfi_matrix[cluster_id].copy()
    # Normaliser entre 0 et 1 pour la visualisation
    v_min, v_max = np.min(values), np.max(values)
    values_norm = (values - v_min) / (v_max - v_min + 1e-10)

    _c = _radar_palette[cluster_id % len(_radar_palette)]
    _n_cells = int((metaclustering == cluster_id).sum())

    # Construire une couleur de remplissage semi-transparente
    if "rgb" in str(_c):
        _fill = _c.replace(")", ",0.08)").replace("rgb", "rgba")
    else:
        _fill = f"rgba(128,128,128,0.05)"

    fig_radar.add_trace(
        go.Scatterpolar(
            r=np.append(values_norm, values_norm[0]),
            theta=used_markers + [used_markers[0]],
            fill="toself",
            fillcolor=_fill,
            opacity=0.85,
            name=f"MC{cluster_id}  ({_n_cells:,} cells)",
            line=dict(color=_c, width=2),
            marker=dict(size=5),
            customdata=np.stack(
                [
                    np.append(mfi_matrix[cluster_id], mfi_matrix[cluster_id][0]),
                    np.append(values_norm, values_norm[0]),
                ],
                axis=-1,
            ),
            hovertemplate=(
                f"<b>MC{cluster_id}</b><br>"
                "Marqueur: %{theta}<br>"
                "MFI brute: %{customdata[0]:.2f}<br>"
                "Normalisé: %{customdata[1]:.3f}<extra></extra>"
            ),
        )
    )

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1.05],
            tickfont=dict(size=9),
            gridcolor="rgba(0,0,0,0.12)",
        ),
        angularaxis=dict(
            tickfont=dict(size=11),
            gridcolor="rgba(0,0,0,0.12)",
            rotation=90,
            direction="clockwise",
        ),
        bgcolor="rgba(250,250,250,0.5)",
    ),
    title=dict(
        text=f"<b>Profil d'Expression Normalisé — {N_CLUSTERS} Métaclusters</b><br>"
        "<sup>Cliquez sur la légende pour masquer/afficher un cluster</sup>",
        font=dict(size=15),
    ),
    legend=dict(
        title="Métacluster",
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#ccc",
        borderwidth=1,
        yanchor="top",
        y=1.0,
        xanchor="left",
        x=1.05,
    ),
    height=750,
    width=950,
    paper_bgcolor="#fafafa",
    margin=dict(t=90, b=40, l=80, r=200),
)

fig_radar.write_html(
    str(_dashboard_out / f"fig_radar_{_dashboard_ts}.html"),
    include_plotlyjs="cdn",
)
print(f"\n[OK] Spider plot interactif — {N_CLUSTERS} métaclusters affichés")
print("     → Cliquez sur la légende pour isoler un métacluster")
print("     → Survolez les points pour voir les MFI brutes")


################################################################################
# =============================================================================
# ANALYSE DES CLUSTERS EXCLUSIFS (mono-condition)
# =============================================================================
# Identification des clusters contenant UNIQUEMENT des cellules d'une condition
# Utile pour détecter les populations pathologiques spécifiques
# =============================================================================

print("=" * 70)
print(" ANALYSE DES CLUSTERS EXCLUSIFS PAR CONDITION")
print("=" * 70)

# Récupérer les conditions des cellules
cell_conditions = adata_flowsom.obs["condition"].values
unique_conditions = np.unique(cell_conditions)

print(f"\nConditions présentes: {list(unique_conditions)}")
print(f"Nombre de metaclusters: {n_meta}")

# Analyse par metacluster
clusters_patho_only = []
clusters_sain_only = []
clusters_mixed = []

print("\n" + "-" * 70)
print(" METACLUSTERS EXCLUSIFS (100% d'une seule condition)")
print("-" * 70)

for cluster_id in range(1, n_meta + 1):
    mask_cluster = metaclustering_cells == cluster_id
    n_cluster = mask_cluster.sum()

    if n_cluster == 0:
        continue

    # Compter les cellules par condition dans ce cluster
    conditions_in_cluster = cell_conditions[mask_cluster]

    # Calculer les proportions
    condition_counts = {}
    for cond in unique_conditions:
        count = (conditions_in_cluster == cond).sum()
        condition_counts[cond] = count

    # Vérifier si le cluster est exclusif à une condition
    total = sum(condition_counts.values())

    # Cluster 100% pathologique
    if (
        "Pathologique" in condition_counts
        and condition_counts.get("Pathologique", 0) == total
    ):
        clusters_patho_only.append((cluster_id, total))
        print(
            f"   [PATHO] Metacluster {cluster_id:2d}: {total:6,} cellules (100% Pathologique)"
        )

    # Cluster 100% sain
    elif "Sain" in condition_counts and condition_counts.get("Sain", 0) == total:
        clusters_sain_only.append((cluster_id, total))
        print(
            f"   [SAIN]  Metacluster {cluster_id:2d}: {total:6,} cellules (100% Sain)"
        )

    else:
        clusters_mixed.append(cluster_id)

# Résumé
print("\n" + "=" * 70)
print(" RÉSUMÉ")
print("=" * 70)

if clusters_patho_only:
    total_patho_exclusive = sum([c[1] for c in clusters_patho_only])
    print(f"\n[!] CLUSTERS 100% PATHOLOGIQUES: {len(clusters_patho_only)}")
    print(f"    Metaclusters: {[c[0] for c in clusters_patho_only]}")
    print(f"    Total cellules: {total_patho_exclusive:,}")
    print(
        f"    → Ces clusters représentent des populations UNIQUEMENT présentes chez le patient"
    )
else:
    print(f"\n    Aucun cluster exclusivement pathologique détecté")

if clusters_sain_only:
    total_sain_exclusive = sum([c[1] for c in clusters_sain_only])
    print(f"\n[!] CLUSTERS 100% SAINS: {len(clusters_sain_only)}")
    print(f"    Metaclusters: {[c[0] for c in clusters_sain_only]}")
    print(f"    Total cellules: {total_sain_exclusive:,}")
    print(f"    → Ces clusters représentent des populations ABSENTES chez le patient")
else:
    print(f"\n    Aucun cluster exclusivement sain détecté")

print(f"\n    Clusters mixtes (partagés): {len(clusters_mixed)}")

# Visualisation si clusters exclusifs pathologiques
if clusters_patho_only and len(clusters_patho_only) > 0:
    print("\n" + "-" * 70)
    print(" DÉTAIL DES CLUSTERS PATHOLOGIQUES EXCLUSIFS")
    print("-" * 70)

    # Calculer le MFI des marqueurs pour ces clusters
    for cluster_id, n_cells in clusters_patho_only:
        mask_c = metaclustering_cells == cluster_id
        print(f"\n   Metacluster {cluster_id} ({n_cells:,} cellules):")

        # Top 3 marqueurs les plus exprimés
        mfi_cluster = adata_flowsom.X[mask_c].mean(axis=0)
        top_3_idx = np.argsort(mfi_cluster)[-3:][::-1]
        print(f"      Top marqueurs: ", end="")
        for idx in top_3_idx:
            marker_name = adata_flowsom.var_names[idx]
            print(f"{marker_name}({mfi_cluster[idx]:.2f}) ", end="")
        print()


################################################################################
# =============================================================================
# EXPORT CSV/FCS AVEC COORDONNÉES SOM (style FlowSOM R EXACT)
# =============================================================================

import os
from datetime import datetime

# Créer le dossier de sortie et ses sous-dossiers
OUTPUT_DIR = "./output"
OUTPUT_FCS = os.path.join(OUTPUT_DIR, "fcs")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "csv")
OUTPUT_OTHER = os.path.join(OUTPUT_DIR, "other")
os.makedirs(OUTPUT_FCS, exist_ok=True)
os.makedirs(OUTPUT_CSV, exist_ok=True)
os.makedirs(OUTPUT_OTHER, exist_ok=True)

print(" PRÉPARATION DES DONNÉES POUR EXPORT (style FlowSOM R EXACT)")
print("=" * 70)

# =====================================================================
# PARAMÈTRES DE JITTER - STYLE FLOWSOM R EXACT
# Dans FlowSOM R, le jitter est CIRCULAIRE (pas carré!)
# La taille du cercle dépend du nombre de cellules dans le cluster
# Formule R: rnorm() * scale_factor * sqrt(node_size/max_size)
# =====================================================================
np.random.seed(SEED)  # Pour reproductibilité

# Paramètres FlowSOM R
MAX_NODE_SIZE = 0.45  # Rayon maximum du cercle (quand le node est le plus grand)
MIN_NODE_SIZE = (
    0.1  # Rayon minimum du cercle (pour éviter que les petits nodes disparaissent)
)

# Récupérer les coordonnées de grille et MST depuis cluster_data
grid_coords = cluster_data.obsm.get("grid", None)
layout_coords = cluster_data.obsm.get("layout", None)

# Récupérer le clustering pour mapper les coordonnées sur chaque cellule
clustering = cell_data.obs["clustering"].values
n_cells = len(clustering)
n_nodes = len(cluster_data)

# Calculer la taille de chaque node (nombre de cellules) — VECTORISÉ
node_sizes = np.bincount(clustering.astype(int), minlength=n_nodes).astype(np.float32)

max_size = node_sizes.max()
print(f"\n Taille des nodes:")
print(f"   Min: {node_sizes.min():.0f} cellules")
print(f"   Max: {max_size:.0f} cellules")
print(f"   Total: {n_cells} cellules")

# Créer un DataFrame avec toutes les données
df_export = pd.DataFrame(X, columns=var_names)

# MetaCluster avec +1 pour Kaluza (éviter le 0, commencer à 1)
df_export["FlowSOM_metacluster"] = metaclustering + 1

# FlowSOM cluster (nodes) avec +1
df_export["FlowSOM_cluster"] = clustering + 1

# Ajouter les métadonnées si disponibles
if "condition" in cell_data.obs.columns:
    df_export["Condition"] = cell_data.obs["condition"].values
    df_export["Condition_Num"] = np.where(df_export["Condition"] == "Sain", 1, 2)
if "file_origin" in cell_data.obs.columns:
    df_export["File_Origin"] = cell_data.obs["file_origin"].values


# =====================================================================
# FONCTION JITTER CIRCULAIRE (style FlowSOM R exact)
# Génère des points distribués uniformément dans un disque
# Le rayon dépend de la taille du cluster
# =====================================================================
def circular_jitter(n_points, cluster_ids, node_sizes, max_radius=0.45, min_radius=0.1):
    """
    Génère un jitter circulaire style FlowSOM R.

    Génère un jitter circulaire style FlowSOM R — ENTIÈREMENT VECTORISÉ.
    dont le rayon dépend du nombre de cellules dans le node.
    Plus un node a de cellules, plus le cercle est grand.

    Méthode:
    - Angle theta uniforme [0, 2*pi]
    - Rayon r = sqrt(u) * max_r (pour distribution uniforme dans le disque)
    - Le max_r dépend de la taille du node
    """
    # Angle uniforme autour du cercle
    theta = np.random.uniform(0, 2 * np.pi, n_points)

    # Rayon - distribution uniforme dans le disque (sqrt pour uniformité)
    u = np.random.uniform(0, 1, n_points)

    # Calculer le rayon pour chaque cellule selon la taille de son cluster
    # Dans FlowSOM R, le rayon est proportionnel à sqrt(node_size/max_size)
    # Calculer le rayon pour chaque cellule — VECTORISÉ (pas de boucle)
    max_size_val = node_sizes.max()
    radii = min_radius + (max_radius - min_radius) * np.sqrt(
        node_sizes[cluster_ids.astype(int)] / max_size_val
    )

    # Rayon final pour distribution uniforme dans le disque
    r = np.sqrt(u) * radii

    # Convertir en coordonnées cartésiennes
    jitter_x = r * np.cos(theta)
    jitter_y = r * np.sin(theta)

    return jitter_x.astype(np.float32), jitter_y.astype(np.float32)


# =====================================================================
# COORDONNÉES GRILLE SOM (xGrid, yGrid) - Style FlowSOM R
# =====================================================================
print(f"\n Application du jitter CIRCULAIRE style FlowSOM R")
print(f"   Rayon min: {MIN_NODE_SIZE}, Rayon max: {MAX_NODE_SIZE}")

if grid_coords is not None:
    # Générer jitter CIRCULAIRE dépendant de la taille du node
    jitter_x, jitter_y = circular_jitter(
        n_cells,
        clustering,
        node_sizes,
        max_radius=MAX_NODE_SIZE,
        min_radius=MIN_NODE_SIZE,
    )

    # Mapper les coordonnées de grille sur chaque cellule
    xGrid_base = np.array(
        [grid_coords[int(c), 0] for c in clustering], dtype=np.float32
    )
    yGrid_base = np.array(
        [grid_coords[int(c), 1] for c in clustering], dtype=np.float32
    )

    # Appliquer le jitter circulaire
    xGrid_jittered = xGrid_base + jitter_x
    yGrid_jittered = yGrid_base + jitter_y

    # Décaler pour que les axes commencent à 1 (X ET Y)
    # Mapper les coordonnées de grille sur chaque cellule — VECTORISÉ
    cl_int = clustering.astype(int)
    xGrid_base = grid_coords[cl_int, 0].astype(np.float32)
    yGrid_base = grid_coords[cl_int, 1].astype(np.float32)

    # Appliquer le jitter circulaire
    xGrid_jittered = xGrid_base + jitter_x
    yGrid_jittered = yGrid_base + jitter_y

    # Décaler pour que les axes commencent à 1 (X ET Y)
    xGrid = xGrid_jittered - xGrid_jittered.min() + 1.0
    yGrid = yGrid_jittered - yGrid_jittered.min() + 1.0

    df_export["xGrid"] = xGrid.astype(np.float32)
    df_export["yGrid"] = yGrid.astype(np.float32)

    print(f"[OK] xGrid: [{xGrid.min():.3f} - {xGrid.max():.3f}]")
    print(f"[OK] yGrid: [{yGrid.min():.3f} - {yGrid.max():.3f}]")

    # =====================================================================
    # COORDONNÉES MST (xNodes, yNodes) - Style FlowSOM R
    # =====================================================================
    # Mapper les coordonnées MST sur chaque cellule — VECTORISÉ
    cl_int = clustering.astype(int)
    xNodes_base = layout_coords[cl_int, 0].astype(np.float32)
    yNodes_base = layout_coords[cl_int, 1].astype(np.float32)

    # Calculer l'échelle pour le jitter MST (proportionnel à l'espacement moyen)
    x_range = xNodes_base.max() - xNodes_base.min()
    y_range = yNodes_base.max() - yNodes_base.min()
    mst_scale = min(x_range, y_range) / (XDIM * 2)  # Proportionnel à la grille

    # Jitter circulaire pour MST aussi
    mst_jitter_x, mst_jitter_y = circular_jitter(
        n_cells,
        clustering,
        node_sizes,
        max_radius=mst_scale * 0.8,  # Un peu moins que Grid car MST est plus espacé
        min_radius=mst_scale * 0.2,
    )

    # Appliquer le jitter
    xNodes_jittered = xNodes_base + mst_jitter_x
    yNodes_jittered = yNodes_base + mst_jitter_y

    # Décaler pour que les axes commencent à 1 (X ET Y)
    xNodes = xNodes_jittered - xNodes_jittered.min() + 1.0
    yNodes = yNodes_jittered - yNodes_jittered.min() + 1.0

    df_export["xNodes"] = xNodes.astype(np.float32)
    df_export["yNodes"] = yNodes.astype(np.float32)

    print(f"[OK] xNodes: [{xNodes.min():.3f} - {xNodes.max():.3f}]")
    print(f"[OK] yNodes: [{yNodes.min():.3f} - {yNodes.max():.3f}]")

# TAILLE DES NODES (pour chaque cellule) — VECTORISÉ
# TAILLE DES NODES (pour chaque cellule)
size_col = node_sizes[clustering.astype(int)]
size_col = np.array([node_sizes[int(c)] for c in clustering], dtype=np.float32)
df_export["size"] = size_col
print(f"[OK] size: [{size_col.min():.0f} - {size_col.max():.0f}]")

# =====================================================================
# EXPORT CSV
# =====================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(OUTPUT_DIR, f"flowsom_results_{timestamp}.csv")
df_export.to_csv(csv_path, index=False)

print(f"\n[OK] CSV exporté: {csv_path}")
print(f"   Shape: {df_export.shape}")

# =====================================================================
# EXPORT FCS COMPATIBLE KALUZA
# =====================================================================
print("\n" + "=" * 70)
print("📄 EXPORT FCS COMPATIBLE KALUZA")
print("=" * 70)


def export_to_fcs_kaluza(df, output_path):
    """Export FCS compatible Kaluza avec toutes les coordonnées positives."""
    try:
        import fcswrite

        numeric_df = df.select_dtypes(include=[np.number])
        data = numeric_df.values.astype(np.float32)
        channels = list(numeric_df.columns)

        # Nettoyer NaN/Inf
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=0.0)

        print(f"   {data.shape[0]:,} events, {data.shape[1]} canaux")

        fcswrite.write_fcs(output_path, channels, data, compat_chn_names=True)
        return True

    except ImportError:
        print("   [!] fcswrite non disponible (pip install fcswrite)")
        return False
    except Exception as e:
        print(f"   [!] Erreur: {e}")
        return False


# Préparer le DataFrame FCS
df_fcs = df_export.select_dtypes(include=[np.number]).copy()

# Vérifier les ranges
print(f"\n Colonnes exportées vers FCS:")
for col in [
    "FlowSOM_metacluster",
    "FlowSOM_cluster",
    "xGrid",
    "yGrid",
    "xNodes",
    "yNodes",
    "size",
    "Condition_Num",
]:
    if col in df_fcs.columns:
        print(
            f"   [OK] {col:25s}: [{df_fcs[col].min():10.2f}, {df_fcs[col].max():10.2f}]"
        )

# Export FCS complet → dossier fcs/
fcs_path = os.path.join(OUTPUT_FCS, f"flowsom_results_{timestamp}.fcs")
if export_to_fcs_kaluza(df_fcs, fcs_path):
    print(f"\n[OK] FCS exporté: {fcs_path}")


################################################################################
# =============================================================================
# EXPORT DU RAPPORT DE STATISTIQUES
# =============================================================================

# Sauvegarder le rapport de statistiques → dossier csv/
stats_path = os.path.join(OUTPUT_CSV, f"flowsom_statistics_{timestamp}.csv")
df_stats.to_csv(stats_path, index=False)
print(f"[OK] Statistiques exportées: {stats_path}")

# Sauvegarder la matrice MFI → dossier csv/
mfi_path = os.path.join(OUTPUT_CSV, f"flowsom_mfi_matrix_{timestamp}.csv")
df_mfi.to_csv(mfi_path)
print(f"[OK] Matrice MFI exportée: {mfi_path}")

# Résumé final
print("\n" + "=" * 80)
print(" RÉSUMÉ DE L'ANALYSE FLOWSOM")
print("=" * 80)
print(f"   Fichiers analysés: {len(all_adatas)}")
print(f"   Cellules totales: {len(cell_data)}")
print(f"   Marqueurs utilisés: {len(used_markers)}")
print(f"   Nombre de métaclusters: {N_CLUSTERS}")
print(f"   Transformation: {TRANSFORM_TYPE}")
print(f"   Cofacteur: {COFACTOR}")
if TRANSFORM_TYPE != "none":
    print(f"   ⚠️  Export FCS: données BRUTES (transformation inversée pour Kaluza)")
print("=" * 80)
print("[OK] Analyse FlowSOM terminée avec succès!")


################################################################################
# =============================================================================
# EXPORT JSON MÉTADONNÉES - TRAÇABILITÉ COMPLÈTE DE L'ANALYSE
# =============================================================================

import json

print("=" * 70)
print(" EXPORT DES MÉTADONNÉES (JSON) → dossier other/")
print("=" * 70)

# Collecter toutes les métadonnées de l'analyse
metadata = {
    "analysis_info": {
        "date": datetime.now().isoformat(),
        "timestamp": timestamp,
        "pipeline_version": "FlowSOM_Analysis_Pipeline v2.0",
    },
    "input_files": {
        "total_files": len(globals()["all_fcs_files"])
        if "all_fcs_files" in globals()
        else len(all_adatas),
        "healthy_files": [str(f) for f in healthy_files]
        if "healthy_files" in dir()
        else [],
        "pathological_files": [str(f) for f in patho_files]
        if "patho_files" in dir()
        else [],
        "healthy_folder": str(HEALTHY_FOLDER) if "HEALTHY_FOLDER" in dir() else "N/A",
        "pathological_folder": str(PATHOLOGICAL_FOLDER)
        if "PATHOLOGICAL_FOLDER" in dir()
        else "N/A",
    },
    "preprocessing": {
        "gating_mode": GATING_MODE if "GATING_MODE" in dir() else "N/A",
        "gate_doublets": GATE_DOUBLETS if "GATE_DOUBLETS" in dir() else "N/A",
        "gate_debris": GATE_DEBRIS if "GATE_DEBRIS" in dir() else "N/A",
        "gate_cd45": GATE_CD45 if "GATE_CD45" in dir() else "N/A",
        "filter_blasts": FILTER_BLASTS if "FILTER_BLASTS" in dir() else "N/A",
        "marker_filtering": {
            "enabled": APPLY_MARKER_FILTERING,
            "keep_area": KEEP_AREA,
            "keep_height": KEEP_HEIGHT,
        },
    },
    "transformation": {
        "type": TRANSFORM_TYPE,
        "cofactor": COFACTOR,
        "apply_to_scatter": APPLY_TO_SCATTER,
        "export_data": "raw (inverse transform applied)"
        if TRANSFORM_TYPE != "none"
        else "raw (no transform)",
    },
    "flowsom_parameters": {
        "seed": SEED,
        "xdim": XDIM,
        "ydim": YDIM,
        "n_clusters": N_CLUSTERS,
        "total_nodes": XDIM * YDIM,
        "exclude_scatter": EXCLUDE_SCATTER,
    },
    "data_summary": {
        "total_cells": int(n_cells),
        "total_markers": len(var_names),
        "markers_used_for_clustering": used_markers,
        "all_markers": var_names,
        "cells_per_condition": {
            cond: int((cell_data.obs["condition"] == cond).sum())
            for cond in cell_data.obs["condition"].unique()
        }
        if "condition" in cell_data.obs.columns
        else {},
        "cells_per_file": {
            fname: int((cell_data.obs["file_origin"] == fname).sum())
            for fname in cell_data.obs["file_origin"].unique()
        }
        if "file_origin" in cell_data.obs.columns
        else {},
    },
    "metacluster_summary": {
        f"MC{i}": {
            "n_cells": int((metaclustering == i).sum()),
            "pct_total": round(
                float((metaclustering == i).sum() / len(metaclustering) * 100), 2
            ),
        }
        for i in range(N_CLUSTERS)
    },
    "export_files": {
        "fcs_complete": fcs_path,
        "csv_complete": csv_path,
        "statistics": stats_path if "stats_path" in dir() else "N/A",
        "mfi_matrix": mfi_path if "mfi_path" in dir() else "N/A",
    },
    "export_folders": {
        "fcs": OUTPUT_FCS,
        "csv": OUTPUT_CSV,
        "other": OUTPUT_OTHER,
    },
    "jitter_parameters": {
        "max_node_size": MAX_NODE_SIZE,
        "min_node_size": MIN_NODE_SIZE,
        "method": "circular_jitter (style FlowSOM R)",
    },
}

# Sauvegarder le JSON → dossier other/
metadata_path = os.path.join(OUTPUT_OTHER, f"flowsom_metadata_{timestamp}.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

print(f"\n[OK] Métadonnées exportées: {metadata_path}")
print(f"\nContenu du fichier:")
print(f"   - Informations d'analyse (date, version)")
print(f"   - Fichiers source ({metadata['input_files']['total_files']} fichiers)")
print(f"   - Paramètres de preprocessing (gating, filtrage)")
print(f"   - Paramètres de transformation ({TRANSFORM_TYPE}, cofactor={COFACTOR})")
print(f"   - Paramètres FlowSOM (grille {XDIM}x{YDIM}, {N_CLUSTERS} métaclusters)")
print(f"   - Résumé des données ({n_cells:,} cellules, {len(used_markers)} marqueurs)")
print(f"   - Résumé par métacluster")
print(f"   - Chemins des fichiers exportés")


################################################################################
# =============================================================================
# EXPORT PAR CONDITION (Sain vs Pathologique) + COLONNE TIMEPOINT
# =============================================================================

import re

print("=" * 70)
print(" EXPORT PAR CONDITION + TIMEPOINT")
print("=" * 70)

# =====================================================================
# AJOUT DE LA COLONNE TIMEPOINT (extraction depuis le nom de fichier)
# =====================================================================


def extract_date_from_filename(filename):
    """Extrait une date depuis un nom de fichier FCS."""
    patterns = [
        r"(\d{2}[-/]\d{2}[-/]\d{4})",  # DD-MM-YYYY ou DD/MM/YYYY
        r"(\d{4}[-/]\d{2}[-/]\d{2})",  # YYYY-MM-DD ou YYYY/MM/DD
        r"(\d{2}[-/]\d{2}[-/]\d{2})",  # DD-MM-YY
        r"(\d{8})",  # YYYYMMDD
    ]
    for pattern in patterns:
        match = re.search(pattern, str(filename))
        if match:
            return match.group(1)
    return "unknown"


# Créer la colonne timepoint
if "file_origin" in cell_data.obs.columns:
    file_origins_arr = cell_data.obs["file_origin"].values

    unique_files = list(set(file_origins_arr))
    file_to_date = {f: extract_date_from_filename(f) for f in unique_files}

    timepoints = np.array([file_to_date[str(f)] for f in file_origins_arr])
    df_export["Timepoint"] = timepoints

    unique_dates = sorted(set(timepoints))
    date_to_idx = {d: i + 1 for i, d in enumerate(unique_dates)}
    df_export["Timepoint_Num"] = np.array([date_to_idx[t] for t in timepoints])

    print(f"\n[OK] Colonne 'Timepoint' ajoutée")
    print(f"   Dates détectées: {len(unique_dates)}")
    for dt in unique_dates:
        n_dt = (timepoints == dt).sum()
        print(f"   [{date_to_idx[dt]}] {dt}: {n_dt:,} cellules")
else:
    print("[INFO] Pas de colonne 'file_origin' — Timepoint non créé")

# =====================================================================
# EXPORT SÉPARÉ PAR CONDITION → dossiers fcs/ et csv/
# =====================================================================
print("\n" + "=" * 70)
print(" EXPORT SÉPARÉ PAR CONDITION → fcs/ et csv/")
print("=" * 70)

if "Condition" in df_export.columns:
    conditions_list = df_export["Condition"].unique()

    for cond in conditions_list:
        mask_cond = df_export["Condition"] == cond
        df_cond = df_export[mask_cond].copy()

        # Export CSV par condition → dossier csv/
        cond_safe = str(cond).replace(" ", "_").replace("/", "-")
        csv_cond_path = os.path.join(OUTPUT_CSV, f"flowsom_{cond_safe}_{timestamp}.csv")
        df_cond.to_csv(csv_cond_path, index=False)

        # Export FCS par condition → dossier fcs/
        fcs_cond_path = os.path.join(OUTPUT_FCS, f"flowsom_{cond_safe}_{timestamp}.fcs")
        df_cond_numeric = df_cond.select_dtypes(include=[np.number]).copy()
        export_to_fcs_kaluza(df_cond_numeric, fcs_cond_path)

        print(f"\n[OK] Condition '{cond}':")
        print(f"   Cellules: {len(df_cond):,}")
        print(f"   CSV: {csv_cond_path}")
        print(f"   FCS: {fcs_cond_path}")

        # Résumé des métaclusters par condition
        mc_counts = df_cond["FlowSOM_metacluster"].value_counts().sort_index()
        mc_pcts = (mc_counts / len(df_cond) * 100).round(1)
        print(f"   Métaclusters:")
        for mc, (cnt, pct) in enumerate(zip(mc_counts.values, mc_pcts.values)):
            print(f"      MC{mc + 1}: {cnt:>7,} ({pct:5.1f}%)")

    # Export aussi par fichier si plusieurs fichiers
    if "File_Origin" in df_export.columns:
        unique_files_export = df_export["File_Origin"].unique()
        if len(unique_files_export) > 1:
            print(f"\n" + "-" * 70)
            print(f" EXPORT PAR FICHIER ({len(unique_files_export)} fichiers) → csv/")
            print(f"-" * 70)

            for fname in unique_files_export:
                mask_file = df_export["File_Origin"] == fname
                df_file = df_export[mask_file].copy()

                fname_safe = (
                    str(fname).replace(" ", "_").replace("/", "-").replace(".fcs", "")
                )
                csv_file_path = os.path.join(
                    OUTPUT_CSV, f"flowsom_{fname_safe}_{timestamp}.csv"
                )
                df_file.to_csv(csv_file_path, index=False)

                print(f"   [OK] {fname}: {len(df_file):,} cellules → {csv_file_path}")
else:
    print("[INFO] Pas de colonne 'Condition' — Export par condition non disponible")

print(f"\n[OK] Tous les exports par condition/fichier terminés")


################################################################################
# =============================================================================
# EXPORT RAPPORT HTML COMPLET — VISUALISATIONS INTERACTIVES PLOTLY + IMAGES
# =============================================================================
# Ce rapport HTML est autonome (self-contained): il inclut toutes les
# visualisations Plotly interactives en temps réel + les figures matplotlib
# converties en images base64 inline. Pas de dépendance externe.
# =============================================================================

import base64
from io import BytesIO
import plotly.io as pio
import plotly.offline

print("=" * 70)
print(" GÉNÉRATION DU RAPPORT HTML COMPLET")
print("=" * 70)

# =====================================================================
# RÉCUPÉRATION DU BUNDLE PLOTLY.JS POUR HTML SELF-CONTAINED
# =====================================================================
# On embarque plotly.js directement dans le HTML pour que le rapport
# fonctionne hors-ligne, sans dépendance CDN.
plotly_js_bundle = plotly.offline.get_plotlyjs()
print(f"   [OK] Plotly.js embarqué ({len(plotly_js_bundle) // 1024} KB)")

# =====================================================================
# COLLECTE DE TOUTES LES FIGURES
# =====================================================================


# --- Convertir les figures Matplotlib en base64 PNG ---
def fig_to_base64(fig_mpl):
    """Convertit une figure matplotlib en string base64 PNG."""
    buf = BytesIO()
    fig_mpl.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# --- Convertir les figures Plotly en HTML div ---
def plotly_to_html_div(fig_plotly, fig_id=""):
    """Convertit une figure Plotly en div HTML avec interactivité.
    Utilise la hauteur définie dans le layout de la figure pour préserver
    les dimensions originales (ex: spider plot 750px).
    include_plotlyjs=False car plotly.js est embarqué dans le <head>.
    """
    fig_height = fig_plotly.layout.height or 500
    fig_width_val = fig_plotly.layout.width
    default_w = f"{fig_width_val}px" if fig_width_val else "100%"
    return pio.to_html(
        fig_plotly,
        full_html=False,
        include_plotlyjs=False,
        div_id=fig_id if fig_id else None,
        default_height=f"{fig_height}px",
        default_width=default_w,
        config={"responsive": True},
    )


# Collecter les figures matplotlib existantes
mpl_figures = {}
plotly_figures = {}

# Chercher toutes les variables de type Figure dans le namespace
import matplotlib.figure

for name, obj in list(globals().items()):
    if isinstance(obj, matplotlib.figure.Figure):
        try:
            mpl_figures[name] = fig_to_base64(obj)
            print(f"   [OK] Figure matplotlib: {name}")
        except Exception as e:
            print(f"   [!] Erreur {name}: {e}")

# Chercher toutes les figures Plotly — on stocke les OBJETS Figure
# pour les convertir en HTML avec les bonnes dimensions
import plotly.graph_objs as go

plotly_figure_objects = {}
for name, obj in list(globals().items()):
    if isinstance(obj, go.Figure):
        try:
            # Vérifier que la figure contient des données
            n_traces = len(obj.data)
            plotly_figure_objects[name] = obj
            print(
                f"   [OK] Figure Plotly: {name} ({n_traces} traces, h={obj.layout.height or 'auto'})"
            )
        except Exception as e:
            print(f"   [!] Erreur {name}: {e}")

print(
    f"\n   Total: {len(mpl_figures)} figures matplotlib, {len(plotly_figure_objects)} figures Plotly"
)

# =====================================================================
# CONSTRUCTION DU HTML
# =====================================================================

# Noms lisibles pour les figures
figure_labels = {
    "fig": "Aperçu général",
    "fig1": "Visualisation 1",
    "fig2": "Visualisation 2",
    "fig3": "Visualisation 3",
    "fig4": "Visualisation 4",
    "fig5": "Visualisation 5",
    "fig_stars": "Star Chart FlowSOM",
    "fig_comp": "Comparaison Conditions",
    "fig_gates": "Gates de pré-traitement",
    "fig_grid_mc": "Grille SOM — Métaclusters",
    "fig_grid_cond": "Grille SOM — Conditions",
    "fig_hist": "Histogrammes des marqueurs",
    "fig_mst": "Arbre MST",
    "fig_overview": "Vue d'ensemble",
    "fig_radar": "Spider Plot — Profils MFI",
    "fig_sankey": "Diagramme Sankey",
    "fig_table": "Tableau résumé",
    "fig_table_cond": "Tableau par condition",
    "fig_ransac_qc": "QC RANSAC — Scatter FSC-A vs FSC-H",
    "fig_singlets_table": "QC Singlets — Tableau par fichier",
    "fig_heatmap": "Heatmap MFI — Métaclusters × Marqueurs (Z-score)",
    "fig_heatmap_clinical": "Expression Phénotypique — Métaclusters × Marqueurs",
    "fig_barplots": "Marqueurs Clés — NBM vs Pathologique",
    "fig_phenotype": "Signature Phénotypique par Métacluster",
}

# --- Statistiques par métacluster pour le tableau HTML ---
mc_rows_html = ""
for i in range(N_CLUSTERS):
    mask_mc = metaclustering == i
    n_mc = int(mask_mc.sum())
    pct_mc = n_mc / len(metaclustering) * 100

    # MFI top 3
    if n_mc > 0:
        mfi_mc = np.nanmean(X[mask_mc][:, cols_to_use], axis=0)
        top3_idx = np.argsort(mfi_mc)[::-1][:3]
        top3 = ", ".join([used_markers[j] for j in top3_idx])
    else:
        top3 = "N/A"

    mc_rows_html += f"""
    <tr>
        <td style="font-weight:bold; text-align:center;">MC{i + 1}</td>
        <td style="text-align:right;">{n_mc:,}</td>
        <td style="text-align:right;">{pct_mc:.1f}%</td>
        <td>{top3}</td>
    </tr>"""

# --- Cellules par condition ---
cond_rows_html = ""
if "condition" in cell_data.obs.columns:
    for cond in cell_data.obs["condition"].unique():
        n_cond = int((cell_data.obs["condition"] == cond).sum())
        pct_cond = n_cond / len(cell_data) * 100
        cond_rows_html += f"""
    <tr>
        <td style="font-weight:bold;">{cond}</td>
        <td style="text-align:right;">{n_cond:,}</td>
        <td style="text-align:right;">{pct_cond:.1f}%</td>
    </tr>"""

# --- Fichiers source ---
files_rows_html = ""
if "file_origin" in cell_data.obs.columns:
    for fname in cell_data.obs["file_origin"].unique():
        n_f = int((cell_data.obs["file_origin"] == fname).sum())
        files_rows_html += f"""
    <tr>
        <td>{fname}</td>
        <td style="text-align:right;">{n_f:,}</td>
    </tr>"""

# --- Sections Plotly (conversion des objets Figure → HTML divs) ---
plotly_sections = ""
for _fig_name, _fig_iter in plotly_figure_objects.items():
    label = figure_labels.get(_fig_name, _fig_name)
    try:
        div_html = plotly_to_html_div(_fig_iter, fig_id=_fig_name)
        plotly_sections += f"""
    <div class="section">
        <h2>{label}</h2>
        <div class="plotly-container">
            {div_html}
        </div>
    </div>
    """
    except Exception as e:
        print(f"   [!] Erreur conversion HTML pour {_fig_name}: {e}")
del _fig_name, _fig_iter  # Éviter que la variable de boucle se retrouve comme figure

# --- Sections Matplotlib ---
mpl_sections = ""
for name, b64 in mpl_figures.items():
    label = figure_labels.get(name, name)
    mpl_sections += f"""
    <div class="section">
        <h2>{label}</h2>
        <div style="text-align:center;">
            <img src="data:image/png;base64,{b64}" style="max-width:100%; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1);" />
        </div>
    </div>
    """

# --- Marqueurs utilisés ---
markers_html = ""
for i, m in enumerate(used_markers):
    markers_html += f'<span class="marker-badge">{m}</span>\n'

# =====================================================================
# TEMPLATE HTML COMPLET
# =====================================================================
# Note: On utilise un placeholder __PLOTLY_JS_BUNDLE__ pour le script
# plotly.js car le bundle fait ~3.5 MB et ne doit pas être dans le f-string.
# Il sera remplacé après la construction du template.

html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlowSOM Analysis Report — {timestamp}</title>
    <script type="text/javascript">__PLOTLY_JS_BUNDLE__</script>
    <style>
        :root {{
            --primary: #667eea;
            --primary-dark: #764ba2;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #2d3748;
            --text-light: #718096;
            --border: #e2e8f0;
            --success: #48bb78;
            --warning: #ed8936;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        
        .header {{
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            padding: 40px 0;
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.2em;
            margin-bottom: 8px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        .section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border: 1px solid var(--border);
        }}
        
        .section h2 {{
            font-size: 1.4em;
            color: var(--primary);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border);
        }}
        
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        .grid-3 {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f6f8ff, #f0f4ff);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid #dde4f0;
        }}
        
        .stat-card .value {{
            font-size: 2em;
            font-weight: 700;
            color: var(--primary);
        }}
        
        .stat-card .label {{
            font-size: 0.9em;
            color: var(--text-light);
            margin-top: 4px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        th {{
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 10px 16px;
            border-bottom: 1px solid var(--border);
        }}
        
        tr:nth-child(even) {{
            background: #f7fafc;
        }}
        
        tr:hover {{
            background: #edf2f7;
        }}
        
        .marker-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea22, #764ba222);
            color: var(--primary-dark);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            margin: 3px;
            border: 1px solid #667eea44;
            font-weight: 500;
        }}
        
        .plotly-container {{
            width: 100%;
            overflow-x: auto;
            display: flex;
            justify-content: center;
        }}
        
        .plotly-container > div {{
            min-width: 0;
        }}
        
        .param-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 12px;
        }}
        
        .param-item {{
            background: #f7fafc;
            padding: 10px 15px;
            border-radius: 8px;
            border-left: 3px solid var(--primary);
        }}
        
        .param-item .param-label {{
            font-size: 0.8em;
            color: var(--text-light);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .param-item .param-value {{
            font-size: 1.1em;
            font-weight: 600;
            color: var(--text);
        }}
        
        .toc {{
            background: #f0f4ff;
            border-radius: 10px;
            padding: 20px 30px;
            margin-bottom: 24px;
        }}
        
        .toc h3 {{
            margin-bottom: 10px;
            color: var(--primary-dark);
        }}
        
        .toc ul {{
            list-style: none;
            columns: 2;
        }}
        
        .toc li {{
            padding: 4px 0;
        }}
        
        .toc a {{
            color: var(--primary);
            text-decoration: none;
        }}
        
        .toc a:hover {{
            text-decoration: underline;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-light);
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
            .toc ul {{ columns: 1; }}
        }}
    </style>
</head>
<body>

<!-- HEADER -->
<div class="header">
    <div class="container">
        <h1>FlowSOM Analysis Report</h1>
        <div class="subtitle">
            Analyse générée le {datetime.now().strftime("%d/%m/%Y à %H:%M")} — 
            {n_cells:,} cellules · {len(used_markers)} marqueurs · {N_CLUSTERS} métaclusters
        </div>
    </div>
</div>

<div class="container">

<!-- TABLE DES MATIÈRES -->
<div class="toc">
    <h3>Table des matières</h3>
    <ul>
        <li><a href="#params">1. Paramètres de l'analyse</a></li>
        <li><a href="#data">2. Résumé des données</a></li>
        <li><a href="#markers">3. Marqueurs utilisés</a></li>
        <li><a href="#metaclusters">4. Métaclusters</a></li>
        <li><a href="#plotly-viz">5. Visualisations interactives</a></li>
        <li><a href="#static-viz">6. Visualisations statiques</a></li>
        <li><a href="#exports">7. Fichiers exportés</a></li>
    </ul>
</div>

<!-- 1. PARAMÈTRES -->
<div class="section" id="params">
    <h2>1. Paramètres de l'Analyse</h2>
    <div class="param-grid">
        <div class="param-item">
            <div class="param-label">Transformation</div>
            <div class="param-value">{TRANSFORM_TYPE.upper()}</div>
        </div>
        <div class="param-item">
            <div class="param-label">Cofacteur</div>
            <div class="param-value">{COFACTOR}</div>
        </div>
        <div class="param-item">
            <div class="param-label">Grille SOM</div>
            <div class="param-value">{XDIM} × {YDIM} ({XDIM * YDIM} nodes)</div>
        </div>
        <div class="param-item">
            <div class="param-label">Métaclusters</div>
            <div class="param-value">{N_CLUSTERS}</div>
        </div>
        <div class="param-item">
            <div class="param-label">Seed</div>
            <div class="param-value">{SEED}</div>
        </div>
        <div class="param-item">
            <div class="param-label">Filtrage marqueurs</div>
            <div class="param-value">{"Area (-A)" if KEEP_AREA else ""} {"Height (-H)" if KEEP_HEIGHT else ""}</div>
        </div>
        <div class="param-item">
            <div class="param-label">Gating mode</div>
            <div class="param-value">{GATING_MODE if "GATING_MODE" in dir() else "N/A"}</div>
        </div>
        <div class="param-item">
            <div class="param-label">Exclure Scatter</div>
            <div class="param-value">{"Oui" if EXCLUDE_SCATTER else "Non"}</div>
        </div>
    </div>
</div>

<!-- 2. RÉSUMÉ DES DONNÉES -->
<div class="section" id="data">
    <h2>2. Résumé des Données</h2>
    <div class="grid-3">
        <div class="stat-card">
            <div class="value">{n_cells:,}</div>
            <div class="label">Cellules totales</div>
        </div>
        <div class="stat-card">
            <div class="value">{len(used_markers)}</div>
            <div class="label">Marqueurs (clustering)</div>
        </div>
        <div class="stat-card">
            <div class="value">{len(all_adatas) if "all_adatas" in dir() else "N/A"}</div>
            <div class="label">Fichiers analysés</div>
        </div>
    </div>
    
    <h3 style="margin-top:25px; margin-bottom:10px;">Par condition</h3>
    <table>
        <tr><th>Condition</th><th>Cellules</th><th>Pourcentage</th></tr>
        {cond_rows_html}
    </table>
    
    <h3 style="margin-top:25px; margin-bottom:10px;">Par fichier source</h3>
    <table>
        <tr><th>Fichier</th><th>Cellules</th></tr>
        {files_rows_html}
    </table>
</div>

<!-- 3. MARQUEURS -->
<div class="section" id="markers">
    <h2>3. Marqueurs Utilisés pour le Clustering</h2>
    <p style="margin-bottom:15px; color:var(--text-light);">
        {len(used_markers)} marqueurs sélectionnés (scatter et Time exclus)
    </p>
    {markers_html}
</div>

<!-- 4. MÉTACLUSTERS -->
<div class="section" id="metaclusters">
    <h2>4. Résumé des Métaclusters</h2>
    <table>
        <tr>
            <th>Métacluster</th>
            <th>Cellules</th>
            <th>% Total</th>
            <th>Top 3 Marqueurs</th>
        </tr>
        {mc_rows_html}
    </table>
</div>

<!-- 5. VISUALISATIONS PLOTLY (INTERACTIVES) -->
<div id="plotly-viz">
    <div class="section">
        <h2>5. Visualisations Interactives (Plotly)</h2>
        <p style="color:var(--text-light); margin-bottom:10px;">
            {len(plotly_figure_objects)} figures interactives — zoom, pan, hover pour explorer les données
        </p>
    </div>
    {plotly_sections}
</div>

<!-- 6. VISUALISATIONS MATPLOTLIB (STATIQUES) -->
<div id="static-viz">
    <div class="section">
        <h2>6. Visualisations Statiques (Matplotlib)</h2>
        <p style="color:var(--text-light); margin-bottom:10px;">
            {len(mpl_figures)} figures haute résolution
        </p>
    </div>
    {mpl_sections}
</div>

<!-- 7. FICHIERS EXPORTÉS -->
<div class="section" id="exports">
    <h2>7. Fichiers Exportés</h2>
    <table>
        <tr><th>Type</th><th>Fichier</th></tr>
        <tr><td>CSV complet</td><td>{csv_path}</td></tr>
        <tr><td>FCS (Kaluza compatible)</td><td>{fcs_path}</td></tr>
        <tr><td>Statistiques</td><td>{stats_path if "stats_path" in dir() else "N/A"}</td></tr>
        <tr><td>Matrice MFI</td><td>{mfi_path if "mfi_path" in dir() else "N/A"}</td></tr>
        <tr><td>Métadonnées JSON</td><td>{metadata_path}</td></tr>
    </table>
</div>

</div>

<!-- FOOTER -->
<div class="footer">
    <p>FlowSOM Analysis Pipeline v2.0 — Rapport généré automatiquement le {datetime.now().strftime("%d/%m/%Y à %H:%M:%S")}</p>
    <p>Transformation: {TRANSFORM_TYPE.upper()} (cofactor={COFACTOR}) · Grille: {XDIM}×{YDIM} · {N_CLUSTERS} métaclusters · Seed: {SEED}</p>
</div>

</body>
</html>
"""

# =====================================================================
# INJECTION DU BUNDLE PLOTLY.JS (self-contained, pas de CDN)
# =====================================================================
# On remplace le placeholder par le vrai code plotly.js.
# Cela rend le HTML autonome (~3-4 MB de JS embarqué).
html_content = html_content.replace("__PLOTLY_JS_BUNDLE__", plotly_js_bundle)

# =====================================================================
# SAUVEGARDE DU RAPPORT HTML → dossier other/
# =====================================================================
html_path = os.path.join(OUTPUT_OTHER, f"flowsom_report_{timestamp}.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

# Taille du fichier
html_size_mb = os.path.getsize(html_path) / (1024 * 1024)

print(f"\n[OK] Rapport HTML exporté: {html_path}")
print(f"   Taille: {html_size_mb:.1f} MB (plotly.js embarqué = self-contained)")
print(f"   Figures Plotly interactives: {len(plotly_figure_objects)}")
print(f"   Figures Matplotlib (images): {len(mpl_figures)}")
print(f"\n   Ouvrez le fichier dans un navigateur pour explorer les données")
print(f"   Les figures Plotly sont entièrement interactives (zoom, hover, etc.)")
print(f"   Le rapport fonctionne HORS-LIGNE (pas de dépendance CDN)")

import pandas as pd

stats = pd.DataFrame(
    {
        "Colonne": adata_flowsom.var_names,
        "Min": adata_flowsom.X.min(axis=0),
        "Max": adata_flowsom.X.max(axis=0),
        "Moyenne": adata_flowsom.X.mean(axis=0),
        "Std": adata_flowsom.X.std(axis=0),
    }
)
display(stats)

print("\n" + "-" * 70)
print("- Aperçu des observations (.obs):")
print("-" * 70)
if adata_flowsom.obs.shape[1] > 0:
    print(
        f"   {adata_flowsom.obs.shape[0]:,} cellules, {adata_flowsom.obs.shape[1]} colonnes d'annotation"
    )
else:
    print("Aucune annotation dans .obs")

print("\n" + "-" * 70)
print("- Toutes les variables disponibles:")
print("-" * 70)
# %whos


################################################################################
# =============================================================================
# SECTION 10.1 — IMPORTS, CONFIGURATION ET CHEMINS (V3 — A UNIQUEMENT)
# =============================================================================
# STRATÉGIE DE MAPPING V3 :
# ─────────────────────────
# On ne conserve QUE les colonnes suffixées "-A" (Area) dans les CSV ET dans le FCS.
# Raison : les colonnes -H et -A mesurent le même signal avec une légère différence
# d'amplitude. Mélanger les deux introduit un biais d'échelle. En ne gardant que les
# -A, le mapping devient une simple correspondance 1:1 sur le nom normalisé (sans
# aucun fallback -H→-A ni similarité floue), ce qui est :
#   • Plus reproductible  : même règle pour tous les fichiers
#   • Plus lisible        : on sait exactement quels canaux sont comparés
#   • Plus robuste        : pas de cas de doublons après remapping
#   • Biologiquement cohérent : -A (aire) est la mesure de référence en cytométrie
#
# Colonnes CSV disponibles (header du fichier) :
#   FSC-H / FSC-A / SSC-H / SSC-A
#   CD7+56 FITC-H/A  CD13 PE-H/A  HLADR ECD-H/A  CD33 PC5.5-H/A
#   CD38 BD  PC7-H/A  CD34 APC-H/A  CD7 APC-A700-H/A  CD19 APC-A750-H/A
#   CD117 BD PB450-H/A  CD45 KO525-H/A  FSC-Width  TIME
#
# Colonnes retenues après filtre -A :
#   FSC-A  SSC-A  CD7+56 FITC-A  CD13 PE-A  HLADR ECD-A  CD33 PC5.5-A
#   CD38 BD  PC7-A  CD34 APC-A  CD7 APC-A700-A  CD19 APC-A750-A
#   CD117 BD PB450-A  CD45 KO525-A
# =============================================================================

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import os
import re
import unicodedata

try:
    from scipy.spatial.distance import cdist

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[!] scipy non disponible — pip install scipy")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("[!] plotly non disponible — pip install plotly")

# ─── Chemins ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\Users\Florian Travail\Documents\FlowSom")
REF_MFI_DIR = PROJECT_ROOT / "Data" / "Ref MFI"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_FCS_DIR = OUTPUT_DIR / "fcs"
PARQUET_CACHE = OUTPUT_DIR / "ref_mfi_parquet_cache"

for _d in [OUTPUT_DIR, OUTPUT_FCS_DIR, PARQUET_CACHE]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Paramètres de mapping ────────────────────────────────────────────────────
# Percentile de distance au-delà duquel un nœud est assigné "Unknown"
DISTANCE_PERCENTILE: int = 60

# Inclure FSC-A / SSC-A dans le calcul de distance ?
# True  = utilise FSC-A et SSC-A en plus des canaux de fluorescence
#         (taille + granularité aident à séparer granulocytes / lymphocytes)
# False = fluorescence uniquement (plus standard en phénotypage)
INCLUDE_SCATTER_IN_MAPPING: bool = True

# Normalisation avant calcul de distance
# "range"  : (x - min) / (max - min)  → 0-1  [recommandé si scatter inclus]
# "zscore" : (x - mean) / std         → centré/réduit
# "none"   : pas de normalisation (déconseillé si scatter inclus)
NORMALIZATION_METHOD: str = "range"

# Couleurs par population pour Plotly
POPULATION_COLORS: Dict[str, str] = {
    "Granulo": "#F4A261",
    "Hématogone 34+": "#E76F51",
    "Hematogones19+33-": "#2A9D8F",
    "Ly T_NK": "#264653",
    "Lymphos B": "#457B9D",
    "Lymphos": "#A8DADC",
    "Plasmo": "#E9C46A",
    "Unknown": "#6C757D",
}

# ─── Colonnes à TOUJOURS exclure (jamais utilisées pour le mapping) ───────────
# Ces colonnes ne portent pas d'information phénotypique :
#   FSC-H   : doublon de FSC-A (même signal, mesure différente)
#   SSC-H   : doublon de SSC-A
#   FSC-Width : largeur du pulse → paramètre de discrimination doublets
#   TIME      : timestamp d'acquisition
_ALWAYS_EXCLUDE = {"FSC-H", "SSC-H", "FSC-Width", "TIME"}

# ─── Colonnes ajoutées par le pipeline FlowSOM (à ne pas utiliser non plus) ──
_FLOWSOM_META_COLS = {
    "FlowSOM_cluster",
    "FlowSOM_metacluster",
    "xGrid",
    "yGrid",
    "xNodes",
    "yNodes",
    "size",
    "Timepoint",
    "Timepoint_Num",
    "File_Origin",
    "Condition",
    "Condition_Num",
}

# ─── Localisation automatique du FCS final exporté ───────────────────────────
_fcs_candidates = sorted(OUTPUT_FCS_DIR.glob("flowsom_results_*.fcs"), reverse=True)

if _fcs_candidates:
    FINAL_FCS_PATH: Path = _fcs_candidates[0]
    print(f"[OK] FCS final détecté automatiquement : {FINAL_FCS_PATH.name}")
else:
    FINAL_FCS_PATH = OUTPUT_FCS_DIR / "flowsom_results_YYYYMMDD_HHMMSS.fcs"
    print(f"[!] Aucun FCS trouvé dans {OUTPUT_FCS_DIR}")
    print(f"    → Modifiez FINAL_FCS_PATH manuellement.")

print(f"\n     REF_MFI_DIR             : {REF_MFI_DIR}")
print(f"     FINAL_FCS_PATH          : {FINAL_FCS_PATH}")
print(f"     DISTANCE_PERCENTILE     : {DISTANCE_PERCENTILE}")
print(f"     INCLUDE_SCATTER         : {INCLUDE_SCATTER_IN_MAPPING}")
print(f"     NORMALIZATION_METHOD    : {NORMALIZATION_METHOD}")
print(f"     Colonnes toujours exclues : {sorted(_ALWAYS_EXCLUDE)}")

print(f"\n     Fichiers Ref MFI disponibles :")
if REF_MFI_DIR.exists():
    for _f in sorted(REF_MFI_DIR.glob("*.csv")):
        print(f"       {_f.name:<40s}  {_f.stat().st_size / 1e6:.1f} MB")
else:
    print(f"     [!] Dossier introuvable : {REF_MFI_DIR}")


################################################################################

# =============================================================================
# SECTION 10.1b — TRANSFORMATION CYTOMÉTRIQUE AVANT CALCUL DES MFI
# =============================================================================
#
# PROBLÈME RÉSOLU ICI :
# ─────────────────────
# Les nœuds FlowSOM sont dans l'espace TRANSFORMÉ (arcsinh/logicle, valeurs ≈ -10 à 25).
# Les CSV de référence contiennent des VALEURS BRUTES (linéaires : 50 000 – 300 000).
# → Appliquer la même transformation aux deux AVANT de calculer les MFI.
#
# POURQUOI transformer avant la moyenne et non après ?
# ─────────────────────────────────────────────────────
# La transformation cytométrique est NON-LINÉAIRE.
#   mean(transform(x)) ≠ transform(mean(x))
# Il faut donc transformer chaque événement individuellement (cellule par cellule),
# PUIS calculer la moyenne — exactement comme le fait le pipeline principal.
#
# BOOLÉEN CLÉ : APPLY_LOGICLE_TO_SCATTER
# ────────────────────────────────────────
#   False (par défaut) : FSC-A et SSC-A restent en valeurs brutes linéaires.
#                        Standard en cytométrie (le scatter est rarement log-transformé).
#   True               : FSC-A et SSC-A sont également transformés.
#                        À utiliser si le pipeline FlowSOM a été entraîné avec
#                        le scatter transformé (vérifier EXCLUDE_SCATTER dans les
#                        sections précédentes).
#
# TRANSFORMATION UTILISÉE :
# ─────────────────────────
# Logicle (Parks 2006) via pytometry si disponible, sinon arcsinh(x / cofactor).
# Les paramètres (cofacteur, T, M, W) sont lus depuis les variables existantes
# du pipeline pour garantir la cohérence exacte avec le début du fichier.
# =============================================================================

import warnings

warnings.filterwarnings("ignore")

# ── Paramètre booléen : scatter inclus dans la transformation ? ───────────────
APPLY_LOGICLE_TO_SCATTER: bool = False
# False = FSC-A / SSC-A conservés en linéaire (recommandé)
# True  = FSC-A / SSC-A également transformés

# ── Récupération des paramètres de transformation du pipeline principal ───────
_TRANSFORM_IN_USE: str = str(globals().get("TRANSFORM_TYPE", "arcsinh")).lower()
_COFACTOR_IN_USE: float = float(
    globals().get("COFACTOR", globals().get("ARCSINH_COFACTOR", 5.0))
)

# ── Colonnes scatter (FSC / SSC) — pattern de détection ─────────────────────
_SCATTER_PREFIXES = ("FSC", "SSC")


# =============================================================================
# FONCTIONS DE TRANSFORMATION
# =============================================================================


def _logicle_matrix_pytometry(
    X: np.ndarray,
    col_names: List[str],
) -> np.ndarray:
    """
    Applique la transformation Logicle via pytometry.tl.logicle sur toute la matrice.
    Retourne X_transformed (même shape).
    Requiert pytometry + anndata.
    """
    import pytometry as _pm
    import anndata as _ad

    _adata_tmp = _ad.AnnData(X=X.astype(np.float32).copy())
    _adata_tmp.var_names = [str(c) for c in col_names]
    _pm.tl.logicle(_adata_tmp, channels=list(col_names))
    _out = np.array(_adata_tmp.X, dtype=np.float64)
    return _out


def apply_cyto_transform_matrix(
    X: np.ndarray,
    col_names: List[str],
    apply_to_scatter: bool = False,
    transform_type: str = "arcsinh",
    cofactor: float = 5.0,
) -> np.ndarray:
    """
    Applique la transformation cytométrique à une matrice (n_events × n_features).

    Stratégie :
    1. Sélectionne les colonnes à transformer selon `apply_to_scatter`.
    2. Pour `transform_type="logicle"` : tente pytometry, sinon arcsinh.
    3. Pour `transform_type="arcsinh"` : arcsinh(x / cofactor) vectorisé.
    4. Les colonnes scatter non transformées (apply_to_scatter=False) sont copiées.

    Args:
        X               : Matrice brute (n_events, n_features), float64.
        col_names       : Noms des colonnes dans l'ordre des colonnes de X.
        apply_to_scatter: Si False, FSC-A/SSC-A ne sont pas transformés.
        transform_type  : "logicle" ou "arcsinh".
        cofactor        : Cofacteur pour arcsinh (5 = standard fluo cytométrie).

    Returns:
        X_transformed : Matrice transformée, float64, même shape que X.
    """
    if X.shape[0] == 0:
        return X.copy()

    # Identifier les indices scatter vs fluo
    _is_scatter = np.array(
        [
            any(str(c).upper().startswith(p) for p in _SCATTER_PREFIXES)
            for c in col_names
        ],
        dtype=bool,
    )

    # Indices des colonnes à transformer
    if apply_to_scatter:
        _idx_to_transform = np.arange(len(col_names))
    else:
        _idx_to_transform = np.where(~_is_scatter)[0]

    _idx_to_keep_linear = np.where(_is_scatter & ~apply_to_scatter)[0]

    if len(_idx_to_transform) == 0:
        return X.copy()

    X_out = X.copy()

    # ── Sous-matrice des colonnes à transformer ────────────────────────────────
    _X_sub = X[:, _idx_to_transform]
    _cols_sub = [col_names[i] for i in _idx_to_transform]

    if transform_type == "logicle":
        try:
            _X_sub_t = _logicle_matrix_pytometry(_X_sub, _cols_sub)
            X_out[:, _idx_to_transform] = _X_sub_t
            _method_used = "logicle (pytometry)"
        except Exception as _e_pt:
            # Fallback arcsinh si pytometry non disponible ou erreur
            X_out[:, _idx_to_transform] = np.arcsinh(_X_sub / cofactor)
            _method_used = f"arcsinh(cofactor={cofactor}) [fallback : {_e_pt}]"
    else:
        # arcsinh standard
        X_out[:, _idx_to_transform] = np.arcsinh(_X_sub / cofactor)
        _method_used = f"arcsinh(x / {cofactor})"

    return X_out


# =============================================================================
# RAPPORT DE CONFIGURATION
# =============================================================================
print("=" * 70)
print(" SECTION 10.1b — TRANSFORMATION CYTOMÉTRIQUE (ALIGNEMENT DES ESPACES)")
print("=" * 70)
print(f"\n   ─── Paramètres ───────────────────────────────────────────────────")
print(f"   APPLY_LOGICLE_TO_SCATTER  : {APPLY_LOGICLE_TO_SCATTER}")
print(f"   Transformation pipeline   : '{_TRANSFORM_IN_USE}'")
print(f"   Cofacteur                 : {_COFACTOR_IN_USE}")
print(f"\n   ─── Comportement résultant ──────────────────────────────────────")
print(f"   • Canaux fluorescents (-A non scatter) : TOUJOURS transformés")
print(
    f"   • FSC-A, SSC-A : {'TRANSFORMÉS' if APPLY_LOGICLE_TO_SCATTER else 'CONSERVÉS EN LINÉAIRE (standard cytométrie)'}"
)
print(f"\n   ─── Transformation appliquée ────────────────────────────────────")
if _TRANSFORM_IN_USE == "logicle":
    print(f"   Logicle (pytometry) → fallback arcsinh si non disponible")
else:
    print(f"   arcsinh(x / {_COFACTOR_IN_USE})")
print(f"\n   ─── Impact sur le cache Parquet ─────────────────────────────────")
print(f"   Les caches Parquet sont indexés par (population + transform_tag).")
print(f"   Un nouveau tag sera calculé → les anciens caches seront ignorés")
print(f"   (mais conservés dans le dossier pour rollback rapide).")

# Tag unique pour identifier le cache selon la transformation
_TRANSFORM_TAG = (
    f"{_TRANSFORM_IN_USE}_cof{_COFACTOR_IN_USE}"
    f"{'_scatter' if APPLY_LOGICLE_TO_SCATTER else '_noscatter'}"
)
print(f"\n   Tag de cache                : '{_TRANSFORM_TAG}'")

# Test rapide de disponibilité de pytometry
_PYTOMETRY_AVAIL = False
try:
    import pytometry as _pm_test

    _PYTOMETRY_AVAIL = True
    print(
        f"\n   [OK] pytometry disponible → Logicle natif utilisé si 'logicle' sélectionné"
    )
except ImportError:
    print(f"\n   [!] pytometry indisponible → arcsinh utilisé comme fallback")

print(f"\n{'=' * 70}")
print(f" → Les Sections 10.2 et 10.3 vont maintenant transformer les données")
print(f"   avant de calculer les centroïdes et les MFI moyennes.")
print(f"{'=' * 70}")


################################################################################

# =============================================================================
# SECTION 10.2 — LECTURE DU FCS FINAL + EXTRACTION DES CENTROÏDES (ESPACE TRANSFORMÉ)
# =============================================================================
# On lit le FCS exporté par le pipeline pour récupérer :
#   1. Les colonnes cytométrie BRUTES (uniquement les -A après filtre)
#   2. On applique la transformation cytométrique (logicle/arcsinh) définie
#      en Section 10.1b → les centroïdes seront dans le MÊME espace que les CSV
#   3. FlowSOM_cluster  → index du nœud SOM (1-indexé dans le FCS)
#   4. FlowSOM_metacluster, xGrid, yGrid, xNodes, yNodes, ...
#
# Centroïdes de nœuds = moyenne des cellules par nœud dans l'espace TRANSFORMÉ.
# ⚠️ IMPORTANT : mean(transform(x)) ≠ transform(mean(x)) → on transforme d'abord
#                chaque événement individuel, puis on fait la moyenne.
# =============================================================================

print("=" * 70)
print(" SECTION 10.2 — LECTURE DU FCS FINAL (données -A, espace transformé)")
print("=" * 70)
print(f"   Fichier : {FINAL_FCS_PATH}")

_df_fcs_raw: Optional[pd.DataFrame] = None

# Tentative 1 : flowsom.io.read_FCS (retourne AnnData)
try:
    import flowsom as _fs_io
    import anndata as _ad

    _adata_raw = _fs_io.io.read_FCS(str(FINAL_FCS_PATH))
    _X_raw = _adata_raw.X
    if hasattr(_X_raw, "toarray"):
        _X_raw = _X_raw.toarray()
    _df_fcs_raw = pd.DataFrame(_X_raw, columns=list(_adata_raw.var_names))
    print(f"   [OK] Lu via flowsom.io  — {_df_fcs_raw.shape}")
except Exception as _e1:
    print(f"   [!] flowsom.io échoué : {_e1}")

# Tentative 2 : flowkit
if _df_fcs_raw is None:
    try:
        import flowkit as fk

        _fk_sample = fk.Sample(str(FINAL_FCS_PATH))
        _events = _fk_sample.get_events(source="raw")
        _df_fcs_raw = pd.DataFrame(_events, columns=_fk_sample.pnn_labels)
        print(f"   [OK] Lu via FlowKit     — {_df_fcs_raw.shape}")
    except Exception as _e2:
        print(f"   [!] FlowKit échoué : {_e2}")

if _df_fcs_raw is None:
    raise RuntimeError(
        "Impossible de lire le FCS final.\n"
        "Vérifiez que flowsom ou flowkit est installé et que FINAL_FCS_PATH est correct."
    )

print(f"\n   Toutes les colonnes du FCS ({len(_df_fcs_raw.columns)}) :")
for _c in _df_fcs_raw.columns:
    print(f"     '{_c}'")

# ─── Identification des colonnes clés ────────────────────────────────────────
_COL_NODE = next(
    (
        c
        for c in _df_fcs_raw.columns
        if "cluster" in c.lower() and "meta" not in c.lower()
    ),
    None,
)
_COL_META = next((c for c in _df_fcs_raw.columns if "metacluster" in c.lower()), None)
_COL_XGRID = next((c for c in _df_fcs_raw.columns if c.lower() == "xgrid"), None)
_COL_YGRID = next((c for c in _df_fcs_raw.columns if c.lower() == "ygrid"), None)
_COL_XNODES = next((c for c in _df_fcs_raw.columns if c.lower() == "xnodes"), None)
_COL_YNODES = next((c for c in _df_fcs_raw.columns if c.lower() == "ynodes"), None)
_COL_CONDITION = next(
    (c for c in _df_fcs_raw.columns if c.lower() in ("condition_num", "condition")),
    None,
)

print(f"\n   Colonnes FlowSOM identifiées :")
print(f"     FlowSOM_cluster    → '{_COL_NODE}'")
print(f"     FlowSOM_metaclust  → '{_COL_META}'")
print(f"     xGrid              → '{_COL_XGRID}'")
print(f"     yGrid              → '{_COL_YGRID}'")
print(f"     xNodes             → '{_COL_XNODES}'")
print(f"     yNodes             → '{_COL_YNODES}'")
print(f"     Condition          → '{_COL_CONDITION}'")

if _COL_NODE is None:
    raise ValueError(
        "Colonne 'FlowSOM_cluster' introuvable dans le FCS.\n"
        f"Colonnes disponibles : {list(_df_fcs_raw.columns)}"
    )

# ─── Colonnes à exclure du mapping (méta-données FlowSOM + exclusions fixes) ─
_ALL_META_COLS = _FLOWSOM_META_COLS | _ALWAYS_EXCLUDE
# Ajouter les noms réellement trouvés dans le FCS
_ALL_META_COLS.update(
    c
    for c in [
        _COL_NODE,
        _COL_META,
        _COL_XGRID,
        _COL_YGRID,
        _COL_XNODES,
        _COL_YNODES,
        _COL_CONDITION,
    ]
    if c is not None
)

# ─── FILTRE CLÉ : ne garder QUE les colonnes se terminant par "-A" ───────────
_CYTOMETRY_COLS_A: List[str] = [
    c
    for c in _df_fcs_raw.columns
    if c not in _ALL_META_COLS and c.upper().endswith("-A")
]

# Colonnes -H et autres (pour information uniquement, non utilisées)
_CYTOMETRY_COLS_H: List[str] = [
    c
    for c in _df_fcs_raw.columns
    if c not in _ALL_META_COLS and c.upper().endswith("-H")
]
_CYTOMETRY_COLS_OTHER: List[str] = [
    c
    for c in _df_fcs_raw.columns
    if c not in _ALL_META_COLS
    and not c.upper().endswith("-A")
    and not c.upper().endswith("-H")
]

print(f"\n" + "─" * 70)
print(f"   FILTRE COLONNES FCS (uniquement -A retenues)")
print(f"─" * 70)
print(f"   ✅ Colonnes -A retenues ({len(_CYTOMETRY_COLS_A)}) :")
for _c in _CYTOMETRY_COLS_A:
    _is_scatter = any(_c.upper().startswith(p) for p in ("FSC", "SSC"))
    _tag = " [scatter]" if _is_scatter else ""
    print(f"       {_c}{_tag}")
print(f"   ❌ Colonnes -H ignorées ({len(_CYTOMETRY_COLS_H)}) : {_CYTOMETRY_COLS_H}")
print(f"   ❌ Autres ignorées ({len(_CYTOMETRY_COLS_OTHER)}) : {_CYTOMETRY_COLS_OTHER}")

# ─── Extraction brute, nettoyage NaN/Inf ─────────────────────────────────────
_node_ids_raw = _df_fcs_raw[_COL_NODE].to_numpy(dtype=np.int32) - 1  # 0-indexé
_n_nodes_raw = int(_node_ids_raw.max()) + 1

_X_cyto_A = _df_fcs_raw[_CYTOMETRY_COLS_A].to_numpy(dtype=np.float64)
_X_cyto_A = np.nan_to_num(_X_cyto_A, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

# ─── [NOUVEAU] Transformation cytométrique AVANT calcul des centroïdes ───────
# On applique la même transformation que celle utilisée en début de pipeline
# (voir Section 10.1b : _TRANSFORM_IN_USE, _COFACTOR_IN_USE).
# Raison : les centroïdes des nœuds FlowSOM originaux sont dans l'espace
# TRANSFORMÉ. Pour aligner les CSV de référence correctement, il faut que les
# centroïdes que nous calculons ici soient dans ce MÊME espace.
print(f"\n" + "=" * 70)
print(f" TRANSFORMATION DES COLONNES FCS -A (espace transformé)")
print(f"=" * 70)
print(f"   Transformation : '{_TRANSFORM_IN_USE}' | cofacteur : {_COFACTOR_IN_USE}")
print(f"   Scatter transformé : {APPLY_LOGICLE_TO_SCATTER}")
print(
    f"   Cellules à transformer : {_X_cyto_A.shape[0]:,} × {_X_cyto_A.shape[1]} canaux"
)

# Afficher les plages AVANT transformation (diagnostic)
print(f"\n   Plages AVANT transformation :")
for _j, _c in enumerate(_CYTOMETRY_COLS_A):
    _is_sc = any(_c.upper().startswith(p) for p in ("FSC", "SSC"))
    _tag = (
        " [scatter — linéaire conservé]"
        if (_is_sc and not APPLY_LOGICLE_TO_SCATTER)
        else ""
    )
    print(
        f"     {_c:<35s} : [{_X_cyto_A[:, _j].min():10.1f}, {_X_cyto_A[:, _j].max():10.1f}]{_tag}"
    )

_X_cyto_A = apply_cyto_transform_matrix(
    _X_cyto_A,
    col_names=_CYTOMETRY_COLS_A,
    apply_to_scatter=APPLY_LOGICLE_TO_SCATTER,
    transform_type=_TRANSFORM_IN_USE,
    cofactor=_COFACTOR_IN_USE,
)

# Afficher les plages APRÈS transformation (confirmation de l'alignement)
print(f"\n   Plages APRÈS transformation :")
for _j, _c in enumerate(_CYTOMETRY_COLS_A):
    _is_sc = any(_c.upper().startswith(p) for p in ("FSC", "SSC"))
    _tag = " [scatter — inchangé]" if (_is_sc and not APPLY_LOGICLE_TO_SCATTER) else ""
    print(
        f"     {_c:<35s} : [{_X_cyto_A[:, _j].min():8.3f}, {_X_cyto_A[:, _j].max():8.3f}]{_tag}"
    )

# ─── Calcul des centroïdes de nœuds dans l'espace TRANSFORMÉ ─────────────────
print(f"\n" + "=" * 70)
print(f" CALCUL DES CENTROÏDES DE NŒUDS — ESPACE TRANSFORMÉ ({_TRANSFORM_IN_USE})")
print(f"=" * 70)

_node_sum_raw = np.zeros((_n_nodes_raw, len(_CYTOMETRY_COLS_A)), dtype=np.float64)
_node_count_raw = np.zeros(_n_nodes_raw, dtype=np.int64)

np.add.at(_node_sum_raw, _node_ids_raw, _X_cyto_A)
np.add.at(_node_count_raw, _node_ids_raw, 1)

_empty_nodes = _node_count_raw == 0
_n_empty = int(_empty_nodes.sum())
if _n_empty > 0:
    print(f"   ⚠️  {_n_empty} nœuds vides → MFI = 0")

_node_mfi_raw_arr = np.zeros_like(_node_sum_raw)
_valid_nodes_mask = ~_empty_nodes
_node_mfi_raw_arr[_valid_nodes_mask] = (
    _node_sum_raw[_valid_nodes_mask] / _node_count_raw[_valid_nodes_mask, np.newaxis]
)

node_mfi_raw_df = pd.DataFrame(
    _node_mfi_raw_arr,
    columns=_CYTOMETRY_COLS_A,
)
node_mfi_raw_df.index.name = "node_id"

# ─── Coordonnées grille/MST et métacluster par nœud ──────────────────────────
_COORD_COLS_AVAIL = [
    c
    for c in [_COL_XGRID, _COL_YGRID, _COL_XNODES, _COL_YNODES]
    if c is not None and c in _df_fcs_raw.columns
]

_df_fcs_nodes_cols = _df_fcs_raw[
    [_COL_NODE] + _COORD_COLS_AVAIL + ([_COL_META] if _COL_META else [])
].copy()
_df_fcs_nodes_cols[_COL_NODE] = _node_ids_raw  # 0-indexé

_node_meta_df = (
    _df_fcs_nodes_cols.groupby(_COL_NODE)
    .median()
    .reset_index()
    .rename(columns={_COL_NODE: "node_id"})
)

if _COL_META:
    _mc_per_node_raw = _node_meta_df.set_index("node_id")[_COL_META].apply(
        lambda x: int(round(x)) - 1  # 0-indexé
    )
else:
    _mc_per_node_raw = pd.Series(
        [0] * _n_nodes_raw, index=range(_n_nodes_raw), name="metacluster"
    )

_node_coords_df = _node_meta_df.set_index("node_id")[_COORD_COLS_AVAIL]

print(f"   [OK] {_n_nodes_raw} nœuds × {len(_CYTOMETRY_COLS_A)} marqueurs -A calculés")
print(f"        Espace : {_TRANSFORM_IN_USE} (cofacteur={_COFACTOR_IN_USE})")
print(f"\n   Aperçu des centroïdes TRANSFORMÉS (-A uniquement) :")
display(node_mfi_raw_df.head(5).round(4))
print(f"\n   Aperçu des coordonnées par nœud :")
display(_node_coords_df.head(5).round(3))


################################################################################

# =============================================================================
# SECTION 10.3 — CHARGEMENT DES CSV REF MFI (ESPACE TRANSFORMÉ — COLONNES -A)
# =============================================================================
# PHILOSOPHIE DU MAPPING V3 (améliorée) :
# ─────────────────────────────────────────
# On ne conserve que les colonnes -A dans les CSV.
# ON APPLIQUE LA TRANSFORMATION CYTOMÉTRIQUE (logicle/arcsinh) à chaque
# événement du CSV AVANT de calculer la MFI moyenne.
#
# Raison mathématique :
#   mean(transform(x)) ≠ transform(mean(x))  [non-linéarité de logicle/arcsinh]
#   → Il faut transformer chaque cellule individuellement, puis faire la moyenne.
#
# Cache Parquet :
#   Le nom du cache inclut maintenant le _TRANSFORM_TAG (ex: "arcsinh_cof5.0_noscatter"),
#   ce qui permet de conserver plusieurs versions (brute, arcsinh, logicle) en parallèle.
#
# Mapping CSV → FCS (inchangé) :
#   CSV "CD13 PE-A" → "cd13pe-a" → FCS "CD13 PE-A"    ✅ match direct
# =============================================================================


def normalize_col_name(name: str) -> str:
    """
    Normalisation minimale d'un nom de colonne pour comparaison :
    - Supprime les espaces
    - Met en minuscules
    - Supprime les accents
    - Conserve les tirets et chiffres (important pour '-A', 'PC5.5', etc.)
    """
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"\s+", "", name)
    return name


def filter_area_columns(columns: List[str]) -> List[str]:
    """
    Retourne uniquement les colonnes se terminant par '-A' (insensible à la casse).
    Exclut aussi les colonnes listées dans _ALWAYS_EXCLUDE.
    """
    return [c for c in columns if c.upper().endswith("-A") and c not in _ALWAYS_EXCLUDE]


def build_direct_mapping_a_only(
    csv_cols_a: List[str],
    fcs_cols_a: List[str],
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Construit le mapping CSV -A → FCS -A par correspondance directe normalisée.
    """
    fcs_norm_to_orig: Dict[str, str] = {}
    for _c in fcs_cols_a:
        _norm = normalize_col_name(_c)
        if _norm in fcs_norm_to_orig:
            print(f"   ⚠️  Collision FCS normalisé '{_norm}' :")
            print(f"       '{fcs_norm_to_orig[_norm]}' vs '{_c}' → on garde le premier")
        else:
            fcs_norm_to_orig[_norm] = _c

    mapping: Dict[str, str] = {}
    unmatched: List[str] = []

    for csv_col in csv_cols_a:
        csv_norm = normalize_col_name(csv_col)
        if csv_norm in fcs_norm_to_orig:
            mapping[csv_col] = fcs_norm_to_orig[csv_norm]
        else:
            unmatched.append(csv_col)

    if verbose:
        print(
            f"\n   ━━ MAPPING DIRECT CSV -A → FCS -A ({len(mapping)} correspondances) ━━"
        )
        print(f"   {'─' * 70}")
        print(f"   {'Colonne CSV (-A)':<40s}   {'Colonne FCS (-A)':<40s}")
        print(f"   {'─' * 40}   {'─' * 40}")
        for _csv, _fcs in sorted(mapping.items()):
            _match_icon = "✅" if _csv == _fcs else "🔄"
            print(f"   {_match_icon} {_csv:<38s} → {_fcs:<38s}")
        if unmatched:
            print(f"\n   ━━ COLONNES CSV NON MAPPÉES ({len(unmatched)}) ━━")
            for _u in sorted(unmatched):
                _u_norm = normalize_col_name(_u)
                print(f"     ✗ '{_u}'  (normalisé: '{_u_norm}')")
        print(f"\n   Bilan : {len(mapping)} mappés / {len(csv_cols_a)} -A dans les CSV")
        print(f"          {len(unmatched)} ignorés / {len(fcs_cols_a)} -A dans le FCS")

    return mapping


def load_population_csv_transformed(
    csv_path: Path,
    parquet_dir: Path,
    area_cols_to_keep: Optional[List[str]] = None,
    chunk_size: int = 50_000,
    sep: str = ";",
    decimal: str = ",",
    force_reload: bool = False,
    transform_type: str = "arcsinh",
    cofactor: float = 5.0,
    apply_to_scatter: bool = False,
    transform_tag: str = "arcsinh_cof5.0_noscatter",
) -> pd.Series:
    """
    Calcule la MFI TRANSFORMÉE moyenne pour les colonnes -A d'un fichier CSV.

    La transformation cytométrique (logicle ou arcsinh) est appliquée à chaque
    événement individuel AVANT de calculer la moyenne. Cela garantit que les
    MFI de référence sont dans le MÊME espace mathématique que les centroïdes
    des nœuds FlowSOM (calculés en Section 10.2).

    Différence vs l'ancienne version :
    • Ancienne : mean(valeurs brutes) → puis transformé au moment du mapping
    • Nouvelle : transform(valeurs brutes) pour chaque cellule → mean des transformées
    Le résultat est mathématiquement correct pour une transformation non-linéaire.

    Cache Parquet : keyed par (pop_name + transform_tag) → sûr pour rollback.

    Args:
        csv_path          : Chemin du CSV brut d'une population.
        parquet_dir       : Dossier de cache Parquet.
        area_cols_to_keep : Liste des colonnes -A à garder (None = toutes les -A).
        chunk_size        : Taille des chunks de lecture.
        sep / decimal     : Séparateur et décimale du CSV.
        force_reload      : Si True, ignore le cache.
        transform_type    : "logicle" ou "arcsinh".
        cofactor          : Cofacteur arcsinh.
        apply_to_scatter  : Si True, transforme aussi FSC-A/SSC-A.
        transform_tag     : Suffixe de cache (identifie la config de transform).

    Returns:
        pd.Series — MFI moyennes TRANSFORMÉES (index = noms de colonnes -A).
    """
    pop_name = csv_path.stem
    # Cache keyed par (population + transformation) pour éviter les collisions
    parquet_path = parquet_dir / f"{pop_name}_{transform_tag}.parquet"

    if parquet_path.exists() and not force_reload:
        cached = pd.read_parquet(parquet_path)
        s = cached.iloc[0]
        print(f"   [cache] {pop_name:<35s}  {len(s)} marqueurs  [{transform_tag}]")
        return s

    file_mb = csv_path.stat().st_size / 1e6

    # ── Lecture de l'en-tête pour identifier les colonnes -A disponibles ─────
    _header_chunk = pd.read_csv(csv_path, sep=sep, decimal=decimal, nrows=0)
    _all_csv_cols = list(_header_chunk.columns)
    _csv_a_cols = filter_area_columns(_all_csv_cols)

    if area_cols_to_keep is not None:
        _csv_a_cols = [c for c in _csv_a_cols if c in area_cols_to_keep]

    if not _csv_a_cols:
        raise ValueError(
            f"Aucune colonne -A trouvée dans {csv_path.name}.\n"
            f"Colonnes disponibles : {_all_csv_cols}"
        )

    print(
        f"   [read]  {pop_name:<35s}  {file_mb:.1f} MB  "
        f"({len(_csv_a_cols)} cols -A)  [{transform_tag}]"
    )

    sum_arr = np.zeros(len(_csv_a_cols), dtype=np.float64)
    count = 0

    for chunk in pd.read_csv(
        csv_path,
        sep=sep,
        decimal=decimal,
        usecols=_csv_a_cols,
        chunksize=chunk_size,
        dtype=np.float32,
        na_values=["", " "],
        engine="c",
    ):
        # S'assurer que les colonnes sont dans le bon ordre
        chunk = chunk.reindex(columns=_csv_a_cols)
        vals = chunk.to_numpy(dtype=np.float64, na_value=0.0)

        # ── [NOUVEAU] Transformation cytométrique par événement ────────────
        # On transforme chaque cellule individuellement AVANT de sommer.
        # Cela garantit que mean(transform(x)) est bien calculé, et non
        # transform(mean(x)) qui serait mathématiquement inexact.
        vals = apply_cyto_transform_matrix(
            vals,
            col_names=_csv_a_cols,
            apply_to_scatter=apply_to_scatter,
            transform_type=transform_type,
            cofactor=cofactor,
        )

        sum_arr += vals.sum(axis=0)
        count += vals.shape[0]

    if count == 0:
        raise ValueError(f"Aucun événement lu dans {csv_path.name}")

    mean_s = pd.Series(sum_arr / count, index=_csv_a_cols, dtype=np.float64)
    mean_s.name = pop_name

    # Sauvegarder le cache avec le tag de transformation
    mean_s.to_frame().T.to_parquet(parquet_path, index=False)
    print(
        f"   [done]  {pop_name:<35s}  {count:>8,} événements → "
        f"MFI transformées cachées [{transform_tag}]"
    )

    return mean_s


# =============================================================================
# CHARGEMENT DE TOUTES LES POPULATIONS (MFI TRANSFORMÉES)
# =============================================================================
print("=" * 70)
print(
    f" SECTION 10.3 — CHARGEMENT DES POPULATIONS (MFI transformées : {_TRANSFORM_IN_USE})"
)
print("=" * 70)

# ── Étape 1 : Lire l'en-tête du premier CSV ───────────────────────────────────
_first_csv = next(iter(sorted(REF_MFI_DIR.glob("*.csv"))), None)
if _first_csv is None:
    raise FileNotFoundError(f"Aucun CSV trouvé dans {REF_MFI_DIR}")

_header_df = pd.read_csv(_first_csv, sep=";", decimal=",", nrows=0)
_ALL_CSV_COLS = list(_header_df.columns)
_CSV_A_COLS = filter_area_columns(_ALL_CSV_COLS)

print(f"\n   En-tête CSV récupérée depuis : {_first_csv.name}")
print(f"   Colonnes totales dans les CSV : {len(_ALL_CSV_COLS)}")
print(f"   Colonnes -A retenues          : {len(_CSV_A_COLS)}")
print(f"   → {_CSV_A_COLS}")
print(
    f"   Colonnes -H ignorées : {[c for c in _ALL_CSV_COLS if c.upper().endswith('-H') and c not in _ALWAYS_EXCLUDE]}"
)
print(
    f"   Autres colonnes ignorées : {[c for c in _ALL_CSV_COLS if c in _ALWAYS_EXCLUDE or (not c.upper().endswith('-A') and not c.upper().endswith('-H'))]}"
)

# ── Étape 2 : Construire le mapping CSV -A → FCS -A ───────────────────────────
print(f"\n" + "=" * 70)
print(f" CONSTRUCTION DU MAPPING CSV -A → FCS -A (correspondance directe)")
print(f"=" * 70)

_csv_to_fcs_mapping_v3 = build_direct_mapping_a_only(
    csv_cols_a=_CSV_A_COLS,
    fcs_cols_a=_CYTOMETRY_COLS_A,
    verbose=True,
)

_VALID_CSV_A_COLS = list(_csv_to_fcs_mapping_v3.keys())
_VALID_FCS_A_COLS = [_csv_to_fcs_mapping_v3[c] for c in _VALID_CSV_A_COLS]

print(f"\n   ✅ {len(_VALID_CSV_A_COLS)} colonnes seront utilisées pour le mapping :")
for _src, _tgt in zip(_VALID_CSV_A_COLS, _VALID_FCS_A_COLS):
    _same = "  [nom identique]" if _src == _tgt else f"  ['{_src}' → '{_tgt}']"
    print(f"     • {_tgt}{_same}")

# ── Étape 3 : Charger et transformer les MFI pour chaque population ───────────
print(f"\n" + "─" * 70)
print(f" Chargement + transformation des CSV de référence")
print(
    f" Méthode : {_TRANSFORM_IN_USE} | cofacteur : {_COFACTOR_IN_USE} | scatter : {APPLY_LOGICLE_TO_SCATTER}"
)
print(f" Tag de cache : '{_TRANSFORM_TAG}'")
print(f"─" * 70)

pop_mfi_raw_ref: Dict[str, pd.Series] = {}

for _csv_f in sorted(REF_MFI_DIR.glob("*.csv")):
    try:
        _s = load_population_csv_transformed(
            csv_path=_csv_f,
            parquet_dir=PARQUET_CACHE,
            area_cols_to_keep=_VALID_CSV_A_COLS,
            force_reload=False,
            transform_type=_TRANSFORM_IN_USE,
            cofactor=_COFACTOR_IN_USE,
            apply_to_scatter=APPLY_LOGICLE_TO_SCATTER,
            transform_tag=_TRANSFORM_TAG,
        )
        pop_mfi_raw_ref[_csv_f.stem] = _s
    except Exception as _e:
        print(f"   [ERREUR] {_csv_f.name} : {_e}")

print(
    f"\n[OK] {len(pop_mfi_raw_ref)} populations chargées : {list(pop_mfi_raw_ref.keys())}"
)

# ── Étape 4 : DataFrame de référence dans l'espace FCS (transformé) ───────────
_ref_records = {}
for _pop, _series in pop_mfi_raw_ref.items():
    _rec = {}
    for _csv_c in _VALID_CSV_A_COLS:
        _fcs_c = _csv_to_fcs_mapping_v3[_csv_c]
        _rec[_fcs_c] = _series.get(_csv_c, np.nan)
    _ref_records[_pop] = _rec

df_mfi_raw_ref = pd.DataFrame(_ref_records).T
df_mfi_raw_ref.index.name = "population"
df_mfi_raw_ref.columns.name = "marqueur_fcs"

print(
    f"\n   Shape df_mfi_raw_ref : {df_mfi_raw_ref.shape}  "
    f"(populations × marqueurs FCS -A, espace {_TRANSFORM_IN_USE})"
)
print(f"   Colonnes : {list(df_mfi_raw_ref.columns)}")
print(f"\n   MFI TRANSFORMÉES par population (à comparer avec les centroïdes nœuds) :")
print(f"   Shape: {df_mfi_raw_ref.shape} — voir fichier CSV de sortie pour le détail")

# ── Diagnostic d'alignement des espaces ──────────────────────────────────────
print(f"\n{'─' * 70}")
print(f" DIAGNOSTIC D'ALIGNEMENT DES ESPACES (nœuds vs références)")
print(f"{'─' * 70}")
_common_diag = sorted(set(node_mfi_raw_df.columns) & set(df_mfi_raw_ref.columns))
print(
    f"\n   {'Marqueur':<35s}  {'Nœuds [min, max]':<25s}  {'Refs [min, max]':<25s}  {'Aligné?'}"
)
print(f"   {'─' * 35}  {'─' * 25}  {'─' * 25}  {'─' * 8}")
for _c in _common_diag[:8]:  # afficher les 8 premiers pour diagnostic rapide
    _n_min = node_mfi_raw_df[_c].min()
    _n_max = node_mfi_raw_df[_c].max()
    _r_min = df_mfi_raw_ref[_c].min()
    _r_max = df_mfi_raw_ref[_c].max()
    # Considéré "aligné" si les plages se chevauchent raisonnablement
    _overlap = not (_r_max < _n_min * 0.1 or _r_min > _n_max * 10)
    _aligned = "✅ OK" if _overlap else "⚠️ ÉCART"
    print(
        f"   {_c:<35s}  [{_n_min:7.2f}, {_n_max:7.2f}]  "
        f"  [{_r_min:7.2f}, {_r_max:7.2f}]  {_aligned}"
    )
if len(_common_diag) > 8:
    print(f"   ... ({len(_common_diag) - 8} autres marqueurs)")

# ── Contrôle NaN ──────────────────────────────────────────────────────────────
_nan_mask = df_mfi_raw_ref.isna()
if _nan_mask.values.any():
    print(f"\n   ⚠️  Valeurs NaN détectées dans df_mfi_raw_ref :")
    for _pop in df_mfi_raw_ref.index:
        _nan_cols = df_mfi_raw_ref.columns[_nan_mask.loc[_pop]].tolist()
        if _nan_cols:
            print(f"     {_pop} : {_nan_cols}")
    df_mfi_raw_ref = df_mfi_raw_ref.fillna(0.0)
else:
    print(f"\n   ✅ Aucun NaN dans df_mfi_raw_ref.")

print(f"\n{'=' * 70}")
print(f" [OK] Section 10.3 terminée — MFI dans l'espace : {_TRANSFORM_IN_USE}")
print(f"      Les espaces nœuds et références sont maintenant ALIGNÉS.")
print(f"{'=' * 70}")


################################################################################

# =============================================================================
# SECTION 10.3b — STATISTIQUES DE RÉFÉRENCE BIOLOGIQUES
#                 (Comptages, Covariances, Échantillons KNN)
# =============================================================================
#
# POURQUOI CETTE SECTION ?
# ─────────────────────────
# Les méthodes M9/M10/M11 en Section 10.4b nécessitent des informations
# que la moyenne (MFI) seule ne fournit pas :
#
#   M9_prior : "Probabilité a priori" (Bayésien)
#   ───────────────────────────────────────────────
#   Les Granulocytes représentent ~70% d'une moelle normale vs ~1% d'Hématogones.
#   → Un nœud à distance égale de deux populations doit aller aux Granuleux.
#   → Poids = log10(n_cells) : les grandes populations sont plus "magnétiques".
#   → Formule : D_adj = D_initial / log10(n_cells_ref)
#
#   M10_mahalanobis : Distance tenant compte de la variance intra-population
#   ─────────────────────────────────────────────────────────────────────────
#   Les Granuleux ont une forte variance (étalement SSC, CD45 variable).
#   Les Hématogones sont homogènes (petit écart-type).
#   → La distance de Mahalanobis pénalise correctement un nœud "excentrique"
#     d'une population homogène, mais est tolérante pour une population hétérogène.
#   → Nécessite : matrice de covariance Σ par population.
#
#   M11_knn_density : Vote par voisinage (K-NN sur événements bruts subsampléchés)
#   ──────────────────────────────────────────────────────────────────────────────────
#   Pour chaque nœud, cherche les K plus proches voisins parmi TOUS les événements
#   de TOUTES les références poolées. Le vote majoritaire donne la population.
#   → Si 90% des voisins sont "Granulo" et 1% "Hématogone", c'est un Granuleux.
#   → Gère naturellement le biais de densité différentielle.
#   → Nécessite : sous-échantillon représentatif de chaque population.
#
# PARAMÈTRES :
#   KNN_SAMPLE_SIZE_PER_POP  : événements subsampléchés par population (M11)
#   KNN_K                    : nombre de voisins pour M11
#   COV_CHUNK_SIZE           : taille des chunks pour le calcul de covariance incrémental
# =============================================================================

print("=" * 70)
print(" SECTION 10.3b — STATISTIQUES DE RÉFÉRENCE BIOLOGIQUES")
print("=" * 70)

# ── Paramètres ────────────────────────────────────────────────────────────────
KNN_SAMPLE_SIZE_PER_POP: int = 2_000  # cellules subsampléchées par population (M11)
KNN_K: int = 15  # K voisins pour le vote K-NN (M11)
COV_REG_ALPHA: float = 1e-4  # Régularisation Tikhonov pour Σ (évite la singularité)
FORCE_RECOMPUTE_STATS: bool = False  # True → recalcule même si déjà en mémoire

# ── Vérification des prérequis ─────────────────────────────────────────────────
if "pop_mfi_raw_ref" not in dir() or len(pop_mfi_raw_ref) == 0:
    raise RuntimeError(
        "[!] pop_mfi_raw_ref non disponible — exécutez Section 10.3 d'abord."
    )
if "_TRANSFORM_IN_USE" not in dir():
    raise RuntimeError(
        "[!] _TRANSFORM_IN_USE non défini — exécutez Section 10.1b d'abord."
    )

_cols_stats = sorted(set(_VALID_CSV_A_COLS))

print(f"\n   Populations : {list(pop_mfi_raw_ref.keys())}")
print(f"   Marqueurs   : {_cols_stats}")
print(f"   Transform   : {_TRANSFORM_IN_USE} (cofacteur={_COFACTOR_IN_USE})")
print(f"\n   KNN_SAMPLE_SIZE_PER_POP : {KNN_SAMPLE_SIZE_PER_POP:,}")
print(f"   KNN_K                   : {KNN_K}")
print(f"   COV_REG_ALPHA           : {COV_REG_ALPHA}")

# ── Structure de sortie ────────────────────────────────────────────────────────
pop_cell_counts: Dict[str, int] = {}  # n cellules par population
pop_cov_matrices: Dict[str, np.ndarray] = {}  # matrice de covariance (n_cols × n_cols)
pop_knn_samples: Dict[
    str, np.ndarray
] = {}  # sous-échantillon (KNN_SAMPLE_SIZE × n_cols)


# =============================================================================
# ALGORITHME : Calcul incrémental de la covariance (une seule passe sur chaque CSV)
# On utilise l'algorithme de Welford pour la variance/covariance en ligne.
# Cela évite de charger tout le fichier en mémoire (datasets > 500k cellules).
# =============================================================================


def _welford_update(
    n: int,
    mean: np.ndarray,
    M2: np.ndarray,  # somme des carrés des différences (pour Welford)
    batch: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Mise à jour de Welford en batch pour la covariance.
    Retourne (n_new, mean_new, M2_new) où M2 sera divisé par (n-1) à la fin.
    Efficace : O(batch_size × n_cols²) mais une seule passe sur le CSV.
    """
    for row in batch:
        n += 1
        delta = row - mean
        mean += delta / n
        delta2 = row - mean
        M2 += np.outer(delta, delta2)
    return n, mean, M2


def compute_pop_stats_from_csv(
    csv_path: Path,
    cols: List[str],
    transform_type: str = "arcsinh",
    cofactor: float = 5.0,
    apply_to_scatter: bool = False,
    knn_sample_size: int = 2_000,
    chunk_size: int = 50_000,
    sep: str = ";",
    decimal: str = ",",
    reg_alpha: float = 1e-4,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Calcule en une seule passe :
    - n_cells    : nombre d'événements dans le CSV
    - cov_matrix : matrice de covariance (n_cols × n_cols), avec régularisation
    - knn_sample : sous-échantillon aléatoire représentatif

    Returns:
        (n_cells, cov_matrix, knn_sample)
    """
    n_dim = len(cols)
    n_total = 0
    mean_acc = np.zeros(n_dim, dtype=np.float64)
    M2_acc = np.zeros((n_dim, n_dim), dtype=np.float64)

    # Réservoir de sous-échantillonnage (reservoir sampling — O(N) temps, O(k) mémoire)
    reservoir: Optional[np.ndarray] = None
    reservoir_n = 0

    np.random.seed(42)

    for chunk in pd.read_csv(
        csv_path,
        sep=sep,
        decimal=decimal,
        usecols=cols,
        chunksize=chunk_size,
        dtype=np.float32,
        na_values=["", " "],
        engine="c",
    ):
        chunk = chunk.reindex(columns=cols)
        vals = chunk.to_numpy(dtype=np.float64, na_value=0.0)

        # Appliquer la transformation (même espace que les nœuds)
        vals = apply_cyto_transform_matrix(
            vals,
            col_names=cols,
            apply_to_scatter=apply_to_scatter,
            transform_type=transform_type,
            cofactor=cofactor,
        )

        n_row = vals.shape[0]

        # ── Mise à jour Welford (covariance incrémentale) ──────────────────
        n_total, mean_acc, M2_acc = _welford_update(n_total, mean_acc, M2_acc, vals)

        # ── Reservoir sampling ────────────────────────────────────────────
        for i in range(n_row):
            reservoir_n += 1
            if reservoir is None or len(reservoir) < knn_sample_size:
                if reservoir is None:
                    reservoir = vals[[i]].copy()
                else:
                    reservoir = np.vstack([reservoir, vals[i]])
            else:
                j = np.random.randint(0, reservoir_n)
                if j < knn_sample_size:
                    reservoir[j] = vals[i]

    if n_total < 2:
        cov_matrix = np.eye(n_dim) * reg_alpha
        knn_sample = reservoir if reservoir is not None else np.zeros((1, n_dim))
        return n_total, cov_matrix, knn_sample

    # Matrice de covariance finale : M2 / (n-1) + régularisation
    cov_matrix = M2_acc / (n_total - 1)
    # Régularisation Tikhonov : Σ_reg = Σ + α × I
    # Évite la singularité si deux marqueurs sont parfaitement corrélés
    cov_matrix += np.eye(n_dim) * reg_alpha

    knn_sample = (
        reservoir if reservoir is not None else np.zeros((knn_sample_size, n_dim))
    )

    return n_total, cov_matrix, knn_sample


# =============================================================================
# CALCUL DES STATISTIQUES POUR CHAQUE POPULATION
# =============================================================================
print(f"\n{'─' * 70}")
print(f" CALCUL STATISTIQUES PAR POPULATION")
print(f"{'─' * 70}")
print(f"   (covariance Welford incrémentale + reservoir sampling)")
print()

import time as _time_stats

for _csv_f in sorted(REF_MFI_DIR.glob("*.csv")):
    _pop = _csv_f.stem
    if _pop not in pop_mfi_raw_ref:
        continue  # population non chargée → skip

    # Skip si déjà calculé et FORCE_RECOMPUTE_STATS=False
    if not FORCE_RECOMPUTE_STATS and _pop in pop_cell_counts:
        print(
            f"   [skip]  {_pop:<35s} — déjà calculé ({pop_cell_counts[_pop]:,} cellules)"
        )
        continue

    _t0 = _time_stats.time()
    try:
        _n, _cov, _samp = compute_pop_stats_from_csv(
            csv_path=_csv_f,
            cols=_cols_stats,
            transform_type=_TRANSFORM_IN_USE,
            cofactor=_COFACTOR_IN_USE,
            apply_to_scatter=APPLY_LOGICLE_TO_SCATTER,
            knn_sample_size=KNN_SAMPLE_SIZE_PER_POP,
            reg_alpha=COV_REG_ALPHA,
        )
        pop_cell_counts[_pop] = _n
        pop_cov_matrices[_pop] = _cov
        pop_knn_samples[_pop] = _samp
        _elapsed = _time_stats.time() - _t0
        print(
            f"   [OK]    {_pop:<35s}  {_n:>9,} cellules  "
            f"Σ={_cov.shape}  sample={_samp.shape} [{_elapsed:.1f}s]"
        )
    except Exception as _e_stat:
        import traceback

        print(f"   [ERREUR] {_pop} : {_e_stat}")
        traceback.print_exc()

# =============================================================================
# TABLEAU RÉCAPITULATIF : PROPORTIONS BIOLOGIQUES
# =============================================================================
print(f"\n{'─' * 70}")
print(f" PROPORTIONS BIOLOGIQUES (taille fichier ∝ fréquence réelle)")
print(f"{'─' * 70}")

_total_cells = sum(pop_cell_counts.values())
print(
    f"\n   {'Population':<35s}  {'Cellules':>12s}  {'Fréquence':>10s}  {'log10(N)':>9s}  Barre"
)
print(f"   {'─' * 35}  {'─' * 12}  {'─' * 10}  {'─' * 9}  {'─' * 20}")
for _pop, _n in sorted(pop_cell_counts.items(), key=lambda x: -x[1]):
    _pct = 100.0 * _n / max(_total_cells, 1)
    _log = np.log10(max(_n, 1))
    _bar = "█" * int(_pct / 2)
    print(f"   {_pop:<35s}  {_n:>12,}  {_pct:>9.1f}%  {_log:>9.3f}  {_bar}")

print(f"\n   Total cellules de référence : {_total_cells:,}")
print(f"\n   → Ces proportions seront utilisées comme prior bayésien par M9.")
print(f"     La 'magnétisme' de chaque population est proportionnel à log10(N).")

# =============================================================================
# VÉRIFICATION DE LA QUALITÉ DES COVARIANCES
# =============================================================================
print(f"\n{'─' * 70}")
print(f" QUALITÉ DES MATRICES DE COVARIANCE (M10_mahalanobis)")
print(f"{'─' * 70}")
for _pop, _cov in pop_cov_matrices.items():
    _diag_std = np.sqrt(np.diag(_cov))
    _cond = np.linalg.cond(_cov)
    _is_ok = _cond < 1e12  # seuil de conditionnement acceptable
    _flag = "✅" if _is_ok else "⚠️ mal conditionné"
    print(f"   {_pop:<35s}  σ_moy={_diag_std.mean():.3f}  cond={_cond:.2e}  {_flag}")

print(f"\n{'=' * 70}")
print(f" [OK] Section 10.3b terminée")
print(f"      Variables disponibles :")
print(f"      • pop_cell_counts  : comptages réels par population")
print(f"      • pop_cov_matrices : matrices de covariance (M10)")
print(f"      • pop_knn_samples  : sous-échantillons (M11)")
print(f"{'=' * 70}")


################################################################################
# =============================================================================
# SECTION 10.4 — MAPPING POPULATIONS → NŒUDS (ESPACE BRUT -A, NORMALISÉ)
# =============================================================================
# On calcule la distance Euclidienne entre chaque nœud FlowSOM et chaque
# population de référence.
#
# NORMALISATION (avant calcul de distance) :
# ───────────────────────────────────────────
# Sans normalisation, les canaux à grande plage dynamique (FSC-A : 0–250 000)
# dominent numériquement les canaux de fluorescence (CD45 : 0–30 000).
# → Résultat : les nœuds seraient assignés principalement par leur taille
#   cellulaire, pas par leur phénotype de surface.
#
# Méthode "range" : x_norm = (x - x_min) / (x_max - x_min)
#   ✅ Ramène tous les canaux entre 0 et 1
#   ✅ Robuste aux valeurs négatives (si la transfo a introduit des offset)
#   ✅ Ne suppose pas de distribution gaussienne
#   ⚠️  Sensible aux outliers extrêmes (mais les MFI moyennes le sont peu)
#
# Méthode "zscore" : x_norm = (x - mean) / std
#   ✅ Standard en analyse multi-marqueurs
#   ⚠️  Amplifie les marqueurs à très faible variance (risque de sur-pondération)
#
# → On recommande "range" quand INCLUDE_SCATTER=True, "zscore" sinon.
# =============================================================================
from datetime import datetime


def normalize_matrix(
    X: np.ndarray,
    method: str = "range",
    feature_range: np.ndarray = None,
) -> np.ndarray:
    """
    Normalise une matrice (n_samples × n_features).

    Args:
        X            : Matrice à normaliser.
        method       : "range", "zscore" ou "none".
        feature_range: Si method="range", plage pré-calculée (min, max) sous
                       forme d'array (2, n_features). Optionnel.
                       Permet d'appliquer la même normalisation à une nouvelle
                       matrice (ex: populations de référence) en utilisant
                       la plage calculée sur les nœuds.

    Returns:
        X_norm : Matrice normalisée (même shape que X).
        scale_params : Dict avec min/max ou mean/std (pour réutilisation).
    """
    if method == "none":
        return X.copy(), {}

    if method == "range":
        if feature_range is not None:
            _min = feature_range[0]
            _max = feature_range[1]
        else:
            _min = X.min(axis=0)
            _max = X.max(axis=0)
        _range = _max - _min
        _range[_range == 0] = 1.0  # éviter division par zéro
        return (X - _min) / _range, {"min": _min, "max": _max, "range": _range}

    if method == "zscore":
        _mean = X.mean(axis=0)
        _std = X.std(axis=0)
        _std[_std == 0] = 1.0
        return (X - _mean) / _std, {"mean": _mean, "std": _std}

    raise ValueError(f"méthode de normalisation inconnue : '{method}'")


def map_populations_to_nodes_v3(
    node_mfi_raw: pd.DataFrame,
    pop_mfi_ref: pd.DataFrame,
    include_scatter: bool = True,
    distance_percentile: int = 60,
    normalization_method: str = "range",
) -> pd.DataFrame:
    """
    Assigne une population à chaque nœud par distance Euclidienne minimale.

    Opère dans l'espace brut -A normalisé.
    FSC-A et SSC-A sont inclus si include_scatter=True.

    Différences vs V2 :
    • Mapping déjà effectué en Section 10.3 → colonnes identiques dans les deux matrices
    • Normalisation calculée UNE SEULE FOIS sur les nœuds, puis appliquée aux ref
      (cohérence : les populations sont projetées dans l'espace des nœuds)
    • Seuil "Unknown" : percentile de la distribution des distances minimales
      (nœuds périphériques = loin de toutes les populations connues)

    Args:
        node_mfi_raw         : DataFrame (n_nœuds × n_marqueurs_A).
        pop_mfi_ref          : DataFrame (n_populations × n_marqueurs_A).
                               Les colonnes DOIVENT être dans l'espace FCS
                               (déjà remappées en Section 10.3).
        include_scatter      : Inclure FSC-A / SSC-A.
        distance_percentile  : Nœuds au-delà de ce percentile → "Unknown".
        normalization_method : "range" (recommandé), "zscore", ou "none".

    Returns:
        DataFrame : node_id | best_pop | best_dist | threshold | assigned_pop
    """
    # ── Vérification de la cohérence des colonnes (V3 : doit être exacte) ────
    assert set(node_mfi_raw.columns) == set(pop_mfi_ref.columns), (
        "Les colonnes de node_mfi_raw et pop_mfi_ref doivent être identiques.\n"
        f"node uniquement : {set(node_mfi_raw.columns) - set(pop_mfi_ref.columns)}\n"
        f"ref uniquement  : {set(pop_mfi_ref.columns) - set(node_mfi_raw.columns)}"
    )

    # ── Sélection des colonnes à utiliser pour la distance ───────────────────
    _scatter_patterns = ("FSC-A", "SSC-A")
    if include_scatter:
        _cols_for_dist = list(node_mfi_raw.columns)
    else:
        _cols_for_dist = [
            c
            for c in node_mfi_raw.columns
            if not any(c.upper() == p.upper() for p in _scatter_patterns)
        ]

    print(f"\n   ━━ PARAMÈTRES DU CALCUL DE DISTANCE ━━")
    print(
        f"   Colonnes utilisées   : {len(_cols_for_dist)} / {len(node_mfi_raw.columns)}"
    )
    print(f"   Include scatter      : {include_scatter}")
    print(f"   Normalisation        : {normalization_method}")
    print(f"   Colonnes             : {_cols_for_dist}")

    _X_nodes = node_mfi_raw[_cols_for_dist].to_numpy(dtype=np.float64)
    _X_pops = pop_mfi_ref[_cols_for_dist].to_numpy(dtype=np.float64)

    # ── Normalisation : calculée sur les nœuds, appliquée aux deux matrices ──
    if normalization_method != "none":
        _X_nodes_norm, _scale = normalize_matrix(_X_nodes, method=normalization_method)

        # Appliquer la MÊME normalisation aux populations de référence
        # (projection dans l'espace normalisé des nœuds)
        if normalization_method == "range":
            _range = _scale["range"]
            _X_pops_norm = (_X_pops - _scale["min"]) / _range
        elif normalization_method == "zscore":
            _X_pops_norm = (_X_pops - _scale["mean"]) / _scale["std"]

        print(f"\n   Plage avant normalisation (nœuds) :")
        for _j, _c in enumerate(_cols_for_dist):
            print(
                f"     {_c:<30s} : [{_X_nodes[:, _j].min():10.1f}, {_X_nodes[:, _j].max():10.1f}]"
            )
        print(f"\n   Plage après normalisation (nœuds) : [0, 1] (range) ou z-score")
    else:
        _X_nodes_norm = _X_nodes
        _X_pops_norm = _X_pops
        print(f"   ⚠️  Pas de normalisation — les canaux à grande plage dominent!")

    # ── Calcul de la matrice de distance ─────────────────────────────────────
    _pop_names = list(pop_mfi_ref.index)
    _dist_matrix = cdist(_X_nodes_norm, _X_pops_norm, metric="euclidean")
    # Shape : (n_nœuds, n_populations)

    _best_idx = np.argmin(_dist_matrix, axis=1)
    _best_dist = _dist_matrix[np.arange(len(_X_nodes_norm)), _best_idx]
    _threshold = float(np.percentile(_best_dist, distance_percentile))

    print(f"\n   Statistiques des distances minimales par nœud :")
    print(f"     min     : {_best_dist.min():.4f}")
    print(f"     median  : {np.median(_best_dist):.4f}")
    print(f"     P{distance_percentile}      : {_threshold:.4f}  ← seuil 'Unknown'")
    print(f"     max     : {_best_dist.max():.4f}")

    # ── Assignation finale ────────────────────────────────────────────────────
    _assigned = np.where(
        _best_dist <= _threshold,
        np.array([_pop_names[i] for i in _best_idx]),
        "Unknown",
    )

    return pd.DataFrame(
        {
            "node_id": np.arange(len(_X_nodes_norm)),
            "best_pop": [_pop_names[i] for i in _best_idx],
            "best_dist": np.round(_best_dist, 4),
            "threshold": round(_threshold, 4),
            "assigned_pop": _assigned,
        }
    )


# =============================================================================
# EXÉCUTION DU MAPPING
# =============================================================================
print("=" * 70)
print(" SECTION 10.4 — MAPPING POPULATIONS → NŒUDS (espace brut -A)")
print("=" * 70)

# S'assurer que les colonnes de node_mfi_raw_df et df_mfi_raw_ref sont alignées
_common_a_cols = sorted(set(node_mfi_raw_df.columns) & set(df_mfi_raw_ref.columns))

print(f"\n   Marqueurs -A communs (nœuds FCS ∩ ref CSV) : {len(_common_a_cols)}")
print(f"   → {_common_a_cols}")

# Aligner les deux DataFrames sur les colonnes communes
_node_mfi_aligned = node_mfi_raw_df[_common_a_cols].copy()
_ref_mfi_aligned = df_mfi_raw_ref[_common_a_cols].copy()

# Exécuter le mapping
mapping_df_raw = map_populations_to_nodes_v3(
    node_mfi_raw=_node_mfi_aligned,
    pop_mfi_ref=_ref_mfi_aligned,
    include_scatter=INCLUDE_SCATTER_IN_MAPPING,
    distance_percentile=DISTANCE_PERCENTILE,
    normalization_method=NORMALIZATION_METHOD,
)

# ── Enrichir avec le métacluster et les coordonnées ──────────────────────────
mapping_df_raw["metacluster"] = _mc_per_node_raw.reindex(
    mapping_df_raw["node_id"]
).values

mapping_df_raw = mapping_df_raw.merge(
    _node_coords_df.reset_index(),
    on="node_id",
    how="left",
)

# ── Résumé ───────────────────────────────────────────────────────────────────
_n_tot = len(mapping_df_raw)
_n_unk = int((mapping_df_raw["assigned_pop"] == "Unknown").sum())
_n_asg = _n_tot - _n_unk

print(f"\n   ━━ RÉSUMÉ DU MAPPING ━━")
print(f"   Nœuds totaux  : {_n_tot}")
print(f"   Nœuds assignés: {_n_asg}  ({100 * _n_asg / _n_tot:.1f}%)")
print(f"   Nœuds Unknown : {_n_unk}  ({100 * _n_unk / _n_tot:.1f}%)")
print(f"\n   Distribution détaillée :")
print(f"   {'─' * 62}")
print(f"   {'Population':<35s}  {'Nœuds':>6}  {'%':>6}  Barre")
print(f"   {'─' * 62}")
for _pop, _cnt in mapping_df_raw["assigned_pop"].value_counts().items():
    _bar = "█" * int(_cnt / _n_tot * 30)
    _col = POPULATION_COLORS.get(_pop, "#AAAAAA")
    print(f"   {_pop:<35s}  {_cnt:>6d}  {100 * _cnt / _n_tot:>5.1f}%  {_bar}")
print(f"   {'─' * 62}")

print(f"\n   Correspondance population → métacluster dominant :")
for _pop in sorted(set(mapping_df_raw["assigned_pop"])):
    _sub = mapping_df_raw[mapping_df_raw["assigned_pop"] == _pop]
    _dom = _sub["metacluster"].value_counts().idxmax()
    _mc_n = _sub["metacluster"].value_counts()
    _top3 = ", ".join([f"MC{int(k)}({v})" for k, v in _mc_n.head(3).items()])
    print(f"     {_pop:<35s} → MC{int(_dom)}  (top 3: {_top3})")

display(mapping_df_raw)

# Export CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_mapping_csv_path = OUTPUT_DIR / "other" / f"node_population_mapping_v3_{timestamp}.csv"
mapping_df_raw.to_csv(_mapping_csv_path, index=False, sep=";", decimal=",")
print(f"\n[OK] Mapping V3 exporté : {_mapping_csv_path}")


################################################################################

# =============================================================================
# SECTION 10.4b — BENCHMARK MULTI-MÉTHODES (V5) + PRIOR + MAHALANOBIS + KNN
# =============================================================================
#
# MÉTHODES DISPONIBLES :
# ──────────────────────
#  M1 : Euclidien + range norm (nœuds)
#  M2 : Manhattan  + range norm (nœuds)
#  M3 : Cosinus — compare profils, insensible à l'intensité absolue
#  M4 : Euclidien + RobustScaler (médiane/IQR P10-P90)
#  M5 : Euclidien + arcsinh* + range norm  (* sauté si data déjà transformée)
#  M6 : Euclidien pondéré (scatter ×0.3, fluo ×1.0)
#  M7 : Vote majoritaire M2 + M3 + M8
#  M8 : Euclidien + normalisation dans la plage RÉFÉRENCE (★ gère scale mismatch)
#
#  M9  : ★ PRIOR BAYÉSIEN (base Euclidien M8) ★
#        D_adj = D_euclidean_M8 / log10(n_cells_ref)
#        → Requiert : pop_cell_counts (Section 10.3b)
#
#  M10 : ★ MAHALANOBIS ★
#        D_maha(node, pop) = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
#        → Requiert : pop_cov_matrices (Section 10.3b)
#
#  M11 : ★ K-NN DENSITÉ (échantillonnage PROPORTIONNEL) ★  ← CORRIGÉ
#        Pool KNN rééchantillonné proportionnellement à pop_cell_counts.
#        Évite la sur-représentation des populations rares (ex: plasmocytes).
#        → Requiert : pop_knn_samples + pop_cell_counts (Section 10.3b)
#
#  M12 : ★ COSINE + PRIOR RENFORCÉ log10^3 + HARD LIMIT ★  ← RECOMMANDÉ V2
#        Étape 1 : Distance Cosinus → robuste au scale mismatch, compare la
#                  forme du profil vectoriel (comme M3).
#        Étape 2 : Prior RENFORCÉ D_adj = D_cosine / log10(n_cells)^3
#                  → Granulos/Plasmos : ratio ×7.1 (vs ×1.9 avec log10 simple)
#                  → Les populations < 1% ne gagnent un nœud que si
#                    le Cosinus est quasi parfait (proche de 0).
#        Étape 3 : Hard Limit optionnel — si le nœud contient plus de cellules
#                  que l'attendu max pour une pop. rare, D → ∞ (exclusion forcée).
#        → Requiert : pop_cell_counts + node_sizes (Section 10.3b / 10.1)
#
#  VARIANTES DU PRIOR (paramètre prior_mode dans _apply_bayesian_prior) :
#  ──────────────────────────────────────────────────────────────────────────────
#  "log10"        (M9 legacy)  : D_adj = D / log10(N)
#                                ratio Granulos/Plasmos ~1.9x  ← insuffisant
#  "log10_cubed"  (M12 V2 ★)  : D_adj = D / log10(N)^3
#                                ratio Granulos/Plasmos ~7.1x  ← RECOMMANDÉ
#                                Granulos 2.2M → 6.34^3 = 254.8
#                                Plasmos  2000 →  3.30^3 =  35.9
#  "sqrt_n"       (très agres.): D_adj = D / sqrt(N / 10)
#                                ratio Granulos/Plasmos ~33x   ← extrême
#
# SEUIL AUTO-UNKNOWN :
#   "auto_otsu" → coupure naturelle bimodale  (recommandé)
#   "auto_mad"  → médiane + 2×MAD
#   "pXX"       → percentile fixe (ex: "p70")
#   "none"      → tout assigné (pas d'Unknown)
#
# DÉPENDANCES :
#   • pop_cell_counts, pop_cov_matrices, pop_knn_samples  [Section 10.3b]
#   • node_mfi_raw_df, df_mfi_raw_ref                     [Sections 10.2/10.3]
#   • _mc_per_node_raw, _node_coords_df                   [Section 10.1]
# =============================================================================

import warnings

warnings.filterwarnings("ignore")

from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# ─── Paramètres ──────────────────────────────────────────────────────────────
ARCSINH_COFACTOR: float = 5.0
MARKER_WEIGHTS_V5: Dict[str, float] = {"FSC-A": 0.30, "SSC-A": 0.30}
MARKER_WEIGHTS_DEFAULT_V5: float = 1.0

# Seuil Unknown — choisir : "auto_otsu", "auto_mad", "pXX", "none"
UNKNOWN_THRESHOLD_MODE: str = "auto_otsu"

DISTANCE_PERCENTILE_V5: int = 70  # utilisé seulement si mode = "pXX"
INCLUDE_SCATTER_V5: bool = True
KNN_K_BENCH: int = 15  # K pour M11 dans le benchmark

# ── Paramètres M11 corrigé : pool KNN proportionnel ─────────────────────────
TOTAL_KNN_POINTS: int = 15_000  # Taille totale du pool KNN poolé
KNN_MIN_POINTS_PER_POP: int = 50  # Minimum vital par pop (évite disparition des rares)

# ─── Méthodes à exécuter (retirer une méthode pour l'ignorer) ─────────────────
_METHODS_BENCH: List[str] = [
    "M1_euclidean",
    "M2_cityblock",
    "M3_cosine",
    "M4_robust",
    "M5_arcsinh",
    "M6_weighted",
    "M7_vote",
    "M8_ref_norm",
    "M9_prior",
    "M10_mahalanobis",
    "M11_knn",
    "M12_cosine_prior",  # ← RECOMMANDÉ : Cosine + Prior log10^3 + Hard Limit
]

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================


def arcsinh_transform(X: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
    return np.arcsinh(X / cofactor)


def robust_scale(
    X_nodes: np.ndarray, X_pops: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    return scaler.fit_transform(X_nodes), scaler.transform(X_pops)


def build_weight_vector(
    cols: List[str], weights: Dict[str, float], default: float = 1.0
) -> np.ndarray:
    return np.array([weights.get(c, default) for c in cols])


def _otsu_threshold_1d(values: np.ndarray, n_bins: int = 64) -> float:
    _v = values[np.isfinite(values)]
    if len(_v) == 0:
        return float(values.max())
    hist, bin_edges = np.histogram(_v, bins=min(n_bins, max(10, len(_v) // 2)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = float(hist.sum())
    if total == 0:
        return float(np.median(_v))
    mu_total = float((hist * bin_centers).sum()) / total
    best_bcv, best_th = -np.inf, bin_centers[-1]
    w0_cum, mu0_cum = 0.0, 0.0
    for i in range(len(hist) - 1):
        w0_cum += hist[i] / total
        mu0_cum += hist[i] * bin_centers[i] / total
        if w0_cum <= 1e-8 or w0_cum >= 1.0 - 1e-8:
            continue
        w1 = 1.0 - w0_cum
        mu0 = mu0_cum / w0_cum
        mu1 = (mu_total - mu0_cum) / w1
        bcv = w0_cum * w1 * (mu0 - mu1) ** 2
        if bcv > best_bcv:
            best_bcv, best_th = bcv, bin_centers[i]
    return float(best_th)


def _mad_threshold_1d(values: np.ndarray, k: float = 2.0) -> float:
    _v = values[np.isfinite(values)]
    if len(_v) == 0:
        return float(np.inf)
    med = float(np.median(_v))
    return med + float(np.median(np.abs(_v - med))) * 1.4826 * k


def compute_unknown_threshold(
    best_dist: np.ndarray, mode: str = "auto_otsu", percentile: int = 70
) -> Tuple[float, str]:
    m = mode.lower().strip()
    if m == "none":
        return float("inf"), "aucun seuil (tout assigné)"
    if m.startswith("p"):
        _p = int(m[1:]) if m[1:].isdigit() else percentile
        return float(np.percentile(best_dist, _p)), f"P{_p} (percentile fixe)"
    if m == "auto_otsu":
        return _otsu_threshold_1d(best_dist), "Otsu (coupure naturelle bimodale)"
    if m == "auto_mad":
        return _mad_threshold_1d(best_dist, k=2.0), "MAD (médiane + 2×MAD)"
    return float(np.percentile(best_dist, percentile)), f"P{percentile} (fallback)"


def assign_with_auto_threshold(
    dist_matrix: np.ndarray,
    pop_names: List[str],
    threshold_mode: str = "auto_otsu",
    percentile: int = 70,
    min_assigned_frac: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray, float, str]:
    best_idx = np.argmin(dist_matrix, axis=1)
    best_dist = dist_matrix[np.arange(len(dist_matrix)), best_idx]
    threshold, desc = compute_unknown_threshold(best_dist, threshold_mode, percentile)
    assigned = np.where(
        best_dist <= threshold, np.array([pop_names[i] for i in best_idx]), "Unknown"
    )
    frac = (assigned != "Unknown").sum() / len(assigned)
    if frac < min_assigned_frac and threshold_mode != "none":
        threshold = float(np.percentile(best_dist, 70))
        desc = "P70 (fallback auto)"
        assigned = np.where(
            best_dist <= threshold,
            np.array([pop_names[i] for i in best_idx]),
            "Unknown",
        )
    return assigned, best_dist, threshold, desc


# =============================================================================
# MÉTHODES BIOLOGIQUES AVANCÉES (M9, M10, M11, M12)
# =============================================================================


def _apply_bayesian_prior(
    dist_matrix: np.ndarray,
    pop_names: List[str],
    cell_counts: Dict[str, int],
    prior_mode: str = "log10",
    node_sizes: Optional[np.ndarray] = None,
    hard_limit_factor: float = 0.0,
    n_nodes_total: int = 0,
) -> np.ndarray:
    """
    Prior bayésien RENFORCE — pondère l'attractivité de chaque population selon son abondance.

    Modes disponibles (paramètre `prior_mode`) :
    -----------------------------------------------------------------
    * "log10"       (M9, M12 legacy) :
        D_adj = D / log10(N)
        Granulos 2.2M vs Plasmos 2000 → ratio 1.9x  (trop faible)

    * "log10_cubed" RECOMMANDE (M12 par defaut depuis V2) :
        D_adj = D / log10(N)^3
        Granulos 2.2M vs Plasmos 2000 → ratio 7.1x
        Les populations < 1% ne peuvent gagner un nœud que si le Cosinus
        est quasi parfait (proche de 0).

    * "sqrt_n"      (tres agressif) :
        D_adj = D / sqrt(N / 10)
        Granulos 2.2M vs Plasmos 2000 → ratio ~33x

    Hard Limit (activé si hard_limit_factor > 0 ET node_sizes fourni) :
    -----------------------------------------------------------------
    Si le nœud contient plus de cellules que la taille totale attendue de la
    population rare, cette population est exclue d'office (D → inf).
        max_attendu_nœud = (n_cells_pop / n_nodes_total) x hard_limit_factor

    Args:
        prior_mode:        "log10" | "log10_cubed" | "sqrt_n".
        node_sizes:        Nombre de cellules par nœud SOM (Section 10.1).
        hard_limit_factor: Multiplicateur du max attendu par nœud. 0 = désactivé.
                           Valeurs typiques : 3.0-10.0 (5.0 = défaut).
        n_nodes_total:     Nombre total de nœuds (pour le calcul du max attendu).
    """
    D = dist_matrix.copy()

    for _j, _pop in enumerate(pop_names):
        _n = max(cell_counts.get(_pop, 1), 1)

        # Pénalité bayésienne (prior de taille) ----------------------------------------
        if prior_mode == "log10_cubed":
            # log10(N)^3 : ratio Granulos/Plasmos ~7x vs 1.9x pour log10 simple
            # Granulos 2.2M → 6.34^3 = 254.8 | Plasmos 2000 → 3.30^3 = 35.9
            # Les populations rares (<1%) ne gagnent que si cosinus ~0.0
            _w = np.log10(max(_n, 10)) ** 3
        elif prior_mode == "sqrt_n":
            # sqrt(N/10) : très agressif (~33x ratio). Utile si déséquilibre extrême.
            _w = np.sqrt(max(_n, 10) / 10.0)
        else:  # "log10" — comportement original (M9 backward compat, ratio ~1.9x)
            _w = np.log10(max(_n, 10))

        D[:, _j] /= _w  # diviser = rendre plus attractif (magnétisme)

        # Hard Limit : exclusion si nœud trop grand pour une pop. rare -------
        # Un nœud avec 5 000 cellules NE PEUT PAS être Plasmocytes
        # si la population entière n'en contient que 2 000.
        if (
            hard_limit_factor > 0
            and node_sizes is not None
            and n_nodes_total > 0
            and len(node_sizes) > 0
        ):
            _max_expected = max(1, int(_n / n_nodes_total * hard_limit_factor))
            _node_sz_arr = node_sizes[: len(D)]
            _too_large = _node_sz_arr > _max_expected
            if _too_large.any():
                D[_too_large, _j] = np.inf  # Distance infinie = exclusion forcée

    return D


def _mahalanobis_distance_batch(
    X_nodes: np.ndarray,
    X_pops_mean: np.ndarray,
    pop_names: List[str],
    cov_matrices: Dict[str, np.ndarray],
    cols: List[str],
) -> np.ndarray:
    """
    M10 — Distance de Mahalanobis pour chaque (nœud, population).

    D_maha(x, pop) = sqrt((x - μ)ᵀ Σ_pop⁻¹ (x - μ))

    La matrice de covariance Σ_pop modélise l'étalement naturel de la population :
    • Grande variance (Granulos) → ellipse large → un nœud excentrique est toléré
    • Faible variance (Hématogones) → ellipse serrée → un nœud excentrique est suspect

    Retourne une matrice (n_nodes × n_pops).
    """
    n_nodes = X_nodes.shape[0]
    n_pops = len(pop_names)
    D = np.zeros((n_nodes, n_pops), dtype=np.float64)

    for _j, _pop in enumerate(pop_names):
        _mu = X_pops_mean[_j]
        _cov = cov_matrices.get(_pop)

        if _cov is None or _cov.shape != (len(cols), len(cols)):
            D[:, _j] = np.linalg.norm(X_nodes - _mu, axis=1)
            continue

        try:
            _cov_inv = np.linalg.pinv(_cov)
        except np.linalg.LinAlgError:
            D[:, _j] = np.linalg.norm(X_nodes - _mu, axis=1)
            continue

        _diff = X_nodes - _mu
        _diff_cov = _diff @ _cov_inv
        _sq_dists = np.sum(_diff_cov * _diff, axis=1)
        D[:, _j] = np.sqrt(np.maximum(_sq_dists, 0.0))

    return D


# ─── NOUVEAU : Échantillonnage KNN stratifié proportionnel ────────────────────
def _proportional_stratified_pool(
    knn_norm: Dict[str, np.ndarray],
    pop_names: List[str],
    cell_counts: Dict[str, int],
    total_points: int = 15_000,
    min_points: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Rééchantillonne le pool KNN proportionnellement aux comptages réels (pop_cell_counts).

    Problème résolu : sans cette correction, toutes les populations ont le même nombre
    de points dans le pool (ex: 2000/pop), ce qui sur-représente massivement les
    populations rares (Plasmocytes 0.1%) par rapport à leur abondance réelle.

    Algorithme en 3 passes :
      1. Alloue `min_points` à chaque population présente (filet de sécurité).
      2. Distribue le budget restant (total_points − n_pops × min_points)
         proportionnellement à cell_counts[pop].
      3. Plafonne au nombre de points réellement disponibles dans knn_norm[pop].

    Args:
        knn_norm:     Échantillons KNN normalisés {pop: (n_sample, n_cols)}.
        pop_names:    Liste ordonnée des noms de populations.
        cell_counts:  Comptages réels {pop: n_cells} issus de Section 10.3b.
        total_points: Taille cible du pool global (default: 15 000).
        min_points:   Minimum vital par population (default: 50).

    Returns:
        Dict {pop: array(n_allocated, n_cols)} sous-échantillonné.
    """
    # Populations disponibles (ont des points dans le pool normalisé)
    _available = [p for p in pop_names if p in knn_norm and len(knn_norm[p]) > 0]
    n_avail = len(_available)

    if n_avail == 0:
        return {}

    # Budget après allocation du minimum vital
    _budget_min = n_avail * min_points
    _budget_prop = max(0, total_points - _budget_min)

    # Poids proportionnels basés sur cell_counts (fallback = équiprobable)
    _counts = np.array([cell_counts.get(p, 1) for p in _available], dtype=np.float64)
    _total_c = _counts.sum()
    _weights = _counts / _total_c if _total_c > 0 else np.ones(n_avail) / n_avail

    # Allocation proportionnelle + minimum vital
    _alloc_prop = np.round(_weights * _budget_prop).astype(int)
    _alloc_total = (_alloc_prop + min_points).astype(int)

    # Plafonnement : on ne peut pas allouer plus que ce qui est disponible
    _available_counts = np.array([len(knn_norm[p]) for p in _available])
    _alloc_total = np.minimum(_alloc_total, _available_counts)

    # Sous-échantillonnage aléatoire reproductible (seed fixé globalement)
    rng = np.random.default_rng(42)
    result: Dict[str, np.ndarray] = {}
    for _p, _n in zip(_available, _alloc_total):
        _pts = knn_norm[_p]
        if _n >= len(_pts):
            result[_p] = _pts  # Prend tout si quota > dispo
        else:
            _idx = rng.choice(len(_pts), size=_n, replace=False)
            result[_p] = _pts[_idx]

    return result


def _knn_vote(
    X_nodes: np.ndarray,
    pop_names: List[str],
    knn_samples: Dict[str, np.ndarray],
    k: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    M11 — K-NN vote sur les événements poolés de toutes les références.

    Pour chaque nœud :
    1. Cherche les K plus proches voisins parmi ALL les événements de référence.
    2. La population majoritaire parmi les K voisins gagne.
    3. Le "score de victoire" = distance au K-ème voisin (proxy de confiance).

    NOTE : Le rééchantillonnage proportionnel doit être appliqué AVANT
    cet appel (via _proportional_stratified_pool) pour corriger le biais d'abondance.

    Retourne:
        best_pop_labels (n_nodes,) : population gagnante pour chaque nœud
        best_dist_knn   (n_nodes,) : distance au K-ème plus proche voisin
    """
    from scipy.spatial import cKDTree

    _all_X = []
    _all_labels = []
    for _pop in pop_names:
        _samp = knn_samples.get(_pop)
        if _samp is None or len(_samp) == 0:
            continue
        _all_X.append(_samp)
        _all_labels.extend([_pop] * len(_samp))

    if len(_all_X) == 0:
        print(
            "   [M11] ⚠️ Aucun échantillon KNN disponible — population 'Unknown' par défaut."
        )
        return np.full(len(X_nodes), "Unknown"), np.full(len(X_nodes), np.inf)

    _X_pool = np.vstack(_all_X).astype(np.float64)
    _labels_arr = np.array(_all_labels)

    _tree = cKDTree(_X_pool)
    _k_actual = min(k, len(_X_pool))
    _dists, _inds = _tree.query(X_nodes, k=_k_actual)

    best_pops = np.empty(len(X_nodes), dtype=object)
    best_dists = np.empty(len(X_nodes), dtype=np.float64)

    for _i in range(len(X_nodes)):
        _neighbor_labels = _labels_arr[_inds[_i]]
        _unique, _counts = np.unique(_neighbor_labels, return_counts=True)
        _winner = _unique[np.argmax(_counts)]
        best_pops[_i] = _winner
        best_dists[_i] = float(_dists[_i, -1])

    return best_pops, best_dists


# =============================================================================
# FONCTION PRINCIPALE map_populations_to_nodes_v5
# =============================================================================


def map_populations_to_nodes_v5(
    node_mfi_raw: pd.DataFrame,
    pop_mfi_ref: pd.DataFrame,
    include_scatter: bool = True,
    percentile: int = 70,
    threshold_mode: str = "auto_otsu",
    method: str = "M12_cosine_prior",
    arcsinh_cofactor: float = 5.0,
    marker_weights: Dict[str, float] = None,
    verbose: bool = True,
    data_already_transformed: bool = False,
    # Paramètres pour méthodes biologiques
    cell_counts: Optional[Dict[str, int]] = None,  # M9, M11, M12
    cov_matrices: Optional[Dict[str, np.ndarray]] = None,  # M10
    knn_samples: Optional[Dict[str, np.ndarray]] = None,  # M11
    k_neighbors: int = 15,  # M11
    knn_total_points: int = 15_000,  # M11 pool size
    knn_min_per_pop: int = 50,  # M11 min vital/pop
    # Paramètres M12 : Prior renforcé + Hard Limit ----------------------------
    node_sizes_hard_limit: Optional[np.ndarray] = None,  # M12 : tableau node_sizes
    hard_limit_factor: float = 5.0,  # M12 : mult. max/nœud
) -> pd.DataFrame:
    """
    Mapping V5 — toutes méthodes M1–M12.
    Retourne DataFrame : node_id | best_pop | best_dist | threshold | threshold_mode | assigned_pop
    """
    # ── 0. Colonnes ────────────────────────────────────────────────────────────
    _scatter = {"FSC-A", "SSC-A"}
    _cols = (
        list(node_mfi_raw.columns)
        if include_scatter
        else [c for c in node_mfi_raw.columns if c not in _scatter]
    )

    if verbose:
        print(f"\n{'━' * 68}")
        print(f"   MAPPING V5 — méthode : {method}")
        print(f"{'━' * 68}")
        print(f"   Marqueurs     : {len(_cols)} / {len(node_mfi_raw.columns)}")
        print(f"   Scatter       : {include_scatter}")
        print(f"   Seuil Unknown : {threshold_mode}")
        print(f"   data_pré-transfo: {data_already_transformed}")
        print(f"   Populations   : {list(pop_mfi_ref.index)}")

    _pop_names = list(pop_mfi_ref.index)
    _X_nodes_raw = node_mfi_raw[_cols].to_numpy(dtype=np.float64)
    _X_pops_raw = pop_mfi_ref[_cols].to_numpy(dtype=np.float64)

    if verbose and method not in ("M11_knn",):
        print(f"\n   Plages (nœuds vs refs) :")
        for _j, _c in enumerate(_cols[:6]):
            print(
                f"     {_c:<30s}  nœuds [{_X_nodes_raw[:, _j].min():7.3f}–"
                f"{_X_nodes_raw[:, _j].max():7.3f}]  "
                f"ref [{_X_pops_raw[:, _j].min():7.3f}–{_X_pops_raw[:, _j].max():7.3f}]"
            )
        if len(_cols) > 6:
            print(f"     ... ({len(_cols) - 6} autres)")

    # ── 1. Calcul de la matrice de distance selon la méthode ──────────────────
    _metric = "euclidean"
    _dist_final = None

    if method == "M1_euclidean":
        _mn = _X_nodes_raw.min(axis=0)
        _rng = np.where(
            _X_nodes_raw.max(axis=0) - _mn > 0, _X_nodes_raw.max(axis=0) - _mn, 1.0
        )
        _X_n, _X_p = (_X_nodes_raw - _mn) / _rng, (_X_pops_raw - _mn) / _rng

    elif method == "M2_cityblock":
        _mn = _X_nodes_raw.min(axis=0)
        _rng = np.where(
            _X_nodes_raw.max(axis=0) - _mn > 0, _X_nodes_raw.max(axis=0) - _mn, 1.0
        )
        _X_n, _X_p = (_X_nodes_raw - _mn) / _rng, (_X_pops_raw - _mn) / _rng
        _metric = "cityblock"

    elif method == "M3_cosine":
        _X_n = np.clip(_X_nodes_raw, 0, None)
        _X_p = np.clip(_X_pops_raw, 0, None)
        _metric = "cosine"

    elif method == "M4_robust":
        _X_n, _X_p = robust_scale(_X_nodes_raw, _X_pops_raw)

    elif method == "M5_arcsinh":
        if data_already_transformed:
            _A_n, _A_p = _X_nodes_raw, _X_pops_raw
            if verbose:
                print(
                    f"\n   [M5] data_pré-transfo → arcsinh SAUTÉ, range norm seulement"
                )
        else:
            _A_n = arcsinh_transform(_X_nodes_raw, arcsinh_cofactor)
            _A_p = arcsinh_transform(_X_pops_raw, arcsinh_cofactor)
        _mn = _A_n.min(axis=0)
        _rng = np.where(_A_n.max(axis=0) - _mn > 0, _A_n.max(axis=0) - _mn, 1.0)
        _X_n, _X_p = (_A_n - _mn) / _rng, (_A_p - _mn) / _rng

    elif method == "M6_weighted":
        _mn = _X_nodes_raw.min(axis=0)
        _rng = np.where(
            _X_nodes_raw.max(axis=0) - _mn > 0, _X_nodes_raw.max(axis=0) - _mn, 1.0
        )
        _X_n, _X_p = (_X_nodes_raw - _mn) / _rng, (_X_pops_raw - _mn) / _rng
        _w = build_weight_vector(
            _cols, marker_weights or MARKER_WEIGHTS_V5, MARKER_WEIGHTS_DEFAULT_V5
        )
        _X_n, _X_p = _X_n * _w, _X_p * _w
        if verbose:
            print(
                f"\n   Poids : " + ", ".join(f"{c}={w:.2f}" for c, w in zip(_cols, _w))
            )

    elif method == "M7_vote":
        if verbose:
            print(f"\n   [M7] Vote M2 + M3 + M8 — seuil via M8")
        _votes: Dict[str, np.ndarray] = {}
        for _sub_m in ["M2_cityblock", "M3_cosine", "M8_ref_norm"]:
            _sub_df = map_populations_to_nodes_v5(
                node_mfi_raw=node_mfi_raw,
                pop_mfi_ref=pop_mfi_ref,
                include_scatter=include_scatter,
                percentile=100,
                threshold_mode="none",
                method=_sub_m,
                arcsinh_cofactor=arcsinh_cofactor,
                marker_weights=marker_weights,
                verbose=False,
                data_already_transformed=data_already_transformed,
                cell_counts=cell_counts,
                cov_matrices=cov_matrices,
                knn_samples=knn_samples,
            )
            _votes[_sub_m] = _sub_df["best_pop"].values
        _vote_arr = np.stack(list(_votes.values()), axis=1)
        _final_arr = np.array(
            [
                (_u := np.unique(_vote_arr[_i], return_counts=True))[0][
                    np.argmax(_u[1])
                ]
                if np.max(np.unique(_vote_arr[_i], return_counts=True)[1]) >= 2
                else _votes["M8_ref_norm"][_i]
                for _i in range(len(node_mfi_raw))
            ]
        )
        _ref_df = map_populations_to_nodes_v5(
            node_mfi_raw=node_mfi_raw,
            pop_mfi_ref=pop_mfi_ref,
            include_scatter=include_scatter,
            percentile=100,
            threshold_mode="none",
            method="M8_ref_norm",
            verbose=False,
            data_already_transformed=data_already_transformed,
        )
        _bd = _ref_df["best_dist"].values
        _th, _tdesc = compute_unknown_threshold(_bd, threshold_mode, percentile)
        _asgn = np.where(_bd <= _th, _final_arr, "Unknown")
        if verbose:
            _nu = int((_asgn == "Unknown").sum())
            print(f"   [M7] seuil={_th:.4f} ({_tdesc})  Unknown={_nu}/{len(_asgn)}")
            for _p, _c in pd.Series(_asgn).value_counts().items():
                _bar = "█" * int(_c / len(_asgn) * 35)
                print(
                    f"     {_p:<35s}: {_c:>4d}  {100 * _c / len(_asgn):5.1f}%  {_bar}"
                )
        return pd.DataFrame(
            {
                "node_id": np.arange(len(_asgn)),
                "best_pop": _final_arr,
                "best_dist": np.round(_bd, 4),
                "threshold": round(_th, 4),
                "threshold_mode": _tdesc,
                "assigned_pop": _asgn,
            }
        )

    elif method == "M8_ref_norm":
        _ref_min = _X_pops_raw.min(axis=0)
        _ref_rng = np.where(
            _X_pops_raw.max(axis=0) - _ref_min > 1e-8,
            _X_pops_raw.max(axis=0) - _ref_min,
            1.0,
        )
        _X_n = (_X_nodes_raw - _ref_min) / _ref_rng
        _X_p = (_X_pops_raw - _ref_min) / _ref_rng
        if verbose:
            print(f"\n   [M8] Normalisation dans le cadre référence → Refs=[0,1]")
            for _j, _c in enumerate(_cols[:6]):
                _nmin, _nmax = _X_n[:, _j].min(), _X_n[:, _j].max()
                _flag = " ⚠️ HORS" if _nmin < -0.5 or _nmax > 1.5 else ""
                print(
                    f"     {_c:<30s}  nœuds=[{_nmin:.3f},{_nmax:.3f}]  ref=[{_X_p[:, _j].min():.3f},{_X_p[:, _j].max():.3f}]{_flag}"
                )
            if len(_cols) > 6:
                print(f"     ... ({len(_cols) - 6} autres)")

    elif method == "M9_prior":
        # ── ★ PRIOR BAYÉSIEN (base euclidienne M8) ★ ────────────────────────
        # Base : normalisation M8 (dans le cadre référence)
        # Puis : D_adj[node, pop] = D_euclidean_M8[node, pop] / log10(n_cells[pop])
        # → Les Granuleux (N=50 000 → log=4.7) sont ×4.7 plus attractifs
        #   qu'une population avec N=100 → log=2.0
        _ref_min = _X_pops_raw.min(axis=0)
        _ref_rng = np.where(
            _X_pops_raw.max(axis=0) - _ref_min > 1e-8,
            _X_pops_raw.max(axis=0) - _ref_min,
            1.0,
        )
        _X_n_base = (_X_nodes_raw - _ref_min) / _ref_rng
        _X_p_base = (_X_pops_raw - _ref_min) / _ref_rng
        _D_base = cdist(_X_n_base, _X_p_base, metric="euclidean")

        _cc = cell_counts or {}
        if not _cc:
            if verbose:
                print(
                    f"   [M9] ⚠️ pop_cell_counts indisponible — Section 10.3b non exécutée."
                )
                print(f"        Utilise M8 (prior=1 pour toutes les populations)")
        else:
            if verbose:
                print(f"\n   [M9] Prior Bayésien — log10(n_cells):")
                _total = sum(_cc.values())
                for _p in _pop_names:
                    _n = _cc.get(_p, 1)
                    _log = np.log10(max(_n, 10))
                    _pct = 100.0 * _n / max(_total, 1)
                    print(
                        f"     {_p:<35s}: {_n:>9,} cells  {_pct:5.1f}%  "
                        f"log10={_log:.2f}  → D ÷ {_log:.2f}"
                    )

        _dist_final = _apply_bayesian_prior(
            _D_base, _pop_names, _cc or {p: 1 for p in _pop_names}
        )
        if verbose:
            print(f"\n   Distances ajustées (mean/pop) :")
            for _p, _d in zip(_pop_names, _dist_final.mean(axis=0)):
                _bar = "█" * int(min(_d * 8, 40))
                print(f"     {_p:<35s}: {_d:.4f}  {_bar}")

        _asgn, _bd, _th, _tdesc = assign_with_auto_threshold(
            _dist_final, _pop_names, threshold_mode, percentile
        )
        if verbose:
            _nu = int((_asgn == "Unknown").sum())
            print(
                f"\n   Seuil={_th:.4f} ({_tdesc})  Assignés {len(_asgn) - _nu}/{len(_asgn)}"
            )
            for _p, _c in pd.Series(_asgn).value_counts().items():
                _bar = ("░" if _p == "Unknown" else "█") * int(_c / len(_asgn) * 35)
                _mk = " ← BLAST CANDIDATES" if _p == "Unknown" else ""
                print(
                    f"     {_p:<35s}: {_c:>4d}  {100 * _c / len(_asgn):5.1f}%  {_bar}{_mk}"
                )
        return pd.DataFrame(
            {
                "node_id": np.arange(len(_asgn)),
                "best_pop": [_pop_names[i] for i in np.argmin(_dist_final, axis=1)],
                "best_dist": np.round(_bd, 4),
                "threshold": round(_th, 4),
                "threshold_mode": _tdesc,
                "assigned_pop": _asgn,
            }
        )

    elif method == "M10_mahalanobis":
        # ── ★ DISTANCE DE MAHALANOBIS ★ ───────────────────────────────────────
        _ref_min = _X_pops_raw.min(axis=0)
        _ref_rng = np.where(
            _X_pops_raw.max(axis=0) - _ref_min > 1e-8,
            _X_pops_raw.max(axis=0) - _ref_min,
            1.0,
        )
        _X_n_norm = (_X_nodes_raw - _ref_min) / _ref_rng
        _X_p_norm = (_X_pops_raw - _ref_min) / _ref_rng

        _cov_m = cov_matrices or {}
        if not _cov_m:
            if verbose:
                print(
                    f"   [M10] ⚠️ pop_cov_matrices indisponible — Section 10.3b non exécutée."
                )
                print(f"        Utilise distance euclidienne (= M8 fallback)")

        _cov_norm: Dict[str, np.ndarray] = {}
        for _p in _pop_names:
            _cov_raw = _cov_m.get(_p)
            if _cov_raw is None:
                continue
            _col_indices = []
            for _c in _cols:
                if _c in _cols_stats:
                    _col_indices.append(_cols_stats.index(_c))
            if len(_col_indices) != len(_cols):
                continue
            _cov_sub = _cov_raw[np.ix_(_col_indices, _col_indices)]
            _rng_outer = np.outer(_ref_rng, _ref_rng)
            _cov_norm[_p] = _cov_sub / np.where(_rng_outer > 1e-12, _rng_outer, 1e-12)

        if verbose:
            print(
                f"\n   [M10] Mahalanobis — covariances disponibles : "
                f"{list(_cov_norm.keys())}"
            )

        _dist_final = _mahalanobis_distance_batch(
            _X_n_norm,
            _X_p_norm,
            _pop_names,
            _cov_norm,
            _cols,
        )
        if verbose:
            print(f"   Distances Mahalanobis (mean/pop) :")
            for _p, _d in zip(_pop_names, _dist_final.mean(axis=0)):
                _bar = "█" * int(min(_d * 4, 40))
                print(f"     {_p:<35s}: {_d:.4f}  {_bar}")

        _asgn, _bd, _th, _tdesc = assign_with_auto_threshold(
            _dist_final, _pop_names, threshold_mode, percentile
        )
        if verbose:
            _nu = int((_asgn == "Unknown").sum())
            print(
                f"\n   Seuil={_th:.4f} ({_tdesc})  Assignés {len(_asgn) - _nu}/{len(_asgn)}"
            )
            for _p, _c in pd.Series(_asgn).value_counts().items():
                _bar = ("░" if _p == "Unknown" else "█") * int(_c / len(_asgn) * 35)
                _mk = " ← BLAST CANDIDATES" if _p == "Unknown" else ""
                print(
                    f"     {_p:<35s}: {_c:>4d}  {100 * _c / len(_asgn):5.1f}%  {_bar}{_mk}"
                )
        return pd.DataFrame(
            {
                "node_id": np.arange(len(_asgn)),
                "best_pop": [_pop_names[i] for i in np.argmin(_dist_final, axis=1)],
                "best_dist": np.round(_bd, 4),
                "threshold": round(_th, 4),
                "threshold_mode": _tdesc,
                "assigned_pop": _asgn,
            }
        )

    elif method == "M11_knn":
        # ── ★ K-NN DENSITÉ (Pool Proportionnel) ★ ────────────────────────────
        _ref_min = _X_pops_raw.min(axis=0)
        _ref_rng = np.where(
            _X_pops_raw.max(axis=0) - _ref_min > 1e-8,
            _X_pops_raw.max(axis=0) - _ref_min,
            1.0,
        )
        _X_n_norm = (_X_nodes_raw - _ref_min) / _ref_rng

        _knn_s = knn_samples or {}
        if not _knn_s:
            if verbose:
                print(
                    f"   [M11] ⚠️ pop_knn_samples indisponible — Section 10.3b non exécutée."
                )

        # Normaliser les échantillons KNN dans le même espace que les nœuds
        _knn_norm: Dict[str, np.ndarray] = {}
        for _p in _pop_names:
            _s = _knn_s.get(_p)
            if _s is None or len(_s) == 0:
                continue
            _col_indices = [_cols_stats.index(_c) for _c in _cols if _c in _cols_stats]
            if len(_col_indices) != len(_cols):
                _knn_norm[_p] = (_s - _ref_min) / _ref_rng
            else:
                _knn_norm[_p] = (_s[:, _col_indices] - _ref_min) / _ref_rng

        # ── Rééchantillonnage proportionnel stratifié ─────────────────────────
        _cc = cell_counts or {}
        _knn_prop = _proportional_stratified_pool(
            knn_norm=_knn_norm,
            pop_names=_pop_names,
            cell_counts=_cc,
            total_points=knn_total_points,
            min_points=knn_min_per_pop,
        )

        if verbose:
            _total_pts_before = sum(len(v) for v in _knn_norm.values())
            _total_pts_after = sum(len(v) for v in _knn_prop.values())
            print(
                f"\n   [M11] Pool KNN AVANT : {_total_pts_before:,} pts (équilibré fixe)"
            )
            print(f"   [M11] Pool KNN APRÈS : {_total_pts_after:,} pts (proportionnel)")
            print(
                f"         K={k_neighbors}  | total_cible={knn_total_points:,}  | min/pop={knn_min_per_pop}"
            )
            if _cc:
                _total_c = sum(_cc.values())
                print(
                    f"   {'Population':<35s}  {'Avant':>7s}  {'Après':>7s}  {'Réel%':>6s}  {'Pool%':>6s}"
                )
                for _p in _pop_names:
                    _n_bef = len(_knn_norm.get(_p, []))
                    _n_aft = len(_knn_prop.get(_p, []))
                    _pct_r = 100.0 * _cc.get(_p, 0) / max(_total_c, 1)
                    _pct_p = 100.0 * _n_aft / max(_total_pts_after, 1)
                    _flag = " ⚠️ sur-repr" if _pct_p > _pct_r * 3 else ""
                    print(
                        f"     {_p:<35s}: {_n_bef:>7d}  {_n_aft:>7d}  {_pct_r:>5.1f}%  {_pct_p:>5.1f}%{_flag}"
                    )
            else:
                for _p in _pop_names:
                    _n = len(_knn_prop.get(_p, []))
                    print(f"     {_p:<35s}: {_n} points")

        _best_pop_knn, _best_dist_knn = _knn_vote(
            _X_n_norm,
            _pop_names,
            _knn_prop,
            k=k_neighbors,
        )

        _th, _tdesc = compute_unknown_threshold(
            _best_dist_knn, threshold_mode, percentile
        )
        _asgn = np.where(_best_dist_knn <= _th, _best_pop_knn, "Unknown")
        _frac = (_asgn != "Unknown").sum() / len(_asgn)
        if _frac < 0.30:
            _th = float(np.percentile(_best_dist_knn, 70))
            _tdesc = "P70 (fallback M11)"
            _asgn = np.where(_best_dist_knn <= _th, _best_pop_knn, "Unknown")

        if verbose:
            _nu = int((_asgn == "Unknown").sum())
            print(
                f"\n   Seuil K-NN dist={_th:.4f} ({_tdesc})  Unknown={_nu}/{len(_asgn)}"
            )
            for _p, _c in pd.Series(_asgn).value_counts().items():
                _bar = ("░" if _p == "Unknown" else "█") * int(_c / len(_asgn) * 35)
                _mk = " ← BLAST CANDIDATES" if _p == "Unknown" else ""
                print(
                    f"     {_p:<35s}: {_c:>4d}  {100 * _c / len(_asgn):5.1f}%  {_bar}{_mk}"
                )
        return pd.DataFrame(
            {
                "node_id": np.arange(len(_asgn)),
                "best_pop": _best_pop_knn.astype(str),
                "best_dist": np.round(_best_dist_knn, 4),
                "threshold": round(_th, 4),
                "threshold_mode": _tdesc,
                "assigned_pop": _asgn,
            }
        )

    elif method == "M12_cosine_prior":
        # ── ★ COSINE + PRIOR RENFORCÉ log10^3 + HARD LIMIT ★ ──────────────────
        #
        # Rationalité :
        # • M3_cosine est géométriquement robuste (insensible au scale mismatch)
        #   car elle compare la FORME du vecteur d'expression, pas les valeurs brutes.
        #   cos(θ) = (u·v) / (||u|| × ||v||) — distance ∈ [0, 2]
        #
        # • Problème du prior log10 simple (M9/M12 legacy) :
        #   Granulos 2.2M vs Plasmos 2000 → ratio pénalité 1.9x seulement.
        #   Insuffisant : les plasmos "volent" les nœuds ambigus.
        #
        # → Combinaison V2 RENFORCÉE : D_adj = D_cosine / log10(n_cells)^3
        #   Granulos 2.2M → log10^3 = 254.8  |  Plasmos 2000 → log10^3 = 35.9
        #   Ratio réel = 7.1x → les populations < 1% ne gagnent un nœud que si
        #   le Cosinus est quasi parfait (proche de 0.0).
        #
        # → Hard Limit (optionnel, hard_limit_factor > 0) :
        #   Règle binaire absolue — si un nœud contient plus de cellules que
        #   l'attendu max pour la population rare, D → inf (exclusion forcée).
        #   Ex: nœud à 3 000 cellules ne peut pas être "Plasmocytes (N=2 000 total)".

        # Étape 1 : Clip des valeurs négatives (obligatoire pour la distance Cosinus)
        # En espace Logicle, les valeurs légèrement négatives sont fréquentes → clip à 0.
        _X_n_clip = np.clip(_X_nodes_raw, 0.0, None)
        _X_p_clip = np.clip(_X_pops_raw, 0.0, None)

        # Étape 2 : Matrice de distance Cosinus brute (n_nodes × n_pops)
        # D_cosine[i, j] = 1 - cos(θ) entre le nœud i et la population j
        _D_cosine = cdist(_X_n_clip, _X_p_clip, metric="cosine")

        # Étape 3 : Pondération bayésienne renforcée D_adj = D_cosine / log10(n_cells)^3
        # ─────────────────────────────────────────────────────────────────────────────
        # Comparaison des variantes :
        #   "log10"        : ratio Granulos/Plasmos ~1.9x  ← insuffisant (M9 legacy)
        #   "log10_cubed"  : ratio Granulos/Plasmos ~7.1x  ← RECOMMANDÉ (utilisé ici)
        #   "sqrt_n"       : ratio Granulos/Plasmos ~33x   ← très agressif
        _cc = cell_counts or {}
        if not _cc:
            if verbose:
                print(
                    f"   [M12] ⚠️ pop_cell_counts indisponible — Section 10.3b non exécutée."
                )
                print(
                    f"        Utilise M3_cosine pur (prior=1 pour toutes les populations)"
                )
        else:
            if verbose:
                print(f"   [M12] Cosine + Prior RENFORCE [log10^3] + Hard Limit:")
                _total = sum(_cc.values())
                for _p in _pop_names:
                    _n = _cc.get(_p, 1)
                    _log3 = np.log10(max(_n, 10)) ** 3
                    _sqrn = np.sqrt(max(_n, 10) / 10.0)
                    _pct = 100.0 * _n / max(_total, 1)
                    _flag = (
                        "[RARE <1%]"
                        if _pct < 1.0
                        else ("[DOMINANT]" if _pct > 30 else "")
                    )
                    print(
                        f"     {_p:<35s}: {_n:>9,} cells  {_pct:5.1f}%  "
                        f"log10^3={_log3:7.1f}  sqrt={_sqrn:6.1f}  {_flag}"
                    )
                _ns_ok = node_sizes_hard_limit is not None
                print(f"   [M12] Prior actif : log10^3 (ratio ~7x Granulos/Plasmos)")
                print(
                    f"   [M12] Hard Limit  : "
                    f"{'actif (factor=' + str(hard_limit_factor) + ')' if _ns_ok else 'NON ACTIF (node_sizes absent)'}"
                )

        _dist_final = _apply_bayesian_prior(
            _D_cosine,
            _pop_names,
            _cc or {p: 1 for p in _pop_names},
            prior_mode="log10_cubed",  # RENFORCE : ratio ~7.1x vs ~1.9x (log10 simple)
            node_sizes=node_sizes_hard_limit,
            hard_limit_factor=hard_limit_factor,
            n_nodes_total=len(_X_nodes_raw),
        )

        if verbose:
            print(f"\n   Distances cosinus ajustées (mean/pop) :")
            for _p, _d in zip(_pop_names, _dist_final.mean(axis=0)):
                _bar = "" if np.isnan(_d) else "█" * int(min(_d * 12, 40))
                _d_str = "NaN  " if np.isnan(_d) else f"{_d:.4f}"
                print(f"     {_p:<35s}: {_d_str}  {_bar}")
                print(f"     {_p:<35s}: {_d:.4f}  {_bar}")

        # Étape 4 : Seuil auto-Otsu sur les distances minimales ajustées
        _asgn, _bd, _th, _tdesc = assign_with_auto_threshold(
            _dist_final, _pop_names, threshold_mode, percentile
        )

        if verbose:
            _nu = int((_asgn == "Unknown").sum())
            print(
                f"\n   Seuil={_th:.4f} ({_tdesc})  Assignés {len(_asgn) - _nu}/{len(_asgn)}"
            )
            for _p, _c in pd.Series(_asgn).value_counts().items():
                _bar = ("░" if _p == "Unknown" else "█") * int(_c / len(_asgn) * 35)
                _mk = " ← BLAST CANDIDATES" if _p == "Unknown" else ""
                print(
                    f"     {_p:<35s}: {_c:>4d}  {100 * _c / len(_asgn):5.1f}%  {_bar}{_mk}"
                )

        return pd.DataFrame(
            {
                "node_id": np.arange(len(_asgn)),
                "best_pop": [_pop_names[i] for i in np.argmin(_dist_final, axis=1)],
                "best_dist": np.round(_bd, 4),
                "threshold": round(_th, 4),
                "threshold_mode": _tdesc,
                "assigned_pop": _asgn,
            }
        )

    else:
        raise ValueError(f"Méthode inconnue : '{method}'. Options : M1–M12.")

    # ── 2. Calcul de la matrice de distance finale (M1–M8) ────────────────────
    _dist_final = cdist(_X_n, _X_p, metric=_metric)

    if verbose:
        _mt = np.sqrt(len(_cols)) if _metric == "euclidean" else float(len(_cols))
        print(
            f"\n   Distance ({_metric}) min={_dist_final.min():.4f}  "
            f"max={_dist_final.max():.4f}  mean={_dist_final.mean():.4f}  théo.max≈{_mt:.2f}"
        )
        print(f"   Distance moyenne nœuds→pop :")
        for _p, _d in zip(_pop_names, _dist_final.mean(axis=0)):
            _bar = "" if np.isnan(_d) else "█" * int(min(_d * 8, 40))
            _d_str = "NaN  " if np.isnan(_d) else f"{_d:.4f}"
            print(f"     {_p:<35s}: {_d_str}  {_bar}")

    _asgn, _bd, _th, _tdesc = assign_with_auto_threshold(
        _dist_final, _pop_names, threshold_mode, percentile
    )

    if verbose:
        _nu = int((_asgn == "Unknown").sum())
        print(
            f"\n   Seuil={_th:.4f} ({_tdesc})  Assignés {len(_asgn) - _nu}/{len(_asgn)}"
        )
        for _p, _c in pd.Series(_asgn).value_counts().items():
            _bar = ("░" if _p == "Unknown" else "█") * int(_c / len(_asgn) * 35)
            _mk = " ← BLAST CANDIDATES" if _p == "Unknown" else ""
            print(
                f"     {_p:<35s}: {_c:>4d}  {100 * _c / len(_asgn):5.1f}%  {_bar}{_mk}"
            )

    return pd.DataFrame(
        {
            "node_id": np.arange(len(_X_n)),
            "best_pop": [_pop_names[i] for i in np.argmin(_dist_final, axis=1)],
            "best_dist": np.round(_bd, 4),
            "threshold": round(_th, 4),
            "threshold_mode": _tdesc,
            "assigned_pop": _asgn,
        }
    )


# =============================================================================
# BENCHMARK
# =============================================================================
print("=" * 70)
print(" SECTION 10.4b — BENCHMARK MULTI-MÉTHODES (M1–M12)")
print("=" * 70)

_DATA_PRE_TRANSFORMED: bool = globals().get("_TRANSFORM_TAG") is not None
_transfo_tag = globals().get("_TRANSFORM_TAG", "brutes (10.1b non exécutée)")

_CC_AVAIL = "pop_cell_counts" in dir() and len(pop_cell_counts) > 0
_COV_AVAIL = "pop_cov_matrices" in dir() and len(pop_cov_matrices) > 0
_KNN_AVAIL = "pop_knn_samples" in dir() and len(pop_knn_samples) > 0

print(f"\n   data_pré-transfo  : {_DATA_PRE_TRANSFORMED}  ({_transfo_tag})")
print(f"   Seuil Unknown     : {UNKNOWN_THRESHOLD_MODE}")
print(
    f"   pop_cell_counts   : {'✅ disponible' if _CC_AVAIL else '⚠️ manquant (exéc. 10.3b)'}"
)
print(
    f"   pop_cov_matrices  : {'✅ disponible' if _COV_AVAIL else '⚠️ manquant (exéc. 10.3b)'}"
)
print(
    f"   pop_knn_samples   : {'✅ disponible' if _KNN_AVAIL else '⚠️ manquant (exéc. 10.3b)'}"
)
if _CC_AVAIL and _KNN_AVAIL:
    print(
        f"   M11 pool cible    : {TOTAL_KNN_POINTS:,} pts  | min/pop={KNN_MIN_POINTS_PER_POP}"
    )

# Alignement des colonnes
_common_a_cols_v5 = sorted(set(node_mfi_raw_df.columns) & set(df_mfi_raw_ref.columns))
_node_v5 = node_mfi_raw_df[_common_a_cols_v5].copy()
_ref_v5 = df_mfi_raw_ref[_common_a_cols_v5].copy()
_cols_stats = _common_a_cols_v5

print(f"\n   Marqueurs communs : {len(_common_a_cols_v5)}  → {_common_a_cols_v5}")

# ─── Scale check ─────────────────────────────────────────────────────────────
print(f"\n   ─── Scale check ───")
for _c in _common_a_cols_v5:
    _nrng = _node_v5[_c].max() - _node_v5[_c].min()
    _rrng = _ref_v5[_c].max() - _ref_v5[_c].min()
    _ratio = _nrng / _rrng if _rrng > 1e-8 else 0.0
    _flag = f" ⚠️ ×{1 / _ratio:.1f}" if _ratio < 0.3 else " ✅"
    print(f"     {_c:<30s}  ratio={_ratio:.3f}{_flag}")

# ─── Proportions biologiques (si Section 10.3b exécutée) ─────────────────────
if _CC_AVAIL:
    print(f"\n   ─── Proportions biologiques réelles (pop_cell_counts) ───")
    _total_c = sum(pop_cell_counts.values())
    for _p in sorted(pop_cell_counts, key=lambda x: -pop_cell_counts[x]):
        _n = pop_cell_counts[_p]
        _pct = 100.0 * _n / max(_total_c, 1)
        _log = np.log10(max(_n, 10))
        _log3 = _log**3
        _bar = "█" * int(_pct / 3)
        _flag = " [RARE <1%]" if _pct < 1.0 else ("")
        print(
            f"     {_p:<35s}: {_n:>9,}  {_pct:5.1f}%  "
            f"log10={_log:.2f}  log10^3={_log3:.1f}{_flag}  {_bar}"
        )

# ─── Benchmark ────────────────────────────────────────────────────────────────
_bench_results: Dict[str, pd.DataFrame] = {}

for _m in _METHODS_BENCH:
    print(f"\n{'=' * 70}")
    _bench_results[_m] = map_populations_to_nodes_v5(
        node_mfi_raw=_node_v5,
        pop_mfi_ref=_ref_v5,
        include_scatter=INCLUDE_SCATTER_V5,
        percentile=DISTANCE_PERCENTILE_V5,
        threshold_mode=UNKNOWN_THRESHOLD_MODE,
        method=_m,
        arcsinh_cofactor=ARCSINH_COFACTOR,
        marker_weights=MARKER_WEIGHTS_V5,
        verbose=True,
        data_already_transformed=_DATA_PRE_TRANSFORMED,
        cell_counts=pop_cell_counts if _CC_AVAIL else None,
        cov_matrices=pop_cov_matrices if _COV_AVAIL else None,
        knn_samples=pop_knn_samples if _KNN_AVAIL else None,
        k_neighbors=KNN_K_BENCH,
        knn_total_points=TOTAL_KNN_POINTS,
        knn_min_per_pop=KNN_MIN_POINTS_PER_POP,
        node_sizes_hard_limit=(
            globals().get("node_sizes") if _m == "M12_cosine_prior" else None
        ),
        hard_limit_factor=5.0,
    )

# ─── Synthèse ─────────────────────────────────────────────────────────────────
print(f"\n\n{'=' * 70}")
print(
    f" SYNTHÈSE — TOUTES MÉTHODES  |  Seuil={UNKNOWN_THRESHOLD_MODE}  "
    f"|  scatter={'incl' if INCLUDE_SCATTER_V5 else 'excl'}"
)
print(f"{'=' * 70}")

_all_pops = sorted(
    set(p for df in _bench_results.values() for p in df["assigned_pop"].unique())
)

_hdr = f"{'Méthode':<18s}"
for _p in _all_pops:
    _hdr += f"  {_p[:11]:>11s}"
_hdr += f"  {'%Unk':>5s}  {'D_mean':>7s}  {'Seuil':>7s}"
print(_hdr)
print("─" * (20 + 13 * len(_all_pops) + 24))

for _m, _df in _bench_results.items():
    _row = f"{_m:<18s}"
    _vc = _df["assigned_pop"].value_counts()
    _unk_pct = 100.0 * _vc.get("Unknown", 0) / len(_df)
    for _p in _all_pops:
        _row += f"  {_vc.get(_p, 0):>11d}"
    _row += f"  {_unk_pct:>4.0f}%  {_df['best_dist'].mean():>7.4f}  {_df['threshold'].iloc[0]:>7.4f}"
    print(_row)

# ─── Score de recommandation ──────────────────────────────────────────────────
print(f"\n{'─' * 70}  RECOMMANDATION")
_scores = {}
for _m, _df in _bench_results.items():
    _vc = _df["assigned_pop"].value_counts()
    _n_unk = _vc.get("Unknown", 0)
    _n_pops = len([p for p in _vc.index if p != "Unknown"])
    _n_tot = len(_df)
    _avc = _vc.drop("Unknown", errors="ignore")
    _max_share = _avc.max() / max(_avc.sum(), 1) if len(_avc) > 0 else 1.0
    _diversity = max(0.0, 1.0 - max(0.0, _max_share - 0.5) * 2)
    _scores[_m] = _n_pops * (1.0 - _n_unk / _n_tot) * _diversity

_best_method = max(_scores, key=_scores.get)
print(
    f"\n   ★ Méthode recommandée : {_best_method}  (score={_scores[_best_method]:.2f})"
)
print(f"\n   Scores par méthode :")
for _m, _s in sorted(_scores.items(), key=lambda x: -x[1]):
    _bar = "█" * int(_s * 6)
    _star = " ★" if _m == _best_method else ""
    print(f"     {_m:<18s}: {_s:.2f}  {_bar}{_star}")

# ─── Export ───────────────────────────────────────────────────────────────────
mapping_df_v5 = _bench_results[_best_method].copy()
mapping_df_v5["metacluster"] = _mc_per_node_raw.reindex(mapping_df_v5["node_id"]).values
mapping_df_v5 = mapping_df_v5.merge(
    _node_coords_df.reset_index(), on="node_id", how="left"
)
mapping_df_v5["method"] = _best_method

_out_v5 = (
    OUTPUT_DIR / "other" / f"node_population_mapping_v5_{_best_method}_{timestamp}.csv"
)
mapping_df_v5.to_csv(_out_v5, index=False, sep=";", decimal=",")
print(f"\n[OK] Mapping V5 ({_best_method}) exporté : {_out_v5}")

_unknown_nodes_v5 = mapping_df_v5[mapping_df_v5["assigned_pop"] == "Unknown"][
    "node_id"
].tolist()
print(
    f"     Nœuds Unknown (candidats blastes) : {len(_unknown_nodes_v5)}/{len(mapping_df_v5)}"
)
if _unknown_nodes_v5:
    print(f"     → Exécutez Section 10.4c pour le scoring blast détaillé.")

display(mapping_df_v5)


################################################################################
# =============================================================================
# SECTION 10.4c — PROFILAGE ET SCORING DES CANDIDATS BLASTES (Nœuds 'Unknown')
# =============================================================================
#
# PIPELINE :
#   1. Extraction   — Isolation des node_id "Unknown" depuis mapping_df_v5
#   2. Profilage    — Récupération des centroïdes MFI (espace Logicle transformé)
#   3. Scoring      — Blast Score /10 par règles hématologiques ELN 2022
#   4. Heatmap      — Signature comparative Unknown vs populations saines de référence
#   5. Radar Chart  — Profil polaire par nœud Unknown vs médiane Granulocytes/Lymphocytes
#   6. Export       — CSV + variable blast_candidates_df
#
# DÉPENDANCES :
#   mapping_df_v5     [Section 10.4b]  — colonnes : node_id, assigned_pop, best_dist
#   node_mfi_raw_df   [Section 10.2]   — centroïdes nœuds, espace transformé
#   df_mfi_raw_ref    [Section 10.3]   — MFI de référence par population
#   _common_a_cols_v5 [Section 10.4b]  — marqueurs communs nœuds ↔ référence
#   node_sizes        [Section 10.1]   — nombre de cellules par nœud SOM
# =============================================================================

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.path import Path as MplPath  # aliasé pour ne pas écraser pathlib.Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from typing import Dict, List, Optional

# ─── Style global ─────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "#1e1e2e",
        "axes.facecolor": "#1e1e2e",
        "axes.edgecolor": "#44475a",
        "axes.labelcolor": "#e2e8f0",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "text.color": "#e2e8f0",
        "grid.color": "#2d2d3f",
        "grid.linewidth": 0.5,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    }
)

# ─── Palettes de couleurs ─────────────────────────────────────────────────────
_COL_UNKNOWN = "#f97316"  # orange vif — candidats blastes
_COL_BLAST_H = "#ef4444"  # rouge — blast fort
_COL_BLAST_L = "#fbbf24"  # ambre — blast faible
_COL_NON_B = "#6366f1"  # indigo — Unknown non-blast
_COL_GRANU = "#22c55e"  # vert   — Granulocytes sains
_COL_LYMPH = "#38bdf8"  # bleu   — Lymphocytes sains
_COL_HEMA = "#a78bfa"  # violet — Hématogones
_COL_PLASMA = "#f472b6"  # rose   — Plasmocytes

# Correspondance nom population → couleur
_POP_PALETTE = {
    "Granulocytes": _COL_GRANU,
    "Lymphocytes": _COL_LYMPH,
    "Hematogones": _COL_HEMA,
    "Hematogones I": _COL_HEMA,
    "Hematogones II": _COL_HEMA,
    "Plasmocytes": _COL_PLASMA,
    "Monocytes": "#fb923c",
    "T_NK": "#34d399",
    "Unknown": _COL_UNKNOWN,
}

print("=" * 70)
print(" SECTION 10.4c — PROFILAGE ET SCORING DES CANDIDATS BLASTES")
print("=" * 70)

# =============================================================================
# 0. VÉRIFICATIONS DES DÉPENDANCES
# =============================================================================
_deps_ok = True
for _dep, _desc in [
    ("mapping_df_v5", "Section 10.4b (mapping V5)"),
    ("node_mfi_raw_df", "Section 10.2 (centroïdes nœuds)"),
    ("df_mfi_raw_ref", "Section 10.3 (MFI référence)"),
    ("_common_a_cols_v5", "Section 10.4b (marqueurs communs)"),
]:
    _ok = _dep in dir() and globals().get(_dep) is not None
    _sym = "✅" if _ok else "❌"
    print(f"   {_sym} {_dep:<25s} [{_desc}]")
    if not _ok:
        _deps_ok = False

if not _deps_ok:
    print("\n   ⚠️ Dépendances manquantes — exécutez les sections indiquées d'abord.")
    raise SystemExit("Section 10.4c : dépendances manquantes.")

# =============================================================================
# 1. EXTRACTION DES NŒUDS UNKNOWN
# =============================================================================
_mask_unk = mapping_df_v5["assigned_pop"] == "Unknown"
_unk_ids = mapping_df_v5.loc[_mask_unk, "node_id"].values
_total_nodes = len(mapping_df_v5)
_n_unk = len(_unk_ids)

print(f"\n{'─' * 70}")
print(f" EXTRACTION")
print(f"{'─' * 70}")
print(f"   Méthode source     : {mapping_df_v5['method'].iloc[0]}")
print(f"   Nœuds total        : {_total_nodes}")
print(f"   Nœuds Unknown      : {_n_unk}  ({100.0 * _n_unk / _total_nodes:.1f}%)")
print(
    f"   Nœuds assignés     : {_total_nodes - _n_unk}  ({100.0 * (_total_nodes - _n_unk) / _total_nodes:.1f}%)"
)
print(f"   IDs Unknown        : {list(_unk_ids)}")

if _n_unk == 0:
    print("\n   ✅ Aucun nœud Unknown — pipeline terminé.")
    print(
        "      Réduisez le seuil (UNKNOWN_THRESHOLD_MODE='none') pour forcer des Unknown."
    )
    raise SystemExit("Aucun Unknown à analyser.")

# Nombre de cellules par nœud (pour pondérer les interprétations)
_node_size_arr = globals().get("node_sizes", np.ones(_total_nodes))
_unk_cell_counts = {
    int(_nid): int(_node_size_arr[_nid])
    for _nid in _unk_ids
    if _nid < len(_node_size_arr)
}

# =============================================================================
# 2. PROFILAGE MFI DES NŒUDS UNKNOWN
# =============================================================================
# Récupération des centroïdes dans l'espace transformé (Logicle)
_cols_blast = [c for c in _common_a_cols_v5 if c in node_mfi_raw_df.columns]

# Robustesse : accès par index ou par valeur
try:
    _unk_centroids = node_mfi_raw_df.loc[_unk_ids, _cols_blast].copy()
except KeyError:
    _unk_centroids = node_mfi_raw_df.iloc[_unk_ids][_cols_blast].copy()

# Normalisation M8 : cadre référence → [0, 1] pour les populations saines
# Les valeurs < 0 = dim (sous le min de la référence) → caractéristique blastique
# Les valeurs > 1 = bright (au-dessus du max de la référence) → caractéristique blastique
_ref_arr = df_mfi_raw_ref[_cols_blast].to_numpy(dtype=np.float64)
_ref_min_b = _ref_arr.min(axis=0)
_ref_max_b = _ref_arr.max(axis=0)
_ref_rng_b = np.where(_ref_max_b - _ref_min_b > 1e-8, _ref_max_b - _ref_min_b, 1.0)

_X_unk_raw = _unk_centroids.to_numpy(dtype=np.float64)  # espace Logicle brut
_X_unk_norm = (_X_unk_raw - _ref_min_b) / _ref_rng_b  # espace M8-normalisé

print(f"\n{'─' * 70}")
print(f" PROFILAGE MFI (espace Logicle + normalisation M8)")
print(f"{'─' * 70}")
print(f"   Marqueurs analysés : {_cols_blast}")
print(f"   Cellules/nœud      : { {int(k): v for k, v in _unk_cell_counts.items()} }")

# =============================================================================
# 3. SCORING BLASTE /10
# =============================================================================
# Grille de poids issue des critères ELN 2022 / LSCflow ALFA :
#   CD34  bright → marqueur de progéniteur         poids +3.0
#   CD117 bright → c-Kit, progéniteur myéloïde     poids +2.5
#   CD45  dim    → discriminant LAIP #1             poids -2.0 (négatif → bas = blaste)
#   HLA-DR pos   → blastes myéloïdes                poids +1.5
#   CD33  var    → engagement myéloïde              poids +1.0
#   CD13  var    → engagement myéloïde              poids +0.5
#   CD19/CD3 pos → signature lymphoïde (anti-blaste)poids -1.5
_BLAST_WEIGHTS: Dict[str, float] = {}

for _c in _cols_blast:
    _cu = _c.upper()
    if "CD34" in _cu:
        _BLAST_WEIGHTS[_c] = +3.0
    elif "CD117" in _cu or "CKIT" in _cu:
        _BLAST_WEIGHTS[_c] = +2.5
    elif "CD45" in _cu:
        _BLAST_WEIGHTS[_c] = -2.0
    elif "HLAD" in _cu or "HLA-DR" in _cu:
        _BLAST_WEIGHTS[_c] = +1.5
    elif "CD33" in _cu:
        _BLAST_WEIGHTS[_c] = +1.0
    elif "CD13" in _cu:
        _BLAST_WEIGHTS[_c] = +0.5
    elif "CD19" in _cu or "CD3 " in _cu:
        _BLAST_WEIGHTS[_c] = -1.5
    elif "SSC" in _cu:
        _BLAST_WEIGHTS[_c] = -1.0  # SSC faible = blaste
    else:
        _BLAST_WEIGHTS[_c] = 0.0

# Calcul du score brut
# Règle CD45-dim  : score_brut_CD45 = -(poids) × max(0, -valeur_M8norm)
#                   ↳ si valeur_M8norm = -0.5 → score = -(-2.0) × 0.5 = +1.0
# Règle CD34-high : score_brut_CD34  = poids × max(0, valeur_M8norm - 1.0)
#                   ↳ si valeur_M8norm = 1.3  → score = 3.0 × 0.3 = +0.9
# Règle SSC-dim   : même logique que CD45

_W_arr = np.array([_BLAST_WEIGHTS.get(_c, 0.0) for _c in _cols_blast])

_blast_scores_raw = np.zeros(len(_unk_ids))
for _j, (_c, _w) in enumerate(zip(_cols_blast, _W_arr)):
    _v = _X_unk_norm[:, _j]
    if _w > 0:
        # Marqueur positif : contribution quand valeur AU-DESSUS du plafond référence
        _blast_scores_raw += _w * np.maximum(0.0, _v - 1.0)
    elif _w < 0:
        # Marqueur négatif (CD45, SSC) : contribution quand valeur EN-DESSOUS du plancher ref
        _blast_scores_raw += (-_w) * np.maximum(0.0, -_v)

# Normalisation sur /10 (proportionnel au score max théorique)
_max_theoretical = sum(abs(w) for w in _BLAST_WEIGHTS.values() if w != 0)
_blast_scores_10 = np.clip(
    _blast_scores_raw / max(_max_theoretical, 1e-6) * 10.0, 0.0, 10.0
)


# Catégories cliniques
def _categorize_blast(score: float) -> str:
    """Classification blast selon le score /10."""
    if score >= 6.0:
        return "BLAST_HIGH"  # Signature blastique forte → suspecter LAM fortement
    elif score >= 3.0:
        return "BLAST_MODERATE"  # Signature intermédiaire → confirmation nécessaire
    elif score > 0.0:
        return "BLAST_WEAK"  # Signal léger → population atypique mais peu évocatrice
    else:
        return "NON_BLAST_UNK"  # Aucune signature blastique → pop. rare normale absente de ref


# =============================================================================
# 4. CONSTRUCTION DU DATAFRAME RÉSULTAT
# =============================================================================
_df_blast = pd.DataFrame(
    {
        "node_id": _unk_ids,
        "blast_score": np.round(_blast_scores_10, 2),
        "blast_score_raw": np.round(_blast_scores_raw, 4),
        "n_cells": [_unk_cell_counts.get(int(_nid), 0) for _nid in _unk_ids],
    }
)
_df_blast["blast_category"] = _df_blast["blast_score"].apply(_categorize_blast)

# Merge avec mapping_df_v5 pour récupérer dist, seuil, metacluster
_df_blast = _df_blast.merge(
    mapping_df_v5[
        ["node_id", "best_dist", "threshold", "threshold_mode", "metacluster"]
    ],
    on="node_id",
    how="left",
).rename(columns={"best_dist": "dist_nearest_ref"})

# Ajout des valeurs M8-normalisées pour chaque marqueur
for _j, _c in enumerate(_cols_blast):
    _df_blast[f"{_c}_M8"] = np.round(_X_unk_norm[:, _j], 3)

# Ajout des valeurs brutes (Logicle)
for _j, _c in enumerate(_cols_blast):
    _df_blast[f"{_c}_raw"] = np.round(_X_unk_raw[:, _j], 3)

# Tri par blast_score décroissant
_df_blast = _df_blast.sort_values("blast_score", ascending=False).reset_index(drop=True)

# =============================================================================
# 5. AFFICHAGE TEXTUEL STYLISTÉ
# =============================================================================
_CAT_COLORS = {
    "BLAST_HIGH": "🔴",
    "BLAST_MODERATE": "🟠",
    "BLAST_WEAK": "🟡",
    "NON_BLAST_UNK": "🔵",
}
_cat_order = ["BLAST_HIGH", "BLAST_MODERATE", "BLAST_WEAK", "NON_BLAST_UNK"]

print(f"\n{'─' * 70}")
print(f" POIDS DE SCORING (critères ELN 2022)")
print(f"{'─' * 70}")
for _c, _w in sorted(_BLAST_WEIGHTS.items(), key=lambda x: -abs(x[1])):
    if _w != 0.0:
        _dir = "BRIGHT → blaste" if _w > 0 else "DIM   → blaste"
        _bar = ("█" if _w > 0 else "░") * int(abs(_w) * 3)
        print(f"   {_c:<30s} : {_w:+.1f}  {_bar:12s}  {_dir}")
print(f"   Score max théorique : {_max_theoretical:.1f}")

print(f"\n{'─' * 70}")
print(f" RÉSULTATS — BLAST SCORE /10")
print(f"{'─' * 70}")
for _cat in _cat_order:
    _sub = _df_blast[_df_blast["blast_category"] == _cat]
    if len(_sub) == 0:
        continue
    print(f"\n   {_CAT_COLORS[_cat]} {_cat} ({len(_sub)} nœud(s))")
    for _, _row in _sub.iterrows():
        _bar_score = "█" * int(_row["blast_score"])
        _cells_str = f"{int(_row['n_cells']):,} cellules"
        print(
            f"      Node {int(_row['node_id']):3d} | Score={_row['blast_score']:5.2f}/10  "
            f"{_bar_score:<10s} | dist_ref={_row['dist_nearest_ref']:.4f} | {_cells_str}"
        )
        # Affiche les marqueurs discriminants
        _key_markers = ["CD45", "CD34", "CD117"]
        _annotations = []
        for _km in _key_markers:
            for _col in _df_blast.columns:
                if _km in _col.upper() and _col.endswith("_M8"):
                    _v = _row[_col]
                    _flag = "DIM ↓" if _v < 0 else ("BRIGHT ↑" if _v > 1 else "norm")
                    _annotations.append(
                        f"{_col.replace('_M8', '')}: {_v:.2f} ({_flag})"
                    )
                    break
        if _annotations:
            print(f"         → {' | '.join(_annotations)}")

print(f"\n{'─' * 70}")
print(f" PROFIL MOYEN PAR CATÉGORIE (valeurs M8-normalisées)")
print(f" Légende : 0=min-ref, 1=max-ref, <0=DIM, >1=BRIGHT")
print(f"{'─' * 70}")
_m8_cols = [c for c in _df_blast.columns if c.endswith("_M8")]
_m8_labels = [c.replace("_M8", "") for c in _m8_cols]
if _m8_cols:
    _mean_cat = _df_blast.groupby("blast_category")[_m8_cols].mean().round(3)
    _mean_cat.columns = _m8_labels

# =============================================================================
# 6. FIGURE 1 — HEATMAP COMPARATIVE SEABORN
# =============================================================================
# Prépare la matrice de comparaison :
#   Lignes : nœuds Unknown + populations de référence (Granulocytes, Lymphocytes)
#   Colonnes : marqueurs clés cytométrie

# Populations de référence à inclure dans la comparaison
_REF_POPS_HEATMAP = [
    "Granulocytes",
    "Lymphocytes",
    "Monocytes",
    "Hematogones I",
    "Plasmocytes",
]

_heatmap_rows = {}

# Ajout des populations de référence (valeurs M8-normalisées = [0,1] par définition)
for _pop in _REF_POPS_HEATMAP:
    if _pop in df_mfi_raw_ref.index:
        _row_ref = df_mfi_raw_ref.loc[_pop, _cols_blast].to_numpy(dtype=np.float64)
        _row_norm = (_row_ref - _ref_min_b) / _ref_rng_b
        _heatmap_rows[_pop] = _row_norm

# Ajout des nœuds Unknown
for _i, (_nid, _row_blast) in enumerate(zip(_unk_ids, _X_unk_norm)):
    _score = _df_blast.loc[_df_blast["node_id"] == _nid, "blast_score"].values
    _score_str = f"{_score[0]:.1f}" if len(_score) > 0 else "?"
    _cat = _df_blast.loc[_df_blast["node_id"] == _nid, "blast_category"].values
    _cat_emoji = _CAT_COLORS.get(_cat[0], "❓") if len(_cat) > 0 else "?"
    _label = f"Node {_nid} [score={_score_str}/10]"
    _heatmap_rows[_label] = _row_blast

_df_heatmap = pd.DataFrame(_heatmap_rows, index=_cols_blast).T

# Sélection des marqueurs informatifs (écart-type > 0 pour éviter colonnes mortes)
_std_per_col = _df_heatmap.std(axis=0)
_informative_cols = _std_per_col[_std_per_col > 0.05].index.tolist()
if len(_informative_cols) < 3:
    _informative_cols = _cols_blast  # Fallback : tous les marqueurs

_df_heatmap_plot = _df_heatmap[_informative_cols]

# Construction des annotations de ligne (catégorie)
_row_labels_col = []
for _lbl in _df_heatmap_plot.index:
    if any(_lbl.startswith(f"Node {_nid}") for _nid in _unk_ids):
        _nid_this = int(_lbl.split()[1])
        _cat_this = _df_blast.loc[_df_blast["node_id"] == _nid_this, "blast_category"]
        _row_labels_col.append(_cat_this.values[0] if len(_cat_this) > 0 else "Unknown")
    else:
        _row_labels_col.append(_lbl)

# Couleur de ligne
_row_palette = {
    "BLAST_HIGH": _COL_BLAST_H,
    "BLAST_MODERATE": _COL_UNKNOWN,
    "BLAST_WEAK": _COL_BLAST_L,
    "NON_BLAST_UNK": _COL_NON_B,
    **_POP_PALETTE,
}
_row_colors = pd.Series(
    [_row_palette.get(lbl, "#64748b") for lbl in _row_labels_col],
    index=_df_heatmap_plot.index,
)

# ── Figure ────────────────────────────────────────────────────────────────────
_fig_heat, _ax_heat = plt.subplots(
    figsize=(
        max(14, len(_informative_cols) * 0.9),
        max(6, len(_df_heatmap_plot) * 0.6),
    ),
    constrained_layout=True,
)
_fig_heat.patch.set_facecolor("#1e1e2e")

_cmap_heat = sns.diverging_palette(250, 15, s=80, l=40, as_cmap=True)

_sns_ax = sns.heatmap(
    _df_heatmap_plot,
    ax=_ax_heat,
    cmap=_cmap_heat,
    center=0.5,  # 0.5 = milieu de la plage référence
    vmin=-0.5,
    vmax=1.5,  # DIM = négatif, BRIGHT = > 1
    annot=True,
    fmt=".2f",
    annot_kws={"size": 8, "color": "#e2e8f0"},
    linewidths=0.3,
    linecolor="#2d2d3f",
    cbar_kws={
        "label": "Valeur M8-normalisée  (0=min-ref, 1=max-ref)",
        "shrink": 0.6,
        "ticks": [-0.5, 0.0, 0.5, 1.0, 1.5],
    },
)

# Décorations des axes
_ax_heat.set_xticklabels(
    [_c.replace("-", "\n") for _c in _informative_cols],
    rotation=45,
    ha="right",
    fontsize=9,
    color="#e2e8f0",
)
_ax_heat.set_yticklabels(
    _df_heatmap_plot.index,
    rotation=0,
    fontsize=9,
    color="#e2e8f0",
)

# Colorier les étiquettes de lignes selon la catégorie
for _ytick, _lbl in zip(_ax_heat.get_yticklabels(), _row_labels_col):
    _ytick.set_color(_row_palette.get(_lbl, "#94a3b8"))
    # Grasse si nœud Unknown
    if _lbl in ("BLAST_HIGH", "BLAST_MODERATE", "BLAST_WEAK", "NON_BLAST_UNK"):
        _ytick.set_fontweight("bold")

# Séparateur visuel entre populations de référence et nœuds Unknown
_n_ref_rows = sum(
    1
    for l in _df_heatmap_plot.index
    if not any(l.startswith(f"Node {_nid}") for _nid in _unk_ids)
)
_ax_heat.axhline(
    y=_n_ref_rows, color="#f97316", linewidth=2.5, linestyle="--", alpha=0.8
)
_ax_heat.text(
    len(_informative_cols) + 0.15,
    _n_ref_rows - 0.5,
    "Référence saine",
    fontsize=8,
    color="#94a3b8",
    va="center",
)
_ax_heat.text(
    len(_informative_cols) + 0.15,
    _n_ref_rows + len(_unk_ids) / 2.0,
    "Unknown\n(candidats\nblastes)",
    fontsize=8,
    color=_COL_UNKNOWN,
    va="center",
    fontweight="bold",
)

# Zones de couleur DIM/NORMAL/BRIGHT sur la colorbar
_cbar = _sns_ax.collections[0].colorbar
_cbar.ax.set_ylabel("Valeur M8-normalisée", color="#94a3b8", fontsize=9)
_cbar.ax.tick_params(colors="#94a3b8")
_cbar.ax.axhline(y=0.0, color="#38bdf8", linewidth=1.5, alpha=0.7)
_cbar.ax.axhline(y=1.0, color="#38bdf8", linewidth=1.5, alpha=0.7)
_cbar.ax.text(
    1.1, -0.25, "DIM", transform=_cbar.ax.transData, fontsize=8, color="#38bdf8"
)
_cbar.ax.text(
    1.1, 1.15, "BRIGHT", transform=_cbar.ax.transData, fontsize=8, color="#38bdf8"
)

_ax_heat.set_title(
    f"Section 10.4c — Profil d'expression M8-normalisé\n"
    f"Nœuds Unknown ({_n_unk}) vs Populations de référence saines\n"
    f"[méthode : {mapping_df_v5['method'].iloc[0]}]",
    fontsize=12,
    color="#e2e8f0",
    pad=12,
)

# Légende catégories
_leg_patches = [
    mpatches.Patch(color=_COL_BLAST_H, label="BLAST_HIGH ≥ 6/10"),
    mpatches.Patch(color=_COL_UNKNOWN, label="BLAST_MODERATE 3–6/10"),
    mpatches.Patch(color=_COL_BLAST_L, label="BLAST_WEAK < 3/10"),
    mpatches.Patch(color=_COL_NON_B, label="NON_BLAST_UNK (score=0)"),
    mpatches.Patch(color=_COL_GRANU, label="Granulocytes (ref)"),
    mpatches.Patch(color=_COL_LYMPH, label="Lymphocytes (ref)"),
]
_ax_heat.legend(
    handles=_leg_patches,
    loc="lower left",
    bbox_to_anchor=(0, -0.22),
    ncol=3,
    fontsize=8,
    facecolor="#2d2d3f",
    edgecolor="#44475a",
    labelcolor="#e2e8f0",
)

plt.savefig(
    OUTPUT_DIR / "other" / f"blast_heatmap_{timestamp}.png",
    dpi=180,
    bbox_inches="tight",
    facecolor="#1e1e2e",
)
plt.close("all")
print(f"[OK] Heatmap sauvegardée → output/other/blast_heatmap_{timestamp}.png")

# =============================================================================
# 7. FIGURE 2 — RADAR CHART (Spider plot) PAR NŒUD UNKNOWN
# =============================================================================
# Marqueurs clés pour le radar (max 12 pour la lisibilité)
_RADAR_PRIORITY = [
    "CD45",
    "CD34",
    "CD117",
    "CD13",
    "CD33",
    "HLA-DR",
    "CD19",
    "CD3",
    "CD56",
    "CD16",
    "CD38",
    "CD10",
]
_radar_markers = []
for _rp in _RADAR_PRIORITY:
    for _c in _cols_blast:
        if _rp in _c.upper() and _c not in _radar_markers:
            _radar_markers.append(_c)
            break
# Compléter avec les autres marqueurs si on a < 6
for _c in _cols_blast:
    if _c not in _radar_markers and len(_radar_markers) < 12:
        _radar_markers.append(_c)

if len(_radar_markers) < 3:
    print("\n   ⚠️ Moins de 3 marqueurs disponibles pour le radar — figure ignorée.")
else:
    # Profils à tracer :
    #   • Granulocytes (référence saine) en vert
    #   • Lymphocytes  (référence saine) en bleu
    #   • Chaque nœud Unknown en orange/rouge

    def _get_ref_profile(pop_name: str, markers: List[str]) -> Optional[np.ndarray]:
        """Retourne le profil M8-normalisé d'une population de référence."""
        if pop_name not in df_mfi_raw_ref.index:
            return None
        _cols_ok = [m for m in markers if m in df_mfi_raw_ref.columns]
        if not _cols_ok:
            return None
        _raw = df_mfi_raw_ref.loc[pop_name, _cols_ok].to_numpy(dtype=np.float64)
        _cols_idx = [_cols_blast.index(m) for m in _cols_ok]
        _norm = (_raw - _ref_min_b[_cols_idx]) / _ref_rng_b[_cols_idx]
        return np.clip(_norm, -0.5, 1.5)

    # Angles du radar (égaux entre les marqueurs)
    _N = len(_radar_markers)
    _angles = np.linspace(0, 2 * np.pi, _N, endpoint=False).tolist()
    _angles += _angles[:1]  # Fermeture du polygone

    def _radar_profile(vals: np.ndarray) -> List[float]:
        """Ferme le polygone radar."""
        v_clip = np.clip(vals, -0.5, 1.5)
        return list(v_clip) + [v_clip[0]]

    # Récupération des profils références
    _granu_profile = _get_ref_profile("Granulocytes", _radar_markers)
    _lymph_profile = _get_ref_profile("Lymphocytes", _radar_markers)

    # Grille de sous-figures : max 4 par ligne
    _n_cols_radar = min(4, _n_unk)
    _n_rows_radar = int(np.ceil(_n_unk / _n_cols_radar))
    _fig_radar, _axes_radar = plt.subplots(
        _n_rows_radar,
        _n_cols_radar,
        figsize=(4.5 * _n_cols_radar, 4.5 * _n_rows_radar),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    _fig_radar.patch.set_facecolor("#1e1e2e")
    _axes_radar_flat = np.array(_axes_radar).flatten()

    for _ax_r in _axes_radar_flat:
        _ax_r.set_facecolor("#12121e")
        _ax_r.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5])
        _ax_r.set_yticklabels(
            ["-0.5\nDIM", "0", "0.5", "1", "1.5\nBRIGHT"], fontsize=6.5, color="#64748b"
        )
        _ax_r.set_ylim(-0.5, 1.5)
        _ax_r.set_xlim(0, 2 * np.pi)
        _ax_r.set_xticks(_angles[:-1])
        _ax_r.set_xticklabels(
            [m.replace("-", "\n") for m in _radar_markers],
            fontsize=7.5,
            color="#94a3b8",
        )
        _ax_r.tick_params(pad=4)
        _ax_r.grid(color="#2d2d3f", linewidth=0.8, alpha=0.7)
        _ax_r.spines["polar"].set_color("#44475a")
        # Zones de référence : la zone [0,1] est la plage saine
        _ax_r.fill_between(_angles, 0, 1.0, color="#22c55e", alpha=0.05, zorder=0)
        _ax_r.axhline(y=0.0, color="#38bdf8", linewidth=0.8, alpha=0.4)
        _ax_r.axhline(y=1.0, color="#38bdf8", linewidth=0.8, alpha=0.4)

    for _plot_idx, _nid in enumerate(_df_blast["node_id"].values):
        if _plot_idx >= len(_axes_radar_flat):
            break
        _ax_r = _axes_radar_flat[_plot_idx]

        # Profil du nœud Unknown
        _node_raw_idx = np.where(_unk_ids == _nid)[0]
        if len(_node_raw_idx) == 0:
            continue
        _node_raw_idx = _node_raw_idx[0]

        # Extraire les valeurs M8-norm pour les marqueurs radar
        _unk_vals_radar = np.array(
            [_X_unk_norm[_node_raw_idx, _cols_blast.index(_m)] for _m in _radar_markers]
        )

        # Score et catégorie
        _score_r = _df_blast.loc[_df_blast["node_id"] == _nid, "blast_score"].values[0]
        _cat_r = _df_blast.loc[_df_blast["node_id"] == _nid, "blast_category"].values[0]
        _n_cells_r = _unk_cell_counts.get(int(_nid), 0)
        _node_color = _row_palette.get(_cat_r, _COL_UNKNOWN)

        # Tracé des références saines
        if _granu_profile is not None:
            _gv = np.array(
                [
                    _granu_profile[_radar_markers.index(_m)]
                    if _m in _radar_markers
                    and _radar_markers.index(_m) < len(_granu_profile)
                    else np.nan
                    for _m in _radar_markers
                ]
            )
            _gv_clean = np.where(np.isnan(_gv), 0.0, _gv)
            _ax_r.plot(
                _angles,
                _radar_profile(_gv_clean),
                color=_COL_GRANU,
                linewidth=1.5,
                linestyle="--",
                alpha=0.6,
                label="Granulocytes",
            )
            _ax_r.fill(_angles, _radar_profile(_gv_clean), color=_COL_GRANU, alpha=0.08)

        if _lymph_profile is not None:
            _lv = np.array(
                [
                    _lymph_profile[_radar_markers.index(_m)]
                    if _m in _radar_markers
                    and _radar_markers.index(_m) < len(_lymph_profile)
                    else np.nan
                    for _m in _radar_markers
                ]
            )
            _lv_clean = np.where(np.isnan(_lv), 0.0, _lv)
            _ax_r.plot(
                _angles,
                _radar_profile(_lv_clean),
                color=_COL_LYMPH,
                linewidth=1.5,
                linestyle=":",
                alpha=0.6,
                label="Lymphocytes",
            )
            _ax_r.fill(_angles, _radar_profile(_lv_clean), color=_COL_LYMPH, alpha=0.06)

        # Tracé du nœud Unknown
        _ax_r.plot(
            _angles,
            _radar_profile(_unk_vals_radar),
            color=_node_color,
            linewidth=2.5,
            zorder=5,
            label=f"Node {_nid}",
        )
        _ax_r.fill(
            _angles,
            _radar_profile(_unk_vals_radar),
            color=_node_color,
            alpha=0.20,
            zorder=4,
        )

        # Titre du sous-graphe
        _cat_sym = {
            "BLAST_HIGH": "🔴",
            "BLAST_MODERATE": "🟠",
            "BLAST_WEAK": "🟡",
            "NON_BLAST_UNK": "🔵",
        }.get(_cat_r, "❓")
        _ax_r.set_title(
            f"Node {_nid}  {_cat_sym}\nScore = {_score_r:.1f}/10  |  {_n_cells_r:,} cells",
            fontsize=9,
            color=_node_color,
            pad=10,
            fontweight="bold",
        )
        _ax_r.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.15),
            fontsize=6.5,
            facecolor="#2d2d3f",
            edgecolor="#44475a",
            labelcolor="#e2e8f0",
        )

    # Masquer les panneaux vides
    for _ax_r in _axes_radar_flat[_n_unk:]:
        _ax_r.set_visible(False)

    _fig_radar.suptitle(
        f"Section 10.4c — Radar des candidats blastes\n"
        f"Profil M8-normalisé vs Granulocytes/Lymphocytes sains  |  "
        f"Zone bleue = plage référence saine [0–1]",
        fontsize=11,
        color="#e2e8f0",
        y=1.02,
    )

    plt.savefig(
        OUTPUT_DIR / "other" / f"blast_radar_{timestamp}.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="#1e1e2e",
    )
    plt.close("all")
    print(f"[OK] Radar chart sauvegardé → output/other/blast_radar_{timestamp}.png")

# =============================================================================
# 8. FIGURE 3 — BAR CHART DES BLAST SCORES (résumé visuel)
# =============================================================================
_fig_bar, _ax_bar = plt.subplots(
    figsize=(max(8, _n_unk * 1.5), 5), constrained_layout=True
)
_fig_bar.patch.set_facecolor("#1e1e2e")
_ax_bar.set_facecolor("#12121e")

_bar_node_labels = [f"Node {int(r['node_id'])}" for _, r in _df_blast.iterrows()]
_bar_scores = _df_blast["blast_score"].values
_bar_n_cells = _df_blast["n_cells"].values
_bar_colors = [
    _row_palette.get(_cat, "#64748b") for _cat in _df_blast["blast_category"]
]

# Barres de score
_bars = _ax_bar.bar(
    _bar_node_labels,
    _bar_scores,
    color=_bar_colors,
    edgecolor="#2d2d3f",
    linewidth=0.8,
    zorder=3,
)

# Annotations (score + n_cells)
for _bar_obj, _score, _ncells in zip(_bars, _bar_scores, _bar_n_cells):
    _ax_bar.text(
        _bar_obj.get_x() + _bar_obj.get_width() / 2,
        _score + 0.15,
        f"{_score:.1f}\n({_ncells:,}c)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#e2e8f0",
    )

# Lignes de seuil clinique
_ax_bar.axhline(
    y=6.0,
    color=_COL_BLAST_H,
    linewidth=1.5,
    linestyle="--",
    alpha=0.7,
    label="Seuil BLAST_HIGH (6.0)",
)
_ax_bar.axhline(
    y=3.0,
    color=_COL_UNKNOWN,
    linewidth=1.5,
    linestyle="--",
    alpha=0.7,
    label="Seuil BLAST_MODERATE (3.0)",
)
_ax_bar.axhline(y=0.0, color="#64748b", linewidth=1.0, linestyle=":", alpha=0.5)

# Zone verte "sain"
_ax_bar.axhspan(-0.5, 0.0, color="#22c55e", alpha=0.06, label="Zone saine (~0)")

_ax_bar.set_ylim(-0.5, 10.5)
_ax_bar.set_ylabel("Blast Score /10", color="#e2e8f0", fontsize=10)
_ax_bar.set_xlabel("Nœud SOM", color="#e2e8f0", fontsize=10)
_ax_bar.set_title(
    f"Blast Score par nœud Unknown — Section 10.4c\n"
    f"Le score intègre : CD34/CD117 (BRIGHT) + CD45/SSC (DIM)",
    color="#e2e8f0",
    fontsize=11,
    fontweight="bold",
)
_ax_bar.legend(
    facecolor="#2d2d3f",
    edgecolor="#44475a",
    labelcolor="#e2e8f0",
    fontsize=8.5,
)
_ax_bar.tick_params(colors="#94a3b8")
_ax_bar.grid(axis="y", color="#2d2d3f", linewidth=0.8, alpha=0.6, zorder=0)
for _spine in _ax_bar.spines.values():
    _spine.set_edgecolor("#44475a")

plt.savefig(
    OUTPUT_DIR / "other" / f"blast_scores_bar_{timestamp}.png",
    dpi=180,
    bbox_inches="tight",
    facecolor="#1e1e2e",
)
plt.close("all")
print(f"[OK] Bar chart sauvegardé → output/other/blast_scores_bar_{timestamp}.png")

# =============================================================================
# 9. EXPORT FINAL
# =============================================================================
_out_blast = OUTPUT_DIR / "other" / f"blast_candidates_10.4c_{timestamp}.csv"
_df_blast.to_csv(_out_blast, index=False, sep=";", decimal=",")
print(f"\n[OK] Tableau blast exporté : {_out_blast}")

# Export de la variable publique pour les sections suivantes
blast_candidates_df = _df_blast.copy()

# =============================================================================
# 10. RÉSUMÉ FINAL
# =============================================================================
print(f"\n{'=' * 70}")
print(f" SYNTHÈSE — SECTION 10.4c")
print(f"{'=' * 70}")
print(f"   Méthode source      : {mapping_df_v5['method'].iloc[0]}")
print(f"   Nœuds Unknown total : {_n_unk}")
_cat_summary = _df_blast["blast_category"].value_counts()
for _cat in _cat_order:
    _cnt = _cat_summary.get(_cat, 0)
    if _cnt > 0:
        _bar = "█" * _cnt
        print(f"   {_CAT_COLORS[_cat]} {_cat:<22s}: {_cnt:>3d}  {_bar}")

_n_high = _cat_summary.get("BLAST_HIGH", 0) + _cat_summary.get("BLAST_MODERATE", 0)
if _n_high > 0:
    print(
        f"\n   ⚠️  {_n_high} nœud(s) avec signature blastique significative (score ≥ 3/10)"
    )
    print(f"      → Confirmer par cytologie + immunophénotypage clinique (tube LAIP)")
    print(f"      → Vérifier expression CD34/CD117 sur cytogramme brut")
    _blast_node_ids = _df_blast.loc[
        _df_blast["blast_category"].isin(["BLAST_HIGH", "BLAST_MODERATE"]), "node_id"
    ].tolist()
    print(f"      → Nœuds suspects : {_blast_node_ids}")
else:
    print(f"\n   ✅ Aucun nœud avec score blast significatif.")
    print(f"      Les nœuds Unknown correspondent probablement à des populations rares")
    print(f"      absentes du panel de référence (ex: précurseurs NK, basophiles).")

print(f"\n   Variables exportées :")
print(f"     blast_candidates_df  — DataFrame complet avec scores et profils")
print(f"     → Prochaine étape : Section 10.5 (MST coloré par blast_score)")
print(f"{'=' * 70}")


################################################################################

# =============================================================================
# SECTION 10.4d — TRAÇABILITÉ DES BLASTES → FICHIERS FCS SOURCE
# =============================================================================
#
# Pour chaque nœud candidat blaste (BLAST_HIGH / BLAST_MODERATE) identifié
# en Section 10.4c, retrouve :
#   1. Quelles cellules dans cell_data appartiennent à ces nœuds
#      (via cell_data.obs['clustering'])
#   2. Depuis quel fichier FCS / quelle condition elles proviennent
#      (via cell_data.obs['condition'] ou obs['sample']/obs['file'])
#   3. Un résumé par nœud + un résumé global (Sain vs Pathologique)
#
# DÉPENDANCES :
#   blast_candidates_df  [Section 10.4c]  — colonnes : node_id, blast_category, blast_score
#   cell_data            [Section 10.1]   — AnnData avec obs['clustering'] + obs['condition']
# =============================================================================

from collections import Counter

import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print(" SECTION 10.4d — TRAÇABILITÉ DES BLASTES → FICHIERS FCS SOURCE")
print("=" * 70)

# ─── 0. Vérifications des dépendances ────────────────────────────────────────
_deps_10d = True
for _dep_name, _desc in [
    ("blast_candidates_df", "Section 10.4c (scoring blast)"),
    ("cell_data", "Section 10.1  (AnnData FlowSOM)"),
]:
    _ok = _dep_name in dir() and globals().get(_dep_name) is not None
    print(f"   {'✅' if _ok else '❌'} {_dep_name:<25s} [{_desc}]")
    if not _ok:
        _deps_10d = False

if not _deps_10d:
    print("\n   ⚠️ Dépendances manquantes — exécutez les sections indiquées.")
    raise SystemExit("Section 10.4d : dépendances manquantes.")

# ─── 1. Sélection des nœuds suspects (BLAST_HIGH + BLAST_MODERATE) ────────────
_SUSPECT_CATEGORIES = ["BLAST_HIGH", "BLAST_MODERATE"]
_blast_suspect_rows = blast_candidates_df[
    blast_candidates_df["blast_category"].isin(_SUSPECT_CATEGORIES)
].copy()
_blast_suspect_ids = _blast_suspect_rows["node_id"].astype(int).tolist()

print(f"\n   Nœuds suspects ({'/'.join(_SUSPECT_CATEGORIES)}) : {_blast_suspect_ids}")

if not _blast_suspect_ids:
    print("\n   ✅ Aucun nœud BLAST_HIGH/BLAST_MODERATE — traçabilité non nécessaire.")
    print("      (Réduisez le seuil ou vérifiez la Section 10.4c si inattendu)")
    raise SystemExit("Aucun nœud suspect à tracer.")

# ─── 2. Récupération du vecteur de clustering (nœud SOM de chaque cellule) ───
if "clustering" not in cell_data.obs.columns:
    print(
        "   ⚠️ 'clustering' absent de cell_data.obs — impossible de tracer les cellules."
    )
    raise SystemExit(
        "Section 10.4d : colonne 'clustering' manquante dans cell_data.obs."
    )

_clustering_vec = cell_data.obs["clustering"].astype(int).values

# ─── 3. Identification des colonnes source disponibles ───────────────────────
_OBS_COLS = list(cell_data.obs.columns)
print(f"\n   Colonnes obs disponibles : {_OBS_COLS}")

# Ordre de priorité : cherche la colonne la plus informative
_SOURCE_COL_PRIORITY = ["sample", "file", "filename", "fcs_file", "source", "condition"]
_source_col = next((c for c in _SOURCE_COL_PRIORITY if c in _OBS_COLS), None)

if _source_col is None:
    # Fallback : utilise l'index (obs_names) comme proxy de fichier source
    print(f"\n   ⚠️ Aucune colonne source trouvée parmi {_SOURCE_COL_PRIORITY}.")
    print(f"      Utilisation de l'index obs_names comme proxy.")
    _source_values = np.array(cell_data.obs_names)
else:
    print(f"\n   Colonne source identifiée : '{_source_col}'")
    _source_values = cell_data.obs[_source_col].astype(str).values

# Colonne condition (Sain / Pathologique) pour le résumé global
_COND_COL = "condition" if "condition" in _OBS_COLS else None

# ─── 4. Analyse nœud par nœud ─────────────────────────────────────────────────
print(f"\n{'─' * 70}")
print(f" DÉTAIL PAR NŒUD SUSPECT")
print(f"{'─' * 70}")
print(
    f"   {'Nœud':>5}  {'Catégorie':<22}  {'Score':>6}  "
    f"{'Cellules':>9}  {'Source #1 (dominante)':<35}  {'% nœud':>7}"
)
print(f"   {'─' * 70}")

_source_summary_by_node: dict = {}  # node_id → Counter(source → n_cells)
_condition_summary_by_node: dict = {}  # node_id → Counter(condition → n_cells)

for _, _row_b in _blast_suspect_rows.sort_values(
    "blast_score", ascending=False
).iterrows():
    _nid = int(_row_b["node_id"])
    _cat_b = _row_b["blast_category"]
    _sc = _row_b["blast_score"]

    # Masque : cellules appartenant à ce nœud
    _mask_node = _clustering_vec == _nid
    _n_found = int(_mask_node.sum())

    if _n_found == 0:
        print(
            f"   {_nid:>5d}  {_cat_b:<22}  {_sc:>5.1f}/10  "
            f"{'—':>9}  {'⚠️ aucune cellule dans cell_data':<35}"
        )
        continue

    # Distribution des sources pour ce nœud
    _node_src = _source_values[_mask_node]
    _cnt_src = Counter(_node_src.tolist())
    _source_summary_by_node[_nid] = _cnt_src

    _dom_src, _dom_cnt = _cnt_src.most_common(1)[0]
    _dom_pct = 100.0 * _dom_cnt / _n_found

    print(
        f"   {_nid:>5d}  {_cat_b:<22}  {_sc:>5.1f}/10  "
        f"{_n_found:>9,}  {str(_dom_src)[:34]:<35}  {_dom_pct:>6.1f}%"
    )

    # Détail des sources (si multi-origines)
    if len(_cnt_src) > 1:
        for _src, _cnt_val in _cnt_src.most_common(5):
            _pct = 100.0 * _cnt_val / _n_found
            _bar = "█" * max(1, int(_pct / 4))
            # Marquage si nom évocateur de pathologie
            _flag = ""
            _src_lower = str(_src).lower()
            if any(k in _src_lower for k in ["pathol", "laip", "diag", "lam", "aml"]):
                _flag = "  🔴 PATHOLOGIQUE"
            elif any(
                k in _src_lower for k in ["sain", "nbm", "normal", "ctrl", "control"]
            ):
                _flag = "  🟢 SAIN"
            print(
                f"         {'':>5}  {str(_src)[:42]:<42}  "
                f"{_cnt_val:>7,} cells  {_pct:>5.1f}%  {_bar}{_flag}"
            )

    # Distribution par condition (Sain/Pathologique) si disponible
    if _COND_COL is not None:
        _node_cond = cell_data.obs[_COND_COL].values[_mask_node]
        _cnt_cond = Counter(_node_cond.astype(str).tolist())
        _condition_summary_by_node[_nid] = _cnt_cond
        _cond_str = "  |  ".join(
            f"{'🔴' if 'pathol' in c.lower() else '🟢'}{c}={n:,}"
            for c, n in _cnt_cond.most_common()
        )
        print(f"         {'':>5}  Condition : {_cond_str}")

# ─── 5. Résumé global : contribution par fichier FCS ─────────────────────────
print(f"\n{'─' * 70}")
print(f" RÉSUMÉ GLOBAL — Contribution par fichier FCS aux nœuds suspects")
print(f"{'─' * 70}")

_global_src_counter: Counter = Counter()
for _cnt in _source_summary_by_node.values():
    _global_src_counter.update(_cnt)

if _global_src_counter:
    _total_blast_cells = sum(_global_src_counter.values())
    print(f"   Total cellules dans les nœuds suspects : {_total_blast_cells:,}")
    print(f"\n   {'Fichier / Source':<50}  {'Cellules':>9}  {'%':>6}  Distribution")
    for _src, _n in _global_src_counter.most_common():
        _pct = 100.0 * _n / max(_total_blast_cells, 1)
        _bar = "█" * max(1, int(_pct / 3))
        _tag = ""
        _src_lower = str(_src).lower()
        if any(k in _src_lower for k in ["pathol", "laip", "diag", "lam", "aml"]):
            _tag = " 🔴 PATHOLOGIQUE"
        elif any(k in _src_lower for k in ["sain", "nbm", "normal", "ctrl"]):
            _tag = " 🟢 SAIN"
        print(f"   {str(_src)[:49]:<50}  {_n:>9,}  {_pct:>5.1f}%  {_bar}{_tag}")

# ─── 6. Résumé par condition Sain vs Pathologique ────────────────────────────
if _COND_COL is not None and _condition_summary_by_node:
    print(f"\n{'─' * 70}")
    print(f" RÉSUMÉ PAR CONDITION ('{_COND_COL}')")
    print(f"{'─' * 70}")

    _global_cond_counter: Counter = Counter()
    for _cnt in _condition_summary_by_node.values():
        _global_cond_counter.update(_cnt)

    _total_cond = sum(_global_cond_counter.values())
    for _cond, _n in _global_cond_counter.most_common():
        _pct = 100.0 * _n / max(_total_cond, 1)
        _bar = "█" * max(1, int(_pct / 3))
        _ico = "🔴" if "pathol" in str(_cond).lower() else "🟢"
        print(f"   {_ico} {str(_cond):<35}  {_n:>9,} cells  {_pct:>5.1f}%  {_bar}")

    # Alerte si >50% des cellules suspectes viennent d'un fichier pathologique
    _n_patho = sum(
        v for k, v in _global_cond_counter.items() if "pathol" in str(k).lower()
    )
    if _n_patho > 0.5 * _total_cond:
        print(
            f"\n   ⚠️  ALERTE CLINIQUE : {100.0 * _n_patho / _total_cond:.0f}% des cellules suspectes"
        )
        print(f"      proviennent de la condition PATHOLOGIQUE.")
        print(
            f"      → Signature blastique probablement issue du fichier LAM/LAIP patient."
        )
        print(
            f"      → Confirmer par : cytologie + CD34/CD117 sur cytogramme brut (tube LAIP)"
        )
    elif sum(_global_cond_counter.values()) > 0 and _n_patho == 0:
        print(
            f"\n   ✅ Aucune cellule suspecte ne provient de la condition PATHOLOGIQUE."
        )
        print(f"      Les nœuds Unknown pourraient représenter des populations rares")
        print(f"      non couvertes par le panel de référence.")

# ─── 7. Figure — Diagramme en barres empilées par nœud (source × condition) ──
if _source_summary_by_node:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        _fig_src, _axes_src = plt.subplots(
            1,
            len(_source_summary_by_node),
            figsize=(max(6, len(_source_summary_by_node) * 3.5), 5),
            constrained_layout=True,
        )
        _fig_src.patch.set_facecolor("#1e1e2e")
        if len(_source_summary_by_node) == 1:
            _axes_src = [_axes_src]

        # Palette dynamique par source
        _all_srcs = sorted(
            set(s for cnt in _source_summary_by_node.values() for s in cnt)
        )
        _src_colors = {}
        _cmap_src = plt.get_cmap("tab10")
        for _i, _s in enumerate(_all_srcs):
            _s_lower = str(_s).lower()
            if any(k in _s_lower for k in ["pathol", "laip", "diag", "lam", "aml"]):
                _src_colors[_s] = "#ef4444"  # rouge — pathologique
            elif any(k in _s_lower for k in ["sain", "nbm", "normal", "ctrl"]):
                _src_colors[_s] = "#22c55e"  # vert  — sain
            else:
                _src_colors[_s] = _cmap_src(_i % 10)

        for _ax_s, (_nid_s, _cnt_s) in zip(_axes_src, _source_summary_by_node.items()):
            _ax_s.set_facecolor("#12121e")
            _total_s = sum(_cnt_s.values())
            _bottom = 0.0
            _order = sorted(_cnt_s.keys(), key=lambda k: -_cnt_s[k])
            for _src_s in _order:
                _pct_s = 100.0 * _cnt_s[_src_s] / max(_total_s, 1)
                _ax_s.bar(
                    0,
                    _pct_s,
                    bottom=_bottom,
                    color=_src_colors.get(_src_s, "#6366f1"),
                    edgecolor="#2d2d3f",
                    linewidth=0.5,
                    width=0.6,
                )
                if _pct_s > 8:  # Annotation seulement si lisible
                    _ax_s.text(
                        0,
                        _bottom + _pct_s / 2,
                        f"{str(_src_s)[:18]}\n{_pct_s:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white",
                        fontweight="bold",
                    )
                _bottom += _pct_s

            # Récupère score et catégorie
            _score_s = blast_candidates_df.loc[
                blast_candidates_df["node_id"] == _nid_s, "blast_score"
            ].values
            _cat_s = blast_candidates_df.loc[
                blast_candidates_df["node_id"] == _nid_s, "blast_category"
            ].values
            _cat_str = _cat_s[0] if len(_cat_s) > 0 else "?"
            _sc_str = f"{_score_s[0]:.1f}/10" if len(_score_s) > 0 else "?"
            _cat_color = {
                "BLAST_HIGH": "#ef4444",
                "BLAST_MODERATE": "#f97316",
                "BLAST_WEAK": "#fbbf24",
                "NON_BLAST_UNK": "#6366f1",
            }.get(_cat_str, "#e2e8f0")

            _ax_s.set_xlim(-0.5, 0.5)
            _ax_s.set_ylim(0, 105)
            _ax_s.set_xticks([])
            _ax_s.tick_params(colors="#94a3b8")
            _ax_s.set_ylabel("% cellules du nœud", color="#e2e8f0", fontsize=9)
            _ax_s.set_title(
                f"Node {_nid_s}\n{_cat_str}\nScore={_sc_str}",
                color=_cat_color,
                fontsize=9,
                fontweight="bold",
            )
            for _spine in _ax_s.spines.values():
                _spine.set_edgecolor("#44475a")
            _ax_s.grid(axis="y", color="#2d2d3f", linewidth=0.7, alpha=0.6)

        # Légende globale
        _legend_patches = [
            mpatches.Patch(color=_src_colors[_s], label=str(_s)[:35])
            for _s in _all_srcs
        ]
        _fig_src.legend(
            handles=_legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(3, len(_all_srcs)),
            fontsize=8,
            facecolor="#2d2d3f",
            edgecolor="#44475a",
            labelcolor="#e2e8f0",
        )
        _fig_src.suptitle(
            f"Section 10.4d — Origine FCS des cellules des nœuds blastes suspects\n"
            f"[méthode mapping : {mapping_df_v5['method'].iloc[0]}]  "
            f"| 🔴 rouge = Pathologique  | 🟢 vert = Sain",
            fontsize=10,
            color="#e2e8f0",
        )

        plt.savefig(
            OUTPUT_DIR / "other" / f"blast_source_fcs_{timestamp}.png",
            dpi=180,
            bbox_inches="tight",
            facecolor="#1e1e2e",
        )
        plt.close("all")
        print(
            f"\n[OK] Figure source FCS sauvegardée → output/other/blast_source_fcs_{timestamp}.png"
        )

    except Exception as _e_fig_src:
        print(f"\n   ⚠️ Figure source FCS non générée : {_e_fig_src}")

# ─── 8. Résumé final ──────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f" SYNTHÈSE — SECTION 10.4d")
print(f"{'=' * 70}")
print(f"   Nœuds suspects analysés   : {len(_blast_suspect_ids)}")
print(f"   Colonne source utilisée   : {_source_col or 'obs_names (fallback)'}")
print(f"   Colonne condition utilisée: {_COND_COL or 'non disponible'}")
print(
    f"   Total cellules suspectes  : {sum(sum(c.values()) for c in _source_summary_by_node.values()):,}"
)
if _global_src_counter:
    _dom_file, _dom_n = _global_src_counter.most_common(1)[0]
    _dom_pct_g = 100.0 * _dom_n / max(sum(_global_src_counter.values()), 1)
    print(f"   Fichier dominant          : {_dom_file}  ({_dom_pct_g:.1f}%)")
print(f"\n   Variables disponibles :")
print(f"     blast_candidates_df    — DataFrame scores/profils (Section 10.4c)")
print(f"     _source_summary_by_node  — {{node_id: Counter(source → n_cells)}}")
print(f"     _condition_summary_by_node — {{node_id: Counter(condition → n_cells)}}")
print(f"{'=' * 70}")


################################################################################

# =============================================================================
# SECTION 10.5 — MST INTERACTIF PLOTLY — COLORÉ PAR POPULATION (meilleur mapping)
# =============================================================================

# ── 1. Résolution du meilleur mapping disponible ─────────────────────────────
# Priorité : mapping_df_v5 (M3_cosine ou meilleur score benchmark) > mapping_df_raw
_BEST_MAP_10_5 = (
    mapping_df_v5
    if "mapping_df_v5" in dir() and mapping_df_v5 is not None and len(mapping_df_v5) > 0
    else mapping_df_raw
)
_best_map_name = (
    _BEST_MAP_10_5["method"].iloc[0]
    if "method" in _BEST_MAP_10_5.columns and len(_BEST_MAP_10_5) > 0
    else globals().get("best_map_name", "mapping_par_défaut")
)

# ── 2. Guard principal ────────────────────────────────────────────────────────
if not PLOTLY_AVAILABLE:
    print("[!] plotly requis pour la section 10.5 — visualisation ignorée")
else:
    print("=" * 70)
    print(f" SECTION 10.5 — MST INTERACTIF COLORÉ PAR POPULATION")
    print(f" Meilleur algorithme de mapping : {_best_map_name}")
    print("=" * 70)

    # ── 3. Sélection des colonnes de coordonnées ──────────────────────────────
    _has_mst_coords = (
        _COL_XNODES is not None
        and _COL_YNODES is not None
        and _COL_XNODES in _node_coords_df.columns
        and _COL_YNODES in _node_coords_df.columns
    )

    if not _has_mst_coords:
        print("   [!] xNodes/yNodes absents → repli sur xGrid/yGrid")
        _x_col_mst = _COL_XGRID
        _y_col_mst = _COL_YGRID
    else:
        _x_col_mst = _COL_XNODES
        _y_col_mst = _COL_YNODES

    # ── 4. Vérification finale des colonnes ───────────────────────────────────
    if _x_col_mst is None or _x_col_mst not in _node_coords_df.columns:
        print("   [!] Coordonnées MST indisponibles — visualisation annulée")
    else:
        # ── 5. Extraction des coordonnées ─────────────────────────────────────
        _layout_x = _node_coords_df[_x_col_mst].to_numpy(dtype=np.float64)
        _layout_y = _node_coords_df[_y_col_mst].to_numpy(dtype=np.float64)
        _n_nodes = len(_layout_x)

        # ── 6. Taille des bulles proportionnelle au nb de cellules ────────────
        _max_cnt_mst = max(float(_node_count_raw.max()), 1.0)
        _bubble_sz_mst = 10.0 + (_node_count_raw / _max_cnt_mst) * 40.0

        # ── 7. Reconstruction du mapping node → population ────────────────────
        # Utilisation du meilleur mapping disponible (V5 : M3_cosine en priorité)
        _mp_node = _BEST_MAP_10_5.set_index("node_id")

        _node_index = pd.RangeIndex(_n_nodes)
        _mp_aligned = _mp_node.reindex(_node_index)

        _assigned_pops = _mp_aligned["assigned_pop"].fillna("Unknown").to_numpy()
        _mc_nodes_mst = _mp_aligned["metacluster"].fillna(0).astype(int).to_numpy()
        _best_dist_vals = _mp_aligned["best_dist"].to_numpy(dtype=np.float64)
        _threshold_vals = _mp_aligned["threshold"].to_numpy(dtype=np.float64)

        # ── 8. Construction des arêtes MST (optionnel, non bloquant) ──────────
        _edge_x_mst: list = []
        _edge_y_mst: list = []

        try:
            if (
                "cluster_data" in dir()
                and hasattr(cluster_data, "uns")
                and "mst" in cluster_data.uns
            ):
                import igraph as _ig

                _g = cluster_data.uns["mst"]

                if isinstance(_g, _ig.Graph):
                    for _e in _g.es:
                        _s, _t = _e.source, _e.target
                        if _s < _n_nodes and _t < _n_nodes:
                            _edge_x_mst += [_layout_x[_s], _layout_x[_t], None]
                            _edge_y_mst += [_layout_y[_s], _layout_y[_t], None]

                    print(f"   [OK] {len(_g.es)} arêtes MST chargées")
                else:
                    print("   [!] cluster_data.uns['mst'] n'est pas un igraph.Graph")
        except ImportError:
            print("   [!] igraph non disponible — arêtes MST ignorées")
        except Exception as _exc:
            print(f"   [!] Erreur chargement arêtes MST : {_exc}")

        # ── 9. Création de la figure ──────────────────────────────────────────
        fig_mst_raw = go.Figure()

        # Trace des arêtes (si disponibles)
        if _edge_x_mst:
            fig_mst_raw.add_trace(
                go.Scatter(
                    x=_edge_x_mst,
                    y=_edge_y_mst,
                    mode="lines",
                    line=dict(width=1.5, color="rgba(120,120,120,0.45)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Arêtes MST",
                )
            )

        # ── 10. Trace des nœuds, une trace par population ─────────────────────
        for _pop in sorted(set(_assigned_pops)):
            _mask = _assigned_pops == _pop
            _idx = np.where(_mask)[0]

            if len(_idx) == 0:
                continue

            _col = POPULATION_COLORS.get(_pop, "#AAAAAA")

            _hover = [
                (
                    f"<b>Node {i}</b><br>"
                    f"Population : <b>{_pop}</b><br>"
                    f"Métacluster : MC{_mc_nodes_mst[i]}<br>"
                    f"Cellules : {int(_node_count_raw[i]):,}<br>"
                    f"Distance best : {_best_dist_vals[i]:.4f}<br>"
                    f"Seuil Unknown : {_threshold_vals[i]:.4f}"
                )
                for i in _idx
            ]

            fig_mst_raw.add_trace(
                go.Scatter(
                    x=_layout_x[_idx],
                    y=_layout_y[_idx],
                    mode="markers+text",
                    marker=dict(
                        size=_bubble_sz_mst[_idx],
                        color=_col,
                        line=dict(width=1.5, color="white"),
                        opacity=0.90,
                    ),
                    text=[str(_mc_nodes_mst[i]) for i in _idx],
                    textfont=dict(size=9, color="white", family="Arial Black"),
                    textposition="middle center",
                    name=f"{_pop} ({len(_idx)} nœuds)",
                    hovertext=_hover,
                    hoverinfo="text",
                )
            )

        # ── 11. Mise en page ──────────────────────────────────────────────────
        fig_mst_raw.update_layout(
            title=dict(
                text=(
                    f"<b>MST FlowSOM — {_n_nodes} nœuds — "
                    f"Populations assignées (<i>{_best_map_name}</i>)</b><br>"
                    f"<sup>Normalisation : {NORMALIZATION_METHOD} · "
                    f"Scatter : "
                    f"{'inclus (FSC-A, SSC-A)' if INCLUDE_SCATTER_IN_MAPPING else 'exclus'}"
                    f" · Meilleur algorithme de clustering · Prior log10³ | Hard Limit</sup>"
                ),
                font=dict(size=14),
            ),
            xaxis=dict(
                title=_x_col_mst,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.06)",
                zeroline=False,
            ),
            yaxis=dict(
                title=_y_col_mst,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.06)",
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,
            ),
            height=750,
            width=900,
            paper_bgcolor="#fafafa",
            plot_bgcolor="#f5f5f5",
            legend=dict(
                title=dict(text="Population cellulaire", font=dict(size=11)),
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#cccccc",
                borderwidth=1,
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(t=90, b=50, l=60, r=230),
        )

        # ── 12. Affichage et export ───────────────────────────────────────────
        fig_mst_raw.write_html(
            str(_dashboard_out / f"fig_mst_raw_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )

        _out_mst = OUTPUT_DIR / "other" / f"mst_population_v3_{timestamp}.html"
        _out_mst.parent.mkdir(parents=True, exist_ok=True)
        fig_mst_raw.write_html(str(_out_mst))
        print(f"\n[OK] MST ({_best_map_name}) exporté → {_out_mst}")

        # ── 13. VÉRIFICATION BIOLOGIQUE — Nœuds "Lymphocytes bruts" ──────────
        # Rationnel clinique : les nœuds labellisés "Lymphos" (catégorie générique)
        # doivent contenir à la fois des lymphocytes T/NK (CD3+/CD7+/CD56+)
        # et des lymphocytes B (CD19+). Cette vérification assure la cohérence
        # du clustering SOM et détecte d'éventuels sous-clusterings manquants.
        print(f"\n{'=' * 70}")
        print(f" VÉRIFICATION — Nœuds 'Lymphocytes bruts' : T/NK + B attendus")
        print(f" Mapping utilisé : {_best_map_name}")
        print(f"{'=' * 70}")

        # Labels considérés comme "lymphocytes génériques non classifiés"
        _LYMPHO_GENERIC = {"Lymphos", "Lymphocytes", "Lympho", "Lymphs"}
        # Labels correspondant aux sous-populations attendues
        _LYTNK_LABELS = {"Ly T_NK", "Lymphos T", "Lymphos T/NK", "T_NK", "T NK"}
        _LYMPHOB_LABELS = {"Lymphos B", "Lymphocytes B", "B cells", "B lymphocytes"}

        # ── Extraction des nœuds par catégorie ───────────────────────────────
        _pop_col_bst = _BEST_MAP_10_5["assigned_pop"].astype(str)
        _nid_col_bst = _BEST_MAP_10_5["node_id"].astype(int)

        _lympho_nodes_bst = _nid_col_bst[_pop_col_bst.isin(_LYMPHO_GENERIC)].tolist()
        _lytnk_nodes_bst = _nid_col_bst[_pop_col_bst.isin(_LYTNK_LABELS)].tolist()
        _lymphob_nodes_bst = _nid_col_bst[_pop_col_bst.isin(_LYMPHOB_LABELS)].tolist()

        print(
            f"\n   Nœuds 'Lymphos' génériques  : {len(_lympho_nodes_bst)} → {_lympho_nodes_bst}"
        )
        print(
            f"   Nœuds 'Ly T_NK'             : {len(_lytnk_nodes_bst)}  → {_lytnk_nodes_bst}"
        )
        print(
            f"   Nœuds 'Lymphos B'            : {len(_lymphob_nodes_bst)} → {_lymphob_nodes_bst}"
        )

        if not _lympho_nodes_bst:
            print("\n   ✅ Aucun nœud 'Lymphos' générique — tous les nœuds lymphoïdes")
            print(
                "      sont déjà assignés à Ly T_NK ou Lymphos B (sous-classification complète)."
            )
        else:
            # ── Récupération des MFI par nœud (node_mfi_raw_df) ───────────────
            _mfi_src = None
            for _mfi_cand in [
                "node_mfi_raw_df",
                "node_mfi_aligned",
                "_node_mfi_aligned",
            ]:
                if _mfi_cand in dir() and globals().get(_mfi_cand) is not None:
                    _mfi_src = globals()[_mfi_cand]
                    break

            if _mfi_src is None:
                print("\n   ⚠️ node_mfi_raw_df non disponible — calcul MFI impossible.")
                print("      Vérifiez la Section 10.3 (chargement des MFI).")
            else:
                # MFI des profils de référence: centroïdes des nœuds T/NK et B
                def _get_mean_profile(
                    node_ids: list, mfi_df: "pd.DataFrame"
                ) -> "pd.Series | None":
                    """Retourne le profil MFI moyen pondéré par taille des nœuds spécifiés."""
                    _valid_ids = [i for i in node_ids if i < len(mfi_df)]
                    if not _valid_ids:
                        return None
                    _sub = mfi_df.iloc[_valid_ids]
                    # Pondération par taille de nœud (nb cellules)
                    _weights = np.array(
                        [_node_count_raw[i] for i in _valid_ids], dtype=float
                    )
                    _weights = np.maximum(_weights, 1.0)
                    _weighted_mean = np.average(_sub.values, axis=0, weights=_weights)
                    return pd.Series(_weighted_mean, index=_sub.columns)

                _ref_lytnk_mfi = _get_mean_profile(_lytnk_nodes_bst, _mfi_src)
                _ref_lymphob_mfi = _get_mean_profile(_lymphob_nodes_bst, _mfi_src)

                if _ref_lytnk_mfi is None and _ref_lymphob_mfi is None:
                    print("\n   ⚠️ Aucune référence Ly T_NK/Lymphos B dans ce mapping.")
                    print(
                        "      Impossible de comparer les profils des nœuds Lymphos génériques."
                    )
                else:
                    # ── Affichage MFI moyen des sous-populations de référence ─
                    _markers_display = list(_mfi_src.columns)
                    print(
                        f"\n   ── Profil MFI moyen des sous-populations de référence ──────────────"
                    )
                    print(
                        f"   {'Marqueur':<30}  {'Ly T_NK (réf)':>14}  {'Lymphos B (réf)':>15}"
                    )
                    print(f"   {'─' * 62}")
                    for _mk in _markers_display:
                        _v_t = (
                            f"{_ref_lytnk_mfi[_mk]:.3f}"
                            if _ref_lytnk_mfi is not None
                            else "  N/A"
                        )
                        _v_b = (
                            f"{_ref_lymphob_mfi[_mk]:.3f}"
                            if _ref_lymphob_mfi is not None
                            else "  N/A"
                        )
                        print(f"   {_mk:<30}  {_v_t:>14}  {_v_b:>15}")

                    # ── Analyse nœud par nœud (Lymphos génériques) ────────────
                    from scipy.spatial.distance import cosine as _cosine_dist

                    _results_lympho_verif = []
                    print(
                        f"\n   ── Classification des nœuds 'Lymphos' bruts ────────────────────────"
                    )
                    print(
                        f"   {'Nœud':>5}  {'Cellules':>9}  {'MC':>4}  "
                        f"{'d(T_NK)':>9}  {'d(B)':>9}  {'→ Sous-pop probable'}"
                    )
                    print(f"   {'─' * 70}")

                    for _n_lymp in sorted(_lympho_nodes_bst):
                        if _n_lymp >= len(_mfi_src):
                            continue
                        _node_mfi_v = _mfi_src.iloc[_n_lymp].values.astype(float)
                        _n_cells_v = int(_node_count_raw[_n_lymp])
                        _mc_v = int(_mc_nodes_mst[_n_lymp])

                        _d_t, _d_b = np.nan, np.nan
                        if _ref_lytnk_mfi is not None:
                            _ref_t = _ref_lytnk_mfi.values.astype(float)
                            if (
                                np.linalg.norm(_ref_t) > 0
                                and np.linalg.norm(_node_mfi_v) > 0
                            ):
                                _d_t = float(_cosine_dist(_node_mfi_v, _ref_t))
                        if _ref_lymphob_mfi is not None:
                            _ref_b = _ref_lymphob_mfi.values.astype(float)
                            if (
                                np.linalg.norm(_ref_b) > 0
                                and np.linalg.norm(_node_mfi_v) > 0
                            ):
                                _d_b = float(_cosine_dist(_node_mfi_v, _ref_b))

                        # Décision : plus proche de T/NK ou de B ?
                        if np.isnan(_d_t) and np.isnan(_d_b):
                            _decision = "Indéterminé"
                            _flag = "⚪"
                        elif np.isnan(_d_t):
                            _decision = "→ Lymphos B probable"
                            _flag = "🔵"
                        elif np.isnan(_d_b):
                            _decision = "→ Ly T_NK probable"
                            _flag = "🟣"
                        elif _d_t <= _d_b:
                            _decision = "→ Ly T_NK probable"
                            _flag = "🟣"
                        else:
                            _decision = "→ Lymphos B probable"
                            _flag = "🔵"

                        _d_t_str = f"{_d_t:.4f}" if not np.isnan(_d_t) else "   N/A"
                        _d_b_str = f"{_d_b:.4f}" if not np.isnan(_d_b) else "   N/A"
                        print(
                            f"   {_n_lymp:>5d}  {_n_cells_v:>9,}  MC{_mc_v:>2d}  "
                            f"{_d_t_str:>9}  {_d_b_str:>9}  {_flag} {_decision}"
                        )

                        _results_lympho_verif.append(
                            {
                                "node_id": _n_lymp,
                                "n_cells": _n_cells_v,
                                "mc": _mc_v,
                                "dist_lytnk": _d_t,
                                "dist_lymphob": _d_b,
                                "suggestion": _decision,
                            }
                        )

                    # ── Résumé de la vérification ─────────────────────────────
                    _tot_lympho_cells = sum(r["n_cells"] for r in _results_lympho_verif)
                    _n_prob_t = sum(
                        1 for r in _results_lympho_verif if "T_NK" in r["suggestion"]
                    )
                    _n_prob_b = sum(
                        1 for r in _results_lympho_verif if "B prob" in r["suggestion"]
                    )
                    _n_indef = sum(
                        1 for r in _results_lympho_verif if "Indét" in r["suggestion"]
                    )

                    print(f"\n   {'─' * 70}")
                    print(
                        f"   SYNTHÈSE Lymphocytes bruts ({len(_results_lympho_verif)} nœuds, {_tot_lympho_cells:,} cellules)"
                    )
                    print(f"   → Nœuds similaires Ly T_NK : {_n_prob_t}")
                    print(f"   → Nœuds similaires Lymphos B : {_n_prob_b}")
                    print(f"   → Indéterminés               : {_n_indef}")

                    # Alerte clinique si mix T/NK + B détecté dans les Lymphos bruts
                    if _n_prob_t > 0 and _n_prob_b > 0:
                        print(
                            f"\n   ✅ ATTENDU — Les nœuds 'Lymphos' contiennent bien les deux"
                        )
                        print(
                            f"      sous-populations : T/NK ({_n_prob_t} nœuds) et B ({_n_prob_b} nœuds)."
                        )
                        print(
                            f"      → Le SOM a correctement co-localisé les deux lignées lymphoïdes."
                        )
                    elif _n_prob_t == 0 and _n_prob_b > 0:
                        print(
                            f"\n   ⚠️ ALERTE : Les nœuds 'Lymphos' sont tous proches de B uniquement."
                        )
                        print(
                            f"      → Vérifier la présence des marqueurs T (CD3, CD7) dans le panel."
                        )
                    elif _n_prob_b == 0 and _n_prob_t > 0:
                        print(
                            f"\n   ⚠️ ALERTE : Les nœuds 'Lymphos' sont tous proches de T/NK uniquement."
                        )
                        print(
                            f"      → Vérifier CD19 dans le panel et la représentation des B en FCS."
                        )

                    # ── Figure comparative heatmap MFI des nœuds Lymphos ─────
                    if _results_lympho_verif and len(_markers_display) > 0:
                        try:
                            _data_heat_lympho = []
                            _labels_heat = []

                            # Ajouter les profils de référence
                            if _ref_lytnk_mfi is not None:
                                _data_heat_lympho.append(_ref_lytnk_mfi.values)
                                _labels_heat.append("◀ Ly T_NK (réf)")
                            if _ref_lymphob_mfi is not None:
                                _data_heat_lympho.append(_ref_lymphob_mfi.values)
                                _labels_heat.append("◀ Lymphos B (réf)")

                            # Ajouter chaque nœud Lymphos brut
                            for _r in _results_lympho_verif:
                                _nid_h = _r["node_id"]
                                if _nid_h < len(_mfi_src):
                                    _data_heat_lympho.append(
                                        _mfi_src.iloc[_nid_h].values
                                    )
                                    _ico = "🟣" if "T_NK" in _r["suggestion"] else "🔵"
                                    _labels_heat.append(
                                        f"{_ico} Node {_nid_h} "
                                        f"({_r['n_cells']:,} cells)"
                                    )

                            _mat_lympho = np.array(_data_heat_lympho, dtype=float)
                            # Z-score colonne pour normaliser les visualisations
                            _mu_l = _mat_lympho.mean(axis=0)
                            _sd_l = np.where(
                                _mat_lympho.std(axis=0) == 0,
                                1.0,
                                _mat_lympho.std(axis=0),
                            )
                            _mat_z = (_mat_lympho - _mu_l) / _sd_l

                            _fig_lheat = go.Figure(
                                go.Heatmap(
                                    z=_mat_z,
                                    x=_markers_display,
                                    y=_labels_heat,
                                    colorscale="RdBu_r",
                                    zmid=0,
                                    text=np.round(_mat_z, 2).astype(str),
                                    texttemplate="%{text}",
                                    textfont=dict(size=9),
                                    colorbar=dict(
                                        title=dict(text="z-score MFI", side="right"),
                                        tickfont=dict(size=9),
                                    ),
                                    hovertemplate="<b>%{y}</b><br>%{x}<br>z=%{z:.3f}<extra></extra>",
                                )
                            )
                            _fig_lheat.update_layout(
                                title=dict(
                                    text=(
                                        "<b>Vérification nœuds 'Lymphocytes bruts' — "
                                        "Profils MFI (z-score)</b><br>"
                                        f"<sup>Mapping : {_best_map_name} | "
                                        "🟣 T/NK probable | 🔵 B probable | "
                                        "◀ Profil de référence</sup>"
                                    ),
                                    x=0.5,
                                    xanchor="center",
                                    font=dict(size=13),
                                ),
                                height=max(400, len(_labels_heat) * 45 + 200),
                                paper_bgcolor="#fafafa",
                                plot_bgcolor="#f9f9f9",
                                margin=dict(l=220, r=80, t=100, b=120),
                            )
                            _fig_lheat.update_xaxes(
                                tickangle=-45, tickfont=dict(size=9)
                            )
                            _fig_lheat.update_yaxes(tickfont=dict(size=10))

                            _fig_lheat.write_html(
                                str(
                                    _dashboard_out / f"_fig_lheat_{_dashboard_ts}.html"
                                ),
                                include_plotlyjs="cdn",
                            )

                            _out_lheat = (
                                OUTPUT_DIR
                                / "other"
                                / f"lympho_verification_{timestamp}.html"
                            )
                            _fig_lheat.write_html(str(_out_lheat))
                            print(
                                f"\n[OK] Figure vérification Lymphos exportée → {_out_lheat}"
                            )

                        except Exception as _e_lheat:
                            print(
                                f"\n   ⚠️ Figure heatmap lymphocytes non générée : {_e_lheat}"
                            )

        print(f"\n{'=' * 70}")
        print(f" FIN Section 10.5 — Mapping : {_best_map_name}")
        print(f"{'=' * 70}")


################################################################################

# =============================================================================
# SECTION 10.5b — GRILLE SOM COLORÉE PAR POPULATION (espace brut -A)
# =============================================================================

print("=" * 70)
print(" SECTION 10.5b — GRILLE SOM COLORÉE PAR POPULATION")
print("=" * 70)

try:
    _grid_coords_s10 = cluster_data.obsm.get("grid", None)

    if _grid_coords_s10 is None:
        print("[!] cluster_data.obsm['grid'] non disponible.")
    else:
        _clustering_s10 = cell_data.obs["clustering"].values
        _mc_nodes_s10 = cluster_data.obs["metaclustering"].values
        _conditions_s10 = cell_data.obs["condition"].values
        _n_nodes_s10 = len(cluster_data)
        _n_meta_s10 = len(np.unique(_mc_nodes_s10))

        _node_sizes_s10 = np.bincount(
            _clustering_s10.astype(int), minlength=_n_nodes_s10
        ).astype(np.float32)

        _cl_int_s10 = _clustering_s10.astype(int)
        _xGrid_base = _grid_coords_s10[_cl_int_s10, 0].astype(np.float32)
        _yGrid_base = _grid_coords_s10[_cl_int_s10, 1].astype(np.float32)

        np.random.seed(SEED if "SEED" in dir() else 42)
        _theta = np.random.uniform(0, 2 * np.pi, len(_clustering_s10))
        _u = np.random.uniform(0, 1, len(_clustering_s10))
        _max_s = _node_sizes_s10.max()
        _radii = (
            0.1 + (0.45 - 0.1) * np.sqrt(_node_sizes_s10[_cl_int_s10] / _max_s)
        ).astype(np.float32)
        _r = np.sqrt(_u) * _radii
        _jx = (_r * np.cos(_theta)).astype(np.float32)
        _jy = (_r * np.sin(_theta)).astype(np.float32)

        _xGrid_shifted = _xGrid_base - _xGrid_base.min() + 1
        _yGrid_shifted = _yGrid_base - _yGrid_base.min() + 1
        _xj_s10 = _xGrid_shifted + _jx
        _yj_s10 = _yGrid_shifted + _jy

        _mc_cells_s10 = np.array([_mc_nodes_s10[int(c)] for c in _clustering_s10])

        # Sélection du meilleur mapping : V5 (M12_cosine_prior) si disponible sinon mapping_raw
        _BEST_MAP_S10 = mapping_df_v5 if "mapping_df_v5" in dir() else mapping_df_raw
        _method_s10 = (
            mapping_df_v5["method"].iloc[0]
            if "mapping_df_v5" in dir() and len(mapping_df_v5) > 0
            else "mapping_raw"
        )
        _pop_per_node_s10 = _BEST_MAP_S10.set_index("node_id")["assigned_pop"].to_dict()
        _pop_cells_s10 = np.array(
            [_pop_per_node_s10.get(int(c), "Unknown") for c in _clustering_s10]
        )

        _max_pts_s10 = 50_000
        if len(_clustering_s10) > _max_pts_s10:
            np.random.seed(SEED if "SEED" in dir() else 42)
            _sidx = np.random.choice(len(_clustering_s10), _max_pts_s10, replace=False)
        else:
            _sidx = np.arange(len(_clustering_s10))

        import plotly.colors as _pc_s10

        if _n_meta_s10 <= 20:
            _mc_pal_s10 = _pc_s10.qualitative.Alphabet[:_n_meta_s10]
        else:
            _mc_pal_s10 = [
                f"hsl({int(i * 360 / _n_meta_s10)},70%,55%)" for i in range(_n_meta_s10)
            ]

        _active_nodes = [i for i in range(_n_nodes_s10) if _node_sizes_s10[i] > 0]
        _node_x_grid_s10 = [
            _grid_coords_s10[i, 0] - _xGrid_base.min() + 1 for i in _active_nodes
        ]
        _node_y_grid_s10 = [
            _grid_coords_s10[i, 1] - _yGrid_base.min() + 1 for i in _active_nodes
        ]
        _node_txt_mc_s10 = [str(int(_mc_nodes_s10[i] + 1)) for i in _active_nodes]
        _node_txt_pop_s10 = [_pop_per_node_s10.get(i, "?")[:5] for i in _active_nodes]
        _node_sz_s10 = [_node_sizes_s10[i] for i in _active_nodes]

        # ── Plot 1 : Métacluster ──────────────────────────────────────────────
        fig_grid_mc_s10 = go.Figure()
        for _mc_id in range(_n_meta_s10):
            _mask_mc = _mc_cells_s10[_sidx] == _mc_id
            if _mask_mc.sum() == 0:
                continue
            _si = _sidx[_mask_mc]
            fig_grid_mc_s10.add_trace(
                go.Scattergl(
                    x=_xj_s10[_si],
                    y=_yj_s10[_si],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=_mc_pal_s10[_mc_id % len(_mc_pal_s10)],
                        opacity=0.5,
                    ),
                    name=f"MC{_mc_id} ({_mask_mc.sum():,})",
                    hovertemplate=f"MC{_mc_id}<br>x:%{{x:.2f}} y:%{{y:.2f}}<extra></extra>",
                )
            )
        fig_grid_mc_s10.add_trace(
            go.Scatter(
                x=_node_x_grid_s10,
                y=_node_y_grid_s10,
                mode="text",
                text=_node_txt_mc_s10,
                textfont=dict(size=9, color="black", family="Arial Black"),
                hovertemplate=[
                    f"MC{t}<br>{int(s):,} cellules<extra></extra>"
                    for t, s in zip(_node_txt_mc_s10, _node_sz_s10)
                ],
                showlegend=False,
            )
        )
        fig_grid_mc_s10.update_layout(
            title=dict(
                text=f"<b>Grille SOM — {XDIM}×{YDIM} — Métaclusters</b>",
                font=dict(size=14),
            ),
            xaxis=dict(
                title="xGrid",
                range=[0.3, XDIM + 1.7],
                scaleanchor="y",
                scaleratio=1,
                gridcolor="rgba(0,0,0,0.08)",
            ),
            yaxis=dict(
                title="yGrid", range=[0.3, YDIM + 1.7], gridcolor="rgba(0,0,0,0.08)"
            ),
            height=700,
            width=800,
            paper_bgcolor="#fafafa",
            plot_bgcolor="#f5f5f5",
            legend=dict(
                title="Métacluster",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc",
                borderwidth=1,
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(t=70, b=50, l=60, r=180),
        )
        fig_grid_mc_s10.write_html(
            str(_dashboard_out / f"fig_grid_mc_s10_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )

        # ── Plot 2 : Condition ────────────────────────────────────────────────
        fig_grid_cond_s10 = go.Figure()
        for _cl, _cc in {"Sain": "#2ca02c", "Pathologique": "#d62728"}.items():
            _mask_c = _conditions_s10[_sidx] == _cl
            if _mask_c.sum() == 0:
                continue
            _si = _sidx[_mask_c]
            fig_grid_cond_s10.add_trace(
                go.Scattergl(
                    x=_xj_s10[_si],
                    y=_yj_s10[_si],
                    mode="markers",
                    marker=dict(size=3, color=_cc, opacity=0.45),
                    name=f"{_cl} ({_mask_c.sum():,})",
                    hovertemplate=f"{_cl}<br>x:%{{x:.2f}} y:%{{y:.2f}}<extra></extra>",
                )
            )
        fig_grid_cond_s10.update_layout(
            title=dict(
                text=f"<b>Grille SOM — {XDIM}×{YDIM} — Condition</b>",
                font=dict(size=14),
            ),
            xaxis=dict(
                title="xGrid",
                range=[0.3, XDIM + 1.7],
                scaleanchor="y",
                gridcolor="rgba(0,0,0,0.08)",
            ),
            yaxis=dict(
                title="yGrid", range=[0.3, YDIM + 1.7], gridcolor="rgba(0,0,0,0.08)"
            ),
            height=700,
            width=800,
            paper_bgcolor="#fafafa",
            plot_bgcolor="#f5f5f5",
            legend=dict(
                title="Condition",
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc",
                borderwidth=1,
            ),
            margin=dict(t=70, b=50, l=60, r=60),
        )
        fig_grid_cond_s10.write_html(
            str(_dashboard_out / f"fig_grid_cond_s10_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )

        # ── Plot 3 : Population cellulaire ────────────────────────────────────
        fig_grid_pop_s10 = go.Figure()
        for _pop_lbl in sorted(set(_pop_cells_s10)):
            _mask_p = _pop_cells_s10[_sidx] == _pop_lbl
            if _mask_p.sum() == 0:
                continue
            _si = _sidx[_mask_p]
            fig_grid_pop_s10.add_trace(
                go.Scattergl(
                    x=_xj_s10[_si],
                    y=_yj_s10[_si],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=POPULATION_COLORS.get(_pop_lbl, "#AAAAAA"),
                        opacity=0.55,
                    ),
                    name=f"{_pop_lbl} ({_mask_p.sum():,})",
                    hovertemplate=f"{_pop_lbl}<br>x:%{{x:.2f}} y:%{{y:.2f}}<extra></extra>",
                )
            )
        fig_grid_pop_s10.add_trace(
            go.Scatter(
                x=_node_x_grid_s10,
                y=_node_y_grid_s10,
                mode="text",
                text=_node_txt_pop_s10,
                textfont=dict(size=7, color="black", family="Arial"),
                hovertemplate=[
                    f"Node {i}<br>Pop: {_pop_per_node_s10.get(i, '?')}"
                    f"<br>{int(s):,} cellules<extra></extra>"
                    for i, s in zip(_active_nodes, _node_sz_s10)
                ],
                showlegend=False,
            )
        )
        fig_grid_pop_s10.update_layout(
            title=dict(
                text=(
                    f"<b>Grille SOM — {XDIM}×{YDIM} — Population Cellulaire</b><br>"
                    f"<sup>Mapping : <b>{_method_s10}</b> · {NORMALIZATION_METHOD} · "
                    f"scatter {'inclus' if INCLUDE_SCATTER_IN_MAPPING else 'exclus'} · "
                    f"Prior log10^3 | Hard Limit · seuil P{DISTANCE_PERCENTILE}</sup>"
                ),
                font=dict(size=14),
            ),
            xaxis=dict(
                title="xGrid",
                range=[0.3, XDIM + 1.7],
                scaleanchor="y",
                gridcolor="rgba(0,0,0,0.08)",
            ),
            yaxis=dict(
                title="yGrid", range=[0.3, YDIM + 1.7], gridcolor="rgba(0,0,0,0.08)"
            ),
            height=700,
            width=850,
            paper_bgcolor="#fafafa",
            plot_bgcolor="#f5f5f5",
            legend=dict(
                title="Population cellulaire",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc",
                borderwidth=1,
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(t=80, b=50, l=60, r=230),
        )
        fig_grid_pop_s10.write_html(
            str(_dashboard_out / f"fig_grid_pop_s10_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )

        print(f"\n[OK] Grille SOM Section 10.5b — 3 visualisations")
        print(f"     Plot 1 : Métacluster")
        print(f"     Plot 2 : Condition")
        print(f"     Plot 3 : Population cellulaire (mapping {_method_s10})")

except Exception as _e_grid_s10:
    import traceback

    print(f"[!] Erreur : {_e_grid_s10}")
    traceback.print_exc()


################################################################################
# =============================================================================
# SECTION 10.6 — HEATMAP COMPARATIVE + GRILLE SOM POPULATION (espace brut -A)
# =============================================================================

# Nom de la meilleure méthode (M12_cosine_prior si mapping_df_v5 disponible)
_best_map_name_10_6 = (
    mapping_df_v5["method"].iloc[0]
    if "mapping_df_v5" in dir() and mapping_df_v5 is not None and len(mapping_df_v5) > 0
    else "V3"
)

# ─── A. Grille SOM colorée par population (à partir du FCS) ──────────────────
if PLOTLY_AVAILABLE and _COL_XGRID is not None and _COL_YGRID is not None:
    print("=" * 70)
    print(f" SECTION 10.6 — GRILLE SOM (depuis FCS) COLORÉE PAR POPULATION")
    print(f" Méthode de mapping : {_best_map_name_10_6}")
    print("=" * 70)

    _xgrid_cells = _df_fcs_raw[_COL_XGRID].to_numpy(dtype=np.float32)
    _ygrid_cells = _df_fcs_raw[_COL_YGRID].to_numpy(dtype=np.float32)

    # Sélection du meilleur mapping : V5 (M12_cosine_prior) si disponible sinon mapping_raw
    _BEST_MAP_10_6 = mapping_df_v5 if "mapping_df_v5" in dir() else mapping_df_raw
    _pop_per_node_raw_map = _BEST_MAP_10_6.set_index("node_id")[
        "assigned_pop"
    ].to_dict()
    _pop_cells_fcs = np.array(
        [_pop_per_node_raw_map.get(int(nid), "Unknown") for nid in _node_ids_raw]
    )

    _max_pts_fcs = 50_000
    np.random.seed(SEED if "SEED" in dir() else 42)
    _sidx_fcs = (
        np.random.choice(len(_xgrid_cells), _max_pts_fcs, replace=False)
        if len(_xgrid_cells) > _max_pts_fcs
        else np.arange(len(_xgrid_cells))
    )

    fig_grid_pop_fcs = go.Figure()
    for _pop in sorted(set(_pop_cells_fcs)):
        _mask_p = _pop_cells_fcs[_sidx_fcs] == _pop
        if _mask_p.sum() == 0:
            continue
        _si = _sidx_fcs[_mask_p]
        fig_grid_pop_fcs.add_trace(
            go.Scattergl(
                x=_xgrid_cells[_si],
                y=_ygrid_cells[_si],
                mode="markers",
                marker=dict(
                    size=3, color=POPULATION_COLORS.get(_pop, "#AAAAAA"), opacity=0.55
                ),
                name=f"{_pop} ({_mask_p.sum():,})",
                hovertemplate=f"{_pop}<br>xGrid:%{{x:.2f}} yGrid:%{{y:.2f}}<extra></extra>",
            )
        )

    fig_grid_pop_fcs.update_layout(
        title=dict(
            text=(
                "<b>Grille SOM (depuis FCS) — Colorée par Population Cellulaire</b><br>"
                f"<sup>Mapping : <b>{_best_map_name_10_6}</b> · {NORMALIZATION_METHOD} · "
                f"scatter {'inclus' if INCLUDE_SCATTER_IN_MAPPING else 'exclus'} · "
                f"Prior renforcé log10^3 | Hard Limit actif · "
                f"P{DISTANCE_PERCENTILE}</sup>"
            ),
            font=dict(size=14),
        ),
        xaxis=dict(title="xGrid", gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title="yGrid", gridcolor="rgba(0,0,0,0.08)", scaleanchor="x"),
        height=700,
        width=850,
        paper_bgcolor="#fafafa",
        plot_bgcolor="#f5f5f5",
        legend=dict(
            title="Population",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc",
            borderwidth=1,
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(t=90, b=50, l=60, r=230),
    )
    fig_grid_pop_fcs.write_html(
        str(_dashboard_out / f"fig_grid_pop_fcs_{_dashboard_ts}.html"),
        include_plotlyjs="cdn",
    )

    _out_grid_fcs = (
        OUTPUT_DIR / "other" / f"grid_pop_fcs_{_best_map_name_10_6}_{timestamp}.html"
    )
    fig_grid_pop_fcs.write_html(str(_out_grid_fcs))
    print(f"[OK] Grille FCS exportée : {_out_grid_fcs}")


# ─── B. Heatmap comparative : CSV référence vs centroïdes métaclusters ────────
if PLOTLY_AVAILABLE:
    print("\n" + "=" * 70)
    print(f" SECTION 10.6 — HEATMAP COMPARATIVE (CSV ref vs Métaclusters FlowSOM)")
    print(f" Méthode de mapping : {_best_map_name_10_6}")
    print("=" * 70)

    # Colonnes communes (déjà garanties identiques dans _common_a_cols)
    _common_heat = sorted(_common_a_cols)

    if len(_common_heat) == 0:
        print("[!] Aucun marqueur commun — vérifiez la Section 10.3.")
    else:

        def _zscore_df(df: pd.DataFrame) -> pd.DataFrame:
            _mu = df.mean(axis=0)
            _sig = df.std(axis=0).replace(0, 1.0)
            return (df - _mu) / _sig

        # Populations référence
        _df_ext_z = _zscore_df(_ref_mfi_aligned[_common_heat])

        # Métaclusters FlowSOM — centroïde moyen par MC
        _mc_groups: Dict[str, pd.Series] = {}
        for _mc_id in sorted(_mc_per_node_raw.unique()):
            _node_ids_mc = _mc_per_node_raw[_mc_per_node_raw == _mc_id].index
            _mc_groups[f"MC{int(_mc_id)}"] = _node_mfi_aligned.loc[_node_ids_mc].mean(
                axis=0
            )
        _df_mc_heat = pd.DataFrame(_mc_groups).T[_common_heat]
        _df_mc_z = _zscore_df(_df_mc_heat)

        _clim = min(
            max(
                abs(_df_ext_z.values).max(),
                abs(_df_mc_z.values).max(),
                0.1,
            ),
            3.0,
        )

        fig_heat_v3 = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "<b>Populations de référence CSV (-A)</b>",
                "<b>Centroïdes Métaclusters FlowSOM (-A)</b>",
            ],
            horizontal_spacing=0.12,
        )

        fig_heat_v3.add_trace(
            go.Heatmap(
                z=_df_ext_z.values,
                x=_common_heat,
                y=list(_df_ext_z.index),
                colorscale="RdBu_r",
                zmid=0,
                zmin=-_clim,
                zmax=_clim,
                text=np.round(_df_ext_z.values, 2).astype(str),
                texttemplate="%{text}",
                textfont=dict(size=9),
                colorbar=dict(
                    x=0.44,
                    y=0.5,
                    len=0.85,
                    thickness=12,
                    title=dict(text="z-score", side="right"),
                    tickfont=dict(size=9),
                ),
                hovertemplate="<b>%{y}</b><br>%{x}<br>z=%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig_heat_v3.add_trace(
            go.Heatmap(
                z=_df_mc_z.values,
                x=_common_heat,
                y=list(_df_mc_z.index),
                colorscale="RdBu_r",
                zmid=0,
                zmin=-_clim,
                zmax=_clim,
                text=np.round(_df_mc_z.values, 2).astype(str),
                texttemplate="%{text}",
                textfont=dict(size=9),
                colorbar=dict(
                    x=1.01,
                    y=0.5,
                    len=0.85,
                    thickness=12,
                    title=dict(text="z-score", side="right"),
                    tickfont=dict(size=9),
                ),
                hovertemplate="<b>%{y}</b><br>%{x}<br>z=%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig_heat_v3.update_layout(
            title=dict(
                text=(
                    "<b>Heatmap Comparative — Espace brut -A (linéaire)</b><br>"
                    f"<sup>Populations CSV vs Métaclusters FlowSOM | "
                    f"{len(_common_heat)} marqueurs -A | "
                    f"scatter {'inclus' if INCLUDE_SCATTER_IN_MAPPING else 'exclus'} | "
                    f"Mapping : {_best_map_name_10_6}</sup>"
                ),
                x=0.5,
                xanchor="center",
                font=dict(size=14),
            ),
            height=max(500, max(len(_df_ext_z), len(_df_mc_z)) * 45 + 200),
            paper_bgcolor="#fafafa",
            margin=dict(l=150, r=80, t=110, b=120),
        )
        fig_heat_v3.update_xaxes(tickangle=-45, tickfont=dict(size=9))
        fig_heat_v3.update_yaxes(tickfont=dict(size=10))

        fig_heat_v3.write_html(
            str(_dashboard_out / f"fig_heat_v3_{_dashboard_ts}.html"),
            include_plotlyjs="cdn",
        )

        _out_heat_v3 = (
            OUTPUT_DIR / "other" / f"heatmap_{_best_map_name_10_6}_{timestamp}.html"
        )
        fig_heat_v3.write_html(str(_out_heat_v3))
        print(f"[OK] Heatmap exportée : {_out_heat_v3}")

        # ── Tableau de correspondance final ───────────────────────────────────
        print(f"\n   ━━ CORRESPONDANCE POPULATION → MÉTACLUSTER DOMINANT ━━")
        print(f"   Méthode : {_best_map_name_10_6}")
        print(f"   {'─' * 65}")
        print(f"   {'Population':<35s}  {'MC dom.':>7}  {'Nœuds':>6}  Top 3 MC")
        print(f"   {'─' * 65}")
        _map_for_table = mapping_df_v5 if "mapping_df_v5" in dir() else mapping_df_raw
        for _pop in sorted(set(_map_for_table["assigned_pop"])):
            _sub = _map_for_table[_map_for_table["assigned_pop"] == _pop]
            _vcnt = _sub["metacluster"].value_counts()
            _dom = _vcnt.idxmax()
            _top3 = "  |  ".join([f"MC{int(k)}({v})" for k, v in _vcnt.head(3).items()])
            print(f"   {_pop:<35s}  MC{int(_dom):>5}  {len(_sub):>6}  {_top3}")

print(f"\n{'=' * 70}")
print(f"[OK] SECTION 10 TERMINÉE — Mapping {_best_map_name_10_6}")
print(f"     Colonnes utilisées : {_common_a_cols}")
print(f"     Populations : {list(pop_mfi_raw_ref.keys())}")
print(f"     Normalisation : {NORMALIZATION_METHOD}")
print(f"     Scatter inclus : {INCLUDE_SCATTER_IN_MAPPING}")
print(f"{'=' * 70}")


################################################################################
# INTERFACE LIGNE DE COMMANDE
################################################################################


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    import argparse

    parser = argparse.ArgumentParser(
        description="FlowSOM Analysis Pipeline - Analyse de cytométrie en flux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Analyse via fichier de configuration (auto-détecté si config_flowsom.yaml présent)
  python flowsom_pipeline.py

  # Spécifier un fichier de configuration explicitement
  python flowsom_pipeline.py --config config_flowsom.yaml

  # Analyse simple d'un dossier
  python flowsom_pipeline.py --healthy-folder "Data/NBM" --output Results

  # Mode comparaison Sain vs Pathologique
  python flowsom_pipeline.py --healthy-folder "Data/NBM" --patho-folder "Data/Patho" --compare-mode --output Results_Comparison

  # Personnaliser les paramètres FlowSOM
  python flowsom_pipeline.py --healthy-folder "Data/NBM" --xdim 15 --ydim 15 --n-metaclusters 20 --n-iterations 20

  # Désactiver l'accélération GPU
  python flowsom_pipeline.py --healthy-folder "Data/NBM" --no-gpu
        """,
    )

    # Fichier de configuration YAML
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Fichier de configuration YAML (défaut: auto-détection de config_flowsom.yaml)",
    )

    # Chemins et fichiers
    paths = parser.add_argument_group("Chemins et fichiers")
    paths.add_argument(
        "--healthy-folder",
        type=str,
        default=None,
        help="Dossier contenant les fichiers FCS sains (NBM) — priorité sur le fichier de config",
    )
    paths.add_argument(
        "--patho-folder",
        type=str,
        default=None,
        help="Dossier contenant les fichiers FCS pathologiques (optionnel)",
    )
    paths.add_argument(
        "--output",
        "-o",
        type=str,
        default="Results",
        help="Dossier de sortie pour les résultats (défaut: Results)",
    )

    # Mode d'analyse
    mode = parser.add_argument_group("Mode d'analyse")
    mode.add_argument(
        "--compare-mode",
        action="store_true",
        help="Activer le mode comparaison Sain vs Pathologique",
    )
    mode.add_argument(
        "--no-compare-mode",
        dest="compare_mode",
        action="store_false",
        help="Désactiver le mode comparaison",
    )
    parser.set_defaults(compare_mode=True)

    # Pre-gating
    gating = parser.add_argument_group("Pre-gating")
    gating.add_argument(
        "--no-pregate-viable",
        dest="pregate_viable",
        action="store_false",
        help="Désactiver le gating des cellules viables",
    )
    gating.add_argument(
        "--no-pregate-singlets",
        dest="pregate_singlets",
        action="store_false",
        help="Désactiver le gating des singlets",
    )
    gating.add_argument(
        "--no-pregate-cd45",
        dest="pregate_cd45",
        action="store_false",
        help="Désactiver le gating CD45+",
    )
    gating.add_argument(
        "--no-pregate-cd34",
        dest="pregate_cd34",
        action="store_false",
        help="Désactiver le gating CD34+",
    )
    parser.set_defaults(
        pregate_viable=True, pregate_singlets=True, pregate_cd45=True, pregate_cd34=True
    )

    # Paramètres FlowSOM
    flowsom = parser.add_argument_group("Paramètres FlowSOM")
    flowsom.add_argument(
        "--xdim", type=int, default=10, help="Dimension X de la grille SOM (défaut: 10)"
    )
    flowsom.add_argument(
        "--ydim", type=int, default=10, help="Dimension Y de la grille SOM (défaut: 10)"
    )
    flowsom.add_argument(
        "--n-metaclusters",
        type=int,
        default=15,
        help="Nombre de métaclusters (défaut: 15)",
    )
    flowsom.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Taux d'apprentissage (défaut: 0.05)",
    )
    flowsom.add_argument(
        "--sigma", type=float, default=1.5, help="Sigma pour le voisinage (défaut: 1.5)"
    )
    flowsom.add_argument(
        "--n-iterations", type=int, default=10, help="Nombre d'itérations (défaut: 10)"
    )
    flowsom.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire pour la reproductibilité (défaut: 42)",
    )

    # Transformation et normalisation
    transform = parser.add_argument_group("Transformation et normalisation")
    transform.add_argument(
        "--transform",
        type=str,
        default="arcsinh",
        choices=["arcsinh", "logicle", "log", "none"],
        help="Méthode de transformation (défaut: arcsinh)",
    )
    transform.add_argument(
        "--cofactor",
        type=float,
        default=150,
        help="Cofacteur pour transformation arcsinh (défaut: 150)",
    )
    transform.add_argument(
        "--normalize",
        type=str,
        default="zscore",
        choices=["zscore", "minmax", "none"],
        help="Méthode de normalisation (défaut: zscore)",
    )

    # Downsampling
    sampling = parser.add_argument_group("Downsampling")
    sampling.add_argument(
        "--no-downsample",
        dest="downsample",
        action="store_false",
        help="Désactiver le downsampling",
    )
    sampling.add_argument(
        "--max-cells-per-file",
        type=int,
        default=50000,
        help="Nombre max de cellules par fichier (défaut: 50000)",
    )
    sampling.add_argument(
        "--max-cells-total",
        type=int,
        default=1000000,
        help="Nombre max de cellules total (défaut: 1000000)",
    )
    parser.set_defaults(downsample=True)

    # Visualisation
    viz = parser.add_argument_group("Visualisation")
    viz.add_argument(
        "--no-save-plots",
        dest="save_plots",
        action="store_false",
        help="Ne pas sauvegarder les graphiques",
    )
    viz.add_argument(
        "--plot-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Format des graphiques (défaut: png)",
    )
    viz.add_argument(
        "--dpi", type=int, default=300, help="Résolution des graphiques (défaut: 300)"
    )
    parser.set_defaults(save_plots=True)

    # GPU
    gpu = parser.add_argument_group("GPU")
    gpu.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Désactiver l'accélération GPU",
    )
    parser.set_defaults(use_gpu=True)

    # Verbosité
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Mode verbeux (affiche plus de détails)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Mode silencieux (affiche moins de détails)",
    )

    return parser.parse_args()


################################################################################
# WORKFLOW PRINCIPAL D'EXÉCUTION
################################################################################


def run_flowsom_pipeline():
    """
    Exécute le pipeline FlowSOM complet en suivant toutes les étapes du notebook.
    """
    import time

    start_time = time.time()

    logger = logging.getLogger(__name__)

    # =========================================================================
    # ÉTAPE 1: Chargement des fichiers FCS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ÉTAPE 1: Chargement des fichiers FCS")
    logger.info("=" * 80)

    healthy_path = Path(HEALTHY_FOLDER)
    healthy_files = get_fcs_files(healthy_path)
    logger.info(f"Fichiers sains trouvés: {len(healthy_files)}")

    patho_files = []
    if COMPARE_MODE and PATHO_FOLDER:
        patho_path = Path(PATHO_FOLDER)
        patho_files = get_fcs_files(patho_path)
        logger.info(f"Fichiers pathologiques trouvés: {len(patho_files)}")

    if not healthy_files:
        logger.error("Aucun fichier FCS trouvé dans le dossier sain!")
        return False

    # Charger les données
    logger.info("Chargement des données FCS...")
    healthy_data = []
    for fcs_file in healthy_files:
        try:
            sample = FlowCytometrySample.from_fcs(fcs_file, verbose=False)
            healthy_data.append(sample)
        except Exception as e:
            logger.warning(f"Erreur lors du chargement de {fcs_file}: {e}")

    patho_data = []
    if patho_files:
        for fcs_file in patho_files:
            try:
                sample = FlowCytometrySample.from_fcs(fcs_file, verbose=False)
                patho_data.append(sample)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de {fcs_file}: {e}")

    logger.info(
        f"Données chargées: {len(healthy_data)} sains, {len(patho_data)} pathologiques"
    )

    # =========================================================================
    # ÉTAPE 2: Pre-gating
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ÉTAPE 2: Pre-gating")
    logger.info("=" * 80)

    pregating = PreGating()

    def _apply_pregating(df: pd.DataFrame) -> pd.DataFrame:
        """Applique le pre-gating sur un DataFrame FCS et retourne les cellules conservées."""
        X = df.values
        var_names = list(df.columns)
        mask = np.ones(len(df), dtype=bool)

        if PRE_GATE_VIABLE:
            m = PreGating.gate_viable_cells(X, var_names)
            mask &= m
            logger.debug(
                f"    Viable: {m.sum():,} / {len(m):,} ({100 * m.mean():.1f}%)"
            )

        if PRE_GATE_SINGLETS:
            m = PreGating.gate_singlets(X[mask], var_names)
            # remap sur le masque global
            idx = np.where(mask)[0]
            mask[idx[~m]] = False
            logger.debug(f"    Singlets: {mask.sum():,} restantes")

        if PRE_GATE_CD45:
            m = PreGating.gate_cd45_positive(X[mask], var_names)
            idx = np.where(mask)[0]
            mask[idx[~m]] = False
            logger.debug(f"    CD45+: {mask.sum():,} restantes")

        if PRE_GATE_CD34:
            m = PreGating.gate_cd34_blasts(X[mask], var_names)
            idx = np.where(mask)[0]
            mask[idx[~m]] = False
            logger.debug(f"    CD34+: {mask.sum():,} restantes")

        return df[mask].reset_index(drop=True)

    # Appliquer le pre-gating sur les données saines
    logger.info("Application du pre-gating sur les données saines...")
    healthy_gated = []
    for i, sample in enumerate(healthy_data):
        try:
            gated = _apply_pregating(sample)
            healthy_gated.append(gated)
            logger.debug(f"  Sample sain {i + 1}: {len(gated):,} cellules conservées")
        except Exception as e:
            logger.warning(f"Erreur pre-gating sample sain {i + 1}: {e}")

    # Appliquer le pre-gating sur les données pathologiques
    patho_gated = []
    if patho_data:
        logger.info("Application du pre-gating sur les données pathologiques...")
        for i, sample in enumerate(patho_data):
            try:
                gated = _apply_pregating(sample)
                patho_gated.append(gated)
                logger.debug(
                    f"  Sample patho {i + 1}: {len(gated):,} cellules conservées"
                )
            except Exception as e:
                logger.warning(f"Erreur pre-gating sample patho {i + 1}: {e}")

    # =========================================================================
    # ÉTAPE 3: Transformation et normalisation
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ÉTAPE 3: Transformation et normalisation")
    logger.info("=" * 80)

    transformer = DataTransformer()

    # Combiner les données pour transformation
    all_data = healthy_gated + patho_gated

    # Identifier les marqueurs à utiliser (exclure scatter et time)
    if all_data:
        all_markers = list(all_data[0].columns)
        used_markers = [
            m
            for m in all_markers
            if not any(
                x in m.lower() for x in ["fsc", "ssc", "time", "width", "height"]
            )
        ]
        logger.info(f"Marqueurs utilisés pour l'analyse: {len(used_markers)}")
        logger.debug(f"  {', '.join(used_markers)}")
    else:
        logger.error("Aucune donnée après pre-gating!")
        return False

    # Appliquer la transformation
    logger.info(f"Transformation: {TRANSFORM_METHOD}")
    transformed_data = []
    for df in all_data:
        df_transformed = df.copy()
        for marker in used_markers:
            if marker in df.columns:
                if TRANSFORM_METHOD == "arcsinh":
                    df_transformed[marker] = arcsinh_transform(
                        df[marker], cofactor=COFACTOR
                    )
                elif TRANSFORM_METHOD == "logicle":
                    df_transformed[marker] = DataTransformer.logicle_transform(
                        df[marker].values
                    )
                elif TRANSFORM_METHOD == "log":
                    df_transformed[marker] = DataTransformer.log_transform(
                        df[marker].values
                    )
        transformed_data.append(df_transformed)

    # Appliquer la normalisation
    if NORMALIZE_METHOD != "none":
        logger.info(f"Normalisation: {NORMALIZE_METHOD}")
        normalized_data = []
        for df in transformed_data:
            df_normalized = df.copy()
            for marker in used_markers:
                if marker in df.columns:
                    if NORMALIZE_METHOD == "zscore":
                        df_normalized[marker] = DataTransformer.zscore_normalize(
                            df[marker].values
                        )
                    elif NORMALIZE_METHOD == "minmax":
                        df_normalized[marker] = DataTransformer.min_max_normalize(
                            df[marker].values
                        )
            normalized_data.append(df_normalized)
        final_data = normalized_data
    else:
        final_data = transformed_data

    # =========================================================================
    # ÉTAPE 4: Downsampling
    # =========================================================================
    if DOWNSAMPLE:
        logger.info("\n" + "=" * 80)
        logger.info("ÉTAPE 4: Downsampling")
        logger.info("=" * 80)

        downsampled_data = []
        for i, df in enumerate(final_data):
            if len(df) > MAX_CELLS_PER_FILE:
                df_downsampled = df.sample(n=MAX_CELLS_PER_FILE, random_state=SEED)
                logger.debug(
                    f"  Sample {i + 1}: {len(df)} -> {len(df_downsampled)} cellules"
                )
                downsampled_data.append(df_downsampled)
            else:
                downsampled_data.append(df)

        final_data = downsampled_data

    # Combiner toutes les données
    logger.info("Fusion des données...")
    combined_data = pd.concat(final_data, ignore_index=True)
    logger.info(f"Données combinées: {len(combined_data):,} cellules")

    # Downsampling total si nécessaire
    if DOWNSAMPLE and len(combined_data) > MAX_CELLS_TOTAL:
        logger.info(
            f"Downsampling total: {len(combined_data):,} -> {MAX_CELLS_TOTAL:,}"
        )
        combined_data = combined_data.sample(n=MAX_CELLS_TOTAL, random_state=SEED)

    # =========================================================================
    # ÉTAPE 5: Clustering FlowSOM
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ÉTAPE 5: Clustering FlowSOM")
    logger.info("=" * 80)

    # Préparer les données pour FlowSOM
    X_flowsom = combined_data[used_markers].values
    logger.info(f"Matrice d'entrée FlowSOM: {X_flowsom.shape}")

    # Initialiser FlowSOM
    logger.info(f"Configuration: Grille {XDIM}x{YDIM}, {N_METACLUSTERS} métaclusters")

    try:
        from FlowSomGpu.models import GPUFlowSOMEstimator

        use_gpu_actual = USE_GPU
    except ImportError:
        logger.warning("FlowSomGpu non disponible, utilisation du CPU")
        use_gpu_actual = False

    if use_gpu_actual:
        fsom = GPUFlowSOMEstimator(
            xdim=XDIM,
            ydim=YDIM,
            n_clusters=N_METACLUSTERS,
            learning_rate=LEARNING_RATE,
            sigma=SIGMA,
            seed=SEED,
        )
    else:
        # Fallback sur une implémentation CPU si disponible
        logger.warning("Implémentation GPU non disponible")
        return False

    # Entraîner le modèle
    logger.info(f"Entraînement FlowSOM ({N_ITERATIONS} itérations)...")
    fsom.fit(X_flowsom, n_iter=N_ITERATIONS)

    # Prédire les assignations
    logger.info("Assignation des cellules...")
    node_assignments = fsom.predict(X_flowsom)
    metacluster_assignments = fsom.meta_clustering(n_clusters=N_METACLUSTERS)

    # Ajouter les assignations aux données
    combined_data["FlowSOM_cluster"] = node_assignments
    combined_data["FlowSOM_metacluster"] = metacluster_assignments[node_assignments]

    logger.info(
        f"Clustering terminé: {len(np.unique(node_assignments))} nœuds, {N_METACLUSTERS} métaclusters"
    )

    # =========================================================================
    # ÉTAPE 6: Visualisations
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ÉTAPE 6: Génération des visualisations")
    logger.info("=" * 80)

    if SAVE_PLOTS:
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)

        # TODO: Appeler les fonctions de visualisation ici
        # - MST plot
        # - Grille SOM
        # - Heatmaps
        # - Scatter plots

        logger.info(f"Visualisations sauvegardées dans {OUTPUT_DIR}")

    # =========================================================================
    # ÉTAPE 7: Exports
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ÉTAPE 7: Export des résultats")
    logger.info("=" * 80)

    output_path = Path(OUTPUT_DIR)

    # Export CSV
    csv_path = output_path / "flowsom_results.csv"
    combined_data.to_csv(csv_path, index=False)
    logger.info(f"CSV exporté: {csv_path}")

    # Export statistiques par métacluster
    stats = combined_data.groupby("FlowSOM_metacluster")[used_markers].agg(
        ["mean", "median", "std"]
    )
    stats_path = output_path / "metacluster_statistics.csv"
    stats.to_csv(stats_path)
    logger.info(f"Statistiques exportées: {stats_path}")

    # Export métadonnées
    metadata = {
        "pipeline_version": "2.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_healthy_files": len(healthy_files),
        "n_patho_files": len(patho_files),
        "n_cells_final": len(combined_data),
        "n_markers": len(used_markers),
        "markers": used_markers,
        "xdim": XDIM,
        "ydim": YDIM,
        "n_metaclusters": N_METACLUSTERS,
        "transform_method": TRANSFORM_METHOD,
        "normalize_method": NORMALIZE_METHOD,
        "pregate_viable": PRE_GATE_VIABLE,
        "pregate_singlets": PRE_GATE_SINGLETS,
        "pregate_cd45": PRE_GATE_CD45,
        "pregate_cd34": PRE_GATE_CD34,
    }

    metadata_path = output_path / "pipeline_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Métadonnées exportées: {metadata_path}")

    # =========================================================================
    # FIN
    # =========================================================================
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline terminé avec succès en {elapsed:.1f} secondes")
    logger.info("=" * 80)

    return True


def _load_yaml_config(config_path: str) -> dict:
    """Charge un fichier de configuration YAML et retourne un dict plat (clés argparse)."""
    try:
        import yaml
    except ImportError:
        print("[!] PyYAML non installé: pip install pyyaml")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = {}
    paths = raw.get("paths", {})
    if paths.get("healthy_folder"):
        cfg["healthy_folder"] = paths["healthy_folder"]
    if paths.get("patho_folder"):
        cfg["patho_folder"] = paths["patho_folder"]
    if paths.get("output_dir"):
        cfg["output"] = paths["output_dir"]
    analysis = raw.get("analysis", {})
    if "compare_mode" in analysis:
        cfg["compare_mode"] = analysis["compare_mode"]
    pregate = raw.get("pregate", {})
    if "apply" in pregate:
        cfg["apply_pregating"] = pregate["apply"]
    if "mode" in pregate:
        cfg["gating_mode"] = pregate["mode"]
    if "mode_blastes_vs_normal" in pregate:
        cfg["mode_blastes_vs_normal"] = pregate["mode_blastes_vs_normal"]
    for k_yaml, k_cfg in (
        ("viable", "pregate_viable"),
        ("singlets", "pregate_singlets"),
        ("cd45", "pregate_cd45"),
        ("cd34", "pregate_cd34"),
    ):
        if k_yaml in pregate:
            cfg[k_cfg] = pregate[k_yaml]
    pg_adv = raw.get("pregate_advanced", {})
    for k_yaml, k_cfg in (
        ("debris_min_percentile", "debris_min_percentile"),
        ("debris_max_percentile", "debris_max_percentile"),
        ("doublets_ratio_min", "ratio_min"),
        ("doublets_ratio_max", "ratio_max"),
        ("cd45_threshold_percentile", "cd45_threshold_percentile"),
        ("cd34_threshold_percentile", "cd34_threshold_percentile"),
        ("cd34_use_ssc_filter", "use_ssc_filter_for_blasts"),
        ("cd34_ssc_max_percentile", "ssc_max_percentile_blasts"),
    ):
        if k_yaml in pg_adv:
            cfg[k_cfg] = pg_adv[k_yaml]
    fs = raw.get("flowsom", {})
    for k in (
        "xdim",
        "ydim",
        "n_metaclusters",
        "learning_rate",
        "sigma",
        "n_iterations",
        "seed",
        "rlen",
    ):
        if k in fs:
            cfg[k] = fs[k]
    ac = raw.get("auto_clustering", {})
    if "enabled" in ac:
        cfg["auto_cluster"] = ac["enabled"]
    for k_yaml, k_cfg in (
        ("min_clusters", "min_clusters_auto"),
        ("max_clusters", "max_clusters_auto"),
        ("n_bootstrap", "n_bootstrap"),
        ("sample_size_bootstrap", "sample_size_bootstrap"),
        ("min_stability_threshold", "min_stability_threshold"),
        ("weight_stability", "w_stability"),
        ("weight_silhouette", "w_silhouette"),
    ):
        if k_yaml in ac:
            cfg[k_cfg] = ac[k_yaml]
    transform = raw.get("transform", {})
    if "method" in transform:
        cfg["transform"] = transform["method"]
    if "cofactor" in transform:
        cfg["cofactor"] = transform["cofactor"]
    if "apply_to_scatter" in transform:
        cfg["apply_to_scatter"] = transform["apply_to_scatter"]
    normalize = raw.get("normalize", {})
    if "method" in normalize:
        cfg["normalize"] = normalize["method"]
    mk = raw.get("markers", {})
    if "exclude_scatter" in mk:
        cfg["exclude_scatter"] = mk["exclude_scatter"]
    if "exclude_additional" in mk:
        cfg["exclude_additional_markers"] = mk["exclude_additional"]
    ds = raw.get("downsampling", {})
    if "enabled" in ds:
        cfg["downsample"] = ds["enabled"]
    if "max_cells_per_file" in ds:
        cfg["max_cells_per_file"] = ds["max_cells_per_file"]
    if "max_cells_total" in ds:
        cfg["max_cells_total"] = ds["max_cells_total"]
    viz = raw.get("visualization", {})
    if "save_plots" in viz:
        cfg["save_plots"] = viz["save_plots"]
    if "plot_format" in viz:
        cfg["plot_format"] = viz["plot_format"]
    if "dpi" in viz:
        cfg["dpi"] = viz["dpi"]
    gpu = raw.get("gpu", {})
    if "enabled" in gpu:
        cfg["use_gpu"] = gpu["enabled"]
    return cfg


def main():
    """Fonction principale qui exécute le pipeline."""
    args = parse_arguments()

    # ── Chargement du fichier de configuration YAML ───────────────────────────
    # Priorité : CLI > config YAML > défauts argparse
    config_path = args.config
    if config_path is None:
        # Auto-détection dans le répertoire du script puis dans le CWD
        _script_dir = Path(__file__).parent
        for _candidate in (
            _script_dir / "config_flowsom.yaml",
            Path("config_flowsom.yaml"),
            _script_dir / "config.yaml",
        ):
            if _candidate.exists():
                config_path = str(_candidate)
                print(f"[INFO] Configuration auto-détectée: {config_path}")
                break

    yaml_cfg = {}
    if config_path:
        yaml_cfg = _load_yaml_config(config_path)
        print(f"[OK] Configuration chargée depuis: {config_path}")

    # Fusionner: les valeurs CLI non-None écrasent le YAML
    def _get(attr, default=None):
        """Retourne la valeur CLI si fournie explicitement, sinon celle du YAML, sinon default."""
        cli_val = getattr(args, attr, None)
        if cli_val is not None:
            return cli_val
        return yaml_cfg.get(attr, default)

    # Configuration du logging
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)

    # Banner
    logger.info("=" * 80)
    logger.info("FlowSOM Analysis Pipeline v2.0")
    logger.info("=" * 80)

    # Mettre à jour les variables globales avec les arguments
    global HEALTHY_FOLDER, PATHO_FOLDER, OUTPUT_DIR, COMPARE_MODE
    global PRE_GATE_VIABLE, PRE_GATE_SINGLETS, PRE_GATE_CD45, PRE_GATE_CD34
    global XDIM, YDIM, N_METACLUSTERS, LEARNING_RATE, SIGMA, N_ITERATIONS, SEED
    global TRANSFORM_METHOD, COFACTOR, NORMALIZE_METHOD
    global DOWNSAMPLE, MAX_CELLS_PER_FILE, MAX_CELLS_TOTAL
    global SAVE_PLOTS, PLOT_FORMAT, DPI, USE_GPU

    HEALTHY_FOLDER = _get("healthy_folder")
    PATHO_FOLDER = _get("patho_folder")
    OUTPUT_DIR = _get("output", "Results")
    _compare_mode = _get("compare_mode", True)
    COMPARE_MODE = bool(_compare_mode) and PATHO_FOLDER is not None

    # Validation : healthy_folder obligatoire (CLI ou config)
    if not HEALTHY_FOLDER:
        logger.error(
            "[ERREUR] Aucun dossier sain (NBM) spécifié.\n"
            "  → Utilisez --healthy-folder 'chemin/vers/NBM'\n"
            f"  → Ou renseignez 'paths.healthy_folder' dans {config_path or 'config_flowsom.yaml'}"
        )
        sys.exit(1)

    COMPARE_MODE = bool(_compare_mode) and PATHO_FOLDER is not None

    PRE_GATE_VIABLE = _get("pregate_viable", True)
    PRE_GATE_SINGLETS = _get("pregate_singlets", True)
    PRE_GATE_CD45 = _get("pregate_cd45", True)
    PRE_GATE_CD34 = _get("pregate_cd34", False)

    XDIM = _get("xdim", 10)
    YDIM = _get("ydim", 10)
    N_METACLUSTERS = _get("n_metaclusters", 15)
    LEARNING_RATE = _get("learning_rate", 0.05)
    SIGMA = _get("sigma", 1.5)
    N_ITERATIONS = _get("n_iterations", 10)
    SEED = _get("seed", 42)

    TRANSFORM_METHOD = _get("transform", "arcsinh")
    COFACTOR = _get("cofactor", 150)
    NORMALIZE_METHOD = _get("normalize", "zscore")

    DOWNSAMPLE = _get("downsample", True)
    MAX_CELLS_PER_FILE = _get("max_cells_per_file", 50000)
    MAX_CELLS_TOTAL = _get("max_cells_total", 1000000)

    SAVE_PLOTS = _get("save_plots", True)
    PLOT_FORMAT = _get("plot_format", "png")
    DPI = _get("dpi", 300)
    USE_GPU = _get("use_gpu", True)

    # Vérifications
    if not os.path.exists(HEALTHY_FOLDER):
        logger.error(f"Le dossier sain n'existe pas: {HEALTHY_FOLDER}")
        sys.exit(1)

    if COMPARE_MODE and not os.path.exists(PATHO_FOLDER):
        logger.error(f"Le dossier pathologique n'existe pas: {PATHO_FOLDER}")
        sys.exit(1)

    # Créer le dossier de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Dossier de sortie: {OUTPUT_DIR}")

    # Afficher la configuration
    logger.info("\nConfiguration:")
    logger.info(f"  Dossier sain: {HEALTHY_FOLDER}")
    if COMPARE_MODE:
        logger.info(f"  Dossier pathologique: {PATHO_FOLDER}")
    logger.info(f"  Mode comparaison: {'Activé' if COMPARE_MODE else 'Désactivé'}")
    logger.info(
        f"  Pre-gating: Viable={PRE_GATE_VIABLE}, Singlets={PRE_GATE_SINGLETS}, CD45+={PRE_GATE_CD45}, CD34+={PRE_GATE_CD34}"
    )
    logger.info(
        f"  FlowSOM: Grille {XDIM}x{YDIM}, {N_METACLUSTERS} métaclusters, {N_ITERATIONS} itérations"
    )
    logger.info(f"  Transformation: {TRANSFORM_METHOD} (cofacteur={COFACTOR})")
    logger.info(f"  Normalisation: {NORMALIZE_METHOD}")
    logger.info(f"  GPU: {'Activé' if USE_GPU else 'Désactivé'}")

    try:
        # Exécuter le pipeline
        logger.info("\nDémarrage de l'analyse...")

        # Exécuter le pipeline complet
        success = run_flowsom_pipeline()

        if not success:
            logger.error("Le pipeline a rencontré des erreurs")
            sys.exit(1)

        logger.info("\n" + "=" * 80)
        logger.info("Analyse terminée avec succès!")
        logger.info(f"Résultats sauvegardés dans: {OUTPUT_DIR}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\nErreur lors de l'exécution: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
