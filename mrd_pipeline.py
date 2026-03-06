# -*- coding: utf-8 -*-
"""
mrd_pipeline.py — Pipeline Triple MRD pour cytométrie en flux (LAM/AML)
=======================================================================
Implémente trois méthodes de calcul de MRD en parallèle sur un même
fichier FCS patient, mappé sur un MST de référence NBM (Normal Bone Marrow).

Méthode 1 : Delta métaclusters (proportion patho vs NBM, seuil fold-change ×2).
Méthode 2 : Distance euclidienne MST — nœuds "déviants" vs NBM de référence.
Méthode 3 : Mapping populations (section 10, M12_cosine_prior) — nœuds Unknown.

Auteur  : Florian Magne (adapté pour pipeline batch)
Version : 1.0
Date    : 2026-03-05
"""

from __future__ import annotations

import json
import logging
import re
import traceback
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# ─── Librairies optionnelles ─────────────────────────────────────────────────
try:
    import flowsom as fs
    import anndata as ad
    FLOWSOM_AVAILABLE = True
except ImportError:
    FLOWSOM_AVAILABLE = False
    logging.warning("[!] flowsom non disponible : pip install flowsom anndata")

try:
    import flowkit as fk
    FLOWKIT_AVAILABLE = True
except ImportError:
    FLOWKIT_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")           # backend non-interactif pour batch
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.mixture import GaussianMixture
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    from sklearn.metrics import silhouette_score, r2_score
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("[!] scikit-learn non disponible : pip install scikit-learn")

RANDOM_SEED = 42

# =============================================================================
# STRUCTURES DE GATING — GateResult + état global
# =============================================================================

@dataclass
class GateResult:
    """Résultat structuré d'une opération de gating (compatible notebook principal)."""
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


# Stockage global des rapports de gating (audit + export JSON)
gating_reports: List[GateResult] = []
ransac_scatter_data: Dict[str, Any] = {}
singlets_summary_per_file: List[Dict[str, Any]] = []


def log_gating_event(
    gate_name: str,
    method: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
    warning_msg: Optional[str] = None,
) -> None:
    """Log structuré d'un événement de gating (via le module logging standard)."""
    if warning_msg:
        logging.warning(f"[Gating] {gate_name}/{method} — {warning_msg}")
    else:
        logging.debug(f"[Gating] {gate_name}/{method} → {status} | {details or {}}")


# =============================================================================
# SECTION 1 — PARAMÈTRES PAR DÉFAUT
# =============================================================================

DEFAULT_PARAMS: Dict[str, Any] = {
    # ── Section 1 : Chemins & Mode ───────────────────────────────────────────
    "healthy_folder": Path("Data/Moelle_normale"),
    "pathological_folder": Path("Data/Patho"),
    "compare_mode": True,

    # ── Section 2 : Pré-gating ───────────────────────────────────────────────
    "apply_pregating": True,
    "gating_mode": "auto",          # "auto" (GMM/RANSAC) | "manual" (percentiles)
    "mode_blastes": True,           # Mode Blastes vs Normal

    # Gate 1 — Débris (FSC/SSC)
    "gate_debris": True,
    "debris_min_pct": 1.0,
    "debris_max_pct": 99.0,

    # Gate 2 — Doublets (FSC-A/FSC-H)
    "gate_doublets": True,
    "ratio_min": 0.6,
    "ratio_max": 1.4,

    # Gate 3 — CD45+ leucocytes
    "gate_cd45": True,
    "cd45_pct": 5,

    # Gate 4 — Blastes CD34+
    "filter_blasts": False,
    "cd34_pct": 85,
    "ssc_filter": True,
    "ssc_max_pct": 70,

    # ── Section 3 : Transformation ───────────────────────────────────────────
    "transform": "logicle",         # "logicle" | "arcsinh" | "log10" | "none"
    "cofactor": 5.0,
    "apply_to_scatter": False,

    # ── Section 4 : Sélection marqueurs ──────────────────────────────────────
    "exclude_scatter": True,
    "exclude_markers": [""],

    # ── Section 5 : FlowSOM ──────────────────────────────────────────────────
    "xdim": 10,
    "ydim": 10,
    "rlen": "auto",                  # "auto" = √N × 0.1, ou entier fixe (10, 20, 50, 100)
    "n_clusters": 7,
    "seed": RANDOM_SEED,

    # Auto-clustering (Stabilité ARI + Silhouette — Méthode 2024)
    "auto_cluster": False,
    "min_k": 5,
    "max_k": 35,
    "n_bootstrap": 10,
    "sample_boot": 20000,
    "min_stability": 0.75,
    "w_stability": 0.65,
    "w_silhouette": 0.35,

    # ── Filtrage marqueurs -A vs -H (Area vs Height) ────────────────────────
    "apply_marker_filtering": True,  # Dédupliquer -A et -H pour chaque signal
    "keep_area": True,               # True = garder canaux -A (Area) [RECOMMANDÉ]
    "keep_height": False,            # True = garder canaux -H (Height)

    # ── AutoGating avancé ────────────────────────────────────────────────────
    "gmm_max_samples": 200_000,      # Sous-échantillonnage avant fit GMM
    "ransac_r2_threshold": 0.85,     # Seuil R² RANSAC (< → fallback ratio)

    # ── MRD méthode 1 (delta métaclusters — ELN 2022) ────────────────────────
    "mrd1_fold_change": 2.0,
    "mrd1_min_events": 17,

    # ── MRD méthode 2 (distances euclidiennes MAD — ELN 2022) ────────────────
    "mad_allowed": 4,
    "mrd2_ratio_threshold": 2.0,
    "mrd2_min_cells_per_cluster": 10,

    # ── MRD méthode 3 (benchmark M3/M8/M9/M12) ───────────────────────────────
    "mrd3_mapping_method": "M12_cosine_prior",
    "mrd3_unknown_mode": "auto_otsu",
    "mrd3_hard_limit_factor": 5.0,
    "m3_benchmark_methods": ["M3_cosine", "M8_ref_norm", "M9_prior", "M12_cosine_prior"],
    "m3_log_benchmark": True,

    # ── Comparaison / divergence ─────────────────────────────────────────────
    "divergence_threshold_pct": 0.5,

    # ── Output ───────────────────────────────────────────────────────────────
    "output_dir": "output/mrd_reports",
}


# =============================================================================
# SECTION 2 — UTILITAIRES IO & PRÉ-TRAITEMENT
# =============================================================================

def load_fcs_to_array(fcs_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Lit un fichier FCS et retourne (matrice, liste_marqueurs).
    Utilise fs.io.read_FCS() en priorité (identique au pipeline principal),
    puis flowkit, puis flowio en fallback.
    """
    fcs_path = Path(fcs_path)
    if not fcs_path.exists():
        raise FileNotFoundError(f"FCS introuvable : {fcs_path}")

    # Méthode 1 : fs.io.read_FCS — même appel que FlowSOM_Analysis_Pipeline.ipynb
    if FLOWSOM_AVAILABLE:
        try:
            adata = fs.io.read_FCS(str(fcs_path))
            data = np.array(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=np.float64)
            markers = list(adata.var_names)
            return data, markers
        except Exception as _e:
            logging.warning(f"[IO] fs.io.read_FCS échoué pour {fcs_path.name}: {_e}")

    # Méthode 2 : FlowKit
    if FLOWKIT_AVAILABLE:
        try:
            sample = fk.Sample(str(fcs_path))
            df = sample.as_dataframe(source="raw")
            return df.to_numpy(dtype=np.float64), list(df.columns)
        except Exception as _e:
            logging.warning(f"[IO] FlowKit échoué pour {fcs_path.name}: {_e}")

    # Méthode 3 : flowio (fallback minimal)
    try:
        import flowio
        with open(fcs_path, "rb") as f:
            fcs = flowio.FlowData(f)
        n_events = int(fcs.event_count)
        n_params = int(fcs.channel_count)
        data = np.reshape(np.array(fcs.events, dtype=np.float64), (n_events, n_params))
        # Les clés de fcs.channels sont des entiers et le champ s'appelle 'pnn' (minuscules)
        markers = [fcs.channels.get(i + 1, {}).get("pnn", f"Ch{i+1}") for i in range(n_params)]
        return data, markers
    except Exception as e:
        raise RuntimeError(f"Impossible de lire {fcs_path}: {e}")


# ── Helpers IO identiques au pipeline principal (FlowSOM_Analysis_Pipeline.ipynb) ──

def get_fcs_files(folder: Path) -> List[str]:
    """Récupère les FCS d'un dossier (.fcs et .FCS). Copie exacte du notebook principal."""
    folder = Path(folder)
    if not folder.exists():
        logging.warning(f"[IO] Dossier non trouvé : {folder}")
        return []
    files: set = set()
    for f in folder.glob("*.fcs"):
        files.add(str(f))
    for f in folder.glob("*.FCS"):
        files.add(str(f))
    return sorted(list(files))


def load_fcs_files(
    files: List[str],
    condition: str = "Unknown",
    verbose: bool = True,
) -> List[Any]:
    """
    Charge plusieurs FCS via fs.io.read_FCS() et retourne une liste d'AnnData.
    Copie exacte de load_fcs_files() dans FlowSOM_Analysis_Pipeline.ipynb.
    """
    import anndata as ad
    adatas = []
    for fpath in files:
        try:
            if verbose:
                print(f"    Chargement : {Path(fpath).name}...", end=" ")
            adata = fs.io.read_FCS(fpath)
            n_cells = adata.shape[0]
            adata.obs["condition"] = condition
            adata.obs["file_origin"] = Path(fpath).name
            adatas.append(adata)
            if verbose:
                print(f"{n_cells:,} cellules")
        except Exception as e:
            logging.exception(f"[IO] Erreur lecture {Path(fpath).name} : {e}")
            if verbose:
                print(f"\n    [ERREUR] {Path(fpath).name} : {e} (voir logs pour traceback complet)")
    return adatas


def select_cytometry_cols(
    markers: List[str],
    exclude_scatter: bool = True,
    exclude_additional: Optional[List[str]] = None,
    apply_marker_filtering: bool = True,
    keep_area: bool = True,
    keep_height: bool = False,
) -> Tuple[List[int], List[str]]:
    """
    Retourne les indices et noms des canaux de fluorescence à utiliser.

    Filtrage -A/-H (identique au notebook FlowSOM_Analysis_Pipeline.ipynb) :
        apply_marker_filtering=True → déduplique les canaux Area/Height.
        keep_area=True  → conserver les marqueurs se terminant par '-A' (Area).
        keep_height=True → conserver les marqueurs se terminant par '-H' (Height).
        Les marqueurs sans suffixe -A/-H sont toujours conservés.
    """
    exclude_additional = exclude_additional or []
    scatter_prefixes = ("FSC", "SSC", "TIME", "TIME_")

    # ── Étape 1 : filtrage -A / -H (copie de la logique Section 8 du notebook) ──
    if apply_marker_filtering:
        cols_with_A = {m for m in markers if m.upper().endswith("-A")}
        cols_with_H = {m for m in markers if m.upper().endswith("-H")}
        cols_other  = {m for m in markers
                       if not m.upper().endswith("-A") and not m.upper().endswith("-H")}
        keep_set: set = set(cols_other)   # toujours garder les marqueurs sans suffixe
        if keep_area:
            keep_set |= cols_with_A
        if keep_height:
            keep_set |= cols_with_H
        if not keep_area and not keep_height:
            keep_set = set(markers)       # sécurité : garder tout si les deux sont False
    else:
        keep_set = set(markers)

    # ── Étape 2 : exclusion scatter + marqueurs additionnels ─────────────────
    selected_idx, selected_names = [], []
    for i, m in enumerate(markers):
        if m not in keep_set:
            continue
        mu = m.upper()
        if exclude_scatter and any(mu.startswith(p) for p in scatter_prefixes):
            continue
        if any(ex.upper() in mu for ex in exclude_additional):
            continue
        selected_idx.append(i)
        selected_names.append(m)
    return selected_idx, selected_names


def apply_transform(
    X: np.ndarray,
    transform: str = "logicle",
    cofactor: float = 5.0,
) -> np.ndarray:
    """Applique la transformation cytométrique (logicle approximée ou arcsinh)."""
    if transform == "none":
        return X.copy()
    if transform == "arcsinh":
        return np.arcsinh(X / cofactor)
    if transform == "log10":
        return np.log10(np.maximum(X, 1.0))
    if transform == "logicle":
        if FLOWKIT_AVAILABLE:
            try:
                xf = fk.transforms.LogicleTransform(T=262144.0, M=4.5, W=0.5, A=0.0)
                return xf.apply(X.copy())
            except Exception:
                pass
        # Approximation sans FlowKit
        return np.arcsinh(X / 150.0) / np.log(10)
    return X.copy()


def _gate_viable_for_singlets(X: np.ndarray, markers: List[str]) -> np.ndarray:
    """Pré-filtre FSC/SSC 1-99% utilisé par AutoGating.auto_gate_singlets."""
    n = X.shape[0]
    mask = np.ones(n, dtype=bool)
    for patterns in [["FSC-A", "FSC-H", "FSC"], ["SSC-A", "SSC-H", "SSC"]]:
        ci = _find_col(markers, patterns)
        if ci is not None:
            v = X[:, ci].astype(np.float64)
            v = np.where(np.isfinite(v), v, np.nan)
            mask &= (
                np.isfinite(v)
                & (v >= np.nanpercentile(v, 1.0))
                & (v <= np.nanpercentile(v, 99.0))
            )
    return mask


# =============================================================================
# SECTION 2b — CLASSE AutoGating (GMM adaptatif + RANSAC singlets)
# =============================================================================
# Portée directement depuis FlowSOM_Analysis_Pipeline.ipynb (Section AutoGating V2).
# Utilisée par apply_pregating() en mode gating_mode="auto".
# Seuils clé : RANSAC_R2_THRESHOLD = 0.85, GMM_MAX_SAMPLES = 200 000.
# =============================================================================

class AutoGating:
    """
    Gating automatique adaptatif basé sur GMM (Gaussian Mixture Models) et
    régression linéaire robuste RANSAC.

    Méthodes principales :
        safe_fit_gmm       — Wrapper GMM avec retry + fallback unimodal + sous-échantillonnage
        auto_gate_debris   — GMM 2D sur FSC-A/SSC-A (débris adaptatifs)
        auto_gate_singlets — RANSAC FSC-A vs FSC-H + contrôle R² + fallback ratio
        auto_gate_cd45     — GMM 1D bimodal CD45- / CD45+
        auto_gate_cd34     — GMM 1D CD34+ + filtre SSC low pour blastes

    Constantes de classe (surchargeables via DEFAULT_PARAMS) :
        RANSAC_R2_THRESHOLD : 0.85   — seuil R² RANSAC (< → fallback ratio)
        GMM_MAX_SAMPLES     : 200 000 — sous-échantillonnage avant GMM
    """

    RANSAC_R2_THRESHOLD: float = 0.85
    GMM_MAX_SAMPLES: int = 200_000

    @staticmethod
    def _subsample(data: np.ndarray, max_samples: int) -> np.ndarray:
        """Sous-échantillonne les données si > max_samples (convergence GMM)."""
        if data.shape[0] > max_samples:
            idx = np.random.choice(data.shape[0], size=max_samples, replace=False)
            logging.debug(f"[GMM] Sous-échantillonnage: {data.shape[0]:,} → {max_samples:,}")
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
        max_samples: Optional[int] = None,
    ) -> Any:
        """
        Wrapper robuste GaussianMixture avec gestion d'erreurs.
        Sous-échantillonne à max_samples avant fit, retry × max_retries,
        fallback unimodal si tout échoue.

        Returns:
            GaussianMixture fitté.
        Raises:
            RuntimeError si fit impossible malgré tous les fallbacks.
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn requis pour AutoGating.safe_fit_gmm()")
        ms = max_samples or AutoGating.GMM_MAX_SAMPLES
        data_fit = AutoGating._subsample(data, ms)
        last_error: Optional[Exception] = None
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
                    log_gating_event("GMM", f"n={n_components}", "warning",
                                     {"attempt": attempt + 1},
                                     f"Non-convergé tentative {attempt + 1}/{max_retries}")
                    continue
                return gmm
            except Exception as e:
                last_error = e
                log_gating_event("GMM", f"n={n_components}", "error",
                                 {"attempt": attempt + 1, "error": str(e)})
        # Fallback unimodal
        if n_components > 1:
            warn = f"GMM fallback unimodal après {max_retries} échecs (erreur: {last_error})"
            log_gating_event("GMM", "fallback_unimodal", "fallback",
                             {"original_n": n_components}, warn)
            try:
                gmm = GaussianMixture(
                    n_components=1, random_state=random_state,
                    n_init=1, covariance_type=covariance_type, max_iter=max_iter,
                )
                gmm.fit(data_fit)
                return gmm
            except Exception as e:
                raise RuntimeError(f"GMM fit échoué après {max_retries} tentatives + fallback: {e}")
        raise RuntimeError(f"GMM fit échoué après {max_retries} tentatives: {last_error}")

    @staticmethod
    def auto_gate_debris(
        X: np.ndarray,
        var_names: List[str],
        n_components: int = 3,
        min_cluster_fraction: float = 0.02,
    ) -> np.ndarray:
        """
        Gate débris adaptatif par GMM 2D sur FSC-A / SSC-A.
        Sélection automatique du nombre de composantes par BIC (2 ou 3).

        Returns: masque booléen (True = cellule viable, False = débris).
        """
        if not SKLEARN_AVAILABLE:
            logging.warning("[AutoGating] sklearn absent → gate_debris désactivé")
            return np.ones(X.shape[0], dtype=bool)

        n_cells = X.shape[0]
        fsc_i = _find_col(var_names, ["FSC-A"])
        ssc_i = _find_col(var_names, ["SSC-A"])

        if fsc_i is None or ssc_i is None:
            log_gating_event("debris", "auto_gmm", "error",
                             warning_msg="FSC-A ou SSC-A non trouvé")
            return np.ones(n_cells, dtype=bool)

        fsc = X[:, fsc_i].astype(np.float64)
        ssc = X[:, ssc_i].astype(np.float64)
        valid = np.isfinite(fsc) & np.isfinite(ssc)
        data_2d = np.column_stack([fsc[valid], ssc[valid]])

        if valid.sum() < 200:
            return np.ones(n_cells, dtype=bool)

        scaler = StandardScaler()
        data_sc = scaler.fit_transform(data_2d)
        ms = AutoGating.GMM_MAX_SAMPLES

        best_bic, best_gmm = np.inf, None
        for n_comp in [2, 3]:
            try:
                g = AutoGating.safe_fit_gmm(data_sc, n_components=n_comp,
                                            covariance_type="full", n_init=3, max_iter=200)
                bic = g.bic(AutoGating._subsample(data_sc, ms))
                if bic < best_bic:
                    best_bic, best_gmm = bic, g
            except RuntimeError as e:
                logging.warning(f"[AutoGating] GMM {n_comp} composantes: {e}")

        if best_gmm is None:
            log_gating_event("debris", "auto_gmm", "fallback",
                             warning_msg="Aucun GMM convergé, toutes cellules conservées")
            return np.ones(n_cells, dtype=bool)

        labels = best_gmm.predict(data_sc)
        n_comp = best_gmm.n_components
        cluster_sizes = np.bincount(labels, minlength=n_comp)
        cluster_fsc_means = np.array([data_2d[labels == i, 0].mean() for i in range(n_comp)])
        main_cluster = int(np.argmax(cluster_sizes))
        fsc_threshold = cluster_fsc_means[main_cluster] * 0.25

        mask_valid = np.zeros(valid.sum(), dtype=bool)
        for i in range(n_comp):
            if (cluster_sizes[i] / len(labels) >= min_cluster_fraction
                    and cluster_fsc_means[i] >= fsc_threshold):
                mask_valid |= (labels == i)
        if not mask_valid.any():
            mask_valid = (labels == main_cluster)

        mask = np.zeros(n_cells, dtype=bool)
        mask[valid] = mask_valid
        n_kept = int(mask.sum())

        log_gating_event("debris", "auto_gmm", "success", {
            "n_components": int(n_comp), "bic": float(best_bic),
            "n_kept": n_kept, "n_total": n_cells,
        })
        gating_reports.append(GateResult(
            mask=mask, n_kept=n_kept, n_total=n_cells,
            method="auto_gmm_debris", gate_name="G1_debris",
            details={"n_components": int(n_comp), "bic": float(best_bic),
                     "cluster_fsc_means": cluster_fsc_means.tolist()},
        ))
        logging.info(f"[Auto-GMM débris] {n_comp} composantes → {n_kept:,}/{n_cells:,}")
        return mask

    @staticmethod
    def auto_gate_singlets(
        X: np.ndarray,
        var_names: List[str],
        file_origin: Optional[np.ndarray] = None,
        per_file: bool = False,
        r2_threshold: float = 0.85,
        file_id: str = "",
    ) -> np.ndarray:
        """
        Gate singlets adaptatif par régression RANSAC sur FSC-A vs FSC-H.

        V2 : contrôle R² sur inliers RANSAC + fallback ratio si R² < r2_threshold.
        En mode headless (per_file=False) : régression globale sur toutes les données.

        Returns: masque booléen (True = singlet, False = doublet).
        """
        if not SKLEARN_AVAILABLE:
            logging.warning("[AutoGating] sklearn absent → gate_singlets désactivé")
            return np.ones(X.shape[0], dtype=bool)

        n_cells = X.shape[0]
        fsc_a_i = _find_col(var_names, ["FSC-A"])
        fsc_h_i = _find_col(var_names, ["FSC-H"])

        if fsc_a_i is None or fsc_h_i is None:
            return np.ones(n_cells, dtype=bool)

        fsc_a = X[:, fsc_a_i].astype(np.float64)
        fsc_h = X[:, fsc_h_i].astype(np.float64)
        viable = _gate_viable_for_singlets(X, var_names)
        valid = viable & np.isfinite(fsc_a) & np.isfinite(fsc_h) & (fsc_h > 100) & (fsc_a > 100)

        if valid.sum() < 200:
            return np.ones(n_cells, dtype=bool)

        mask = np.zeros(n_cells, dtype=bool)

        def _fallback_ratio(fa, fh, rmin=0.6, rmax=1.5):
            ratio = fa.ravel() / np.maximum(fh.ravel(), 1.0)
            return (ratio >= rmin) & (ratio <= rmax)

        if per_file and file_origin is not None:
            unique_files = np.unique(file_origin)
            for fname in unique_files:
                fmask = (file_origin == fname) & valid
                if fmask.sum() < 50:
                    mask[fmask] = True
                    continue
                fa_f = fsc_a[fmask].reshape(-1, 1)
                fh_f = fsc_h[fmask].reshape(-1, 1)
                try:
                    ransac = RANSACRegressor(
                        estimator=LinearRegression(), min_samples=50,
                        residual_threshold=None, random_state=42, max_trials=100,
                    )
                    ransac.fit(fh_f, fa_f.ravel())
                    inlier_mask = ransac.inlier_mask_
                    r2_val = None
                    if inlier_mask is not None and inlier_mask.sum() > 50:
                        r2_val = r2_score(fa_f[inlier_mask].ravel(),
                                          ransac.predict(fh_f[inlier_mask]))
                        if r2_val < r2_threshold:
                            # Capture de l'échec RANSAC dans ransac_scatter_data
                            ransac_scatter_data[str(fname)] = {
                                "r2": float(r2_val), "method": "ratio_fallback",
                                "fallback": True, "n_total": int(fmask.sum()),
                            }
                            singlets_f = _fallback_ratio(fa_f, fh_f)
                            mask[np.where(fmask)[0]] = singlets_f
                            continue
                    slope_f = float(ransac.estimator_.coef_[0])
                    intercept_f = float(ransac.estimator_.intercept_)
                    n_inliers_f = int(inlier_mask.sum()) if inlier_mask is not None else None
                    # Enregistrement RANSAC par fichier pour audit
                    ransac_scatter_data[str(fname)] = {
                        "slope": slope_f, "intercept": intercept_f,
                        "r2": float(r2_val) if r2_val is not None else None,
                        "n_inliers": n_inliers_f, "n_total": int(fmask.sum()),
                        "method": "ransac", "fallback": False,
                    }
                    residuals = fa_f.ravel() - ransac.predict(fh_f)
                    med, mad = np.median(residuals), np.median(np.abs(residuals - np.median(residuals)))
                    mask[np.where(fmask)[0]] = residuals <= (med + 3.0 * mad)
                except Exception:
                    mask[fmask] = True
        else:
            fa_v = fsc_a[valid].reshape(-1, 1)
            fh_v = fsc_h[valid].reshape(-1, 1)
            try:
                ransac = RANSACRegressor(
                    estimator=LinearRegression(), min_samples=100,
                    residual_threshold=None, random_state=42, max_trials=100,
                )
                ransac.fit(fh_v, fa_v.ravel())
                inlier_mask = ransac.inlier_mask_
                r2_val = None
                if inlier_mask is not None and inlier_mask.sum() > 50:
                    r2_val = r2_score(fa_v[inlier_mask].ravel(),
                                      ransac.predict(fh_v[inlier_mask]))
                    if r2_val < r2_threshold:
                        log_gating_event("singlets", "ransac_fallback_ratio", "fallback",
                                         {"r2": float(r2_val)},
                                         f"R² faible (R²={r2_val:.2f} < {r2_threshold})")
                        # Enregistrement de l'échec RANSAC pour audit
                        ransac_scatter_data[file_id or "global"] = {
                            "r2": float(r2_val), "method": "ratio_fallback",
                            "fallback": True, "n_total": int(valid.sum()),
                        }
                        mask[valid] = _fallback_ratio(fa_v, fh_v)
                        n_s = int(mask.sum())
                        logging.info(f"[RANSAC fallback ratio] Singlets: {n_s:,}/{valid.sum():,}")
                        gating_reports.append(GateResult(
                            mask=mask, n_kept=n_s, n_total=n_cells,
                            method="ratio_fallback_global", gate_name="G2_singlets",
                            details={"r2": float(r2_val), "file_id": file_id},
                            warnings=["R² faible → fallback ratio"],
                        ))
                        return mask
                residuals = fa_v.ravel() - ransac.predict(fh_v)
                med, mad = np.median(residuals), np.median(np.abs(residuals - np.median(residuals)))
                mask[valid] = residuals <= (med + 3.0 * mad)
                slope = float(ransac.estimator_.coef_[0])
                intercept = float(ransac.estimator_.intercept_)
                r2_str = f", R²={r2_val:.3f}" if r2_val is not None else ""
                logging.info(f"[Auto-RANSAC] y={slope:.3f}x + {intercept:.0f}{r2_str} | "
                             f"singlets: {int(mask.sum()):,}/{valid.sum():,}")
                # Enregistrement RANSAC global pour audit (peuplé ici pour la 1ère fois)
                ransac_scatter_data[file_id or "global"] = {
                    "slope": slope, "intercept": intercept,
                    "r2": float(r2_val) if r2_val is not None else None,
                    "n_inliers": int(inlier_mask.sum()) if inlier_mask is not None else None,
                    "n_total": int(valid.sum()),
                    "method": "ransac", "fallback": False,
                }
            except Exception as e:
                logging.warning(f"[AutoGating] RANSAC singlets échoué: {e} → conservation")
                mask[valid] = True

        n_s = int(mask.sum())
        gating_reports.append(GateResult(
            mask=mask, n_kept=n_s, n_total=n_cells,
            method="ransac_singlets", gate_name="G2_singlets",
            details={"per_file": per_file, "r2_threshold": r2_threshold, "file_id": file_id},
        ))
        # Enregistrement singlets pour résumé par fichier
        if file_id:
            singlets_summary_per_file.append({
                "file": file_id,
                "n_singlets": n_s,
                "n_total": n_cells,
                "pct_singlets": round(n_s / max(n_cells, 1) * 100, 2),
                "ransac_data": ransac_scatter_data.get(file_id),
            })
        return mask

    @staticmethod
    def auto_gate_cd45(
        X: np.ndarray,
        var_names: List[str],
        n_components: int = 2,
        threshold_percentile: float = 5.0,
    ) -> np.ndarray:
        """
        Gate CD45+ adaptatif par GMM 1D bimodal.
        Trouve le creux bimodal CD45-/CD45+ en lieu et place d'un percentile fixe.

        Returns: masque booléen (True = CD45+).
        """
        if not SKLEARN_AVAILABLE:
            logging.warning("[AutoGating] sklearn absent → fallback percentile CD45")
            n_cells = X.shape[0]
            cd45_i = _find_col(var_names, ["CD45"])
            if cd45_i is None:
                return np.ones(n_cells, dtype=bool)
            v = X[:, cd45_i].astype(np.float64)
            v = np.where(np.isfinite(v), v, np.nan)
            thr = np.nanpercentile(v, threshold_percentile)
            return np.where(np.isnan(v), False, v > thr)

        n_cells = X.shape[0]
        cd45_i = _find_col(var_names, ["CD45", "CD45-PECY5", "CD45-PC5"])
        if cd45_i is None:
            return np.ones(n_cells, dtype=bool)

        cd45 = X[:, cd45_i].astype(np.float64)
        valid = np.isfinite(cd45)
        if valid.sum() < 200:
            return np.ones(n_cells, dtype=bool)

        try:
            gmm = AutoGating.safe_fit_gmm(cd45[valid].reshape(-1, 1), n_components=n_components)
        except RuntimeError as e:
            warn = f"GMM CD45 échoué: {e} — fallback percentile"
            log_gating_event("cd45", "gmm_fallback_percentile", "fallback",
                             {"error": str(e)}, warn)
            thr = np.nanpercentile(cd45[valid], threshold_percentile)
            mask = np.zeros(n_cells, dtype=bool)
            mask[valid] = cd45[valid] > thr
            gating_reports.append(GateResult(
                mask=mask, n_kept=int(mask.sum()), n_total=int(valid.sum()),
                method="gmm_cd45_fallback_percentile", gate_name="G3_cd45",
                details={"threshold": float(thr), "fallback": True},
                warnings=[warn],
            ))
            return mask

        labels = gmm.predict(cd45[valid].reshape(-1, 1))
        means = gmm.means_.flatten()
        pos_component = int(np.argmax(means))
        sorted_means = np.sort(means)
        stds = np.sqrt(gmm.covariances_.flatten())
        sorted_stds = stds[np.argsort(means)]
        denom = sorted_stds[0] + sorted_stds[1]
        threshold_approx = float(
            (sorted_means[0] * sorted_stds[1] + sorted_means[1] * sorted_stds[0]) / max(denom, 1e-8)
        )

        mask = np.zeros(n_cells, dtype=bool)
        mask[valid] = (labels == pos_component)
        n_pos = int(mask.sum())
        logging.info(f"[Auto-GMM CD45] μ={means.round(0)} → seuil≈{threshold_approx:.0f}, CD45+: {n_pos:,}")

        log_gating_event("cd45", "gmm", "success", {
            "means": means.tolist(), "threshold": threshold_approx, "n_pos": n_pos,
        })
        gating_reports.append(GateResult(
            mask=mask, n_kept=n_pos, n_total=int(valid.sum()),
            method="gmm_cd45", gate_name="G3_cd45",
            details={"means": means.tolist(), "threshold": threshold_approx, "fallback": False},
        ))
        return mask

    @staticmethod
    def auto_gate_cd34(
        X: np.ndarray,
        var_names: List[str],
        use_ssc_filter: bool = True,
        n_components: int = 2,
    ) -> np.ndarray:
        """
        Gate CD34+ blastes adaptatif par GMM 1D.
        Combine optionnellement avec un filtre GMM SSC low (blastes = faible granularité).

        Returns: masque booléen (True = blaste CD34+).
        """
        if not SKLEARN_AVAILABLE:
            logging.warning("[AutoGating] sklearn absent → gate_cd34 désactivé")
            return np.ones(X.shape[0], dtype=bool)

        n_cells = X.shape[0]
        cd34_i = _find_col(var_names, ["CD34", "CD34-PE", "CD34-APC", "CD34-PECY7"])
        if cd34_i is None:
            return np.ones(n_cells, dtype=bool)

        cd34 = X[:, cd34_i].astype(np.float64)
        valid = np.isfinite(cd34)
        if valid.sum() < 200:
            return np.ones(n_cells, dtype=bool)

        try:
            gmm = AutoGating.safe_fit_gmm(cd34[valid].reshape(-1, 1), n_components=n_components)
        except RuntimeError as e:
            log_gating_event("cd34", "gmm", "error", {"error": str(e)},
                             f"GMM CD34 échoué: {e}")
            return np.ones(n_cells, dtype=bool)

        labels = gmm.predict(cd34[valid].reshape(-1, 1))
        means = gmm.means_.flatten()
        pos_component = int(np.argmax(means))
        mask_cd34 = np.zeros(n_cells, dtype=bool)
        mask_cd34[valid] = (labels == pos_component)
        logging.info(f"[Auto-GMM CD34] μ={means.round(0)}, CD34+ cluster μ={means[pos_component]:.0f}")

        if use_ssc_filter:
            ssc_i = _find_col(var_names, ["SSC-A", "SSC-H", "SSC"])
            if ssc_i is not None:
                ssc = X[:, ssc_i].astype(np.float64)
                valid_ssc = np.isfinite(ssc)
                if valid_ssc.sum() >= 200:
                    try:
                        gmm_ssc = AutoGating.safe_fit_gmm(
                            ssc[valid_ssc].reshape(-1, 1), n_components=2)
                        labels_ssc = gmm_ssc.predict(ssc[valid_ssc].reshape(-1, 1))
                        ssc_means = gmm_ssc.means_.flatten()
                        low_ssc = int(np.argmin(ssc_means))
                        mask_ssc = np.zeros(n_cells, dtype=bool)
                        mask_ssc[valid_ssc] = (labels_ssc == low_ssc)
                        combined = mask_cd34 & mask_ssc
                        logging.info(f"[Auto-GMM CD34+SSC] blastes purs: {int(combined.sum()):,}")
                        gating_reports.append(GateResult(
                            mask=combined, n_kept=int(combined.sum()), n_total=int(valid.sum()),
                            method="gmm_cd34_ssc", gate_name="G4_cd34",
                            details={"cd34_means": means.tolist(), "ssc_means": ssc_means.tolist()},
                        ))
                        return combined
                    except RuntimeError as e:
                        logging.warning(f"[AutoGating] GMM SSC échoué: {e} — CD34 sans filtre SSC")

        gating_reports.append(GateResult(
            mask=mask_cd34, n_kept=int(mask_cd34.sum()), n_total=int(valid.sum()),
            method="gmm_cd34", gate_name="G4_cd34",
            details={"means": means.tolist()},
        ))
        return mask_cd34


def _find_col(markers: List[str], patterns: List[str]) -> Optional[int]:
    """Trouve l'indice d'un marqueur parmi les patterns (insensible à la casse)."""
    mu = [m.upper() for m in markers]
    for p in patterns:
        for i, m in enumerate(mu):
            if p.upper() in m:
                return i
    return None


def apply_pregating(
    X: np.ndarray,
    markers: List[str],
    params: Dict[str, Any],
    file_id: str = "",
    mode: str = "cd45",
) -> np.ndarray:
    """
    Applique le pré-gating et retourne un masque booléen.

    Dispatch selon params["gating_mode"] et le paramètre mode :
        mode="cd45"   → pipeline complet incl. Gate CD45+ (M1/M3)
        mode="global" → seulement débris + singlets RANSAC, pas de CD45 (M2)
        "auto"   → AutoGating (GMM adaptatif + RANSAC singlets)
        "manual" → _apply_pregating_manual (gating par percentiles)

    Si params["apply_pregating"] est False, retourne un masque tout-à-True.
    """
    if not params.get("apply_pregating", True):
        return np.ones(X.shape[0], dtype=bool)

    # mode="global" → désactive le gating CD45 et CD34 (M2 : toutes cellules mappées)
    effective_params = dict(params)
    if mode == "global":
        effective_params["gate_cd45"] = False
        effective_params["filter_blasts"] = False

    gating_mode = effective_params.get("gating_mode", "manual")
    if gating_mode == "auto":
        return _apply_pregating_auto(X, markers, effective_params, file_id=file_id)
    return _apply_pregating_manual(X, markers, effective_params, file_id=file_id)


def _apply_pregating_manual(
    X: np.ndarray,
    markers: List[str],
    params: Dict[str, Any],
    file_id: str = "",
) -> np.ndarray:
    """
    Gating manuel par percentiles (Gate 1-4).
    Respecte les mêmes logiques que PreGating dans le notebook principal.
    """
    n = X.shape[0]
    mask = np.ones(n, dtype=bool)
    _fid = f"[{file_id}] " if file_id else ""

    print(f"  {_fid}Tri cellulaire (mode MANUEL) — {n:,} cellules brutes")

    # Gate 1 — Débris (FSC/SSC percentiles)
    if params.get("gate_debris", True):
        fsc_i = _find_col(markers, ["FSC-A", "FSC"])
        ssc_i = _find_col(markers, ["SSC-A", "SSC"])
        lo, hi = params.get("debris_min_pct", 1.0), params.get("debris_max_pct", 99.0)
        for ci in [fsc_i, ssc_i]:
            if ci is not None:
                v = X[:, ci].astype(float)
                v = np.where(np.isfinite(v), v, np.nan)
                mask &= np.isfinite(v) & (v >= np.nanpercentile(v, lo)) & (v <= np.nanpercentile(v, hi))
        print(f"  {_fid}  G1 Débris   → {mask.sum():>8,}/{n:>8,}  "
              f"({mask.sum()/n*100:5.1f}%)  [FSC/SSC p{lo:.0f}-p{hi:.0f}]")

    # Gate 2 — Doublets (ratio FSC-A/FSC-H)
    if params.get("gate_doublets", True):
        fa_i = _find_col(markers, ["FSC-A"])
        fh_i = _find_col(markers, ["FSC-H"])
        if fa_i is not None and fh_i is not None:
            fa, fh = X[:, fa_i].astype(float), X[:, fh_i].astype(float)
            valid = fh > 100
            ratio = np.full(n, np.nan)
            ratio[valid] = fa[valid] / fh[valid]
            rmin, rmax = params.get("ratio_min", 0.6), params.get("ratio_max", 1.4)
            mask &= np.isfinite(ratio) & (ratio >= rmin) & (ratio <= rmax)
        print(f"  {_fid}  G2 Singlets → {mask.sum():>8,}/{n:>8,}  "
              f"({mask.sum()/n*100:5.1f}%)  [ratio FSC-A/H ∈ [{rmin},{rmax}]]")

    # Gate 3 — CD45+ leucocytes
    if params.get("gate_cd45", True):
        cd45_i = _find_col(markers, ["CD45"])
        if cd45_i is not None:
            v = X[:, cd45_i].astype(float)
            v = np.where(np.isfinite(v), v, np.nan)
            thr = np.nanpercentile(v, params.get("cd45_pct", 5))
            mask &= np.where(np.isnan(v), False, v > thr)
        print(f"  {_fid}  G3 CD45+    → {mask.sum():>8,}/{n:>8,}  "
              f"({mask.sum()/n*100:5.1f}%)  [percentile p{params.get('cd45_pct', 5)}]")

    # Gate 4 — Blastes CD34+ (optionnel)
    if params.get("filter_blasts", False):
        cd34_i = _find_col(markers, ["CD34", "CD34-PE", "CD34-APC", "CD34-PECY7"])
        if cd34_i is not None:
            v = X[:, cd34_i].astype(float)
            v = np.where(np.isfinite(v), v, np.nan)
            thr_cd34 = np.nanpercentile(v, params.get("cd34_pct", 85))
            mask_cd34 = np.where(np.isnan(v), False, v >= thr_cd34)
            if params.get("ssc_filter", True):
                ssc_i = _find_col(markers, ["SSC-A", "SSC-H", "SSC"])
                if ssc_i is not None:
                    sv = X[:, ssc_i].astype(float)
                    sv = np.where(np.isfinite(sv), sv, np.nan)
                    thr_ssc = np.nanpercentile(sv, params.get("ssc_max_pct", 70))
                    mask_cd34 &= np.where(np.isnan(sv), False, sv <= thr_ssc)
            mask &= mask_cd34
        print(f"  {_fid}  G4 CD34+    → {mask.sum():>8,}/{n:>8,}  "
              f"({mask.sum()/n*100:5.1f}%)  [CD34 ≥ p{params.get('cd34_pct', 85)}]")

    print(f"  {_fid}  ✓ Total retenu : {mask.sum():,}/{n:,}  ({mask.sum()/n*100:.1f}%)")
    return mask


def _apply_pregating_auto(
    X: np.ndarray,
    markers: List[str],
    params: Dict[str, Any],
    file_id: str = "",
) -> np.ndarray:
    """
    Gating adaptatif GMM/RANSAC via la classe AutoGating.
    Requiert scikit-learn. Fallback vers gating manuel si sklearn absent.
    """
    if not SKLEARN_AVAILABLE:
        logging.warning("[AutoGating] sklearn non disponible → fallback gating manuel")
        return _apply_pregating_manual(X, markers, params, file_id=file_id)

    n_total = X.shape[0]
    mask = np.ones(n_total, dtype=bool)
    _fid = f"[{file_id}] " if file_id else ""

    print(f"  {_fid}Tri cellulaire (mode AUTO) — {n_total:,} cellules brutes")

    # Mise à jour des constantes AutoGating depuis les params
    AutoGating.GMM_MAX_SAMPLES = int(params.get("gmm_max_samples", AutoGating.GMM_MAX_SAMPLES))
    AutoGating.RANSAC_R2_THRESHOLD = float(
        params.get("ransac_r2_threshold", AutoGating.RANSAC_R2_THRESHOLD)
    )

    # Gate 1 — Débris (GMM 2D FSC-A/SSC-A)
    if params.get("gate_debris", True):
        _idx_before = len(gating_reports)
        mask &= AutoGating.auto_gate_debris(X, markers)
        for _r in gating_reports[_idx_before:]:
            _r.details.setdefault("file_id", file_id)
        print(f"  {_fid}  G1 Débris   → {mask.sum():>8,}/{n_total:>8,}  "
              f"({mask.sum()/n_total*100:5.1f}%)  [GMM 2D FSC/SSC]")

    # Gate 2 — Doublets / Singlets (RANSAC FSC-A vs FSC-H)
    if params.get("gate_doublets", True):
        _idx_before = len(gating_reports)
        mask &= AutoGating.auto_gate_singlets(
            X, markers,
            per_file=False,
            r2_threshold=float(params.get("ransac_r2_threshold", AutoGating.RANSAC_R2_THRESHOLD)),
            file_id=file_id,
        )
        for _r in gating_reports[_idx_before:]:
            _r.details.setdefault("file_id", file_id)
        _ransac = ransac_scatter_data.get(file_id or "global", {})
        _r2_str = f"R²={_ransac['r2']:.3f}" if _ransac.get("r2") is not None else "R²=N/A"
        _method_str = "RANSAC" if not _ransac.get("fallback") else "ratio (fallback)"
        print(f"  {_fid}  G2 Singlets → {mask.sum():>8,}/{n_total:>8,}  "
              f"({mask.sum()/n_total*100:5.1f}%)  [{_method_str}, {_r2_str}]")

    # Gate 3 — CD45+ (GMM 1D bimodal)
    if params.get("gate_cd45", True):
        _idx_before = len(gating_reports)
        mask &= AutoGating.auto_gate_cd45(X, markers)
        for _r in gating_reports[_idx_before:]:
            _r.details.setdefault("file_id", file_id)
        print(f"  {_fid}  G3 CD45+    → {mask.sum():>8,}/{n_total:>8,}  "
              f"({mask.sum()/n_total*100:5.1f}%)  [GMM 1D bimodal]")

    # Gate 4 — Blastes CD34+ (GMM 1D + SSC low optionnel)
    if params.get("filter_blasts", False):
        _idx_before = len(gating_reports)
        mask &= AutoGating.auto_gate_cd34(
            X, markers,
            use_ssc_filter=bool(params.get("ssc_filter", True)),
        )
        for _r in gating_reports[_idx_before:]:
            _r.details.setdefault("file_id", file_id)
        print(f"  {_fid}  G4 CD34+    → {mask.sum():>8,}/{n_total:>8,}  "
              f"({mask.sum()/n_total*100:5.1f}%)  [GMM 1D + SSC]")

    print(f"  {_fid}  ✓ Total retenu : {mask.sum():,}/{n_total:,}  ({mask.sum()/n_total*100:.1f}%)")
    return mask



# =============================================================================
# SECTION 2c — FONCTIONS D'AUDIT GATING (export + résumé)
# =============================================================================

def print_gating_summary() -> None:
    """
    Affiche un tableau récapitulatif du tri cellulaire par fichier et par gate.
    À appeler dans le notebook après le lancement de la pipeline :
        import mrd_pipeline as mp
        mp.print_gating_summary()

    Utilise les globaux gating_reports et singlets_summary_per_file.
    """
    if not gating_reports:
        print("[Gating] Aucun rapport de gating disponible. Lancez la pipeline d'abord.")
        return

    # Regroupement par fichier puis par gate
    from collections import defaultdict
    by_file: Dict[str, List[GateResult]] = defaultdict(list)
    no_file: List[GateResult] = []
    for r in gating_reports:
        fid = r.details.get("file_id", "")
        if fid:
            by_file[fid].append(r)
        else:
            no_file.append(r)
    if no_file:
        by_file["(global)"] = no_file

    print("\n" + "=" * 72)
    print("  AUDIT TRI CELLULAIRE — résumé par fichier")
    print("=" * 72)
    for fname, reports in sorted(by_file.items()):
        print(f"\n  Fichier : {fname}")
        # Trouver le total de cellules brutes (n_total du premier gate)
        n_raw = reports[0].n_total if reports else "?"
        print(f"  {'Gate':<20}  {'Retenu':>8}  {'Total':>8}  {'%':>6}  Méthode")
        print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*20}")
        for r in reports:
            pct = f"{r.pct_kept:.1f}%" if r.n_total > 0 else "?"
            print(f"  {r.gate_name:<20}  {r.n_kept:>8,}  {r.n_total:>8,}  {pct:>6}  {r.method}")
            if r.warnings:
                for w in r.warnings:
                    print(f"    ⚠️  {w}")
    print("=" * 72 + "\n")


def print_ransac_summary() -> None:
    """
    Affiche un tableau récapitulatif des résultats RANSAC singlets par fichier.
    À appeler dans le notebook après le lancement de la pipeline :
        import mrd_pipeline as mp
        mp.print_ransac_summary()

    Utilise le global ransac_scatter_data.
    """
    if not ransac_scatter_data:
        print("[RANSAC] Aucune donnée RANSAC disponible. Lancez la pipeline avec gating_mode='auto'.")
        return

    print("\n" + "=" * 80)
    print("  AUDIT RANSAC SINGLETS — résumé par fichier")
    print("=" * 80)
    print(f"  {'Fichier':<35}  {'Méthode':<16}  {'R²':>6}  {'Pente':>8}  {'Intercept':>10}  "
          f"{'Inliers':>8}  {'Total':>8}")
    print(f"  {'-'*35}  {'-'*16}  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}")
    for fname, d in sorted(ransac_scatter_data.items()):
        r2_str = f"{d['r2']:.3f}" if d.get("r2") is not None else "N/A"
        slope_str = f"{d['slope']:.3f}" if d.get("slope") is not None else "N/A"
        intercept_str = f"{d['intercept']:.0f}" if d.get("intercept") is not None else "N/A"
        inliers_str = f"{d['n_inliers']:,}" if d.get("n_inliers") is not None else "N/A"
        total_str = f"{d['n_total']:,}" if d.get("n_total") is not None else "N/A"
        method_str = d.get("method", "?") + (" ⚠️" if d.get("fallback") else "")
        print(f"  {fname:<35}  {method_str:<16}  {r2_str:>6}  {slope_str:>8}  "
              f"{intercept_str:>10}  {inliers_str:>8}  {total_str:>8}")
    print("=" * 80 + "\n")


def reset_gating_logs() -> None:
    """
    Remet à zéro tous les logs de gating (utile en cas de ré-exécution du notebook).
    À appeler avant de relancer la pipeline pour repartir d'un état propre :
        import mrd_pipeline as mp
        mp.reset_gating_logs()
    """
    global gating_reports, ransac_scatter_data, singlets_summary_per_file
    gating_reports.clear()
    ransac_scatter_data.clear()
    singlets_summary_per_file.clear()
    print("[Gating] Logs remis à zéro.")


def preprocess_fcs(
    fcs_path: Path,
    params: Dict[str, Any],
    is_nbm_reference: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Pipeline complet : lecture FCS → gating → transformation → sélection marqueurs.
    Retourne (X_transformed, marker_names) sur les canaux de fluorescence.

    Args:
        is_nbm_reference: Si True, skip le gating (les NBM sont poolés bruts,
                          comme dans FlowSOM_Analysis_Pipeline.ipynb).
    """
    X_raw, markers = load_fcs_to_array(fcs_path)

    # Vérification NaN — critique pour FlowSOM
    if np.isnan(X_raw).any():
        nan_count = int(np.isnan(X_raw).sum())
        logging.warning(f"   ⚠️  {nan_count} NaN détectés → remplacement par 0")
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Gating — désactivé pour les références NBM (fidèle au notebook principal)
    if is_nbm_reference or not params.get("apply_pregating", True):
        X_gated = X_raw
        logging.info(f"   Gating : SKIP (is_nbm_reference={is_nbm_reference})")
    else:
        logging.info(f"   Mode gating : {params.get('gating_mode', 'manual').upper()}")
        gate_mask = apply_pregating(X_raw, markers, params, file_id=str(fcs_path.name))
        X_gated = X_raw[gate_mask]
        logging.info(f"   Gating : {gate_mask.sum():,}/{len(gate_mask):,} cellules retenues")

    # Transformation
    X_tf = apply_transform(X_gated, params.get("transform", "logicle"), params.get("cofactor", 5.0))

    # Sélection marqueurs
    sel_idx, sel_names = select_cytometry_cols(
        markers,
        exclude_scatter=params.get("exclude_scatter", True),
        exclude_additional=params.get("exclude_markers", ["CD45"]),
        apply_marker_filtering=params.get("apply_marker_filtering", True),
        keep_area=params.get("keep_area", True),
        keep_height=params.get("keep_height", False),
    )
    X_final = X_tf[:, sel_idx]

    return X_final, sel_names


def prepare_patho_adata(
    fcs_path: Path,
    nbm: "NBMReference",
    params: Dict[str, Any],
    mode: str = "cd45",
) -> Any:
    """
    Charge un FCS patho + gating selon le mode + transformation.
    Retourne un AnnData aligné sur les marqueurs NBM (same order).

    Args:
        fcs_path : chemin FCS pathologique
        nbm      : référence NBM construite (pour aligner les marqueurs)
        params   : paramètres de pipeline
        mode     : "cd45"   → gating complet incl. CD45+ (M1/M3)
                   "global" → seulement débris + singlets, pas CD45 (M2)

    Returns:
        AnnData avec condition='patho', var_names=nbm.markers (zéro-padded)
    """
    _mode = mode
    import anndata as ad

    if not FLOWSOM_AVAILABLE:
        raise ImportError("flowsom requis pour prepare_patho_adata()")

    fcs_path = Path(fcs_path)

    # ── Chargement FCS brut via fs.io.read_FCS (identique aux notebooks) ─────
    try:
        adata_raw = fs.io.read_FCS(str(fcs_path))
        X_raw = np.array(
            adata_raw.X.toarray() if hasattr(adata_raw.X, "toarray") else adata_raw.X,
            dtype=np.float64,
        )
        var_names = list(adata_raw.var_names)
    except Exception as e:
        raise RuntimeError(f"Impossible de lire {fcs_path.name} via fs.io.read_FCS: {e}")

    # NaN check — critique (FlowSOM plante silencieusement)
    if np.isnan(X_raw).any():
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Gating : débris + doublets + CD45+ (identique à la Section 5 du notebook) ─
    gate_mask = apply_pregating(
        X_raw, var_names, params, file_id=str(fcs_path.name), mode=_mode
    )
    X_gated = X_raw[gate_mask]
    n_gated = int(gate_mask.sum())
    _mode_label = "CD45+" if _mode == "cd45" else "global (sans CD45)"
    logging.info(f"[Patho] Gating {_mode_label} : {n_gated:,}/{len(gate_mask):,} cellules retenues")

    # ── Transformation cytométrique (même paramètres que build() NBM) ────────
    X_tf = apply_transform(X_gated, params.get("transform", "logicle"), params.get("cofactor", 5.0))

    # ── Sélection marqueurs (identique à build() dans NBMReference) ──────────
    sel_idx, sel_names = select_cytometry_cols(
        var_names,
        exclude_scatter=params.get("exclude_scatter", True),
        exclude_additional=params.get("exclude_markers", ["CD45"]),
        apply_marker_filtering=params.get("apply_marker_filtering", True),
        keep_area=params.get("keep_area", True),
        keep_height=params.get("keep_height", False),
    )
    X_sel = X_tf[:, sel_idx]

    # ── Alignement sur les marqueurs NBM (zero-padding pour marqueurs absents) ─
    # Garantit que patho_adata a EXACTEMENT les mêmes colonnes que _nbm_adata_for_fsom
    # → new_data(patho_adata) fonctionne sans réindexation (cols_to_use entiers)
    nbm_markers = nbm.markers if nbm.markers else sel_names
    X_aligned = np.zeros((n_gated, len(nbm_markers)), dtype=np.float64)
    for i, m in enumerate(nbm_markers):
        if m in sel_names:
            j = sel_names.index(m)
            X_aligned[:, i] = X_sel[:, j]

    if np.isnan(X_aligned).any():
        X_aligned = np.nan_to_num(X_aligned, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Créer AnnData avec condition='patho' ──────────────────────────────────
    adata_patho = ad.AnnData(X=X_aligned)
    adata_patho.var_names = list(nbm_markers)
    adata_patho.obs_names = [f"patho_{i}" for i in range(n_gated)]
    adata_patho.obs["condition"] = "patho"
    adata_patho.obs["file_origin"] = fcs_path.name

    return adata_patho


# =============================================================================
# SECTION 3 — MODÈLE FlowSOM DE RÉFÉRENCE NBM
# =============================================================================

class NBMReference:
    """
    Encapsule le modèle FlowSOM de référence NBM.
    Construit une seule fois, réutilisé pour tous les patients (batch).

    Attributs publics :
        som_weights  : (n_nodes, n_markers) — poids des nœuds SOM
        node_mfi     : (n_nodes, n_markers) — MFI de référence par nœud
        node_counts  : (n_nodes,)           — nombre de cellules NBM par nœud
        metaclusters : (n_nodes,)           — métacluster de chaque nœud
        markers      : liste des marqueurs utilisés
        n_nodes      : xdim × ydim
        xdim, ydim   : dimensions de la grille SOM
        node_distances: distances intra-NBM entre nœuds (pour Méthode 2)
    """

    def __init__(
        self,
        nbm_folder: Path,
        params: Dict[str, Any],
    ) -> None:
        self.nbm_folder = Path(nbm_folder)
        self.params = params
        self.xdim: int = params.get("xdim", 10)
        self.ydim: int = params.get("ydim", 10)
        self.n_nodes: int = self.xdim * self.ydim
        self.markers: List[str] = []
        self.som_weights: Optional[np.ndarray] = None
        self.node_mfi: Optional[np.ndarray] = None
        self.node_counts: Optional[np.ndarray] = None
        self.metaclusters: Optional[np.ndarray] = None
        self.node_distances: Optional[np.ndarray] = None
        # Pour Méthode 3 : pop_mfi_ref (CSV référence)
        self.pop_mfi_ref: Optional[pd.DataFrame] = None
        self.pop_cell_counts: Optional[Dict[str, int]] = None
        self._fsom_model: Any = None
        self._built: bool = False
        # Données brutes NBM + seuils MAD par cluster (calculés lors du build)
        self._X_nbm: Optional[np.ndarray] = None
        self._nbm_cell_assignments: Optional[np.ndarray] = None
        self._cluster_thresholds: Optional[np.ndarray] = None
        # AnnData NBM transformé (pour arbre mixte M1/M3 et new_data M2)
        self._nbm_adata_for_fsom: Any = None
        # Attributs de compatibilité notebook
        self._n_clusters_used: int = params.get("n_clusters", 7)
        self._total_cells: int = 0

    # ── Propriétés de compatibilité (notebook FlowSOM_Analysis_Pipeline) ──────
    @property
    def n_metaclusters(self) -> int:
        """Nombre de métaclusters uniques (équivalent de nbm.n_metaclusters dans le notebook)."""
        if self.metaclusters is None:
            return 0
        return int(len(np.unique(self.metaclusters)))

    @property
    def total_cells(self) -> int:
        """Nombre total de cellules NBM poolées utilisées pour le build."""
        if self.node_counts is not None:
            return int(self.node_counts.sum())
        return self._total_cells

    def build(
        self,
        ref_mfi_csv_folder: Optional[Path] = None,
        verbose: bool = False,
    ) -> "NBMReference":
        """
        Construit le modèle FlowSOM sur tous les FCS du dossier NBM.

        Nouveautés v2 :
            - rlen   : "auto" → calcule √N × 0.1 (N = cellules poolées)
            - auto_cluster : True → optimise k par stabilité ARI + silhouette
            - verbose: True → affiche des informations détaillées

        Optionnel : charge les CSV de référence MFI pour la Méthode 3.
        """
        import anndata as ad

        if not FLOWSOM_AVAILABLE:
            raise ImportError("flowsom requis pour construire le modèle NBM.")

        # ── Étape 1 : découverte des FCS (case-insensitive, .fcs et .FCS) ────
        nbm_file_paths = get_fcs_files(self.nbm_folder)
        if not nbm_file_paths:
            raise FileNotFoundError(f"Aucun FCS dans {self.nbm_folder}")

        if verbose:
            print(f"[NBM] {len(nbm_file_paths)} FCS trouvés :")
            for p in nbm_file_paths:
                print(f"   • {Path(p).name}")
        logging.info(f"[NBM] {len(nbm_file_paths)} FCS trouvés dans {self.nbm_folder}")

        # ── Étape 2 : chargement via fs.io.read_FCS — identique au pipeline ──
        # FlowSOM_Analysis_Pipeline.ipynb:load_fcs_files() → fs.io.read_FCS()
        healthy_adatas = load_fcs_files(nbm_file_paths, condition="NBM", verbose=verbose)

        if not healthy_adatas:
            raise RuntimeError(
                f"Impossible de charger les FCS NBM ({len(nbm_file_paths)} fichier(s) trouvé(s), "
                f"0 chargé):\n"
                f"  Dossier : {self.nbm_folder.resolve()}\n"
                f"  Vérifiez les logs (logging.exception) ci-dessus pour le traceback complet.\n"
                f"  Causes fréquentes : flowsom non installé, FCS corrompu, marqueurs absents."
            )

        # ── Étape 3 : pooling — identique à FlowSOM_Analysis_Pipeline.ipynb ──
        # ad.concat(join='inner') → intersection des marqueurs communs
        if len(healthy_adatas) > 1:
            nbm_pooled = ad.concat(healthy_adatas, join="inner")
        else:
            nbm_pooled = healthy_adatas[0].copy()

        if verbose:
            print(f"[NBM] Pool : {nbm_pooled.shape[0]:,} cellules × {nbm_pooled.shape[1]} marqueurs")

        # ── Étape 4 : transformation + sélection marqueurs (PAS de gating NBM) ─
        # Le pipeline principal ne gate pas les NBM lors du pooling → is_nbm_reference=True
        X_pool_raw = np.array(nbm_pooled.X.toarray() if hasattr(nbm_pooled.X, "toarray")
                              else nbm_pooled.X, dtype=np.float64)
        raw_markers = list(nbm_pooled.var_names)

        # NaN check
        if np.isnan(X_pool_raw).any():
            X_pool_raw = np.nan_to_num(X_pool_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Étape 4b : filtrage -A/-H (identique à la Section 8 du notebook) ──
        # Sélection marqueurs fluorescence avec déduplication -A/-H
        apply_mf = self.params.get("apply_marker_filtering", True)
        keep_A   = self.params.get("keep_area", True)
        keep_H   = self.params.get("keep_height", False)
        sel_idx, sel_names = select_cytometry_cols(
            raw_markers,
            exclude_scatter=self.params.get("exclude_scatter", True),
            exclude_additional=self.params.get("exclude_markers", ["CD45"]),
            apply_marker_filtering=apply_mf,
            keep_area=keep_A,
            keep_height=keep_H,
        )
        self.markers = sel_names
        if verbose:
            suffix_kept = ("-A" if keep_A and not keep_H
                           else "-H" if keep_H and not keep_A
                           else "-A et -H" if keep_A and keep_H else "aucun")
            print(f"[NBM] Filtrage marqueurs : {'activé (conserve ' + suffix_kept + ')' if apply_mf else 'désactivé'}")

        # Transformation cytométrique sur les colonnes sélectionnées uniquement
        X_sel_raw = X_pool_raw[:, sel_idx]
        X_nbm = apply_transform(
            X_sel_raw,
            self.params.get("transform", "logicle"),
            self.params.get("cofactor", 5.0),
        )

        if verbose:
            print(f"[NBM] Après sélection marqueurs : {X_nbm.shape[0]:,} × {len(sel_names)} | "
                  f"marqueurs : {sel_names}")

        self._total_cells = X_nbm.shape[0]
        logging.info(f"[NBM] Matrice poolée : {X_nbm.shape[0]:,} × {X_nbm.shape[1]} marqueurs")

        # Vérification NaN finale
        if np.isnan(X_nbm).any():
            X_nbm = np.nan_to_num(X_nbm, nan=0.0)

        # AnnData pour FlowSOM : sous-sélection sur le vrai AnnData issu de fs.io.read_FCS
        # (on préserve la structure interne attendue par flowsom, sans recréer un AnnData vide)
        import scipy.sparse as sp
        adata_for_fsom = nbm_pooled[:, sel_names].copy()
        X_tf_dense = X_nbm  # déjà transformé + dense
        adata_for_fsom.X = (sp.csr_matrix(X_tf_dense)
                            if isinstance(adata_for_fsom.X, sp.spmatrix)
                            else X_tf_dense)

        # ── Calcul de rlen (itérations SOM) ──────────────────────────────────
        rlen_param = self.params.get("rlen", "auto")
        if rlen_param == "auto":
            rlen_val = max(10, int(np.sqrt(X_nbm.shape[0]) * 0.1))
            logging.info(f"[NBM] rlen auto → {rlen_val} (√{X_nbm.shape[0]:,} × 0.1)")
        else:
            rlen_val = int(rlen_param)

        # ── Nombre de métaclusters (fixe ou auto-sélection) ──────────────────
        if self.params.get("auto_cluster", False):
            n_clusters_val = self._auto_select_n_clusters(X_nbm)
            logging.info(f"[NBM] Auto-clustering → k optimal = {n_clusters_val}")
        else:
            n_clusters_val = self.params.get("n_clusters", 7)
        self._n_clusters_used = n_clusters_val

        # Sauvegarder l'AnnData NBM transformé pour :
        #   - build_mixed_tree() : concat NBM+patho → arbre mixte (M1/M3)
        #   - new_data()         : projection patho sur arbre NBM (M2)
        # Contrainte : var_names = sel_names, X = transformé + dense, obs['condition']='NBM'
        self._nbm_adata_for_fsom = adata_for_fsom.copy()

        # Construction FlowSOM sur le pool NBM.
        # On passe cols_to_use comme INDICES ENTIERS et non des noms de colonnes strings.
        #   - cols_to_use=None → flowsom stocke dans un type non supporté par anndata → IndexError.
        #   - cols_to_use=strings → get_channels() les utilise comme regex ('^name$') et les
        #     caractères spéciaux (+, ., (, ), etc.) dans les noms de marqueurs font planter.
        #   - cols_to_use=list(range(n)) → accès direct par indice, aucun regex → SAFE.
        # adata_for_fsom ne contient QUE les marqueurs sélectionnés → on utilise tous.
        fsom_cols = list(range(adata_for_fsom.n_vars))
        self._fsom_model = fs.FlowSOM(
            adata_for_fsom,
            cols_to_use=fsom_cols,
            xdim=self.xdim,
            ydim=self.ydim,
            rlen=rlen_val,
            n_clusters=n_clusters_val,
            seed=self.params.get("seed", RANDOM_SEED),
        )

        # Extraction des poids SOM et métaclusters
        cluster_data = self._fsom_model.mudata["cluster_data"]
        cell_data_nbm = self._fsom_model.mudata["cell_data"]

        self.som_weights = np.array(cluster_data.X, dtype=np.float64)  # (n_nodes, n_markers)
        self.metaclusters = np.array(cluster_data.obs["metaclustering"].values, dtype=np.int32)
        node_assignments = np.array(cell_data_nbm.obs["clustering"].values, dtype=np.int32)

        # MFI et comptages par nœud
        n_nodes_actual = self.som_weights.shape[0]
        self.n_nodes = n_nodes_actual
        self.node_mfi = np.zeros((n_nodes_actual, len(self.markers)), dtype=np.float64)
        self.node_counts = np.zeros(n_nodes_actual, dtype=np.int64)
        node_sum = np.zeros_like(self.node_mfi)
        np.add.at(node_sum, node_assignments, X_nbm)
        np.add.at(self.node_counts, node_assignments, 1)
        valid = self.node_counts > 0
        self.node_mfi[valid] = node_sum[valid] / self.node_counts[valid, np.newaxis]

        # Distances intra-NBM (pour seuil Méthode 2)
        # distance de chaque nœud à ses voisins — sert à définir le bruit de fond
        self.node_distances = cdist(self.node_mfi, self.node_mfi, metric="euclidean")

        # Sauvegarder les cellules NBM brutes et leurs assignations par nœud
        # Nécessaire pour calculer les seuils MAD par cluster (Méthode 2 cell-level)
        self._X_nbm: np.ndarray = X_nbm               # (n_nbm_cells, n_markers)
        self._nbm_cell_assignments: np.ndarray = node_assignments  # (n_nbm_cells,)

        # Calcul des seuils MAD par cluster (ELN 2022 — Méthode 2)
        mad_allowed = self.params.get("mad_allowed", 4)
        self._cluster_thresholds: np.ndarray = np.full(n_nodes_actual, np.inf, dtype=np.float64)
        for _nid in range(n_nodes_actual):
            _mask = node_assignments == _nid
            if _mask.sum() < 2:
                continue
            _dists = np.linalg.norm(X_nbm[_mask] - self.som_weights[_nid], axis=1)
            _median = np.median(_dists)
            _mad = np.median(np.abs(_dists - _median))
            self._cluster_thresholds[_nid] = _median + mad_allowed * _mad

        logging.debug(f"[NBM] Seuils MAD calculés sur {n_nodes_actual} clusters (mad_allowed={mad_allowed})")

        # Chargement optionnel des CSV de référence MFI (pour Méthode 3)
        if ref_mfi_csv_folder is not None:
            self._load_ref_mfi_csvs(Path(ref_mfi_csv_folder))

        self._built = True
        logging.info(f"[NBM] Modèle construit : {n_nodes_actual} nœuds, {len(self.markers)} marqueurs")
        if verbose:
            print(f"[NBM] {n_nodes_actual} nœuds | {len(self.markers)} marqueurs | "
                  f"k={n_clusters_val} | rlen={rlen_val} | {X_nbm.shape[0]:,} cellules")
        return self

    def _auto_select_n_clusters(self, X_nbm: np.ndarray) -> int:
        """
        Pipeline 3 phases d'optimisation du nombre de métaclusters k.

        Méthode littérature 2024 (Weber, Van Gassen et al.) :
            Phase 1 : Silhouette sur codebook SOM  → screening rapide (1 seul SOM, re-MC par k)
            Phase 2 : Bootstrap ARI pairwise       → stabilité sur top candidats
            Phase 3 : Score composite pondéré      → sélection finale

        Stocke self._auto_cluster_report (dict) pour visualisation notebook.

        Paramètres lus depuis self.params :
            min_k, max_k, n_bootstrap, sample_boot,
            min_stability, w_stability, w_silhouette, seed
        """
        if not SKLEARN_AVAILABLE or not FLOWSOM_AVAILABLE:
            logging.warning("[Auto-cluster] sklearn/flowsom requis → k fixe")
            return int(self.params.get("n_clusters", 7))

        import anndata as ad
        import time as _time
        from sklearn.metrics import adjusted_rand_score, silhouette_score as sk_sil

        min_k         = int(self.params.get("min_k", 5))
        max_k         = int(self.params.get("max_k", 35))
        n_boot        = int(self.params.get("n_bootstrap", 10))
        sample_boot   = int(min(self.params.get("sample_boot", 20000), X_nbm.shape[0]))
        min_stability = float(self.params.get("min_stability", 0.75))
        w_stab        = float(self.params.get("w_stability", 0.65))
        w_sil         = float(self.params.get("w_silhouette", 0.35))
        seed          = int(self.params.get("seed", RANDOM_SEED))
        fallback_k    = int(self.params.get("n_clusters", 7))

        n_cells  = X_nbm.shape[0]
        rlen_val = max(10, min(100, int(np.sqrt(n_cells) * 0.1)))
        k_range  = list(range(min_k, min(max_k + 1, 50)))
        fsom_cols = list(range(X_nbm.shape[1]))

        logging.info(
            f"[Auto-cluster] 3 phases | k={min_k}..{max_k} | {n_boot} bootstraps | "
            f"{n_cells:,} cellules | rlen={rlen_val}"
        )

        # ── Phase 1 : Silhouette sur codebook (entraîner SOM une seule fois) ──
        # Économique : on re-métaclustère le codebook (~n_nodes points) pour chaque k,
        # au lieu de recalculer le SOM entier. Idéal pour un screening large de k.
        logging.info("[Auto-cluster] Phase 1 : silhouette codebook (screening rapide)")
        t0 = _time.time()

        adata_full = ad.AnnData(X=X_nbm.copy())
        adata_full.var_names = self.markers
        adata_full.obs["condition"] = "NBM"

        try:
            fsom_ref = fs.FlowSOM(
                adata_full, cols_to_use=fsom_cols,
                xdim=self.xdim, ydim=self.ydim, rlen=rlen_val,
                n_clusters=max(k_range), seed=seed,
            )
        except Exception as e:
            logging.warning(f"[Auto-cluster] Phase 1 SOM échoué : {e} → k fixe")
            self._auto_cluster_report = {"best_k": fallback_k, "error": str(e)}
            return fallback_k

        codebook = np.array(fsom_ref.mudata["cluster_data"].X, dtype=np.float64)
        if hasattr(codebook, "toarray"):
            codebook = codebook.toarray()  # type: ignore[union-attr]

        sil_results: List[Dict] = []
        for k in k_range:
            try:
                fsom_ref.metacluster(n_clusters=k)
                node_labels = np.array(
                    fsom_ref.mudata["cluster_data"].obs["metaclustering"].values, dtype=np.int32
                )
                n_unique = len(np.unique(node_labels))
                sil = sk_sil(codebook, node_labels) if 1 < n_unique < len(codebook) else -1.0
            except Exception:
                sil = -1.0
            sil_results.append({"k": k, "silhouette": sil})
            logging.debug(f"[Auto-cluster P1] k={k}: sil={sil:.4f}")

        sil_df = pd.DataFrame(sil_results)
        elapsed_p1 = _time.time() - t0
        logging.info(f"[Auto-cluster] Phase 1 terminée en {elapsed_p1:.1f}s")

        # ── Sélection top candidats pour Phase 2 ──────────────────────────────
        n_top = min(8, len(sil_df))
        top_candidates: List[int] = sil_df.nlargest(n_top, "silhouette")["k"].values.tolist()
        best_sil_k = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])
        for delta in (-2, -1, 1, 2):
            nb = best_sil_k + delta
            if min_k <= nb <= max_k and nb not in top_candidates:
                top_candidates.append(nb)
        top_candidates = sorted(set(top_candidates))
        logging.info(f"[Auto-cluster] Top candidats Phase 2 : {top_candidates}")

        # ── Phase 2 : Bootstrap ARI pairwise (stabilité) ──────────────────────
        # Sous-échantillon FIXE (mêmes cellules, seeds SOM différentes)
        # → mesure si le clustering est reproductible indépendamment des seeds.
        logging.info(f"[Auto-cluster] Phase 2 : stabilité bootstrap ({n_boot} runs × {len(top_candidates)} k)")
        t0 = _time.time()

        rng = np.random.default_rng(seed)
        eval_idx = rng.choice(n_cells, size=sample_boot, replace=False)
        X_eval = X_nbm[eval_idx].copy()
        adata_eval = ad.AnnData(X=X_eval)
        adata_eval.var_names = self.markers
        adata_eval.obs["condition"] = "NBM"

        stability_results: Dict[int, Dict[str, Any]] = {}
        for k in top_candidates:
            t_k = _time.time()
            labels_runs: List[np.ndarray] = []
            for b in range(n_boot):
                try:
                    fsom_b = fs.FlowSOM(
                        adata_eval, cols_to_use=fsom_cols,
                        xdim=self.xdim, ydim=self.ydim, rlen=rlen_val,
                        n_clusters=k, seed=seed + 100 + b,
                    )
                    lbl = np.array(
                        fsom_b.mudata["cell_data"].obs["metaclustering"].values, dtype=np.int32
                    )
                    labels_runs.append(lbl)
                except Exception as e:
                    logging.debug(f"[Auto-cluster P2] k={k} boot={b}: {e}")

            ari_pairs = [
                adjusted_rand_score(labels_runs[i], labels_runs[j])
                for i in range(len(labels_runs))
                for j in range(i + 1, len(labels_runs))
            ]
            mean_ari = float(np.mean(ari_pairs)) if ari_pairs else 0.0
            std_ari  = float(np.std(ari_pairs))  if ari_pairs else 0.0
            stability_results[k] = {
                "mean_ari": mean_ari, "std_ari": std_ari,
                "n_valid_runs": len(labels_runs), "n_pairs": len(ari_pairs),
            }
            elapsed_k = _time.time() - t_k
            status = "✓" if mean_ari >= min_stability else "✗"
            logging.info(
                f"[Auto-cluster P2] k={k}: ARI={mean_ari:.4f}±{std_ari:.4f} {status} ({elapsed_k:.1f}s)"
            )

        elapsed_p2 = _time.time() - t0
        logging.info(f"[Auto-cluster] Phase 2 terminée en {elapsed_p2:.1f}s")

        # ── Phase 3 : Score composite pondéré ─────────────────────────────────
        # Score = w_stability × ARI_norm + w_silhouette × Sil_norm
        # Les k avec ARI < min_stability reçoivent un malus de 30%.
        composite = sil_df.copy()
        composite["stability"] = composite["k"].map(
            lambda k: stability_results.get(k, {}).get("mean_ari", np.nan)      # type: ignore[arg-type]
        )
        composite["stability_std"] = composite["k"].map(
            lambda k: stability_results.get(k, {}).get("std_ari", np.nan)       # type: ignore[arg-type]
        )
        valid_comp = composite.dropna(subset=["stability"]).copy()

        if valid_comp.empty:
            logging.warning("[Auto-cluster] Phase 3 : aucun candidat valide → k fixe")
            self._auto_cluster_report = {
                "best_k": fallback_k, "sil_df": sil_df,
                "stability_results": stability_results, "composite_df": composite,
            }
            return fallback_k

        sil_min, sil_max = valid_comp["silhouette"].min(), valid_comp["silhouette"].max()
        sta_min, sta_max = valid_comp["stability"].min(), valid_comp["stability"].max()

        valid_comp["sil_norm"] = (
            (valid_comp["silhouette"] - sil_min) / max(sil_max - sil_min, 1e-9)
        )
        valid_comp["sta_norm"] = (
            (valid_comp["stability"] - sta_min) / max(sta_max - sta_min, 1e-9)
        )
        valid_comp["composite_score"] = w_stab * valid_comp["sta_norm"] + w_sil * valid_comp["sil_norm"]
        # Malus si stabilité insuffisante (littérature : robustesse > score absolu)
        valid_comp.loc[valid_comp["stability"] < min_stability, "composite_score"] *= 0.7

        best_idx = valid_comp["composite_score"].idxmax()
        best_k   = int(valid_comp.loc[best_idx, "k"])

        logging.info(
            f"[Auto-cluster] Phase 3 → k={best_k} | "
            f"sil={valid_comp.loc[best_idx,'silhouette']:.4f} | "
            f"ARI={valid_comp.loc[best_idx,'stability']:.4f} | "
            f"score={valid_comp.loc[best_idx,'composite_score']:.4f}"
        )

        # Stocker le rapport complet pour affichage notebook
        self._auto_cluster_report: Dict[str, Any] = {
            "best_k": best_k,
            "sil_df": sil_df,
            "stability_results": stability_results,
            "composite_df": valid_comp,
            "w_stability": w_stab,
            "w_silhouette": w_sil,
            "min_stability": min_stability,
            "rlen_used": rlen_val,
            "elapsed_phase1_s": round(elapsed_p1, 1),
            "elapsed_phase2_s": round(elapsed_p2, 1),
        }
        return best_k

    def build_from_precomputed(
        self,
        node_mfi: np.ndarray,
        node_counts: np.ndarray,
        metaclusters: np.ndarray,
        markers: List[str],
        som_weights: Optional[np.ndarray] = None,
    ) -> "NBMReference":
        """
        Initialise le modèle à partir de données précalculées.
        Utile si le FlowSOM NBM a déjà été exécuté dans le notebook principal.

        Hypothèses de nommage :
            node_mfi     : tableau node_mfi_raw_df.values (n_nodes × n_markers)
            node_counts  : np.bincount(clustering, minlength=n_nodes)
            metaclusters : cluster_data.obs["metaclustering"].values
            markers      : colonnes de node_mfi_raw_df
        """
        self.node_mfi = np.array(node_mfi, dtype=np.float64)
        self.node_counts = np.array(node_counts, dtype=np.int64)
        self.metaclusters = np.array(metaclusters, dtype=np.int32)
        self.markers = list(markers)
        self.n_nodes = self.node_mfi.shape[0]
        self.som_weights = np.array(som_weights, dtype=np.float64) if som_weights is not None else self.node_mfi.copy()
        self.node_distances = cdist(self.node_mfi, self.node_mfi, metric="euclidean")
        # Les seuils MAD ne sont pas disponibles dans ce mode — Méthode 2 utilisera le fallback
        self._cluster_thresholds = None
        self._X_nbm = None
        self._nbm_cell_assignments = None
        self._built = True
        return self

    def _load_ref_mfi_csvs(self, csv_folder: Path) -> None:
        """
        Charge les CSV de référence MFI (format : colonnes = marqueurs, index = population).
        S'aligne sur self.markers.
        """
        csv_files = list(csv_folder.glob("*.csv"))
        if not csv_files:
            logging.warning(f"[NBM] Aucun CSV dans {csv_folder}")
            return

        rows = {}
        for csv_p in csv_files:
            try:
                # Fichiers Kaluza standard : séparateur ';', décimale ','
                try:
                    df = pd.read_csv(csv_p, sep=";", decimal=",")
                except Exception:
                    df = pd.read_csv(csv_p, sep=None, engine="python", decimal=",")
                # Forcer la conversion numérique sur toutes les colonnes
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                pop_name = csv_p.stem
                if set(self.markers).issubset(set(df.columns)):
                    rows[pop_name] = df[self.markers].mean().values
                else:
                    # Essayer de mapper les colonnes -A
                    common = [m for m in self.markers if m in df.columns]
                    if common:
                        row = np.zeros(len(self.markers))
                        for j, m in enumerate(self.markers):
                            if m in common:
                                row[j] = df[m].mean()
                        rows[pop_name] = row
            except Exception as e:
                logging.warning(f"   Erreur CSV {csv_p.name}: {e}")

        if rows:
            self.pop_mfi_ref = pd.DataFrame(rows, index=self.markers).T
            self.pop_cell_counts = {pop: 1000 for pop in rows}  # placeholder
            logging.info(f"[NBM] {len(rows)} populations de référence chargées depuis CSV")

    def get_metacluster_proportions(self) -> Dict[int, float]:
        """Proportions de cellules NBM par métacluster (0-indexé)."""
        total = self.node_counts.sum()
        mc_counts: Dict[int, int] = {}
        for nid, mc in enumerate(self.metaclusters):
            mc_counts[int(mc)] = mc_counts.get(int(mc), 0) + int(self.node_counts[nid])
        return {mc: cnt / max(total, 1) for mc, cnt in mc_counts.items()}

    def build_mixed_tree(
        self,
        patho_adata: Any,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Construit un arbre FlowSOM MIXTE NBM+patho pour les Méthodes 1 et 3.

        Identique à FlowSOM_Analysis_Pipeline.ipynb :
            combined = ad.concat([nbm_pooled, patho_cd45], join='inner')
            fsom_mixte = fs.FlowSOM(combined, ...)

        Le mixte permet de classer chaque cellule (NBM ou patho) dans les
        mêmes métaclusters → delta proportions (M1) ou mapping pop (M3).

        Args:
            patho_adata : AnnData patho, condition='patho', créé par prepare_patho_adata().
            params      : paramètres (xdim, ydim, n_clusters, seed, rlen).

        Returns dict:
            fsom              : objet FlowSOM mixte
            cluster_data      : MuData cluster_data du mixed FlowSOM
            metaclusters_nbm  : (n_nbm,)   — métacluster de chaque cellule NBM
            metaclusters_patho: (n_patho,) — métacluster de chaque cellule patho
            node_assignments_nbm  : (n_nbm,)   — nœud assigné (cellules NBM)
            node_assignments_patho: (n_patho,) — nœud assigné (cellules patho)
            n_nbm, n_patho    : nombres de cellules de chaque condition
            n_nodes           : nombre total de nœuds SOM
            markers           : liste des marqueurs du modèle mixte
        """
        import anndata as ad

        if self._nbm_adata_for_fsom is None:
            raise RuntimeError(
                "NBMReference._nbm_adata_for_fsom non disponible. "
                "Appelez NBMReference.build() plutôt que build_from_precomputed()."
            )
        if not FLOWSOM_AVAILABLE:
            raise ImportError("flowsom requis pour build_mixed_tree()")

        # Pooling NBM + patho — identique à FlowSOM_Analysis_Pipeline.ipynb Section 3
        mixed = ad.concat(
            [self._nbm_adata_for_fsom, patho_adata],
            join="inner",
            index_unique="-",   # suffixe pour rendre les obs_names uniques
        )

        if verbose_flag := params.get("verbose", False):
            print(f"[Mixte] Pool : {mixed.n_obs:,} cellules ({self._nbm_adata_for_fsom.n_obs:,} NBM + {patho_adata.n_obs:,} patho) × {mixed.n_vars} marqueurs")

        # rlen auto
        rlen_p = params.get("rlen", "auto")
        if rlen_p == "auto":
            rlen_val = max(10, int(np.sqrt(mixed.n_obs) * 0.1))
        else:
            rlen_val = int(rlen_p)

        n_clusters_val = params.get("n_clusters", 7)
        fsom_cols = list(range(mixed.n_vars))

        logging.info(f"[Mixte] FlowSOM {self.xdim}×{self.ydim}, k={n_clusters_val}, rlen={rlen_val}")
        fsom_mixed = fs.FlowSOM(
            mixed,
            cols_to_use=fsom_cols,
            xdim=self.xdim,
            ydim=self.ydim,
            rlen=rlen_val,
            n_clusters=n_clusters_val,
            seed=params.get("seed", RANDOM_SEED),
        )

        cell_data_mixed = fsom_mixed.mudata["cell_data"]
        cluster_data_mixed = fsom_mixed.mudata["cluster_data"]

        metaclusters_all = np.array(cell_data_mixed.obs["metaclustering"].values, dtype=np.int32)
        node_assignments_all = np.array(cell_data_mixed.obs["clustering"].values, dtype=np.int32)
        conditions = mixed.obs["condition"].values

        is_nbm = np.array([str(c) == "NBM" for c in conditions])
        is_patho = ~is_nbm

        return {
            "fsom": fsom_mixed,
            "cluster_data": cluster_data_mixed,
            "metaclusters_nbm": metaclusters_all[is_nbm],
            "metaclusters_patho": metaclusters_all[is_patho],
            "node_assignments_nbm": node_assignments_all[is_nbm],
            "node_assignments_patho": node_assignments_all[is_patho],
            "n_nbm": int(is_nbm.sum()),
            "n_patho": int(is_patho.sum()),
            "n_nodes": int(cluster_data_mixed.shape[0]),
            "markers": list(mixed.var_names),
        }


# =============================================================================
# SECTION 4 — MAPPING PATIENT SUR MST NBM
# =============================================================================

def map_patient_to_nbm(
    X_patient: np.ndarray,
    nbm: NBMReference,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assigne chaque cellule patient au nœud NBM le plus proche (espace SOM des poids).
    
    Retourne :
        node_assignments : (n_cells,) — indice de nœud pour chaque cellule
        patient_node_mfi : (n_nodes, n_markers) — MFI patient par nœud NBM
    """
    if nbm.som_weights is None:
        raise RuntimeError("Le modèle NBM n'est pas initialisé.")
    weights = nbm.som_weights  # (n_nodes, n_markers)

    # Distances euclidiennes cellule → chaque nœud NBM
    # Opération vectorisée sur tout X_patient pour éviter les boucles
    diff = X_patient[:, np.newaxis, :] - weights[np.newaxis, :, :]  # (n_cells, n_nodes, n_markers)
    dists = np.sqrt((diff ** 2).sum(axis=2))  # (n_cells, n_nodes)
    node_assignments = np.argmin(dists, axis=1)  # (n_cells,)

    # Calcul des MFI patient par nœud
    n_nodes = weights.shape[0]
    n_markers = X_patient.shape[1]
    node_sum = np.zeros((n_nodes, n_markers), dtype=np.float64)
    node_cnt = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(node_sum, node_assignments, X_patient)
    np.add.at(node_cnt, node_assignments, 1)
    patient_node_mfi = np.zeros_like(node_sum)
    valid = node_cnt > 0
    patient_node_mfi[valid] = node_sum[valid] / node_cnt[valid, np.newaxis]

    return node_assignments, patient_node_mfi


# =============================================================================
# SECTION 4b — MÉTHODES MRD INTERNES (arbre mixte / new_data)
# =============================================================================
# Ces fonctions implémentent les 3 méthodes MRD selon la nouvelle architecture :
#   _mrd_method_1_mixed()    — M1 sur arbre mixte NBM+patho (Section 10.1 notebook principal)
#   _mrd_method_2_new_data() — M2 avec FlowSOM.new_data()   (notebook MRD_Test Section 10.2)
#   _mrd_method_3_mixed()    — M3 sur arbre mixte           (Section 10.4b notebook principal)
# =============================================================================


def _mrd_method_1_mixed(
    mixed_result: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    MRD Méthode 1 sur arbre FlowSOM MIXTE NBM+patho.

    Identique à FlowSOM_Analysis_Pipeline.ipynb Section 10.1 :
    → compare les proportions patho vs NBM par métacluster de l'arbre mixte.
    Un métacluster est « MRD-contributif » si prop_patho/prop_NBM > fold_change (ELN 2022).
    """
    fold_change_thr = params.get("mrd1_fold_change", 2.0)
    min_events = params.get("mrd1_min_events", 17)

    mc_nbm = mixed_result["metaclusters_nbm"]
    mc_patho = mixed_result["metaclusters_patho"]
    n_nbm = mixed_result["n_nbm"]
    n_patho = mixed_result["n_patho"]

    # Comptages NBM par métacluster
    nbm_mc_counts: Dict[int, int] = {}
    for mc in mc_nbm:
        k = int(mc)
        nbm_mc_counts[k] = nbm_mc_counts.get(k, 0) + 1
    nbm_mc_props = {mc: cnt / max(n_nbm, 1) for mc, cnt in nbm_mc_counts.items()}

    # Comptages patho par métacluster
    patho_mc_counts: Dict[int, int] = {}
    for mc in mc_patho:
        k = int(mc)
        patho_mc_counts[k] = patho_mc_counts.get(k, 0) + 1
    patho_mc_props = {mc: cnt / max(n_patho, 1) for mc, cnt in patho_mc_counts.items()}

    all_mcs = sorted(set(list(nbm_mc_counts.keys()) + list(patho_mc_counts.keys())))

    mc_deltas: Dict[int, float] = {}
    contributing_mcs: List[int] = []
    for mc in all_mcs:
        prop_p = patho_mc_props.get(mc, 0.0)
        prop_n = nbm_mc_props.get(mc, 1e-6)
        n_cells_mc = patho_mc_counts.get(mc, 0)
        fc = prop_p / max(prop_n, 1e-6)
        mc_deltas[mc] = fc
        if fc > fold_change_thr and n_cells_mc >= min_events:
            contributing_mcs.append(mc)

    # MRD M1 (ELN 2022) = somme des DELTAS (excès pathologique) dans les MC contributifs.
    # delta_mc = prop_patho - prop_nbm  pour chaque MC où prop_patho >= fold_change × prop_nbm.
    mc_mrd_details: Dict[int, Dict] = {}
    for mc in contributing_mcs:
        p_p = patho_mc_props.get(mc, 0.0)
        p_n = nbm_mc_props.get(mc, 0.0)
        delta = max(0.0, p_p - p_n)
        mc_mrd_details[mc] = {
            "prop_patho": round(p_p, 6),
            "prop_nbm": round(p_n, 6),
            "delta": round(delta, 6),
            "fold_change": round(mc_deltas[mc], 4),
            "n_cells_patho": patho_mc_counts.get(mc, 0),
        }
    # La MRD = somme des deltas (cellules excédentaires / total patho)
    mrd_value = sum(d["delta"] for d in mc_mrd_details.values())
    mrd_value = max(0.0, mrd_value)
    mrd_events = int(round(mrd_value * n_patho))
    delta_max_mc = max(mc_deltas, key=mc_deltas.get) if mc_deltas else None

    logging.info(
        f"[M1 mixte] MRD={mrd_value*100:.3f}%  contributing_mcs={contributing_mcs}  "
        f"fold_change_thr={fold_change_thr}"
    )
    return {
        "method": "method_1",
        "mrd_value": float(mrd_value),
        "mrd_pct": round(float(mrd_value) * 100, 4),
        "mrd_percent": round(float(mrd_value) * 100, 4),   # compat
        "mrd_events": mrd_events,
        "contributing_mcs": contributing_mcs,
        "mc_mrd_details": mc_mrd_details,
        "mc_patient_props": {k: round(v, 6) for k, v in patho_mc_props.items()},
        "mc_nbm_props": {k: round(v, 6) for k, v in nbm_mc_props.items()},
        "mc_fold_changes": {k: round(v, 4) for k, v in mc_deltas.items()},
        "delta_max_mc": int(delta_max_mc) if delta_max_mc is not None else None,
        "delta_max_value": round(mc_deltas.get(delta_max_mc, 0.0), 4) if delta_max_mc else 0.0,
        "fold_change_threshold": fold_change_thr,
        "n_cells_patient": n_patho,
        "n_contributing_cells": int(sum(patho_mc_counts.get(mc, 0) for mc in contributing_mcs)),
        "tree_mode": "mixed_nbm_patho",
        # is_mrd par métacluster (compatible avec affichage Kaluza)
        "is_mrd_per_mc": {mc: True for mc in contributing_mcs},
    }


def _mrd_method_2_new_data(
    nbm: "NBMReference",
    patho_adata: Any,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    MRD Méthode 2 avec FlowSOM.new_data() — double arbre séquentiel.

    Identique à FlowSOM_Analysis_Pipeline_MRD_Test.ipynb Section 10.2 :
    1. fsom_nbm.new_data(patho_adata, mad_allowed=MAD_ALLOWED)  → projection patho
    2. Distance cellule → centroïde assigné
    3. Seuil MAD = médiane_NBM(cluster) + mad_allowed × MAD_NBM(cluster) [pré-calculé]
    4. Cellules MRD = distance > seuil de leur cluster
    """
    if nbm._fsom_model is None:
        raise RuntimeError("[M2] nbm._fsom_model absent — appelez NBMReference.build()")

    mad_allowed = params.get("mad_allowed", 4)
    ratio_threshold = params.get("mrd2_ratio_threshold", 2.0)
    min_cells = params.get("mrd2_min_cells_per_cluster", 10)

    # ── Récupérer les poids SOM (centroïdes nœuds NBM) ───────────────────────
    nbm_cluster_data_ref = nbm._fsom_model.get_cluster_data()
    som_codes = np.array(
        nbm_cluster_data_ref.X.toarray()
        if hasattr(nbm_cluster_data_ref.X, "toarray")
        else nbm_cluster_data_ref.X,
        dtype=np.float64,
    )
    som_markers = list(nbm_cluster_data_ref.var_names)
    patho_var_names = list(patho_adata.var_names)

    # ── Aligner les marqueurs patho avec ceux du SOM ─────────────────────────
    patho_cols, som_cols = [], []
    for j, m in enumerate(som_markers):
        if m in patho_var_names:
            patho_cols.append(patho_var_names.index(m))
            som_cols.append(j)

    X_patho = np.array(
        patho_adata.X.toarray() if hasattr(patho_adata.X, "toarray") else patho_adata.X,
        dtype=np.float64,
    )
    X_patho_sub = X_patho[:, patho_cols]
    som_sub = som_codes[:, som_cols]     # (n_nodes, n_common_markers)
    n_patho_cells = X_patho_sub.shape[0]

    # ── PROJECTION : assignation manuelle au nœud SOM le plus proche ─────────
    # On évite new_data() car FlowSOM utilise les noms de marqueurs comme regex
    # (pattern '^nom$'), ce qui fait planter sur les marqueurs contenant des
    # caractères spéciaux comme '+', '.', '(', ')' (ex : CD7+56 FITC-A).
    # Ici on utilise des indices entiers → pas de regex, toujours safe.
    CHUNK = 10_000  # traitement par blocs pour éviter l'OOM sur grandes cohortes
    patho_cluster_assignments = np.empty(n_patho_cells, dtype=np.int32)
    for _start in range(0, n_patho_cells, CHUNK):
        _end = min(_start + CHUNK, n_patho_cells)
        # (chunk, 1, n_markers) - (1, n_nodes, n_markers) → (chunk, n_nodes)
        _d = np.linalg.norm(
            X_patho_sub[_start:_end, np.newaxis, :] - som_sub[np.newaxis, :, :],
            axis=2,
        )
        patho_cluster_assignments[_start:_end] = np.argmin(_d, axis=1).astype(np.int32)

    # ── DISTANCES vectorisées cellule → centroïde assigné ────────────────────
    centroids_per_cell = som_sub[patho_cluster_assignments]   # (n_cells, n_markers)
    patho_distances = np.linalg.norm(X_patho_sub - centroids_per_cell, axis=1)

    # ── SEUILS MAD par cluster (pré-calculés dans build() ou recalcul à la volée) ─
    n_nodes = som_codes.shape[0]
    if nbm._cluster_thresholds is not None and len(nbm._cluster_thresholds) == n_nodes:
        cluster_thresholds = nbm._cluster_thresholds
    else:
        # Fallback : recalcul depuis les cellules NBM brutes
        logging.warning("[M2] _cluster_thresholds absent → recalcul MAD à la volée")
        nbm_cell_data_ref = nbm._fsom_model.get_cell_data()
        nbm_clustering_ref = np.array(
            nbm_cell_data_ref.obs["clustering"].values, dtype=np.int32
        )
        X_nbm_raw = nbm._X_nbm
        if X_nbm_raw is None:
            raise RuntimeError("[M2] X_nbm absent pour recalcul MAD")
        nbm_cols = [nbm.markers.index(m) for m in som_markers if m in nbm.markers]
        X_nbm_sub = X_nbm_raw[:, nbm_cols]
        cluster_thresholds = np.full(n_nodes, np.inf, dtype=np.float64)
        for cid in range(n_nodes):
            mask = nbm_clustering_ref == cid
            if mask.sum() < 2:
                continue
            dists = np.linalg.norm(X_nbm_sub[mask] - som_sub[min(cid, som_sub.shape[0]-1)], axis=1)
            med = np.median(dists)
            mad = np.median(np.abs(dists - med))
            cluster_thresholds[cid] = med + mad_allowed * mad

    # ── DÉTECTION MRD — cellules outliers ────────────────────────────────────
    is_mrd_cell = patho_distances > cluster_thresholds[patho_cluster_assignments]
    n_mrd_cells = int(is_mrd_cell.sum())
    mrd_value = n_mrd_cells / max(n_patho_cells, 1)

    # ── CLUSTERS MRD (ratio patho/NBM, comme le notebook) ────────────────────
    nbm_cell_data_ref2 = nbm._fsom_model.get_cell_data()
    nbm_clustering_ref2 = np.array(nbm_cell_data_ref2.obs["clustering"].values, dtype=np.int32)
    nbm_node_counts = np.bincount(nbm_clustering_ref2, minlength=n_nodes)
    patho_node_counts = np.bincount(patho_cluster_assignments, minlength=n_nodes)
    n_nbm_total = int(nbm_node_counts.sum())

    mrd_clusters: List[int] = []
    mrd_cluster_details: Dict[int, Dict] = {}
    for nid in range(n_nodes):
        n_p = int(patho_node_counts[nid])
        n_n = int(nbm_node_counts[nid])
        if n_n == 0 and n_p > min_cells:
            mrd_clusters.append(nid)
            mrd_cluster_details[nid] = {"ratio": float("inf"), "n_patho": n_p, "n_nbm": 0}
        elif n_p > min_cells and n_nbm_total > 0 and n_patho_cells > 0:
            ratio = (n_p / n_patho_cells) / max(n_n / max(n_nbm_total, 1), 1e-9)
            if ratio > ratio_threshold:
                mrd_clusters.append(nid)
                mrd_cluster_details[nid] = {"ratio": round(ratio, 4), "n_patho": n_p, "n_nbm": n_n}

    logging.info(
        f"[M2 new_data] MRD={mrd_value*100:.3f}%  "
        f"mrd_cells={n_mrd_cells}/{n_patho_cells}  mrd_clusters={len(mrd_clusters)}"
    )
    return {
        "method": "method_2",
        "mrd_value": float(mrd_value),
        "mrd_pct": round(float(mrd_value) * 100, 4),
        "mrd_percent": round(float(mrd_value) * 100, 4),   # compat
        "mrd_events": n_mrd_cells,
        "mrd_clusters": mrd_clusters,
        "mrd_cluster_details": mrd_cluster_details,
        "n_mrd_cells": n_mrd_cells,
        "is_mrd_cell_count": n_mrd_cells,   # alias explicite
        "n_cells_patient": n_patho_cells,
        "mad_allowed": mad_allowed,
        "ratio_threshold": ratio_threshold,
        "threshold_mode": "new_data_mad",
        "preprocessing_mode": "global",   # toutes les cellules, sans gating CD45
        "cell_distances_stats": {
            "mean": round(float(np.mean(patho_distances)), 6),
            "median": round(float(np.median(patho_distances)), 6),
            "max": round(float(np.max(patho_distances)), 6),
        },
    }


def _mrd_method_3_mixed(
    mixed_result: Dict[str, Any],
    nbm: "NBMReference",
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    MRD Méthode 3 sur arbre mixte — mapping populations (Section 10.4b).

    Identique à mrd_method_3() mais utilise les nœuds + assignations
    de l'arbre mixte (NBM+patho).
    Nœuds Unknown pour les cellules patho = MRD potentielle.
    """
    n_nodes = mixed_result["n_nodes"]
    node_assignments_patho = mixed_result["node_assignments_patho"]
    n_patho = mixed_result["n_patho"]
    mixed_markers = mixed_result["markers"]

    cluster_data = mixed_result["cluster_data"]
    node_mfi_mixed = np.array(
        cluster_data.X.toarray() if hasattr(cluster_data.X, "toarray") else cluster_data.X,
        dtype=np.float64,
    )
    patho_node_counts = np.bincount(node_assignments_patho, minlength=n_nodes)
    occupied_nodes = np.where(patho_node_counts > 0)[0]

    if len(occupied_nodes) == 0:
        return {"method": "method_3", "mrd_value": 0.0, "mrd_percent": 0.0,
                "unknown_nodes": [], "error": "Aucun nœud patho occupé dans l'arbre mixte"}

    # ── Référence populations ─────────────────────────────────────────────────
    pop_names: List[str] = []
    X_ref: Optional[np.ndarray] = None
    cell_counts: Dict[str, int] = {}

    if nbm.pop_mfi_ref is not None:
        common_ref = [m for m in mixed_markers if m in nbm.pop_mfi_ref.columns]
        if common_ref:
            pop_names = list(nbm.pop_mfi_ref.index)
            X_ref = nbm.pop_mfi_ref[common_ref].values.astype(np.float64)
            cell_counts = nbm.pop_cell_counts or {p: 1000 for p in pop_names}
            ref_idx = [list(mixed_markers).index(m) for m in common_ref]
            X_nodes = node_mfi_mixed[occupied_nodes][:, ref_idx]

    if X_ref is None or len(pop_names) == 0:
        # Fallback : centroïdes des métaclusters NBM comme référence
        mc_nbm = mixed_result["metaclusters_nbm"]
        node_ass_nbm = mixed_result["node_assignments_nbm"]
        mc_ids = sorted(set(mc_nbm.tolist()))
        pop_names = [f"MC{mc}" for mc in mc_ids]
        mc_sums: Dict[int, np.ndarray] = {}
        mc_counts_: Dict[int, int] = {}
        for mc, nid in zip(mc_nbm, node_ass_nbm):
            mc_int = int(mc)
            if mc_int not in mc_sums:
                mc_sums[mc_int] = np.zeros(node_mfi_mixed.shape[1])
                mc_counts_[mc_int] = 0
            mc_sums[mc_int] += node_mfi_mixed[int(nid)]
            mc_counts_[mc_int] += 1
        X_ref = np.array([mc_sums.get(mc, np.zeros(node_mfi_mixed.shape[1])) /
                          max(mc_counts_.get(mc, 1), 1) for mc in mc_ids])
        cell_counts = {f"MC{mc}": mc_counts_.get(mc, 1) for mc in mc_ids}
        X_nodes = node_mfi_mixed[occupied_nodes]

    node_sizes = patho_node_counts[occupied_nodes].astype(np.float64)
    n_tot = len(occupied_nodes)
    hard_limit = params.get("mrd3_hard_limit_factor", 5.0)
    threshold_mode = params.get("mrd3_unknown_mode", "auto_otsu")
    benchmark_methods = params.get(
        "m3_benchmark_methods", ["M3_cosine", "M8_ref_norm", "M9_prior", "M12_cosine_prior"]
    )

    # ── Benchmark multi-méthodes (identique à mrd_method_3) ──────────────────
    benchmark_scores: Dict[str, float] = {}
    benchmark_results_: Dict[str, Any] = {}
    for method_name in benchmark_methods:
        try:
            asgn, bdist, th, tdesc = _run_single_m3_method(
                method_name=method_name, X_nodes=X_nodes, X_ref=X_ref,
                pop_names=pop_names, cell_counts=cell_counts, node_sizes=node_sizes,
                hard_limit_factor=hard_limit, n_nodes_total=n_nodes,
                threshold_mode=threshold_mode,
            )
            score = _score_m3_assignment(asgn, n_tot)
            benchmark_scores[method_name] = round(score, 4)
            benchmark_results_[method_name] = (asgn, bdist, th, tdesc)
        except Exception as e:
            benchmark_scores[method_name] = 0.0
            logging.warning(f"[M3 mixte] {method_name}: {e}")

    if not benchmark_results_:
        return {"method": "method_3", "mrd_value": 0.0, "mrd_percent": 0.0,
                "unknown_nodes": [], "error": "Toutes les méthodes M3 ont échoué"}

    best_m = max(benchmark_scores, key=benchmark_scores.get)  # type: ignore[arg-type]
    assigned_pop, best_dist_arr, threshold, threshold_desc = benchmark_results_[best_m]

    node_mapping: Dict[int, str] = {}
    unknown_nodes: List[int] = []
    for k_i, nid in enumerate(occupied_nodes):
        pop = str(assigned_pop[k_i])
        node_mapping[int(nid)] = pop
        if pop == "Unknown":
            unknown_nodes.append(int(nid))

    n_unknown_cells = int(sum(patho_node_counts[nid] for nid in unknown_nodes))
    mrd_value = n_unknown_cells / max(n_patho, 1)
    pop_distribution: Dict[str, int] = {}
    for nid, pop in node_mapping.items():
        pop_distribution[pop] = pop_distribution.get(pop, 0) + int(patho_node_counts[nid])

    logging.info(
        f"[M3 mixte] MRD={mrd_value*100:.3f}%  best={best_m}  "
        f"unknown_nodes={len(unknown_nodes)}/{len(occupied_nodes)}"
    )
    # is_mrd par nœud (True = Unknown = candidat MRD)
    is_mrd_per_node: Dict[int, bool] = {
        int(nid): (pop == "Unknown") for nid, pop in node_mapping.items()
    }
    return {
        "method": "method_3",
        "mrd_value": float(mrd_value),
        "mrd_pct": round(float(mrd_value) * 100, 4),
        "mrd_percent": round(float(mrd_value) * 100, 4),   # compat
        "mrd_events": n_unknown_cells,
        "best_m3_method": best_m,
        "m3_benchmark": benchmark_scores,
        "unknown_nodes": unknown_nodes,
        "n_unknown_cells": n_unknown_cells,
        "n_unknown_nodes": len(unknown_nodes),
        "n_cells_patient": int(n_patho),
        "threshold": round(float(threshold), 6),
        "threshold_desc": threshold_desc,
        "node_mapping": node_mapping,
        "is_mrd_per_node": is_mrd_per_node,
        "pop_distribution": pop_distribution,
        "n_occupied_nodes": int(len(occupied_nodes)),
        "n_populations_ref": int(len(pop_names)),
        "tree_mode": "mixed_nbm_patho",
    }


# =============================================================================
# SECTION 5 — MÉTHODE 1 : DELTA MÉTACLUSTERS
# =============================================================================

def mrd_method_1(
    patient_node_assignments: np.ndarray,
    nbm: NBMReference,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    MRD Méthode 1 — Delta entre métaclusters patho vs NBM.

    Logique (ELN 2022 adaptée) :
    ─────────────────────────────
    1. Pour chaque métacluster, calculer la proportion de cellules dans le
       fichier patient vs la proportion correspondante dans le NBM de référence.
    2. Un métacluster est considéré « MRD-contributif » si :
           proportion_patient / proportion_NBM > fold_change_threshold  (défaut 2×)
    3. La MRD globale = somme des proportions patient des métaclusters contributifs.

    Args :
        patient_node_assignments : sortie de map_patient_to_nbm()
        nbm                      : modèle NBM de référence
        params                   : dictionnaire de paramètres

    Returns dict :
        mrd_value         : float [0,1] — proportion MRD globale
        contributing_mcs  : liste des métaclusters contributifs
        mc_patient_props  : dict mc → proportion patient
        mc_nbm_props      : dict mc → proportion NBM
        mc_deltas         : dict mc → (prop_patient / prop_nbm)
        method            : "method_1"
    """
    fold_change_thr = params.get("mrd1_fold_change", 2.0)
    min_events = params.get("mrd1_min_events", 17)

    # Proportions par nœud → agrégation par métacluster
    n_cells_patient = len(patient_node_assignments)
    nbm_props = nbm.get_metacluster_proportions()

    # Comptage patient par nœud puis par métacluster
    patient_node_counts = np.bincount(patient_node_assignments, minlength=nbm.n_nodes)
    patient_mc_counts: Dict[int, int] = {}
    for nid, mc in enumerate(nbm.metaclusters):
        mc = int(mc)
        patient_mc_counts[mc] = patient_mc_counts.get(mc, 0) + int(patient_node_counts[nid])

    patient_mc_props = {mc: cnt / max(n_cells_patient, 1) for mc, cnt in patient_mc_counts.items()}
    all_mcs = sorted(set(list(nbm_props.keys()) + list(patient_mc_props.keys())))

    mc_deltas: Dict[int, float] = {}
    contributing_mcs: List[int] = []

    for mc in all_mcs:
        prop_p = patient_mc_props.get(mc, 0.0)
        prop_n = nbm_props.get(mc, 1e-6)       # éviter division par zéro
        n_cells_mc = patient_mc_counts.get(mc, 0)
        fc = prop_p / max(prop_n, 1e-6)
        mc_deltas[mc] = fc
        if fc > fold_change_thr and n_cells_mc >= min_events:
            contributing_mcs.append(mc)

    # MRD M1 (ELN 2022) = somme des DELTAS (excès pathologique) dans les MC contributifs.
    mc_mrd_details_leg: Dict[int, Dict] = {}
    for mc in contributing_mcs:
        p_p = patient_mc_props.get(mc, 0.0)
        p_n = nbm_props.get(mc, 0.0)
        delta = max(0.0, p_p - p_n)
        mc_mrd_details_leg[mc] = {
            "prop_patho": round(p_p, 6),
            "prop_nbm": round(p_n, 6),
            "delta": round(delta, 6),
            "fold_change": round(mc_deltas[mc], 4),
            "n_cells_patho": patient_mc_counts.get(mc, 0),
        }
    mrd_value = sum(d["delta"] for d in mc_mrd_details_leg.values())
    mrd_value = max(0.0, mrd_value)
    mrd_events = int(round(mrd_value * n_cells_patient))
    delta_max_mc = max(mc_deltas, key=mc_deltas.get) if mc_deltas else None

    return {
        "method": "method_1",
        "mrd_value": float(mrd_value),
        "mrd_pct": round(float(mrd_value) * 100, 4),
        "mrd_percent": round(float(mrd_value) * 100, 4),
        "mrd_events": mrd_events,
        "contributing_mcs": contributing_mcs,
        "mc_mrd_details": mc_mrd_details_leg,
        "mc_patient_props": {k: round(v, 6) for k, v in patient_mc_props.items()},
        "mc_nbm_props": {k: round(v, 6) for k, v in nbm_props.items()},
        "mc_fold_changes": {k: round(v, 4) for k, v in mc_deltas.items()},
        "delta_max_mc": int(delta_max_mc) if delta_max_mc is not None else None,
        "delta_max_value": round(mc_deltas.get(delta_max_mc, 0.0), 4) if delta_max_mc else 0.0,
        "fold_change_threshold": fold_change_thr,
        "n_cells_patient": n_cells_patient,
        "n_contributing_cells": int(sum(patient_mc_counts.get(mc, 0) for mc in contributing_mcs)),
        "is_mrd_per_mc": {mc: True for mc in contributing_mcs},
    }


# =============================================================================
# SECTION 6 — MÉTHODE 2 : DISTANCES EUCLIDIENNES SUR MST
# =============================================================================

def mrd_method_2(
    patient_node_mfi: np.ndarray,
    nbm: "NBMReference",
    patient_node_assignments: np.ndarray,
    params: Dict[str, Any],
    X_patient: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    MRD Méthode 2 — Distance euclidienne cell-level avec seuils MAD par cluster.

    Implémentation exacte du notebook FlowSOM_Analysis_Pipeline_MRD_Test.ipynb.

    Logique (ELN 2022 approuvée) :
    ──────────────────────────────
    1. Pour chaque cellule patient, calcul de la distance euclidienne à son centroïde
       de cluster assigné (nœud SOM le plus proche dans l'espace NBM).

    2. Seuil par cluster = médiane_NBM(cluster) + mad_allowed × MAD_NBM(cluster),
       calculé sur les distances des cellules NBM de référence à leur centroïde.

    3. Une cellule est « MRD » si sa distance dépasse le seuil de son cluster.

    4. Un cluster est « MRD » si :
           (n_patho / total_patho) / (n_nbm / total_nbm) > ratio_threshold  ET  n_patho > min_cells
       OU si n_nbm == 0  et  n_patho > min_cells.

    5. MRD% = n_cellules_MRD / n_cellules_patient × 100.

    Args :
        patient_node_mfi         : (n_nodes, n_markers) MFI patient par nœud (conservé pour compat.)
        nbm                      : modèle NBM avec _cluster_thresholds, _X_nbm, _nbm_cell_assignments
        patient_node_assignments : (n_cells,) assignation de chaque cellule patient au nœud NBM
        params                   : dictionnaire de paramètres
        X_patient                : (n_cells, n_markers) cellules patient brutes (recommandé)

    Returns dict :
        mrd_value   : float [0,1]
        mrd_percent : float [0, 100]
        mrd_clusters: liste des nœuds identifiés comme MRD
        n_mrd_cells : nombre de cellules MRD
        threshold_mode: "cell_level_mad" | "node_level_fallback"
    """
    mad_allowed = params.get("mad_allowed", 4)
    ratio_threshold = params.get("mrd2_ratio_threshold", 2.0)
    min_cells = params.get("mrd2_min_cells_per_cluster", 10)

    patient_node_counts = np.bincount(patient_node_assignments, minlength=nbm.n_nodes)
    n_patho_total = int(patient_node_counts.sum())
    n_nbm_total = int(nbm.node_counts.sum()) if nbm.node_counts is not None else 0

    # ─────────────────────────────────────────────────────────────────────────
    # Mode 1 — Cell-level MAD (approche exacte du notebook MRD Test)
    # Requiert : X_patient + _cluster_thresholds + _X_nbm + _nbm_cell_assignments
    # ─────────────────────────────────────────────────────────────────────────
    if (
        X_patient is not None
        and nbm._cluster_thresholds is not None
        and nbm._X_nbm is not None
        and nbm._nbm_cell_assignments is not None
        and nbm.som_weights is not None
    ):
        n_cells = X_patient.shape[0]
        n_markers = X_patient.shape[1]
        n_markers_som = nbm.som_weights.shape[1]

        # Aligner les marqueurs si nécessaire (X_patient peut avoir moins de colonnes)
        if n_markers > n_markers_som:
            X_pat = X_patient[:, :n_markers_som]
        elif n_markers < n_markers_som:
            X_pat = np.zeros((n_cells, n_markers_som), dtype=np.float64)
            X_pat[:, :n_markers] = X_patient
        else:
            X_pat = X_patient

        # Distances cellule → centroïde assigné
        cluster_ids = patient_node_assignments.astype(int)
        centroids_assigned = nbm.som_weights[cluster_ids]         # (n_cells, n_markers)
        cell_distances = np.linalg.norm(X_pat - centroids_assigned, axis=1)  # (n_cells,)

        # Seuil par cluster (pré-calculé dans NBMReference.build())
        cluster_thresholds_per_cell = nbm._cluster_thresholds[cluster_ids]

        # Cellules MRD : distance > seuil de leur cluster
        is_mrd_cell = cell_distances > cluster_thresholds_per_cell
        n_mrd_cells = int(is_mrd_cell.sum())
        mrd_value = n_mrd_cells / max(n_patho_total, 1)

        # Clusters MRD par ratio patho/NBM (critère ELN complémentaire)
        nbm_assignments = nbm._nbm_cell_assignments.astype(int)
        mrd_clusters: List[int] = []
        mrd_cluster_details: Dict[int, Dict] = {}

        for nid in range(nbm.n_nodes):
            n_p = int(patient_node_counts[nid])
            n_n = int((nbm_assignments == nid).sum())
            if n_n == 0 and n_p > min_cells:
                mrd_clusters.append(nid)
                mrd_cluster_details[nid] = {"ratio": float("inf"), "n_patho": n_p, "n_nbm": 0}
            elif n_p > min_cells and n_nbm_total > 0 and n_patho_total > 0:
                ratio = (n_p / n_patho_total) / max(n_n / max(n_nbm_total, 1), 1e-9)
                if ratio > ratio_threshold:
                    mrd_clusters.append(nid)
                    mrd_cluster_details[nid] = {"ratio": round(ratio, 4), "n_patho": n_p, "n_nbm": n_n}

        logging.info(
            f"[M2 cell-MAD] MRD={mrd_value*100:.3f}%  "
            f"mrd_cells={n_mrd_cells}/{n_patho_total}  "
            f"mrd_clusters={len(mrd_clusters)}  mad_allowed={mad_allowed}"
        )

        return {
            "method": "method_2",
            "mrd_value": float(mrd_value),
            "mrd_percent": round(float(mrd_value) * 100, 4),
            "mrd_clusters": mrd_clusters,
            "mrd_cluster_details": mrd_cluster_details,
            "n_mrd_cells": n_mrd_cells,
            "n_cells_patient": n_patho_total,
            "mad_allowed": mad_allowed,
            "ratio_threshold": ratio_threshold,
            "threshold_mode": "cell_level_mad",
            "cell_distances_stats": {
                "mean": round(float(np.mean(cell_distances)), 6),
                "median": round(float(np.median(cell_distances)), 6),
                "max": round(float(np.max(cell_distances)), 6),
            },
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Mode 2 — Fallback nœud-niveau (quand X_patient ou _cluster_thresholds absent)
    # Conservé pour compatibilité avec build_from_precomputed()
    # ─────────────────────────────────────────────────────────────────────────
    logging.warning(
        "[M2 node-fallback] X_patient ou seuils MAD absents — "
        "utilise la comparaison nœud-niveau (moins précis que cell-level MAD)"
    )
    k = params.get("mrd2_ratio_threshold", 2.0)  # réutilise ratio comme k-sigma

    node_dists = np.zeros(nbm.n_nodes, dtype=np.float64)
    for nid in range(nbm.n_nodes):
        if patient_node_counts[nid] == 0:
            continue
        p_mfi = patient_node_mfi[nid]
        n_mfi = nbm.node_mfi[nid]
        node_dists[nid] = float(np.linalg.norm(p_mfi - n_mfi))

    nbm_ref_dists = nbm.node_distances.copy()
    np.fill_diagonal(nbm_ref_dists, np.nan)
    nbm_nn_dists = np.nanmin(nbm_ref_dists, axis=1)
    nbm_mu = float(np.nanmean(nbm_nn_dists))
    nbm_sigma = float(np.nanstd(nbm_nn_dists))
    threshold = nbm_mu + k * nbm_sigma

    occupied_nodes = np.where(patient_node_counts > 0)[0]
    deviant_nodes = [int(nid) for nid in occupied_nodes if node_dists[nid] > threshold]

    n_deviant_cells = int(sum(patient_node_counts[nid] for nid in deviant_nodes))
    mrd_value = n_deviant_cells / max(n_patho_total, 1)

    return {
        "method": "method_2",
        "mrd_value": float(mrd_value),
        "mrd_percent": round(float(mrd_value) * 100, 4),
        "mrd_clusters": deviant_nodes,
        "n_mrd_cells": n_deviant_cells,
        "n_cells_patient": n_patho_total,
        "threshold": round(threshold, 6),
        "nbm_dist_mean": round(nbm_mu, 6),
        "nbm_dist_std": round(nbm_sigma, 6),
        "threshold_mode": "node_level_fallback",
        "node_distances": {int(nid): round(float(node_dists[nid]), 6) for nid in occupied_nodes},
    }


def _bimodality_coefficient(x: np.ndarray) -> float:
    """
    Coefficient de bimodalité de Bimodality Coefficient (BC) de DeCarlo (1997).
    BC > 0.555 suggère une distribution bimodale (signal séparé du bruit).
    BC = (skew² + 1) / (excess_kurtosis + 3)
    """
    if len(x) < 5:
        return 0.0
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return 0.0
    skew = float(stats.skew(x))
    kurt = float(stats.kurtosis(x))  # excès de kurtosis (Fisher)
    n = len(x)
    # Correction pour petits échantillons
    skew_corr = skew * np.sqrt(n * (n - 1)) / (n - 2) if n > 2 else skew
    kurt_corr = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))  # kurtosis attendue pour normale
    return float((skew_corr ** 2 + 1) / max(abs(kurt) + kurt_corr, 1e-10))


# =============================================================================
# SECTION 7 — MÉTHODE 3 : MAPPING POPULATIONS (Section 10, benchmark M3/M8/M9/M12)
# =============================================================================
# Implémentation exacte de map_populations_to_nodes_v5() — Section 10.4b du
# notebook principal FlowSOM_Analysis_Pipeline.ipynb.
# Benchmark automatique sur 4 méthodes, auto-sélection de la meilleure, log.
# =============================================================================

def _otsu_threshold_1d(values: np.ndarray, n_bins: int = 64) -> float:
    """Seuil d'Otsu 1D (coupure naturelle bimodale).
    Identique à _otsu_threshold_1d du notebook principal (Section 10.4b).
    """
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return float(values.max()) if len(values) > 0 else 0.0
    hist, edges = np.histogram(v, bins=min(n_bins, max(10, len(v) // 2)))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = float(hist.sum())
    if total == 0:
        return float(np.median(v))
    mu_total = float((hist * centers).sum()) / total
    best_bcv, best_th = -np.inf, centers[-1]
    w0_cum, mu0_cum = 0.0, 0.0
    for i in range(len(hist) - 1):
        w0_cum += hist[i] / total
        mu0_cum += hist[i] * centers[i] / total
        if w0_cum <= 1e-8 or w0_cum >= 1.0 - 1e-8:
            continue
        w1 = 1.0 - w0_cum
        mu0 = mu0_cum / w0_cum
        mu1 = (mu_total - mu0_cum) / w1
        bcv = w0_cum * w1 * (mu0 - mu1) ** 2
        if bcv > best_bcv:
            best_bcv, best_th = bcv, centers[i]
    return float(best_th)


def _apply_bayesian_prior(
    D: np.ndarray,
    pop_names: List[str],
    cell_counts: Dict[str, int],
    prior_mode: str,
    node_sizes: Optional[np.ndarray],
    hard_limit_factor: float,
    n_nodes_total: int,
) -> np.ndarray:
    """
    Pondération bayésienne sur la matrice de distance D_adj = D / weight.
    Identique à _apply_bayesian_prior du notebook principal (Section 10.4b).

    Modes :
        log10_cubed : D / log10(n)^3  — RECOMMANDÉ (ratio Granulos/Plasmos ~7.1x)
        log10       : D / log10(n)    — M9 legacy (~1.9x insuffisant)
        sqrt_n      : D / sqrt(n/10)  — très agressif (~33x)
    """
    D_adj = D.copy()
    for j, pop in enumerate(pop_names):
        n = max(cell_counts.get(pop, 1), 1)
        if prior_mode == "log10_cubed":
            w = np.log10(max(n, 10)) ** 3
        elif prior_mode == "sqrt_n":
            w = np.sqrt(max(n, 10) / 10.0)
        else:
            w = np.log10(max(n, 10))
        D_adj[:, j] /= w
        if hard_limit_factor > 0 and node_sizes is not None and n_nodes_total > 0:
            max_expected = max(1, int(n / n_nodes_total * hard_limit_factor))
            too_large = node_sizes > max_expected
            if too_large.any():
                D_adj[too_large, j] = np.inf
    return D_adj


def _run_single_m3_method(
    method_name: str,
    X_nodes: np.ndarray,
    X_ref: np.ndarray,
    pop_names: List[str],
    cell_counts: Dict[str, int],
    node_sizes: np.ndarray,
    hard_limit_factor: float,
    n_nodes_total: int,
    threshold_mode: str = "auto_otsu",
    percentile: float = 70.0,
) -> Tuple[np.ndarray, np.ndarray, float, str]:
    """
    Calcule la matrice de distance et l'attribution pour une méthode M3.

    Méthodes supportées (identiques au notebook Section 10.4b) :
        M3_cosine       — distance Cosinus pure (pas de prior)
        M8_ref_norm     — euclidien normalisé sur la plage référence
        M9_prior        — Cosinus + prior log10 simple (M9/legacy)
        M12_cosine_prior— Cosinus + prior log10^3 + hard limit (★ recommandé)

    Returns :
        assigned_pop : (n_nodes,) array de str — population assignée ou "Unknown"
        best_dist    : (n_nodes,) distances minimales ajustées
        threshold    : seuil (Otsu ou percentile)
        threshold_desc : description du seuil
    """
    if method_name in ("M3_cosine", "M9_prior", "M12_cosine_prior"):
        # Cosinus : clip des négatifs obligatoire (espace Logicle → valeurs légèrement négatives)
        X_n_c = np.clip(X_nodes, 0.0, None)
        X_r_c = np.clip(X_ref, 0.0, None)
        D = cdist(X_n_c, X_r_c, metric="cosine")
    elif method_name == "M8_ref_norm":
        # Euclidien normalisé sur la plage de la référence (min/max)
        ref_min = X_ref.min(axis=0)
        ref_rng = np.where(X_ref.max(axis=0) - ref_min > 1e-8,
                           X_ref.max(axis=0) - ref_min, 1.0)
        D = cdist((X_nodes - ref_min) / ref_rng, (X_ref - ref_min) / ref_rng, metric="euclidean")
    else:
        D = cdist(X_nodes, X_ref, metric="euclidean")

    # Application du prior bayésien selon la méthode
    if method_name == "M9_prior":
        D_adj = _apply_bayesian_prior(D, pop_names, cell_counts, "log10",
                                      None, 0.0, n_nodes_total)
    elif method_name == "M12_cosine_prior":
        D_adj = _apply_bayesian_prior(D, pop_names, cell_counts, "log10_cubed",
                                      node_sizes, hard_limit_factor, n_nodes_total)
    else:
        D_adj = D.copy()

    # Attribution au plus proche voisin
    best_idx = np.argmin(D_adj, axis=1)
    best_dist = D_adj[np.arange(len(D_adj)), best_idx]

    # Seuil Otsu ou percentile
    v_finite = best_dist[np.isfinite(best_dist)]
    if threshold_mode == "auto_otsu" and len(v_finite) > 3:
        threshold = _otsu_threshold_1d(v_finite)
        threshold_desc = f"Otsu({threshold:.4f})"
    else:
        threshold = float(np.percentile(v_finite, percentile)) if len(v_finite) > 0 else 1.0
        threshold_desc = f"P{int(percentile)}({threshold:.4f})"

    # Nœuds Unknown = candidats blastes
    assigned_pop = np.where(
        best_dist <= threshold,
        np.array([pop_names[i] for i in best_idx]),
        "Unknown",
    )

    return assigned_pop, best_dist, threshold, threshold_desc


def _score_m3_assignment(
    assigned_pop: np.ndarray,
    n_tot: int,
) -> float:
    """
    Score de qualité d'une attribution M3. Identique au notebook Section 10.4b.

    Formule : n_pops × (1 - n_unk/n_tot) × diversity
        n_pops    = nombre de populations différentes assignées (hors Unknown)
        n_unk     = nombre de nœuds Unknown
        diversity = pénalise les attributions monopolistiques (1 pop = tout)
                    = max(0, 1 - max(0, max_share - 0.5) × 2)
    """
    vc = pd.Series(assigned_pop).value_counts()
    n_unk = int(vc.get("Unknown", 0))
    assigned_vc = vc.drop("Unknown", errors="ignore")
    n_pops = len(assigned_vc)
    if n_pops == 0:
        return 0.0
    max_share = float(assigned_vc.max()) / max(float(assigned_vc.sum()), 1.0)
    diversity = max(0.0, 1.0 - max(0.0, max_share - 0.5) * 2.0)
    return float(n_pops) * (1.0 - n_unk / max(n_tot, 1)) * diversity


def mrd_method_3(
    patient_node_mfi: np.ndarray,
    nbm: "NBMReference",
    patient_node_assignments: np.ndarray,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    MRD Méthode 3 — Benchmark multi-méthodes de mapping populations + auto-sélection.

    Algorithme (Section 10.4b du notebook principal) :
    ───────────────────────────────────────────────────
    1. BENCHMARK : teste 4 méthodes de distance (M3_cosine, M8_ref_norm, M9_prior,
       M12_cosine_prior) avec seuil Otsu.
    2. SCORE : pour chaque méthode, calcule le score de qualité d'attribution
       (équilibre entre couverture des populations et diversité des assignations).
    3. AUTO-SÉLECTION : la méthode avec le meilleur score est choisie.
    4. LOG : les scores et la méthode choisie sont enregistrés dans les résultats.

    La méthode M12_cosine_prior utilise Cosinus + prior log10^3 + hard limit
    (la plus robuste selon les tests empiriques — ELN 2022).

    Nœuds classifiés "Unknown" = candidats blastes (LAIP) = MRD.

    Args :
        patient_node_mfi         : (n_nodes, n_markers) MFI patient par nœud NBM
        nbm                      : modèle NBM (pop_mfi_ref pour M3, sinon métaclusters)
        patient_node_assignments : (n_cells,) assignation des cellules patient
        params                   : dictionnaire de paramètres

    Returns dict :
        mrd_value        : float [0,1]
        mrd_percent      : float [0, 100]
        best_m3_method   : nom de la méthode sélectionnée par le benchmark
        m3_benchmark     : {méthode: score} pour toutes les méthodes testées
        unknown_nodes    : liste des nœuds Unknown / blastes
        node_mapping     : {nid: population assignée}
    """
    hard_limit = params.get("mrd3_hard_limit_factor", 5.0)
    threshold_mode = params.get("mrd3_unknown_mode", "auto_otsu")
    benchmark_methods = params.get("m3_benchmark_methods",
                                   ["M3_cosine", "M8_ref_norm", "M9_prior", "M12_cosine_prior"])
    do_log = params.get("m3_log_benchmark", True)
    percentile = 70.0

    patient_node_counts = np.bincount(patient_node_assignments, minlength=nbm.n_nodes)
    occupied_nodes = np.where(patient_node_counts > 0)[0]

    # ── Construire la matrice de référence des populations ───────────────────
    # Priorité : pop_mfi_ref (CSV chargés depuis Data/Ref MFI/) → centroïdes métaclusters
    if nbm.pop_mfi_ref is not None and len(nbm.pop_mfi_ref) > 0:
        pop_names = list(nbm.pop_mfi_ref.index)
        X_ref = nbm.pop_mfi_ref.values.astype(np.float64)
        cell_counts = nbm.pop_cell_counts or {p: 1000 for p in pop_names}
    else:
        mc_ids = sorted(set(nbm.metaclusters.tolist()))
        pop_names = [f"MC{mc}" for mc in mc_ids]
        X_ref = np.zeros((len(mc_ids), len(nbm.markers)), dtype=np.float64)
        mc_total_counts: Dict[int, int] = {}
        mc_node_sums: Dict[int, np.ndarray] = {}
        for nid, mc in enumerate(nbm.metaclusters):
            mc_ = int(mc)
            cnt = int(nbm.node_counts[nid])
            mc_total_counts[mc_] = mc_total_counts.get(mc_, 0) + cnt
            if mc_ not in mc_node_sums:
                mc_node_sums[mc_] = np.zeros(len(nbm.markers))
            mc_node_sums[mc_] += nbm.node_mfi[nid] * cnt
        for i, mc in enumerate(mc_ids):
            total = mc_total_counts.get(mc, 1)
            X_ref[i] = mc_node_sums.get(mc, np.zeros(len(nbm.markers))) / max(total, 1)
        cell_counts = {f"MC{mc}": mc_total_counts.get(mc, 1) for mc in mc_ids}

    X_nodes = patient_node_mfi[occupied_nodes]
    node_sizes = patient_node_counts[occupied_nodes].astype(np.float64)

    if X_nodes.shape[0] == 0:
        return {"method": "method_3", "mrd_value": 0.0, "mrd_percent": 0.0,
                "unknown_nodes": [], "error": "Aucun nœud occupé"}

    n_tot = len(X_nodes)

    # ── Benchmark multi-méthodes ─────────────────────────────────────────────
    benchmark_scores: Dict[str, float] = {}
    benchmark_results: Dict[str, Tuple[np.ndarray, np.ndarray, float, str]] = {}

    for method_name in benchmark_methods:
        try:
            asgn, bdist, th, tdesc = _run_single_m3_method(
                method_name=method_name,
                X_nodes=X_nodes,
                X_ref=X_ref,
                pop_names=pop_names,
                cell_counts=cell_counts,
                node_sizes=node_sizes,
                hard_limit_factor=hard_limit,
                n_nodes_total=nbm.n_nodes,
                threshold_mode=threshold_mode,
                percentile=percentile,
            )
            score = _score_m3_assignment(asgn, n_tot)
            benchmark_scores[method_name] = round(score, 4)
            benchmark_results[method_name] = (asgn, bdist, th, tdesc)
        except Exception as e:
            benchmark_scores[method_name] = 0.0
            logging.warning(f"[M3 benchmark] {method_name} a échoué : {e}")

    if not benchmark_results:
        return {"method": "method_3", "mrd_value": 0.0, "mrd_percent": 0.0,
                "unknown_nodes": [], "error": "Toutes les méthodes ont échoué"}

    # ── Auto-sélection de la meilleure méthode ───────────────────────────────
    best_method_name = max(benchmark_scores, key=benchmark_scores.get)  # type: ignore[arg-type]
    best_score = benchmark_scores[best_method_name]
    assigned_pop, best_dist_arr, threshold, threshold_desc = benchmark_results[best_method_name]

    if do_log:
        _score_lines = "  ".join(
            f"{m}={s:.3f}" + (" ★" if m == best_method_name else "")
            for m, s in sorted(benchmark_scores.items(), key=lambda x: -x[1])
        )
        logging.info(
            f"[M3 Benchmark] Meilleure : {best_method_name} (score={best_score:.3f}) | "
            f"Scores : {_score_lines}"
        )

    # ── Reconstruction du mapping complet (tous les nœuds occupés) ───────────
    node_mapping: Dict[int, str] = {}
    node_best_dist: Dict[int, float] = {}
    for k_i, nid in enumerate(occupied_nodes):
        node_mapping[int(nid)] = str(assigned_pop[k_i])
        node_best_dist[int(nid)] = round(float(best_dist_arr[k_i]), 6)

    unknown_nodes = [int(nid) for nid, pop in node_mapping.items() if pop == "Unknown"]
    n_cells_patient = int(patient_node_counts.sum())
    n_unknown_cells = int(sum(patient_node_counts[nid] for nid in unknown_nodes))
    mrd_value = n_unknown_cells / max(n_cells_patient, 1)

    # Distribution des populations assignées (en nombre de cellules)
    pop_distribution: Dict[str, int] = {}
    for nid, pop in node_mapping.items():
        pop_distribution[pop] = pop_distribution.get(pop, 0) + int(patient_node_counts[nid])

    return {
        "method": "method_3",
        "mrd_value": float(mrd_value),
        "mrd_pct": round(float(mrd_value) * 100, 4),
        "mrd_percent": round(float(mrd_value) * 100, 4),   # compat
        "mrd_events": n_unknown_cells,
        "best_m3_method": best_method_name,
        "m3_benchmark": benchmark_scores,
        "unknown_nodes": unknown_nodes,
        "n_unknown_cells": n_unknown_cells,
        "n_unknown_nodes": len(unknown_nodes),
        "n_cells_patient": n_cells_patient,
        "threshold": round(float(threshold), 6),
        "threshold_desc": threshold_desc,
        "node_mapping": node_mapping,
        "is_mrd_per_node": {int(nid): (pop == "Unknown") for nid, pop in node_mapping.items()},
        "node_best_dist": node_best_dist,
        "pop_distribution": pop_distribution,
        "n_occupied_nodes": int(len(occupied_nodes)),
        "n_populations_ref": int(len(pop_names)),
    }


# ── Alias conservé pour compatibilité ascendante avec le notebook ─────────────
def _apply_bayesian_prior_3(
    D: np.ndarray,
    pop_names: List[str],
    cell_counts: Dict[str, int],
    prior_mode: str,
    node_sizes: Optional[np.ndarray],
    hard_limit_factor: float,
    n_nodes_total: int,
) -> np.ndarray:
    """Alias vers _apply_bayesian_prior (rétro-compatibilité)."""
    return _apply_bayesian_prior(D, pop_names, cell_counts, prior_mode,
                                 node_sizes, hard_limit_factor, n_nodes_total)


# =============================================================================
# SECTION 8 — SÉLECTION DE LA MEILLEURE MÉTHODE
# =============================================================================

def select_best_mrd_method(
    m1: Dict[str, Any],
    m2: Dict[str, Any],
    m3: Dict[str, Any],
    params: Dict[str, Any],
) -> Tuple[str, float, bool, Dict[str, Any]]:
    """
    Sélectionne la méthode MRD la plus robuste et produit un rapport de décision.

    Critères :
    1. Méthode 1 favorisée si fold-change très clair (> 5×) sur ≥1 MC contributif.
    2. Méthode 2 favorisée si cell-level MAD disponible + signal cohérent.
    3. Méthode 3 favorisée par défaut (méthode de mapping populations).
    4. Bonus de concordance si les méthodes convergent (écart < divergence_threshold).

    Returns :
        best_method     : "method_1" | "method_2" | "method_3"
        confidence      : score [0,1] — confiance dans la méthode sélectionnée
        divergence_flag : True si les méthodes divergent au-delà du seuil
        decision_report : dict détaillant la logique de décision (pour JSON/HTML)
    """
    div_thr = params.get("divergence_threshold_pct", 0.5) / 100.0
    name_map = {"method_1": "M1", "method_2": "M2", "method_3": "M3"}

    v1 = m1.get("mrd_value", 0.0)
    v2 = m2.get("mrd_value", 0.0)
    v3 = m3.get("mrd_value", 0.0)
    values = {"method_1": v1, "method_2": v2, "method_3": v3}

    max_diff = max(abs(v1 - v2), abs(v1 - v3), abs(v2 - v3))
    divergence_flag = bool(max_diff > div_thr)

    scores: Dict[str, float] = {"method_1": 0.0, "method_2": 0.0, "method_3": 0.0}
    reasons: Dict[str, List[str]] = {"method_1": [], "method_2": [], "method_3": []}

    # ── Méthode 1 : fold-change élevé → signal MRD clair ──────────────────
    delta_max = m1.get("delta_max_value", 0.0)
    n_contributing = len(m1.get("contributing_mcs", []))
    if delta_max > 5.0 and n_contributing >= 1:
        scores["method_1"] += 0.4
        reasons["method_1"].append(
            f"fold-change max élevé ({delta_max:.1f}×) sur {n_contributing} MC contributif(s)"
        )
    elif delta_max > 2.0 and n_contributing >= 1:
        scores["method_1"] += 0.2
        reasons["method_1"].append(
            f"fold-change ({delta_max:.1f}×) sur {n_contributing} MC contributif(s)"
        )
    if n_contributing >= 2:
        scores["method_1"] += 0.15
        reasons["method_1"].append(f"{n_contributing} MC contributifs (signal réparti)")
    if not reasons["method_1"]:
        reasons["method_1"].append("aucun MC contributif significatif")

    # ── Méthode 2 : cell-level MAD (projection totale sans CD45) ──────────
    if m2.get("threshold_mode") in ("cell_level_mad", "new_data_mad"):
        n_mrd_clusters = len(m2.get("mrd_clusters", []))
        mrd_frac = m2.get("mrd_value", 0.0)
        if n_mrd_clusters >= 1:
            scores["method_2"] += 0.35
            reasons["method_2"].append(
                f"MAD cell-level actif — {n_mrd_clusters} cluster(s) MRD détectés"
            )
        if 0.001 < mrd_frac < 0.5:
            scores["method_2"] += 0.15
            reasons["method_2"].append(
                f"signal MRD cohérent ({mrd_frac*100:.3f}% — hors aberration)"
            )
        reasons["method_2"].append("mode global (toutes cellules, sans gating CD45)")
    else:
        bc = m2.get("bimodality_coeff", 0.0)
        if bc > 0.555:
            scores["method_2"] += 0.5 + min(bc - 0.555, 0.3)
            reasons["method_2"].append(f"distribution bimodale claire (BC={bc:.3f} > 0.555)")
        n_deviant = len(m2.get("deviant_nodes", m2.get("mrd_clusters", [])))
        if n_deviant >= 3:
            scores["method_2"] += 0.1
            reasons["method_2"].append(f"{n_deviant} nœuds déviants")
    if not reasons["method_2"]:
        reasons["method_2"].append("signal M2 faible ou absent")

    # ── Méthode 3 : fraction de nœuds Unknown (mapping populations) ────────
    n_unknown = len(m3.get("unknown_nodes", []))
    n_occupied = m3.get("n_occupied_nodes", 1)
    unknown_frac = n_unknown / max(n_occupied, 1)
    if 0.05 < unknown_frac < 0.5:
        scores["method_3"] += 0.35
        reasons["method_3"].append(
            f"{n_unknown} nœuds Unknown / {n_occupied} occupés ({unknown_frac*100:.1f}%)"
        )
    scores["method_3"] += 0.1   # bonus de base (méthode de référence mapping)
    reasons["method_3"].append(
        f"méthode mapping populations (best: {m3.get('best_m3_method', 'N/A')})"
    )

    # ── Bonus de concordance si les méthodes convergent ─────────────────────
    concordance_bonus_applied = False
    if not divergence_flag:
        median_mrd = float(np.median([v1, v2, v3]))
        for meth, v in [("method_1", v1), ("method_2", v2), ("method_3", v3)]:
            closeness = max(0.0, 1.0 - abs(v - median_mrd) / max(median_mrd, 1e-6))
            if closeness > 0.5:
                scores[meth] += closeness * 0.15
                concordance_bonus_applied = True

    best_method = max(scores, key=scores.get)  # type: ignore[arg-type]
    total_score = sum(scores.values())
    confidence = scores[best_method] / max(total_score, 1e-6)
    confidence = min(1.0, confidence)

    # ── Rapport de décision (visible dans JSON + HTML) ─────────────────────
    decision_report: Dict[str, Any] = {
        "best_method": best_method,
        "best_method_label": name_map[best_method],
        "confidence": round(confidence, 3),
        "divergence_flag": divergence_flag,
        "divergence_detail": (
            f"Écart max entre méthodes : {max_diff*100:.3f}% "
            f"(seuil : {div_thr*100:.2f}%)"
        ),
        "mrd_by_method": {
            name_map[m]: round(v * 100, 4) for m, v in values.items()
        },
        "scores": {name_map[m]: round(s, 4) for m, s in scores.items()},
        "reasons": {name_map[m]: r for m, r in reasons.items()},
        "concordance_bonus": concordance_bonus_applied,
        "selection_summary": (
            f"Méthode retenue : {name_map[best_method]} "
            f"(confiance {confidence*100:.0f}%)"
            + (" ⚠️ DIVERGENCE" if divergence_flag else " ✓ concordant")
        ),
    }

    logging.info(
        f"[Sélection MRD] {decision_report['selection_summary']} | "
        f"M1={v1*100:.3f}% M2={v2*100:.3f}% M3={v3*100:.3f}% | "
        f"divergence={divergence_flag}"
    )

    return best_method, round(confidence, 3), divergence_flag, decision_report


# =============================================================================
# SECTION 9 — PIPELINE PRINCIPAL PAR PATIENT
# =============================================================================

def run_mrd_pipeline_for_patient(
    fcs_path: Path,
    nbm: NBMReference,
    params: Optional[Dict[str, Any]] = None,
    patient_id: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline complet MRD pour un patient (1 fichier FCS).

    Nouvelle architecture (si nbm._nbm_adata_for_fsom disponible) :
    ─────────────────────────────────────────────────────────────────
    1. prepare_patho_adata()       → gate CD45+ + transform → AnnData patho
    2. M1 + M3 : build_mixed_tree()→ arbre FlowSOM MIXTE NBM+patho → métaclusters communs
       M1 : delta proportions patho/NBM par métacluster (ELN 2022)
       M3 : mapping cosine_prior sur nœuds mixtes → % Unknown = MRD
    3. M2 : fsom_nbm.new_data()    → projection patho sur arbre NBM → seuils MAD

    Fallback (legacy — si build_from_precomputed() ou echec nouveau pipeline) :
    ─────────────────────────────────────────────────────────────────────────────
    Mapping direct patho → nœuds NBM → distances + métaclusters (ancienne approche).

    Args :
        fcs_path   : chemin vers le FCS pathologique du patient
        nbm        : modèle NBM pré-construit (réutilisé pour tout le batch)
        params     : paramètres (utilise DEFAULT_PARAMS si None)
        patient_id : identifiant patient (déduit du nom de fichier si None)
        verbose    : afficher la progression

    Returns :
        dict structuré pour export JSON, rapport HTML et table Kaluza
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    fcs_path = Path(fcs_path)
    pid = patient_id or fcs_path.stem

    result: Dict[str, Any] = {
        "patient_id": pid,
        "fcs_path": str(fcs_path),
        "timestamp": datetime.now().isoformat(),
        "params": p,
        "preprocessing": {},
        "mrd": {
            "method_1": {},
            "method_2": {},
            "method_3": {},
            "best_method": None,
            "confidence": 0.0,
            "divergence_flag": False,
            "decision_report": {},
        },
        "errors": [],
    }

    if not nbm._built:
        result["errors"].append("Modèle NBM non initialisé.")
        return result

    # =========================================================================
    # CHOIX DU PIPELINE
    # Nouveau pipeline (arbre mixte + new_data) si _nbm_adata_for_fsom disponible
    # Fallback legacy sinon (build_from_precomputed(), ou FlowSOM non dispo)
    # =========================================================================
    use_new_pipeline = (
        nbm._nbm_adata_for_fsom is not None
        and nbm._fsom_model is not None
        and FLOWSOM_AVAILABLE
    )

    try:
        if use_new_pipeline:
            # =================================================================
            # NOUVEAU PIPELINE
            # M1/M3 : arbre mixte NBM+patho
            # M2    : FlowSOM.new_data() — projection patho sur arbre NBM
            # =================================================================
            if verbose:
                print(f"[Pipeline] {pid} | arbre mixte (M1/M3) + new_data (M2)")

            # ── Étape 1a : AnnData patho avec gating CD45+ (M1 / M3) ───────────
            try:
                adata_patho_cd45 = prepare_patho_adata(fcs_path, nbm, p, mode="cd45")
            except Exception as e:
                raise RuntimeError(f"prepare_patho_adata [cd45] échoué: {e}") from e

            # ── Étape 1b : AnnData patho SANS CD45 (M2 — toutes cellules) ────────
            # ELN 2022 M2 : la projection new_data doit couvrir toutes les cellules
            # (GMM débris + RANSAC seulement — pas de gate CD45)
            try:
                adata_patho_global = prepare_patho_adata(fcs_path, nbm, p, mode="global")
            except Exception as e:
                logging.warning(f"[prepare_patho_adata global] {pid}: {e}")
                adata_patho_global = adata_patho_cd45   # fallback sécurisé

            result["preprocessing"] = {
                "n_cells_cd45": adata_patho_cd45.n_obs,
                "n_cells_global": adata_patho_global.n_obs,
                "n_cells": adata_patho_cd45.n_obs,
                "n_markers": adata_patho_cd45.n_vars,
                "markers": list(adata_patho_cd45.var_names),
                "n_common_markers": len([m for m in nbm.markers if m in list(adata_patho_cd45.var_names)]),
                "common_markers": [m for m in nbm.markers if m in list(adata_patho_cd45.var_names)],
                "n_cells_mapped": adata_patho_cd45.n_obs,
                "preprocessing_m1m3": "cd45_gating",
                "preprocessing_m2": "global_no_cd45",
            }

            if adata_patho_cd45.n_obs < 100:
                result["errors"].append(
                    f"Trop peu de cellules après gating CD45 ({adata_patho_cd45.n_obs}) — résultats peu fiables."
                )

            # ── Étape 2 : M1 + M3 via arbre mixte NBM+patho ──────────────────
            if verbose:
                print(f"  → Arbre mixte NBM+patho (M1/M3) [mode cd45]...")
            try:
                mixed_result = nbm.build_mixed_tree(adata_patho_cd45, p)

                try:
                    result["mrd"]["method_1"] = _mrd_method_1_mixed(mixed_result, p)
                except Exception as e:
                    result["mrd"]["method_1"] = {"method": "method_1", "error": str(e), "mrd_value": 0.0}
                    result["errors"].append(f"M1 mixte : {e}")

                try:
                    result["mrd"]["method_3"] = _mrd_method_3_mixed(mixed_result, nbm, p)
                except Exception as e:
                    result["mrd"]["method_3"] = {"method": "method_3", "error": str(e), "mrd_value": 0.0}
                    result["errors"].append(f"M3 mixte : {e}")

            except Exception as e:
                result["mrd"]["method_1"] = {"method": "method_1", "error": str(e), "mrd_value": 0.0}
                result["mrd"]["method_3"] = {"method": "method_3", "error": str(e), "mrd_value": 0.0}
                result["errors"].append(f"Arbre mixte M1/M3 : {e}")
                logging.error(f"[M1/M3 mixte] {pid}: {e}")
                mixed_result = None

            # ── Étape 3 : M2 via FlowSOM.new_data() — sans gating CD45 ─────────
            if verbose:
                print(f"  → Projection patho sur arbre NBM (M2) [mode global — sans CD45]...")
            try:
                result["mrd"]["method_2"] = _mrd_method_2_new_data(nbm, adata_patho_global, p)
            except Exception as e:
                result["mrd"]["method_2"] = {"method": "method_2", "error": str(e), "mrd_value": 0.0}
                result["errors"].append(f"M2 new_data : {e}")
                logging.error(f"[M2 new_data] {pid}: {e}")

            # ── Table Kaluza (nœuds de l'arbre mixte) ────────────────────────
            try:
                if mixed_result is not None:
                    n_nodes_mixed = mixed_result["n_nodes"]
                    node_ass_patho = mixed_result["node_assignments_patho"]
                    patho_node_counts = np.bincount(node_ass_patho, minlength=n_nodes_mixed)
                    node_mc = np.zeros(n_nodes_mixed, dtype=np.int32)
                    cd = mixed_result.get("cluster_data")
                    if cd is not None:
                        node_mc = np.array(cd.obs["metaclustering"].values, dtype=np.int32)
                    contrib_mcs_m1 = set(result["mrd"]["method_1"].get("contributing_mcs", []))
                    deviant_m2 = set(result["mrd"]["method_2"].get("mrd_clusters", []))
                    unknown_m3 = set(result["mrd"]["method_3"].get("unknown_nodes", []))
                    n_total = int(patho_node_counts.sum())
                    rows = []
                    for nid in range(n_nodes_mixed):
                        cnt = int(patho_node_counts[nid])
                        mc = int(node_mc[nid])
                        rows.append({
                            "node_id": nid, "metacluster": mc,
                            "n_cells_patient": cnt,
                            "prop_patient": round(cnt / max(n_total, 1), 6),
                            "mrd_m1_contributing": bool(mc in contrib_mcs_m1),
                            "mrd_m2_deviant": bool(nid in deviant_m2),
                            "mrd_m3_unknown": bool(nid in unknown_m3),
                            "mrd_any": bool(mc in contrib_mcs_m1 or nid in deviant_m2 or nid in unknown_m3),
                        })
                    result["kaluza_table"] = rows
                else:
                    result["kaluza_table"] = []
            except Exception as e:
                result["kaluza_table"] = []
                logging.warning(f"[Kaluza] Table non générée: {e}")

        else:
            # =================================================================
            # PIPELINE LEGACY (fallback : build_from_precomputed ou echec)
            # Mapping direct patho → nœuds NBM → distances + métaclusters
            # =================================================================
            if verbose:
                print(f"[Pipeline Legacy] {pid} | mapping direct sur arbre NBM")

            X_patient, markers_patient = preprocess_fcs(fcs_path, p)
            result["preprocessing"] = {
                "n_cells": int(X_patient.shape[0]),
                "n_markers": int(X_patient.shape[1]),
                "markers": markers_patient,
            }

            if X_patient.shape[0] < 100:
                result["errors"].append(
                    f"Trop peu de cellules après gating ({X_patient.shape[0]}) — résultats peu fiables."
                )

            common_markers = [m for m in nbm.markers if m in markers_patient]
            if not common_markers:
                raise RuntimeError(
                    f"Aucun marqueur commun entre patient ({markers_patient[:5]}...) "
                    f"et NBM ({nbm.markers[:5]}...)"
                )
            p_idx = [markers_patient.index(m) for m in common_markers]
            n_idx = [nbm.markers.index(m) for m in common_markers]
            X_aligned = X_patient[:, p_idx]
            nbm_weights_aligned = nbm.som_weights[:, n_idx]
            nbm_mfi_aligned = nbm.node_mfi[:, n_idx]

            X_nbm_aligned: Optional[np.ndarray] = None
            if nbm._X_nbm is not None:
                X_nbm_aligned = nbm._X_nbm[:, n_idx]
            nbm_tmp = _NBMAligned(
                nbm_weights_aligned, nbm_mfi_aligned,
                nbm.node_counts, nbm.metaclusters,
                common_markers, nbm.n_nodes, nbm.node_distances,
                X_nbm=X_nbm_aligned,
                nbm_cell_assignments=nbm._nbm_cell_assignments,
                mad_allowed=p.get("mad_allowed", 4),
            )

            node_assignments, patient_node_mfi = map_patient_to_nbm(X_aligned, nbm_tmp)
            result["preprocessing"]["n_cells_mapped"] = int(X_aligned.shape[0])
            result["preprocessing"]["n_common_markers"] = len(common_markers)
            result["preprocessing"]["common_markers"] = common_markers

            try:
                result["mrd"]["method_1"] = mrd_method_1(node_assignments, nbm_tmp, p)
            except Exception as e:
                result["mrd"]["method_1"] = {"method": "method_1", "error": str(e), "mrd_value": 0.0}
                result["errors"].append(f"Méthode 1 legacy: {e}")

            try:
                result["mrd"]["method_2"] = mrd_method_2(
                    patient_node_mfi, nbm_tmp, node_assignments, p, X_patient=X_aligned
                )
            except Exception as e:
                result["mrd"]["method_2"] = {"method": "method_2", "error": str(e), "mrd_value": 0.0}
                result["errors"].append(f"Méthode 2 legacy: {e}")

            try:
                nbm_tmp_m3 = nbm_tmp
                if nbm.pop_mfi_ref is not None:
                    common_ref = [m for m in common_markers if m in nbm.pop_mfi_ref.columns]
                    nbm_tmp_m3.pop_mfi_ref = nbm.pop_mfi_ref[common_ref] if common_ref else None
                    nbm_tmp_m3.pop_cell_counts = nbm.pop_cell_counts
                result["mrd"]["method_3"] = mrd_method_3(
                    patient_node_mfi, nbm_tmp_m3, node_assignments, p
                )
            except Exception as e:
                result["mrd"]["method_3"] = {"method": "method_3", "error": str(e), "mrd_value": 0.0}
                result["errors"].append(f"Méthode 3 legacy: {e}")

            result["kaluza_table"] = _build_kaluza_table(
                nbm_tmp, node_assignments, patient_node_mfi,
                result["mrd"]["method_1"],
                result["mrd"]["method_2"],
                result["mrd"]["method_3"],
            )

    except Exception as e:
        result["errors"].append(f"Erreur pipeline : {e}\n{traceback.format_exc()}")
        logging.error(f"[PIPELINE] Patient {pid} : {e}")

    # ── Sélection meilleure méthode ───────────────────────────────────────────
    m1 = result["mrd"]["method_1"]
    m2 = result["mrd"]["method_2"]
    m3 = result["mrd"]["method_3"]
    best, conf, div_flag, decision_report = select_best_mrd_method(m1, m2, m3, p)
    result["mrd"]["best_method"] = best
    result["mrd"]["confidence"] = conf
    result["mrd"]["divergence_flag"] = div_flag
    result["mrd"]["decision_report"] = decision_report
    best_mrd = result["mrd"][best].get("mrd_value", 0.0)
    result["mrd"]["best_mrd_value"] = float(best_mrd)
    result["mrd"]["best_mrd_percent"] = round(float(best_mrd) * 100, 4)

    # Clés de compatibilité notebook (result['n_cells'], result['markers'])
    result["n_cells"] = result["preprocessing"].get("n_cells", 0)
    result["markers"] = result["preprocessing"].get("markers", [])

    return result


class _NBMAligned:
    """Structure légère d'un NBM aligné sur les marqueurs communs.

    Supporte optionnellement les données brutes NBM (pour cell-level MAD en M2).
    """
    def __init__(
        self,
        som_weights,
        node_mfi,
        node_counts,
        metaclusters,
        markers,
        n_nodes,
        node_distances_orig,
        X_nbm: Optional[np.ndarray] = None,
        nbm_cell_assignments: Optional[np.ndarray] = None,
        mad_allowed: int = 4,
    ):
        self.som_weights = som_weights
        self.node_mfi = node_mfi
        self.node_counts = node_counts
        self.metaclusters = metaclusters
        self.markers = markers
        self.n_nodes = n_nodes
        # Recalculer les distances sur les marqueurs alignés
        self.node_distances = cdist(node_mfi, node_mfi, metric="euclidean")
        self._built = True
        self.pop_mfi_ref: Optional[pd.DataFrame] = None
        self.pop_cell_counts: Optional[Dict[str, int]] = None
        # Données brutes NBM pour cell-level MAD (Méthode 2)
        self._X_nbm = X_nbm
        self._nbm_cell_assignments = nbm_cell_assignments
        if X_nbm is not None and nbm_cell_assignments is not None:
            # Recalcul des seuils MAD dans l'espace réduit (marqueurs communs)
            n_nodes_actual = som_weights.shape[0]
            self._cluster_thresholds = np.full(n_nodes_actual, np.inf, dtype=np.float64)
            for _nid in range(n_nodes_actual):
                _mask = nbm_cell_assignments == _nid
                if _mask.sum() < 2:
                    continue
                _dists = np.linalg.norm(X_nbm[_mask] - som_weights[_nid], axis=1)
                _med = np.median(_dists)
                _mad = np.median(np.abs(_dists - _med))
                self._cluster_thresholds[_nid] = _med + mad_allowed * _mad
        else:
            self._cluster_thresholds = None

    def get_metacluster_proportions(self) -> Dict[int, float]:
        total = self.node_counts.sum()
        mc_counts: Dict[int, int] = {}
        for nid, mc in enumerate(self.metaclusters):
            mc_counts[int(mc)] = mc_counts.get(int(mc), 0) + int(self.node_counts[nid])
        return {mc: cnt / max(total, 1) for mc, cnt in mc_counts.items()}


def _build_kaluza_table(
    nbm_aligned, node_assignments, patient_node_mfi, m1, m2, m3
) -> List[Dict[str, Any]]:
    """
    Construit la table tabulaire par nœud/métacluster pour le gating Kaluza.
    Colonnes : node_id, metacluster, n_cells_patient, mrd_m1, mrd_m2, mrd_m3, prop_patient
    """
    patient_node_counts = np.bincount(node_assignments, minlength=nbm_aligned.n_nodes)
    n_total = patient_node_counts.sum()
    deviant_m2 = set(m2.get("mrd_clusters", m2.get("deviant_nodes", [])))
    unknown_m3 = set(m3.get("unknown_nodes", []))
    contrib_mcs_m1 = set(m1.get("contributing_mcs", []))

    rows = []
    for nid in range(nbm_aligned.n_nodes):
        cnt = int(patient_node_counts[nid])
        mc = int(nbm_aligned.metaclusters[nid])
        rows.append({
            "node_id": nid,
            "metacluster": mc,
            "n_cells_patient": cnt,
            "prop_patient": round(cnt / max(n_total, 1), 6),
            "mrd_m1_contributing": bool(mc in contrib_mcs_m1),
            "mrd_m2_deviant": bool(nid in deviant_m2),
            "mrd_m3_unknown": bool(nid in unknown_m3),
            "mrd_any": bool(mc in contrib_mcs_m1 or nid in deviant_m2 or nid in unknown_m3),
        })
    return rows


# =============================================================================
# SECTION 10 — EXPORTS HTML & JSON
# =============================================================================

def export_patient_json(result: Dict[str, Any], output_dir: Path) -> Path:
    """Exporte le résultat patient en JSON (avec types numpy sérialisables)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pid = result.get("patient_id", "unknown")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"mrd_{pid}_{ts}.json"

    # Sérialisation JSON-safe (numpy → python natif)
    def _json_safe(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Type non sérialisable: {type(obj)}")

    # Structure JSON finale (nettoyée : enlève les params volumineux et la table Kaluza)
    json_out = {
        "patient_id": result.get("patient_id"),
        "fcs_path": result.get("fcs_path"),
        "timestamp": result.get("timestamp"),
        "preprocessing": result.get("preprocessing", {}),
        "mrd": {
            "method_1": {
                "value": result["mrd"]["method_1"].get("mrd_value", 0.0),
                "percent": result["mrd"]["method_1"].get("mrd_percent", 0.0),
                "details": {
                    k: v for k, v in result["mrd"]["method_1"].items()
                    if k not in ("mc_patient_props", "mc_nbm_props")
                }
            },
            "method_2": {
                "value": result["mrd"]["method_2"].get("mrd_value", 0.0),
                "percent": result["mrd"]["method_2"].get("mrd_percent", 0.0),
                "details": {
                    k: v for k, v in result["mrd"]["method_2"].items()
                    if k not in ("node_distances",)
                }
            },
            "method_3": {
                "value": result["mrd"]["method_3"].get("mrd_value", 0.0),
                "percent": result["mrd"]["method_3"].get("mrd_percent", 0.0),
                "details": {
                    k: v for k, v in result["mrd"]["method_3"].items()
                    if k not in ("node_mapping", "node_best_dist")
                }
            },
            "best_method": result["mrd"].get("best_method"),
            "best_mrd_percent": result["mrd"].get("best_mrd_percent"),
            "confidence": result["mrd"].get("confidence"),
            "divergence_flag": result["mrd"].get("divergence_flag"),
        },
        "errors": result.get("errors", []),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, default=_json_safe, ensure_ascii=False)

    return out_path


def export_patient_html(result: Dict[str, Any], output_dir: Path) -> Path:
    """
    Génère un rapport HTML autonome par patient.
    Inclut :
    - Résumé des paramètres utilisés
    - Résultats des 3 méthodes MRD
    - Tableau des métaclusters/nœuds contributifs
    - Graphique en barres des 3 MRD (base64 inliné pour portabilité)
    """
    import base64
    import io

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pid = result.get("patient_id", "unknown")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"mrd_report_{pid}_{ts}.html"

    mrd = result.get("mrd", {})
    m1 = mrd.get("method_1", {})
    m2 = mrd.get("method_2", {})
    m3 = mrd.get("method_3", {})
    best = mrd.get("best_method", "N/A")
    div_flag = mrd.get("divergence_flag", False)
    conf = mrd.get("confidence", 0.0)

    v1_pct = m1.get("mrd_percent", 0.0)
    v2_pct = m2.get("mrd_percent", 0.0)
    v3_pct = m3.get("mrd_percent", 0.0)

    # ── Graphique barres ─────────────────────────────────────────────────────
    bar_img_b64 = ""
    if MATPLOTLIB_AVAILABLE:
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["#4f86c6", "#e87d3e", "#55b87c"]
            bars = ax.bar(
                ["MRD Méthode 1\n(Δ Métaclusters)",
                 "MRD Méthode 2\n(Distance MST)",
                 "MRD Méthode 3\n(Mapping Pop.)"],
                [v1_pct, v2_pct, v3_pct],
                color=colors, edgecolor="white", linewidth=1.2, width=0.55,
            )
            for bar, val in zip(bars, [v1_pct, v2_pct, v3_pct]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}%",
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
            # Ligne seuil LOQ ELN 2022 (0.005%)
            ax.axhline(y=0.005, color="#e74c3c", linestyle="--", linewidth=1.5,
                       label="LOQ ELN 2022 (0.005%)")
            ax.axhline(y=0.009, color="#f39c12", linestyle=":", linewidth=1.5,
                       label="LOD ELN 2022 (0.009%)")
            ax.set_ylabel("MRD (%)", fontsize=12)
            ax.set_title(f"Triple MRD — Patient {pid}", fontsize=13, fontweight="bold")
            ax.legend(fontsize=9, loc="upper right")
            ax.set_ylim(0, max(max(v1_pct, v2_pct, v3_pct) * 1.3, 0.1))
            # Surligner la meilleure méthode
            best_idx = {"method_1": 0, "method_2": 1, "method_3": 2}.get(best)
            if best_idx is not None:
                bars[best_idx].set_edgecolor("#ffd700")
                bars[best_idx].set_linewidth(3)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            bar_img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        except Exception as e:
            logging.warning(f"Graphique HTML non généré : {e}")

    # ── Table des nœuds/métaclusters ─────────────────────────────────────────
    kaluza_table = result.get("kaluza_table", [])
    kaluza_rows_html = ""
    for row in sorted(kaluza_table, key=lambda r: -r.get("prop_patient", 0)):
        if row.get("n_cells_patient", 0) == 0:
            continue
        bg = "#fff3cd" if row.get("mrd_any") else ""
        kaluza_rows_html += f"""
        <tr style="background:{bg}">
            <td>{row['node_id']}</td>
            <td>{row['metacluster']}</td>
            <td>{row['n_cells_patient']:,}</td>
            <td>{row['prop_patient']*100:.3f}%</td>
            <td>{'✅' if row['mrd_m1_contributing'] else '—'}</td>
            <td>{'✅' if row['mrd_m2_deviant'] else '—'}</td>
            <td>{'✅' if row['mrd_m3_unknown'] else '—'}</td>
        </tr>"""

    # ── Erreurs ──────────────────────────────────────────────────────────────
    errors = result.get("errors", [])
    errors_html = ""
    if errors:
        errors_html = "<div class='alert'><b>⚠️ Avertissements :</b><ul>" + \
                      "".join(f"<li>{e}</li>" for e in errors) + "</ul></div>"

    # ── Table métaclusters Méthode 1 ─────────────────────────────────────────
    mc_rows = ""
    fc_dict = m1.get("mc_fold_changes", {})
    for mc, fc in sorted(fc_dict.items(), key=lambda x: -x[1]):
        pp = m1.get("mc_patient_props", {}).get(mc, 0.0)
        np_ = m1.get("mc_nbm_props", {}).get(mc, 0.0)
        contrib = mc in m1.get("contributing_mcs", [])
        bg = "#d4edda" if contrib else ""
        mc_rows += f"""
        <tr style="background:{bg}">
            <td>MC{mc}</td>
            <td>{pp*100:.3f}%</td>
            <td>{np_*100:.3f}%</td>
            <td>{fc:.2f}×</td>
            <td>{'✅ Contributif' if contrib else '—'}</td>
        </tr>"""

    # ── Paramètres ────────────────────────────────────────────────────────────
    p = result.get("params", {})
    preproc = result.get("preprocessing", {})

    div_badge = (
        '<span class="badge badge-danger">⚠️ Divergence détectée</span>'
        if div_flag else
        '<span class="badge badge-success">✅ Méthodes concordantes</span>'
    )

    bar_section = ""
    if bar_img_b64:
        bar_section = f'<img src="data:image/png;base64,{bar_img_b64}" style="max-width:700px; margin:16px 0;" alt="MRD Barres">'

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rapport MRD — {pid}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; color: #212529; margin: 0; padding: 0; }}
  .container {{ max-width: 1100px; margin: 24px auto; padding: 0 20px; }}
  .header {{ background: linear-gradient(135deg, #1a237e, #1565c0); color:white; padding: 24px 32px;
             border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
  .header h1 {{ margin: 0 0 4px; font-size: 1.8rem; }}
  .header p {{ margin: 0; opacity: 0.85; font-size: 0.95rem; }}
  .card {{ background: white; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
  .card h2 {{ margin: 0 0 16px; font-size: 1.2rem; color: #1a237e; border-bottom: 2px solid #e3e8f0; padding-bottom: 8px; }}
  .mrd-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 16px 0; }}
  .mrd-box {{ padding: 16px; border-radius: 8px; text-align: center; }}
  .mrd-box.m1 {{ background: #e3f2fd; border: 2px solid #4f86c6; }}
  .mrd-box.m2 {{ background: #fff8e1; border: 2px solid #e87d3e; }}
  .mrd-box.m3 {{ background: #e8f5e9; border: 2px solid #55b87c; }}
  .mrd-box.best {{ box-shadow: 0 0 0 3px #ffd700; }}
  .mrd-val {{ font-size: 2.1rem; font-weight: 800; margin: 8px 0; }}
  .mrd-label {{ font-size: 0.85rem; color: #555; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ background: #1a237e; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #e9ecef; }}
  tr:hover td {{ background: #f1f5ff; }}
  .badge {{ display: inline-block; padding: 4px 10px; border-radius: 20px;
            font-size: 0.85rem; font-weight: 600; margin: 4px 0; }}
  .badge-success {{ background: #d4edda; color: #155724; }}
  .badge-danger {{ background: #f8d7da; color: #721c24; }}
  .alert {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 12px 16px; margin: 12px 0; }}
  .params-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 0.85rem; }}
  .param-item {{ background: #f8f9fa; border-radius: 4px; padding: 6px 10px; }}
  .param-item b {{ color: #1a237e; }}
  .section-tag {{ display: inline-block; background: #1a237e; color: white; border-radius: 4px;
                  padding: 2px 8px; font-size: 0.75rem; font-weight: 600; margin-right: 8px; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>Rapport Triple MRD — <code>{pid}</code></h1>
    <p>Généré le {result.get('timestamp', '')} · FlowSOM × 3 méthodes MRD · Pipeline batch</p>
  </div>

  {errors_html}

  <!-- Résultats MRD -->
  <div class="card">
    <h2>Résultats MRD</h2>
    {div_badge}
    &nbsp;&nbsp; Confiance méthode sélectionnée : <b>{conf*100:.0f}%</b>
    &nbsp;&nbsp; Méthode recommandée : <b>{best}</b>
    <div class="mrd-grid">
      <div class="mrd-box m1{'  best' if best=='method_1' else ''}">
        <div class="mrd-label">🔷 Méthode 1 — Δ Métaclusters</div>
        <div class="mrd-val" style="color:#4f86c6">{v1_pct:.4f}%</div>
        <div class="mrd-label">Fold-change seuil : {p.get('mrd1_fold_change', 2.0)}×</div>
        <div class="mrd-label">Métaclusters contributifs : {len(m1.get('contributing_mcs', []))}</div>
        <div class="mrd-label">Δ max : MC{m1.get('delta_max_mc', '?')} ({m1.get('delta_max_value', 0):.1f}×)</div>
      </div>
      <div class="mrd-box m2{'  best' if best=='method_2' else ''}">
        <div class="mrd-label">🔶 Méthode 2 — Distance MST</div>
        <div class="mrd-val" style="color:#e87d3e">{v2_pct:.4f}%</div>
        <div class="mrd-label">Seuil : μ + {m2.get('threshold_k', 2.0)}×σ = {m2.get('threshold', 0.0):.4f}</div>
        <div class="mrd-label">Nœuds déviants : {len(m2.get('mrd_clusters', m2.get('deviant_nodes', [])))}</div>
        <div class="mrd-label">Bimodalité : {m2.get('bimodality_coeff', 0.0):.3f} {'⭐ (>0.555)' if m2.get('bimodality_coeff', 0) > 0.555 else ''}</div>
      </div>
      <div class="mrd-box m3{'  best' if best=='method_3' else ''}">
        <div class="mrd-label">🟢 Méthode 3 — Mapping Pop.</div>
        <div class="mrd-val" style="color:#55b87c">{v3_pct:.4f}%</div>
        <div class="mrd-label">Méthode : {m3.get('mapping_method', 'N/A')}</div>
        <div class="mrd-label">Nœuds Unknown : {len(m3.get('unknown_nodes', []))}</div>
        <div class="mrd-label">Seuil Otsu : {m3.get('threshold', 0.0):.4f}</div>
      </div>
    </div>
    {bar_section}
  </div>

  <!-- Détail Méthode 1 -->
  <div class="card">
    <h2><span class="section-tag">M1</span> Détail Δ Métaclusters</h2>
    <p>Seuil ELN 2022 : proportion patient &gt; <b>{p.get('mrd1_fold_change', 2.0)}×</b> la proportion NBM et ≥ <b>{p.get('mrd1_min_events', 17)} événements</b> par nœud.</p>
    <table>
      <thead><tr><th>Métacluster</th><th>Prop. Patient</th><th>Prop. NBM</th><th>Fold-Change</th><th>Statut</th></tr></thead>
      <tbody>{mc_rows}</tbody>
    </table>
  </div>

  <!-- Méthode 2 résumé -->
  <div class="card">
    <h2><span class="section-tag">M2</span> Distances Euclidiennes MST</h2>
    <p>
      Nœuds déviants (distance &gt; seuil) : <b>{len(m2.get('mrd_clusters', m2.get('deviant_nodes', [])))}</b>
      · Seuil = μ<sub>NBM</sub>({m2.get('nbm_dist_mean', 0):.4f}) + {m2.get('threshold_k', 2.0)}×σ({m2.get('nbm_dist_std', 0):.4f}) = <b>{m2.get('threshold', 0):.4f}</b>
      · Coefficient de bimodalité : <b>{m2.get('bimodality_coeff', 0):.4f}</b>
    </p>
    <p>Nœuds déviants : {m2.get('mrd_clusters', m2.get('deviant_nodes', []))}</p>
  </div>

  <!-- Méthode 3 résumé -->
  <div class="card">
    <h2><span class="section-tag">M3</span> Mapping Populations (Section 10)</h2>
    <p>
      Méthode : <b>{m3.get('mapping_method', 'N/A')}</b>
      · Seuil Otsu : <b>{m3.get('threshold', 0):.4f}</b>
      · Nœuds Unknown (blastes candidats) : <b>{len(m3.get('unknown_nodes', []))}</b>/{m3.get('n_occupied_nodes', 0)}
      · Populations de référence : {m3.get('n_populations_ref', 0)}
    </p>
    <p>Distribution populations : {m3.get('pop_distribution', {})}</p>
  </div>

  <!-- Table Kaluza -->
  <div class="card">
    <h2>Table de gating Kaluza — Nœuds MRD contributifs</h2>
    <p><small>Surlignés en jaune : nœuds contribuant à au moins une méthode MRD. Utiliser <code>node_id</code> et <code>metacluster</code> pour filtrer dans Kaluza.</small></p>
    <table>
      <thead><tr><th>Node ID</th><th>Métacluster</th><th>Cellules Patient</th><th>Proportion</th><th>MRD M1</th><th>MRD M2</th><th>MRD M3</th></tr></thead>
      <tbody>{kaluza_rows_html}</tbody>
    </table>
  </div>

  <!-- Paramètres -->
  <div class="card">
    <h2>Paramètres de l'analyse</h2>
    <div class="params-grid">
      <div class="param-item"><b>Transformation :</b> {p.get('transform', 'N/A')}</div>
      <div class="param-item"><b>Grille SOM :</b> {p.get('xdim', '?')}×{p.get('ydim', '?')}</div>
      <div class="param-item"><b>Métaclusters :</b> {p.get('n_clusters', '?')}</div>
      <div class="param-item"><b>Seed :</b> {p.get('seed', 42)}</div>
      <div class="param-item"><b>Cellules après gating :</b> {preproc.get('n_cells', '?'):,}</div>
      <div class="param-item"><b>Marqueurs communs :</b> {preproc.get('n_common_markers', '?')}</div>
      <div class="param-item"><b>Fold-change M1 :</b> {p.get('mrd1_fold_change', 2.0)}×</div>
      <div class="param-item"><b>k-sigma M2 :</b> {p.get('mrd2_threshold_k', 2.0)}</div>
      <div class="param-item"><b>Méthode M3 :</b> {p.get('mrd3_mapping_method', 'N/A')}</div>
    </div>
    <p style="margin-top:12px; font-size:0.8rem; color: #888;">
      Marqueurs utilisés : {', '.join(preproc.get('common_markers', [])[:10])}
      {'...' if len(preproc.get('common_markers', [])) > 10 else ''}
    </p>
  </div>

  <p style="text-align:center; color:#888; font-size:0.8rem; margin-top:24px;">
    Rapport généré par <b>mrd_pipeline.py</b> · FlowSOM Analysis Pipeline v1.0 ·
    Référence ELN 2022 · LOD 0.009% · LOQ 0.005%
  </p>
</div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path


# =============================================================================
# SECTION 11 — BATCH SUR COHORTE
# =============================================================================

def run_mrd_batch(
    fcs_file_list: List[Path],
    nbm: NBMReference,
    params: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    patient_id_map: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Lance le pipeline MRD sur une liste de fichiers FCS (1 FCS = 1 patient).

    Le modèle NBM est réutilisé pour tous les patients (pas de recalcul).

    Args :
        fcs_file_list  : liste des chemins vers les FCS pathologiques
        nbm            : modèle NBM pré-construit
        params         : paramètres communs (DEFAULT_PARAMS si None)
        output_dir     : dossier de sortie (créé automatiquement)
        patient_id_map : optionnel — {nom_fichier.fcs → patient_id}
        verbose        : afficher la progression

    Returns :
        DataFrame récapitulatif de la cohorte (un rang par patient) +
        sauvegarde CSV de la cohorte dans output_dir/cohort_summary.csv
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    out_dir = Path(output_dir or p.get("output_dir", "output/mrd_reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pid_map = patient_id_map or {}

    cohort_rows = []
    n = len(fcs_file_list)

    for i, fcs_path in enumerate(fcs_file_list):
        fcs_path = Path(fcs_path)
        pid = pid_map.get(fcs_path.name, fcs_path.stem)

        if verbose:
            print(f"[{i+1}/{n}] Patient {pid} — {fcs_path.name} ...", flush=True)

        result = run_mrd_pipeline_for_patient(fcs_path, nbm, p, patient_id=pid)

        # Exports HTML + JSON par patient
        try:
            json_path = export_patient_json(result, out_dir)
            html_path = export_patient_html(result, out_dir)
        except Exception as e:
            logging.error(f"Export {pid} : {e}")
            json_path = html_path = None

        mrd = result.get("mrd", {})
        cohort_rows.append({
            "patient_id": pid,
            "fcs_file": fcs_path.name,
            "n_cells": result.get("preprocessing", {}).get("n_cells", 0),
            "MRD_method_1_pct": mrd.get("method_1", {}).get("mrd_percent", None),
            "MRD_method_2_pct": mrd.get("method_2", {}).get("mrd_percent", None),
            "MRD_method_3_pct": mrd.get("method_3", {}).get("mrd_percent", None),
            "best_method": mrd.get("best_method"),
            "best_MRD_pct": mrd.get("best_mrd_percent"),
            "confidence": mrd.get("confidence"),
            "divergence_flag": mrd.get("divergence_flag"),
            "errors": "; ".join(result.get("errors", [])) or None,
            "html_report": str(html_path) if html_path else None,
            "json_path": str(json_path) if json_path else None,
        })

        if verbose:
            m1_pct = mrd.get("method_1", {}).get("mrd_percent", 0.0)
            m2_pct = mrd.get("method_2", {}).get("mrd_percent", 0.0)
            m3_pct = mrd.get("method_3", {}).get("mrd_percent", 0.0)
            div = "⚠️  DIVERGENCE" if mrd.get("divergence_flag") else "✅"
            print(
                f"   M1={m1_pct:.3f}%  M2={m2_pct:.3f}%  M3={m3_pct:.3f}%"
                f"  → {mrd.get('best_method', 'N/A')} ({mrd.get('confidence', 0)*100:.0f}%)  {div}"
            )

    cohort_df = pd.DataFrame(cohort_rows)

    # Export récap cohorte
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cohort_csv = out_dir / f"cohort_summary_{ts}.csv"
    cohort_df.to_csv(cohort_csv, index=False, sep=";", decimal=",")

    if verbose:
        print(f"\n{'='*60}")
        print(f" COHORTE — {len(cohort_rows)} patients analysés")
        print(f" Récap : {cohort_csv}")
        print(f" HTML  : {out_dir}")
        print(f"{'='*60}")
        if len(cohort_df) > 0:
            print(cohort_df[["patient_id", "MRD_method_1_pct", "MRD_method_2_pct",
                              "MRD_method_3_pct", "best_method", "divergence_flag"]].to_string(index=False))

    return cohort_df


# =============================================================================
# SECTION 12 — UTILITAIRES KALUZA (export CSV par patient)
# =============================================================================

def export_kaluza_csv(result: Dict[str, Any], output_dir: Path) -> Path:
    """
    Exporte la table tabulaire de gating Kaluza en CSV.
    Colonnes : node_id, metacluster, n_cells_patient, prop_patient,
               mrd_m1_contributing, mrd_m2_deviant, mrd_m3_unknown, mrd_any
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pid = result.get("patient_id", "unknown")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"kaluza_{pid}_{ts}.csv"

    kaluza_df = pd.DataFrame(result.get("kaluza_table", []))
    if not kaluza_df.empty:
        kaluza_df.to_csv(out_path, index=False, sep=";", decimal=",")

    return out_path