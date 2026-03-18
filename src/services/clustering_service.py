"""
clustering_service.py — Orchestration du clustering FlowSOM.

Chaîne de clustering complète:
  1. Sélection et filtrage des marqueurs (exclusion FSC/SSC si configuré)
  2. Empilement des échantillons en une matrice unique
  3. Sous-échantillonnage pour le clustering (ratio par condition si COMPARE_MODE)
  4. FlowSOM (GPU → CPU fallback)
  5. Auto-clustering (optionnel): recherche du k optimal
  6. Calcul de la matrice MFI médiane par métacluster
  7. Propagation des labels à toutes les cellules

Retourne un PipelineResult enrichi avec les métadonnées de clustering.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import anndata as ad

    _ANNDATA_AVAILABLE = True
except ImportError:
    _ANNDATA_AVAILABLE = False

from flowsom_pipeline_pro.config.pipeline_config import PipelineConfig
from flowsom_pipeline_pro.config.constants import SCATTER_PATTERNS
from flowsom_pipeline_pro.src.models.sample import FlowSample
from flowsom_pipeline_pro.src.models.pipeline_result import (
    PipelineResult,
    ClusteringMetrics,
)
from flowsom_pipeline_pro.src.core.clustering import FlowSOMClusterer
from flowsom_pipeline_pro.src.core.metaclustering import find_optimal_clusters
from flowsom_pipeline_pro.src.utils.validators import (
    check_no_fsc_ssc_in_analysis_markers,
    check_nan,
)
from flowsom_pipeline_pro.src.utils.logger import get_logger

_logger = get_logger("services.clustering")


def select_markers_for_clustering(
    var_names: List[str],
    config: PipelineConfig,
) -> List[str]:
    """
    Sélectionne les marqueurs à utiliser pour le clustering FlowSOM.

    Règles (dans l'ordre de priorité):
    1. whitelist dans config.markers.use (si non vide)
    2. Exclusion des patterns FSC/SSC/Time si config.flowsom.exclude_scatter=True
    3. Exclusion de la blacklist config.markers.exclude

    Args:
        var_names: Tous les marqueurs disponibles.
        config: Configuration du pipeline.

    Returns:
        Liste des marqueurs sélectionnés.
    """
    # Cas 1: whitelist explicite
    whitelist = getattr(config.markers, "use", [])
    if whitelist:
        available = [m for m in whitelist if m in var_names]
        missing = [m for m in whitelist if m not in var_names]
        if missing:
            _logger.warning("Marqueurs whitelist absents: %s", missing)
        return available

    # Cas 2: exclusion automatique
    markers = list(var_names)

    if getattr(config.flowsom, "exclude_scatter", True):
        markers = [
            m for m in markers if not any(p in m.upper() for p in SCATTER_PATTERNS)
        ]

    # Cas 3: blacklist (correspondance par sous-chaîne, comme le monolithe)
    blacklist = getattr(config.markers, "exclude_additional", [])
    if blacklist:
        markers = [
            m
            for m in markers
            if not any(excl.upper() in m.upper() for excl in blacklist)
        ]

    # Validation finale
    check_no_fsc_ssc_in_analysis_markers(markers)

    return markers


def stack_samples(
    samples: List[FlowSample],
    selected_markers: List[str],
    seed: int = 42,
    max_cells_total: Optional[int] = None,
    balance_conditions: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Empile plusieurs FlowSample en une matrice unique pour FlowSOM.

    Args:
        samples: Liste des échantillons prétraités.
        selected_markers: Marqueurs à inclure.
        seed: Graine pour le sous-échantillonnage d'équilibrage.
        max_cells_total: Limite totale de cellules (sous-échantillonnage global).
        balance_conditions: Si True, limite chaque condition au même nombre de cellules.

    Returns:
        Tuple (X_stacked [n_cells, n_markers], obs_metadata DataFrame).
    """
    all_X: List[np.ndarray] = []
    all_obs: List[Dict] = []

    for sample in samples:
        X_s = sample.matrix
        var_s = sample.var_names

        # Sélectionner et réordonner les marqueurs
        col_idx = [var_s.index(m) for m in selected_markers if m in var_s]
        if len(col_idx) != len(selected_markers):
            missing = [m for m in selected_markers if m not in var_s]
            _logger.warning(
                "%s: marqueurs manquants %s — rempli avec 0",
                sample.name,
                missing,
            )

        X_sel = np.zeros((X_s.shape[0], len(selected_markers)), dtype=np.float64)
        for j, m in enumerate(selected_markers):
            if m in var_s:
                X_sel[:, j] = X_s[:, var_s.index(m)]

        all_X.append(X_sel)

        for _ in range(X_s.shape[0]):
            all_obs.append(
                {
                    "condition": sample.condition,
                    "file_origin": sample.name,
                }
            )

    if not all_X:
        raise ValueError("Aucun échantillon valide à empiler")

    X = np.vstack(all_X)
    obs = pd.DataFrame(all_obs)

    # Vérification d'équilibre (info seulement — le déséquilibre Sain/Patho est voulu :
    # les NBM servent de référence calibrée et doivent dominer)
    if "condition" in obs.columns:
        conditions = obs["condition"].unique()
        if len(conditions) == 2:
            n1 = int((obs["condition"] == conditions[0]).sum())
            n2 = int((obs["condition"] == conditions[1]).sum())
            ratio = max(n1, n2) / max(min(n1, n2), 1)
            _logger.info(
                "Ratio cellules %s/%s : %.1f× (%d vs %d) — déséquilibre intentionnel NBM/Patho",
                conditions[0],
                conditions[1],
                ratio,
                n1,
                n2,
            )

    # Sous-échantillonnage global si nécessaire
    if max_cells_total and X.shape[0] > max_cells_total:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_cells_total, replace=False)
        X = X[idx]
        obs = obs.iloc[idx].reset_index(drop=True)
        _logger.info(
            "Matrice sous-échantillonnée à %d/%d cellules",
            max_cells_total,
            X.shape[0] + max_cells_total,
        )

    # Contrôle NaN final
    n_nan = check_nan(X)
    if n_nan > 0:
        _logger.warning("%d NaN dans la matrice finale — remplacement par 0", n_nan)
        X = np.nan_to_num(X, nan=0.0)

    _logger.info(
        "Matrice FlowSOM: %d cellules × %d marqueurs",
        X.shape[0],
        X.shape[1],
    )
    return X, obs


def stack_raw_markers(
    samples: List[FlowSample],
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Empile les données brutes pré-transformation (raw_data) pour l'export FCS.

    Utilise ``sample.raw_data`` (valeurs linéaires, avant arcsinh/logicle) si
    disponible. Sinon, replie sur les données transformées (``sample.matrix``).
    Cette matrice est utilisée dans le fichier FCS final pour compatibilité
    Kaluza/FlowJo — les logiciels appliquent eux-mêmes leur transformation
    d'affichage à partir des valeurs brutes.

    Returns:
        Tuple (X_raw [n_cells, n_all_markers], raw_var_names, obs_df).
    """
    if not samples:
        raise ValueError("Aucun échantillon valide")

    has_raw = all(s.raw_data is not None for s in samples)
    if not has_raw:
        _logger.warning(
            "raw_data absent dans un ou plusieurs FlowSample — "
            "utilisation des données transformées pour l'export FCS"
        )

    all_X: List[np.ndarray] = []
    all_obs: List[Dict] = []

    if has_raw:
        raw_var_names: List[str] = list(samples[0].raw_data.columns)
        for s in samples:
            X_s = s.raw_data.values.astype(np.float64)
            all_X.append(X_s)
            for _ in range(X_s.shape[0]):
                all_obs.append({"condition": s.condition, "file_origin": s.name})
    else:
        raw_var_names = list(samples[0].var_names)
        for s in samples:
            all_X.append(s.matrix.astype(np.float64))
            for _ in range(s.matrix.shape[0]):
                all_obs.append({"condition": s.condition, "file_origin": s.name})

    X = np.vstack(all_X)
    obs = pd.DataFrame(all_obs)
    return X, raw_var_names, obs


def run_clustering(
    samples: List[FlowSample],
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, FlowSOMClusterer, List[str]]:
    """
    Exécute le clustering FlowSOM complet sur la liste d'échantillons.

    Args:
        samples: Échantillons prétraités.
        config: Configuration du pipeline.

    Returns:
        Tuple (metaclustering [n_cells], clustering [n_cells], clusterer, selected_markers).
    """
    if not samples:
        raise ValueError("Aucun échantillon à clusteriser")

    # Sélection des marqueurs
    selected_markers = select_markers_for_clustering(samples[0].var_names, config)
    _logger.info(
        "Marqueurs pour FlowSOM (%d): %s", len(selected_markers), selected_markers
    )

    # Empilement des matrices
    flowsom_cfg = config.flowsom
    X, obs = stack_samples(
        samples,
        selected_markers,
        seed=config.flowsom.seed,
        max_cells_total=getattr(config.downsampling, "max_cells_for_clustering", None),
    )

    # Auto-clustering (trouver le k optimal)
    n_clusters = flowsom_cfg.n_metaclusters
    if getattr(config.auto_clustering, "enabled", False):
        _logger.info("Auto-clustering activé — recherche du k optimal...")
        n_clusters = find_optimal_clusters(
            X,
            min_clusters=getattr(config.auto_clustering, "min_clusters", 5),
            max_clusters=getattr(config.auto_clustering, "max_clusters", 35),
            n_bootstrap=getattr(config.auto_clustering, "n_bootstrap", 10),
            sample_size_bootstrap=getattr(
                config.auto_clustering, "sample_size_bootstrap", 20_000
            ),
            min_stability_threshold=getattr(
                config.auto_clustering, "min_stability_threshold", 0.75
            ),
            weight_stability=getattr(config.auto_clustering, "weight_stability", 0.65),
            weight_silhouette=getattr(
                config.auto_clustering, "weight_silhouette", 0.35
            ),
            xdim=flowsom_cfg.xdim,
            ydim=flowsom_cfg.ydim,
            seed=config.flowsom.seed,
            verbose=True,
        )
        _logger.info("k optimal trouvé: %d", n_clusters)
    elif n_clusters is None:
        n_clusters = 10  # défaut

    # FlowSOM
    clusterer = FlowSOMClusterer(
        xdim=flowsom_cfg.xdim,
        ydim=getattr(flowsom_cfg, "ydim", flowsom_cfg.xdim),
        n_metaclusters=n_clusters,
        seed=config.flowsom.seed,
        rlen=getattr(flowsom_cfg, "rlen", "auto"),
        use_gpu=getattr(config.gpu, "enabled", False),
    )

    _logger.info(
        "FlowSOM: grille %d×%d, %d métaclusters...",
        clusterer.xdim,
        clusterer.ydim,
        n_clusters,
    )
    clusterer.fit(X, selected_markers)
    metaclustering = getattr(
        clusterer, "metacluster_assignments_", np.zeros(X.shape[0], dtype=int)
    )
    clustering = getattr(
        clusterer, "node_assignments_", np.zeros(X.shape[0], dtype=int)
    )

    return metaclustering, clustering, clusterer, selected_markers


def extract_date_from_filename(filename: str) -> Tuple[str, int]:
    """
    Extrait la date/timepoint depuis le nom de fichier FCS.

    Essaie successivement:
    1. DD-MM-YYYY ou DD/MM/YYYY
    2. YYYY-MM-DD ou YYYY/MM/DD
    3. DD-MM-YY ou DD/MM/YY
    4. DDMMYYYY (8 chiffres collés)

    Args:
        filename: Nom de fichier (sans chemin ou avec).

    Returns:
        Tuple (timepoint_str, timepoint_num) où timepoint_num est
        un entier YYYYMMDD (0 si non trouvé).
    """
    import re
    from pathlib import Path as _Path

    stem = _Path(filename).stem

    patterns = [
        r"(\d{2})[-/](\d{2})[-/](\d{4})",  # DD-MM-YYYY
        r"(\d{4})[-/](\d{2})[-/](\d{2})",  # YYYY-MM-DD
        r"(\d{2})[-/](\d{2})[-/](\d{2})",  # DD-MM-YY
        r"(\d{8})",  # DDMMYYYY ou YYYYMMDD
    ]

    for pattern in patterns:
        m = re.search(pattern, stem)
        if m is None:
            continue
        groups = m.groups()
        try:
            if len(groups) == 3:
                g1, g2, g3 = groups
                if len(g3) == 4:
                    # DD-MM-YYYY
                    tp_str = f"{g1}/{g2}/{g3}"
                    tp_num = int(f"{g3}{g2}{g1}")
                elif len(g1) == 4:
                    # YYYY-MM-DD
                    tp_str = f"{g3}/{g2}/{g1}"
                    tp_num = int(f"{g1}{g2}{g3}")
                else:
                    # DD-MM-YY → 20YY
                    year = int(g3) + (2000 if int(g3) < 50 else 1900)
                    tp_str = f"{g1}/{g2}/{year}"
                    tp_num = year * 10000 + int(g2) * 100 + int(g1)
            else:
                # 8 digits
                raw = groups[0]
                if len(raw) == 8:
                    # Heuristique: YYYY en tête si raw[0:4] > 1900
                    if 1900 < int(raw[:4]) < 2100:
                        tp_str = f"{raw[6:8]}/{raw[4:6]}/{raw[:4]}"
                        tp_num = int(raw)
                    else:
                        tp_str = f"{raw[:2]}/{raw[2:4]}/{raw[4:8]}"
                        tp_num = int(f"{raw[4:8]}{raw[2:4]}{raw[:2]}")
                else:
                    continue
            return tp_str, tp_num
        except (ValueError, IndexError):
            continue

    return "", 0


def build_cells_dataframe(
    X_raw: np.ndarray,
    var_names: List[str],
    obs: pd.DataFrame,
    metaclustering: np.ndarray,
    clustering: np.ndarray,
) -> pd.DataFrame:
    """
    Construit le DataFrame cellulaire complet (marqueurs + métadonnées + clustering).

    Ajoute automatiquement des colonnes Timepoint/Timepoint_Num extraites
    du nom de fichier FCS (DD/MM/YYYY et entier YYYYMMDD).

    Args:
        X_raw: Matrice des marqueurs (n_cells, n_markers).
        var_names: Noms des colonnes.
        obs: Métadonnées des cellules (condition, file_origin).
        metaclustering: Assignation métacluster (n_cells,).
        clustering: Assignation cluster SOM (n_cells,).

    Returns:
        DataFrame complet.
    """
    df = pd.DataFrame(X_raw, columns=var_names)
    df["condition"] = (
        obs["condition"].values if "condition" in obs.columns else "Unknown"
    )
    df["file_origin"] = (
        obs["file_origin"].values if "file_origin" in obs.columns else ""
    )
    df["FlowSOM_metacluster"] = metaclustering.astype(np.int32)
    df["FlowSOM_cluster"] = clustering.astype(np.int32)

    # Encodage numérique de la condition pour export FCS
    conditions_unique = sorted(df["condition"].unique())
    cond_to_int = {c: i + 1 for i, c in enumerate(conditions_unique)}
    df["Condition_Num"] = df["condition"].map(cond_to_int).astype(np.float32)
    df["Condition"] = df["condition"]

    # Extraction du timepoint depuis le nom de fichier (suivi longitudinal MRD)
    if "file_origin" in df.columns:
        tp_str_list = []
        tp_num_list = []
        for fname in df["file_origin"]:
            tp_str, tp_num = extract_date_from_filename(str(fname))
            tp_str_list.append(tp_str)
            tp_num_list.append(tp_num)
        df["Timepoint"] = tp_str_list
        df["Timepoint_Num"] = np.array(tp_num_list, dtype=np.int64)

    return df
