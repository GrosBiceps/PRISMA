"""
population_mapping.py — Assignation de populations aux nœuds FlowSOM.

Implémente l'algorithme de mapping par distance euclidienne minimale:
pour chaque nœud SOM, trouve la population de référence la plus proche
dans l'espace MFI normalisé.

Référence: ELN 2022 — chaque nœud doit être associé à une population
d'intérêt ou marqué "Unknown" si trop éloigné de toutes les références.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.spatial.distance import cdist

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    warnings.warn("scipy requis pour map_populations_to_nodes: pip install scipy")

from flowsom_pipeline_pro.src.utils.logger import get_logger

_logger = get_logger("analysis.population_mapping")

# Préfixes de canaux scatter à optionnellement exclure du calcul de distance
_SCATTER_PREFIXES = ("FSC", "SSC", "TIME", "WIDTH", "AREA", "HEIGHT")


def normalize_matrix(
    X: np.ndarray,
    method: str = "range",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalise une matrice de MFI par colonne.

    Args:
        X: Matrice (n_nodes, n_markers).
        method: "range" (min-max), "zscore" (mean/std), ou "none".

    Returns:
        Tuple (X_normalized, scale_params).
    """
    if method == "none":
        return X.copy(), {}

    if method == "range":
        col_min = X.min(axis=0)
        col_max = X.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range == 0] = 1.0  # éviter div/0
        X_norm = (X - col_min) / col_range
        return X_norm, {"min": col_min, "max": col_max, "range": col_range}

    if method == "zscore":
        col_mean = X.mean(axis=0)
        col_std = X.std(axis=0)
        col_std[col_std == 0] = 1.0
        X_norm = (X - col_mean) / col_std
        return X_norm, {"mean": col_mean, "std": col_std}

    raise ValueError(
        f"Méthode de normalisation inconnue: {method!r}. Utiliser 'range', 'zscore' ou 'none'."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Filtrage strict des canaux -A pour le mapping
# ─────────────────────────────────────────────────────────────────────────────


def filter_area_channels(
    columns: List[str],
) -> List[str]:
    """
    Filtre les colonnes pour ne garder que les canaux -A (fluorescence Area)
    et exclure FSC-Width, TIME, EVENT_COUNT, etc.

    Le mapping ne doit opérer que sur les canaux de fluorescence -A pour
    que les profils d'expression soient comparables entre nœuds et référence.

    Args:
        columns: Liste de noms de colonnes.

    Returns:
        Colonnes filtrées.
    """
    filtered: List[str] = []
    for c in columns:
        cu = c.upper()
        # Exclure les scatter/time/event
        if any(cu.startswith(p) for p in _SCATTER_PREFIXES):
            continue
        # Exclure les suffixes non -A (FSC-Width etc.)
        if any(cu.endswith(s) for s in _NON_A_SUFFIXES_TO_EXCLUDE):
            continue
        # Si le canal se termine par -H et son -A existe, l'exclure
        if cu.endswith("-H"):
            prefix = cu[:-2]
            if any(col.upper() == prefix + "-A" for col in columns):
                continue
        filtered.append(c)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
#  Transformation cytométrique des MFI de référence + cache Parquet
# ─────────────────────────────────────────────────────────────────────────────


def _cache_key(csv_path: str, method: str, cofactor: float) -> str:
    """Génère une clé de cache unique basée sur le fichier et les paramètres."""
    raw = f"{csv_path}|{method}|{cofactor}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def transform_reference_mfi(
    pop_mfi_ref: pd.DataFrame,
    method: str = "arcsinh",
    cofactor: float = 5.0,
    cache_dir: Optional[Path] = None,
    csv_source_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Applique la transformation cytométrique (arcsinh/logicle) aux données de
    référence MFI AVANT le calcul de distance.

    Rationale: les fichiers CSV de référence contiennent les MFI brutes
    (échelle linéaire). Le réseau FlowSOM opère sur des données transformées
    (logicle/arcsinh). Pour que la distance cosine soit cohérente, il faut
    que la référence soit dans le même espace que les nœuds.

    Si un cache_dir est spécifié, les matrices transformées sont sauvegardées
    en Parquet pour réutilisation rapide (évite de recalculer à chaque run).

    Args:
        pop_mfi_ref: DataFrame [n_populations × n_markers] — MFI brutes de référence.
        method: "arcsinh" | "logicle" | "none".
        cofactor: Cofacteur pour arcsinh (default 5.0 pour la cytométrie en flux).
        cache_dir: Répertoire de cache Parquet (optionnel).
        csv_source_path: Chemin du CSV source (pour la clé de cache).

    Returns:
        DataFrame transformé (même shape).
    """
    if method == "none":
        return pop_mfi_ref.copy()

    # ── Cache Parquet ─────────────────────────────────────────────────────────
    if cache_dir is not None and csv_source_path is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(csv_source_path, method, cofactor)
        parquet_path = cache_dir / f"ref_mfi_{key}.parquet"

        if parquet_path.exists():
            _logger.info(
                "Cache Parquet trouvé: %s — chargement des MFI pré-transformées",
                parquet_path.name,
            )
            return pd.read_parquet(parquet_path)

    # ── Transformation ────────────────────────────────────────────────────────
    result = pop_mfi_ref.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    if method == "arcsinh":
        _logger.info(
            "Transformation arcsinh (cofacteur=%s) sur %d marqueurs de référence",
            cofactor,
            len(numeric_cols),
        )
        result[numeric_cols] = np.arcsinh(result[numeric_cols].values / cofactor)

    elif method == "logicle":
        # Approximation logicle via arcsinh biexponentielle quand FlowKit
        # n'est pas disponible pour les données tabulaires de référence
        _logger.info(
            "Transformation logicle (approx. arcsinh, cofacteur=%s) sur référence",
            cofactor,
        )
        result[numeric_cols] = np.arcsinh(result[numeric_cols].values / cofactor)

    else:
        _logger.warning(
            "Méthode de transformation inconnue '%s' — aucune appliquée", method
        )
        return pop_mfi_ref.copy()

    # ── Sauvegarder dans le cache Parquet ─────────────────────────────────────
    if cache_dir is not None and csv_source_path is not None:
        try:
            result.to_parquet(parquet_path, engine="pyarrow")
            _logger.info("Cache Parquet sauvegardé: %s", parquet_path.name)
        except Exception as exc:
            _logger.warning("Sauvegarde cache Parquet échouée: %s", exc)

    return result


def map_populations_to_nodes_v3(
    node_mfi_raw: pd.DataFrame,
    pop_mfi_ref: pd.DataFrame,
    include_scatter: bool = True,
    distance_percentile: int = 60,
    normalization_method: str = "range",
) -> pd.DataFrame:
    """
    Assigne une population à chaque nœud SOM par distance euclidienne minimale.

    Opère dans l'espace MFI normalisé. Les colonnes de node_mfi_raw et
    pop_mfi_ref DOIVENT être identiques (mapper en amont si nécessaire).

    Le seuil "Unknown" est le percentile distance_percentile de la
    distribution des distances minimales: les nœuds périphériques
    (éloignés de toutes les références) sont marqués "Unknown".

    Args:
        node_mfi_raw: DataFrame [n_nodes × n_markers] — MFI par nœud SOM.
        pop_mfi_ref: DataFrame [n_populations × n_markers] — MFI de référence.
        include_scatter: Inclure FSC-A/SSC-A dans le calcul de distance.
        distance_percentile: Percentile de seuil pour "Unknown" (60 = top 40% distant).
        normalization_method: "range", "zscore", ou "none".

    Returns:
        DataFrame: [node_id, best_pop, best_dist, threshold, assigned_pop].
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy requis: pip install scipy")

    # Vérification de cohérence des colonnes
    cols_node = set(node_mfi_raw.columns)
    cols_ref = set(pop_mfi_ref.columns)
    if cols_node != cols_ref:
        extra_node = cols_node - cols_ref
        extra_ref = cols_ref - cols_node
        raise ValueError(
            f"Colonnes non alignées entre node_mfi_raw et pop_mfi_ref.\n"
            f"node uniquement: {extra_node}\nref uniquement: {extra_ref}\n"
            "Aligner les colonnes via les colonnes communes avant d'appeler cette fonction."
        )

    # Sélection des colonnes pour le calcul de distance
    if include_scatter:
        cols_for_dist = list(node_mfi_raw.columns)
    else:
        cols_for_dist = [
            c
            for c in node_mfi_raw.columns
            if not any(c.upper().startswith(p) for p in _SCATTER_PREFIXES)
        ]

    _logger.info(
        "Mapping populations → nœuds: %d nœuds, %d populations, %d marqueurs",
        len(node_mfi_raw),
        len(pop_mfi_ref),
        len(cols_for_dist),
    )

    X_nodes = node_mfi_raw[cols_for_dist].to_numpy(dtype=np.float64)
    X_pops = pop_mfi_ref[cols_for_dist].to_numpy(dtype=np.float64)

    # Normalisation: calculée sur les nœuds, appliquée aux deux matrices
    if normalization_method != "none":
        X_nodes_norm, scale = normalize_matrix(X_nodes, method=normalization_method)

        if normalization_method == "range":
            X_pops_norm = (X_pops - scale["min"]) / scale["range"]
        elif normalization_method == "zscore":
            X_pops_norm = (X_pops - scale["mean"]) / scale["std"]
        else:
            X_pops_norm = X_pops
    else:
        X_nodes_norm = X_nodes
        X_pops_norm = X_pops
        _logger.warning(
            "Normalisation='none' — les canaux à grande plage dominent la distance!"
        )

    # Matrice de distances euclidiennes (n_nodes, n_populations)
    dist_matrix = cdist(X_nodes_norm, X_pops_norm, metric="euclidean")
    best_idx = np.argmin(dist_matrix, axis=1)
    best_dist = dist_matrix[np.arange(len(X_nodes_norm)), best_idx]
    threshold = float(np.percentile(best_dist, distance_percentile))

    pop_names = list(pop_mfi_ref.index)
    assigned = np.where(
        best_dist <= threshold,
        np.array([pop_names[i] for i in best_idx]),
        "Unknown",
    )

    result = pd.DataFrame(
        {
            "node_id": np.arange(len(X_nodes_norm)),
            "best_pop": [pop_names[i] for i in best_idx],
            "best_dist": np.round(best_dist, 4),
            "threshold": round(threshold, 4),
            "assigned_pop": assigned,
        }
    )

    n_tot = len(result)
    n_unk = int((result["assigned_pop"] == "Unknown").sum())
    _logger.info(
        "Mapping terminé: %d nœuds assignés (%d%%), %d Unknown (%d%%)",
        n_tot - n_unk,
        round(100 * (n_tot - n_unk) / n_tot),
        n_unk,
        round(100 * n_unk / n_tot),
    )

    return result


def map_nodes_to_metaclusters(
    mapping_df: pd.DataFrame,
    metaclustering_per_node: np.ndarray,
) -> pd.DataFrame:
    """
    Enrichit le mapping nœuds→populations avec l'assignation de métacluster dominant.

    Args:
        mapping_df: Résultat de map_populations_to_nodes_v3.
        metaclustering_per_node: Vecteur (n_nodes,) avec l'id de métacluster.

    Returns:
        mapping_df enrichi avec colonnes "metacluster" et "dominant_metacluster".
    """
    result = mapping_df.copy()
    result["metacluster"] = np.array(
        [
            metaclustering_per_node[nid] if nid < len(metaclustering_per_node) else -1
            for nid in result["node_id"]
        ]
    )
    return result


def get_population_summary(mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Résumé de la distribution des nœuds par population assignée.

    Args:
        mapping_df: Résultat de map_populations_to_nodes_v3 (avec "assigned_pop").

    Returns:
        DataFrame avec [population, n_nodes, pct_nodes].
    """
    counts = mapping_df["assigned_pop"].value_counts().reset_index()
    counts.columns = ["population", "n_nodes"]
    counts["pct_nodes"] = (counts["n_nodes"] / len(mapping_df) * 100).round(2)
    return counts


def build_population_color_map(
    populations: List[str],
    default_color: str = "#AAAAAA",
) -> Dict[str, str]:
    """
    Construit un dictionnaire population → couleur pour la visualisation.

    Args:
        populations: Liste des populations connues.
        default_color: Couleur par défaut pour les populations non listées.

    Returns:
        Dict {population: hex_color}.
    """
    # Couleurs standard pour les populations cytométriques
    STANDARD_COLORS = {
        "Granulo": "#FF6B6B",
        "Granulocytes": "#FF6B6B",
        "Mono": "#FFA500",
        "Monocytes": "#FFA500",
        "T_NK": "#4ECDC4",
        "T/NK": "#4ECDC4",
        "B": "#45B7D1",
        "HematI": "#96CEB4",
        "HematogenesI": "#96CEB4",
        "Plasmo": "#DDA0DD",
        "Plasmocytes": "#DDA0DD",
        "Blastes": "#FF0000",
        "Blasts": "#FF0000",
        "Unknown": "#AAAAAA",
    }

    return {pop: STANDARD_COLORS.get(pop, default_color) for pop in populations}


# ─────────────────────────────────────────────────────────────────────────────
#  Population mapping V5 — distance cosine + prior bayésien log10³
# ─────────────────────────────────────────────────────────────────────────────


def _apply_bayesian_prior(
    dist_matrix: np.ndarray,
    pop_names: List[str],
    cell_counts: Any,
    prior_mode: str = "log10_cubed",
    node_sizes: Optional[np.ndarray] = None,
    hard_limit_factor: float = 5.0,
    n_nodes_total: Optional[int] = None,
) -> np.ndarray:
    """
    Ajuste la matrice de distances par un prior bayésien de prévalence cellulaire.

    Les populations fréquentes (Granulocytes 2.2M) sont renduees plus attractives
    que les rares (Plasmocytes 2000). Avec `log10_cubed`, le ratio d'attractivité
    est ≈7× entre ces deux extrêmes — cohérent avec la biologie de la moelle.

    Hard limit : si une large fraction d'un nœud excède ce que la population
    peut biologiquement contenir, la distance est mise à ∞ (rejet ELN).

    Args:
        dist_matrix: Matrice (n_nodes, n_populations).
        pop_names: Noms des populations dans l'ordre des colonnes.
        cell_counts: dict {pop_name: n_cells} ou array de tailles.
        prior_mode: "log10_cubed" | "log10" | "linear" | "none".
        node_sizes: Taille (n_cells) de chaque nœud SOM — pour le hard limit.
        hard_limit_factor: Facteur multiplicatif du max biologique attendu.
        n_nodes_total: Nombre de nœuds (utilisé pour calculer la part théorique).

    Returns:
        Matrice D ajustée (copie).
    """
    D = dist_matrix.copy()

    if isinstance(cell_counts, dict):
        counts = np.array([cell_counts.get(p, 1) for p in pop_names], dtype=np.float64)
    else:
        counts = np.asarray(cell_counts, dtype=np.float64)
        if len(counts) < len(pop_names):
            counts = np.ones(len(pop_names), dtype=np.float64)

    total_cells = float(max(counts.sum(), 1))

    for _j, (_pname, _n) in enumerate(zip(pop_names, counts)):
        # Poids d'attractivité proportionnel à la prévalence
        if prior_mode == "log10_cubed":
            _w = np.log10(max(_n, 10)) ** 3
        elif prior_mode == "log10":
            _w = np.log10(max(_n, 10))
        elif prior_mode == "linear":
            _w = max(_n, 1) / max(total_cells / max(len(pop_names), 1), 1)
        else:
            _w = 1.0

        D[:, _j] /= max(_w, 1e-9)

        # Hard limit : un nœud über-représenté ne peut pas aller dans cette pop.
        if node_sizes is not None and n_nodes_total is not None:
            n_nodes_total_eff = max(n_nodes_total, 1)
            expected_max = (
                (_n / total_cells) * float(node_sizes.sum()) * hard_limit_factor
            )
            too_large_mask = node_sizes > expected_max
            D[too_large_mask, _j] = np.inf

    return D


def compute_unknown_threshold(
    best_dist: np.ndarray,
    threshold_mode: str = "percentile",
    percentile: float = 75.0,
) -> float:
    """
    Calcule le seuil distance au-delà duquel un nœud est classé "Unknown".

    Args:
        best_dist: Vecteur des distances minimales par nœud.
        threshold_mode: "percentile" | "mean_std" | "median_iqr".
        percentile: Percentile utilisé en mode "percentile".

    Returns:
        Seuil (float).
    """
    if threshold_mode == "percentile":
        return float(np.percentile(best_dist, percentile))
    elif threshold_mode == "mean_std":
        return float(np.mean(best_dist) + np.std(best_dist))
    elif threshold_mode == "median_iqr":
        q1, q3 = np.percentile(best_dist, [25.0, 75.0])
        return float(q3 + 1.5 * (q3 - q1))
    else:
        return float(np.percentile(best_dist, percentile))


def assign_with_auto_threshold(
    dist_matrix: np.ndarray,
    pop_names: List[str],
    threshold_mode: str = "percentile",
    percentile: float = 75.0,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Assigne chaque nœud à la population la plus proche et calcule le seuil Unknown.

    Returns:
        Tuple (best_idx, best_dist, threshold, assigned_labels).
    """
    best_idx = np.argmin(dist_matrix, axis=1)
    best_dist = dist_matrix[np.arange(len(dist_matrix)), best_idx]

    # Masquer les inf (nœuds rejetés par le hard limit sur toutes les populations)
    finite_mask = np.isfinite(best_dist)
    if finite_mask.sum() == 0:
        _logger.warning(
            "Tous les nœuds ont une distance infinie — vérifier le hard_limit_factor."
        )
        return (
            best_idx,
            best_dist,
            np.inf,
            np.full(len(dist_matrix), "Unknown", dtype=object),
        )

    threshold = compute_unknown_threshold(
        best_dist[finite_mask], threshold_mode=threshold_mode, percentile=percentile
    )
    assigned = np.where(
        finite_mask & (best_dist <= threshold),
        np.array([pop_names[i] for i in best_idx], dtype=object),
        "Unknown",
    )
    return best_idx, best_dist, threshold, assigned


def map_populations_to_nodes_v5(
    node_mfi_raw: pd.DataFrame,
    pop_mfi_ref: pd.DataFrame,
    node_sizes: Optional[np.ndarray] = None,
    cell_counts: Optional[Any] = None,
    method: str = "cosine_prior",
    include_scatter: bool = False,
    threshold_mode: str = "percentile",
    threshold_percentile: float = 75.0,
    normalization_method: str = "range",
    hard_limit_factor: float = 5.0,
    prior_mode: str = "log10_cubed",
    # V5 — Transformation cytométrique avant MFI
    transform_method: str = "none",
    arcsinh_cofactor: float = 5.0,
    data_already_transformed: bool = False,
    # V5 — Cache Parquet
    cache_dir: Optional[Path] = None,
    csv_source_path: Optional[str] = None,
    # V5 — Filtrage strict -A
    filter_area_only: bool = True,
) -> pd.DataFrame:
    """
    Algorithme V5 — assignation populations → nœuds SOM.

    Méthodes disponibles:
    - "cosine_prior" (M12, recommandé): distance cosine sur données ≥0 +
      prior bayésien log10³ + hard limit biologique. Robuste aux différences
      d'intensité entre tubes et patients.
    - "euclidean_prior" (M11): euclidenne normalisée + prior.
    - "euclidean" (M1): comportement identique à v3, compatibilité ascendante.

    L'algorithme v5 diffère de v3 sur trois points clés:
    1. Distance cosine (insensible aux différences d'intensité absolue).
    2. Prior bayésien: les populations fréquentes capturent plus de nœuds,
       reflétant la biologie (Granulos >> Plasmos dans la moelle).
    3. Hard limit: interdit à un nœud sur-représenté d'aller dans une pop rare.

    Args:
        node_mfi_raw: DataFrame [n_nodes × n_markers] — MFI par nœud.
        pop_mfi_ref: DataFrame [n_populations × n_markers] — MFI de référence.
        node_sizes: Taille (n_cells) de chaque nœud — requis pour hard_limit.
        cell_counts: dict {pop: n_cells} ou array — requis pour prior.
        method: "cosine_prior" | "euclidean_prior" | "euclidean".
        include_scatter: Inclure FSC/SSC/Time dans la distance.
        threshold_mode: "percentile" | "mean_std" | "median_iqr".
        threshold_percentile: Percentile pour le seuil Unknown (default 75).
        normalization_method: "range" | "zscore" | "none".
        hard_limit_factor: Facteur multiplicatif du max biologique (default 5.0).
        prior_mode: "log10_cubed" | "log10" | "linear" (utilisé si prior actif).
        transform_method: "arcsinh" | "logicle" | "none" — transformation des MFI de référence.
        arcsinh_cofactor: Cofacteur pour arcsinh (5.0 flow standard).
        data_already_transformed: Si True, les données de référence sont déjà transformées.
        cache_dir: Répertoire de cache Parquet pour les matrices de référence transformées.
        csv_source_path: Chemin du CSV source pour la clé de cache.
        filter_area_only: Si True, n'utilise que les canaux -A (et ignore -H, -W, TIME).

    Returns:
        DataFrame: [node_id, best_pop, best_dist, threshold, assigned_pop].
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy requis: pip install scipy")

    # ── Transformation des MFI de référence (si nécessaire) ──────────────────
    if transform_method != "none" and not data_already_transformed:
        pop_mfi_ref = transform_reference_mfi(
            pop_mfi_ref,
            method=transform_method,
            cofactor=arcsinh_cofactor,
            cache_dir=cache_dir,
            csv_source_path=csv_source_path,
        )
        _logger.info(
            "MFI de référence transformées (%s, cofacteur=%s)",
            transform_method,
            arcsinh_cofactor,
        )

    # ── Alignement des colonnes ───────────────────────────────────────────────
    cols_common = list(
        dict.fromkeys(  # préserver l'ordre
            c for c in node_mfi_raw.columns if c in pop_mfi_ref.columns
        )
    )
    if not cols_common:
        raise ValueError(
            "Aucune colonne commune entre node_mfi_raw et pop_mfi_ref. "
            "Aligner les marqueurs via les colonnes communes avant d'appeler cette fonction."
        )

    if include_scatter:
        cols_for_dist = cols_common
    else:
        cols_for_dist = [
            c
            for c in cols_common
            if not any(c.upper().startswith(p) for p in _SCATTER_PREFIXES)
        ]

    # V5 — Filtrage strict des canaux -A (exclut -H, -W, TIME, EVENT)
    if filter_area_only:
        cols_for_dist = filter_area_channels(cols_for_dist)

    if not cols_for_dist:
        _logger.warning(
            "Aucun marqueur après exclusion scatter — utilisation de toutes les colonnes."
        )
        cols_for_dist = cols_common

    pop_names = list(pop_mfi_ref.index)
    _logger.info(
        "Mapping V5 [%s]: %d nœuds, %d populations, %d marqueurs",
        method,
        len(node_mfi_raw),
        len(pop_mfi_ref),
        len(cols_for_dist),
    )

    X_nodes_raw = node_mfi_raw[cols_for_dist].to_numpy(dtype=np.float64)
    X_pops_raw = pop_mfi_ref[cols_for_dist].to_numpy(dtype=np.float64)

    # ── node_sizes par défaut si absent ──────────────────────────────────────
    n_nodes = len(X_nodes_raw)
    _node_sizes = node_sizes
    if _node_sizes is None:
        _node_sizes = np.ones(n_nodes, dtype=np.float64)

    # ── cell_counts par défaut si absent ─────────────────────────────────────
    _cell_counts: Any
    if cell_counts is None:
        _cell_counts = {p: 1 for p in pop_names}
        _logger.warning(
            "cell_counts absent — prior bayésien désactivé (toutes les pops équiprobables)."
        )
    else:
        _cell_counts = cell_counts

    # ─────────────────────────────────────────────────────────────────────────
    if method == "cosine_prior":
        # M12 — Cosine + prior log10³ (méthode recommandée en usage clinique)
        # Les données cytométriques après logicle peuvent avoir de petites valeurs
        # négatives (bruit de fond). Le cosine demande des vecteurs positifs.
        X_n_clip = np.clip(X_nodes_raw, 0.0, None)
        X_p_clip = np.clip(X_pops_raw, 0.0, None)

        dist_matrix = cdist(X_n_clip, X_p_clip, metric="cosine")
        dist_matrix = _apply_bayesian_prior(
            dist_matrix,
            pop_names,
            _cell_counts,
            prior_mode=prior_mode,
            node_sizes=_node_sizes,
            hard_limit_factor=hard_limit_factor,
            n_nodes_total=n_nodes,
        )

    elif method == "euclidean_prior":
        # M11 — Euclidienne normalisée + prior (intermédiaire pour comparaison)
        if normalization_method != "none":
            X_nodes_norm, scale = normalize_matrix(
                X_nodes_raw, method=normalization_method
            )
            if normalization_method == "range":
                X_pops_norm = (X_pops_raw - scale["min"]) / scale["range"]
            else:
                X_pops_norm = (X_pops_raw - scale["mean"]) / scale["std"]
        else:
            X_nodes_norm, X_pops_norm = X_nodes_raw, X_pops_raw

        dist_matrix = cdist(X_nodes_norm, X_pops_norm, metric="euclidean")
        dist_matrix = _apply_bayesian_prior(
            dist_matrix,
            pop_names,
            _cell_counts,
            prior_mode=prior_mode,
            node_sizes=_node_sizes,
            hard_limit_factor=hard_limit_factor,
            n_nodes_total=n_nodes,
        )

    else:
        # M1 / "euclidean" — comportement identique à v3 pour compatibilité
        if normalization_method != "none":
            X_nodes_norm, scale = normalize_matrix(
                X_nodes_raw, method=normalization_method
            )
            if normalization_method == "range":
                X_pops_norm = (X_pops_raw - scale["min"]) / scale["range"]
            else:
                X_pops_norm = (X_pops_raw - scale["mean"]) / scale["std"]
        else:
            X_nodes_norm, X_pops_norm = X_nodes_raw, X_pops_raw

        dist_matrix = cdist(X_nodes_norm, X_pops_norm, metric="euclidean")

    # ── Assignation + calcul du seuil Unknown ────────────────────────────────
    best_idx, best_dist, threshold, assigned = assign_with_auto_threshold(
        dist_matrix,
        pop_names,
        threshold_mode=threshold_mode,
        percentile=threshold_percentile,
    )

    result = pd.DataFrame(
        {
            "node_id": np.arange(n_nodes),
            "best_pop": [pop_names[i] for i in best_idx],
            "best_dist": np.round(best_dist, 6),
            "threshold": round(threshold, 6),
            "assigned_pop": assigned,
        }
    )

    n_tot = len(result)
    n_unk = int((result["assigned_pop"] == "Unknown").sum())
    _logger.info(
        "V5 mapping terminé [%s]: %d/%d assignés (%d%%), %d Unknown (%d%%)",
        method,
        n_tot - n_unk,
        n_tot,
        round(100 * (n_tot - n_unk) / max(n_tot, 1)),
        n_unk,
        round(100 * n_unk / max(n_tot, 1)),
    )

    return result
