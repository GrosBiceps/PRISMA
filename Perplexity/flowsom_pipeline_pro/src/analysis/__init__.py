"""
src/analysis/__init__.py — Exports publics de la couche analyse.
"""

from .population_mapping import (
    map_populations_to_nodes_v3,
    map_populations_to_nodes_v5,
    map_nodes_to_metaclusters,
    get_population_summary,
    normalize_matrix,
    build_population_color_map,
    _apply_bayesian_prior,
    compute_unknown_threshold,
    assign_with_auto_threshold,
    filter_area_channels,
    transform_reference_mfi,
)
from .blast_detection import (
    build_blast_weights,
    score_nodes_for_blasts,
    categorize_blast_score,
    build_blast_score_dataframe,
    compute_reference_normalization,
    BLAST_HIGH_THRESHOLD,
    BLAST_MODERATE_THRESHOLD,
)
from .statistics import (
    mann_whitney_u,
    kolmogorov_smirnov,
    compute_fold_change,
    assess_mrd_status,
    compare_conditions_per_cluster,
)

__all__ = [
    # Population mapping
    "map_populations_to_nodes_v3",
    "map_populations_to_nodes_v5",
    "map_nodes_to_metaclusters",
    "get_population_summary",
    "normalize_matrix",
    "build_population_color_map",
    "_apply_bayesian_prior",
    "compute_unknown_threshold",
    "assign_with_auto_threshold",
    # Blast detection
    "build_blast_weights",
    "score_nodes_for_blasts",
    "categorize_blast_score",
    "build_blast_score_dataframe",
    "compute_reference_normalization",
    "BLAST_HIGH_THRESHOLD",
    "BLAST_MODERATE_THRESHOLD",
    # Statistics
    "mann_whitney_u",
    "kolmogorov_smirnov",
    "compute_fold_change",
    "assess_mrd_status",
    "compare_conditions_per_cluster",
]
