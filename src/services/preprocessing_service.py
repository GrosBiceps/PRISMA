"""
preprocessing_service.py — Orchestration du prétraitement des échantillons FCS.

Chaîne de traitement complète pour un FlowSample:
  1. Contrôle QC (NaN, cellules minimales, scatter exclusion)
  2. Sous-échantillonnage (si configuré)
  3. Gating (auto ou manuel) selon PipelineConfig
  4. Transformation cytométrique (logicle, arcsinh)
  5. Normalisation (zscore, minmax)

Retourne une liste de FlowSample prêts pour le clustering FlowSOM.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import anndata as ad

    _ANNDATA_AVAILABLE = True
except ImportError:
    _ANNDATA_AVAILABLE = False

from flowsom_pipeline_pro.config.pipeline_config import PipelineConfig
from flowsom_pipeline_pro.src.models.sample import FlowSample
from flowsom_pipeline_pro.src.core.transformers import DataTransformer
from flowsom_pipeline_pro.src.core.normalizers import DataNormalizer
from flowsom_pipeline_pro.src.core.gating import PreGating
from flowsom_pipeline_pro.src.core.auto_gating import AutoGating
from flowsom_pipeline_pro.src.utils.logger import GatingLogger, get_logger
from flowsom_pipeline_pro.src.utils.validators import (
    check_nan,
    check_min_cells,
    check_no_fsc_ssc_in_analysis_markers,
)

_logger = get_logger("services.preprocessing")


def preprocess_sample(
    sample: FlowSample,
    config: PipelineConfig,
    gating_logger: Optional[GatingLogger] = None,
) -> Optional[FlowSample]:
    """
    Applique la chaîne complète de prétraitement à un FlowSample.

    Args:
        sample: Échantillon FCS chargé.
        config: Configuration du pipeline.
        gating_logger: Logger de gating (optionnel).

    Returns:
        FlowSample prétraité, ou None si l'échantillon échoue au QC minimum.
    """
    if gating_logger is None:
        gating_logger = GatingLogger()

    X = sample.matrix
    var_names = sample.var_names
    file_name = sample.name
    n_original = X.shape[0]

    _logger.info("Prétraitement: %s (%d cellules)", file_name, n_original)

    # ── 1. Contrôle QC ────────────────────────────────────────────────────────
    n_nan = check_nan(X)
    if n_nan > 0:
        _logger.warning("%s: %d NaN détectés — imputation à 0", file_name, n_nan)
        X = np.nan_to_num(X, nan=0.0)

    if not check_min_cells(X.shape[0], min_cells=100):
        _logger.warning("%s: trop peu de cellules — ignoré", file_name)
        return None

    # ── 2. Sous-échantillonnage initial ───────────────────────────────────────
    max_cells = config.downsampling.max_cells_per_file
    if max_cells and max_cells > 0 and X.shape[0] > max_cells:
        rng = np.random.default_rng(config.flowsom.seed)
        idx = rng.choice(X.shape[0], size=max_cells, replace=False)
        X = X[idx]
        _logger.info(
            "%s: sous-échantillonné à %d/%d cellules", file_name, max_cells, n_original
        )

    # ── 3. Gating ─────────────────────────────────────────────────────────────
    pregate_cfg = config.pregate
    condition = getattr(sample, "condition", "Unknown") or "Unknown"
    if pregate_cfg.apply:
        X, var_names, mask_combined = _apply_gating(
            X,
            var_names,
            pregate_cfg,
            gating_logger,
            file_name,
            condition=condition,
        )
        if X.shape[0] < 100:
            _logger.warning(
                "%s: %d cellules après gating — ignoré", file_name, X.shape[0]
            )
            return None

    # ── 3b. Filtrage -A/-H (déduplication marqueurs Area vs Height) ───────────
    keep_area = getattr(config.markers, "keep_area_only", False)
    if keep_area:
        X, var_names = _filter_area_markers(X, var_names)
        _logger.info(
            "%s: filtrage -A uniquement → %d marqueurs", file_name, len(var_names)
        )

    # ── 4. Transformation cytométrique ────────────────────────────────────────
    transform_cfg = config.transform
    if transform_cfg.method != "none":
        X = DataTransformer.apply(
            X,
            method=transform_cfg.method,
            cofactor=transform_cfg.cofactor,
            var_names=var_names,
            apply_to_scatter=transform_cfg.apply_to_scatter,
        )

    # ── 5. Normalisation ─────────────────────────────────────────────────────
    normalize_cfg = config.normalize
    if normalize_cfg.method != "none":
        X = DataNormalizer.apply(X, method=normalize_cfg.method)

    # ── Reconstruire le FlowSample avec les données traitées ─────────────────
    import pandas as _pd

    processed = FlowSample(
        name=sample.name,
        path=sample.path,
        condition=sample.condition,
        data=_pd.DataFrame(X, columns=var_names),
        metadata={
            **sample.metadata,
            "preprocessed": True,
            "n_original_cells": n_original,
        },
        n_cells_raw=n_original,
    )

    _logger.info(
        "%s: %d → %d cellules après prétraitement",
        file_name,
        n_original,
        X.shape[0],
    )
    return processed


def _filter_area_markers(
    X: np.ndarray,
    var_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Supprime les marqueurs -H (Height) UNIQUEMENT quand le doublon -A (Area) existe.

    Les instruments BD et Beckman Coulter génèrent systématiquement des paires
    Marqueur-A / Marqueur-H. Garder les deux introduit de la colinéarité dans
    le SOM. On ne conserve que -A + tout marqueur sans suffixe -A/-H
    (ex: FSC-Width, Time). Un marqueur -H sans -A correspondant est conservé.

    La correspondance est détectée par le préfixe avant le suffixe -A/-H.
    Exemple: « CD13 PE-A » et « CD13 PE-H » → seul « CD13 PE-A » est conservé.
             « FSC-H » sans « FSC-A » → « FSC-H » est conservé.

    Returns:
        Tuple (X_filtré, var_names_filtrés).
    """
    upper_names = [n.upper() for n in var_names]
    # Construire l'ensemble des préfixes des marqueurs -A existants
    area_prefixes: set = set()
    for name_u in upper_names:
        if name_u.endswith("-A"):
            area_prefixes.add(name_u[:-2])  # ex: "CD13 PE"

    cols_keep: List[int] = []
    for i, name_u in enumerate(upper_names):
        if name_u.endswith("-H"):
            prefix = name_u[:-2]
            if prefix in area_prefixes:
                # Le doublon -A existe → supprimer ce -H
                _logger.debug(
                    "Marqueur %s supprimé (doublon -A existant)", var_names[i]
                )
                continue
        cols_keep.append(i)

    if not cols_keep or len(cols_keep) == len(var_names):
        return X, list(var_names)  # rien à supprimer

    n_removed = len(var_names) - len(cols_keep)
    _logger.info(
        "Filtrage -A/-H : %d marqueurs -H supprimés (doublon -A existant)",
        n_removed,
    )
    return X[:, cols_keep], [var_names[i] for i in cols_keep]


def _apply_gating(
    X: np.ndarray,
    var_names: List[str],
    pregate_cfg: object,
    gating_logger: GatingLogger,
    file_name: str,
    condition: str = "Unknown",
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Applique le gating (auto ou manuel) sur la matrice.

    Gating asymétrique (MODE_BLASTES_VS_NORMAL) :
    - Sain / NBM : seuls G1 (débris) et G2 (singlets) sont appliqués.
      G3 (CD45) et G4 (CD34) sont ignorés pour conserver les progéniteurs
      normaux CD45− et les hématogones.
    - Pathologique : gating complet G1→G4.

    Args:
        condition: "Sain", "Pathologique" ou autre. Case-insensitive.

    Returns:
        Tuple (X_gated, var_names, combined_mask).
    """
    mode = getattr(pregate_cfg, "mode", "auto")
    mode_blastes_vs_normal = getattr(pregate_cfg, "mode_blastes_vs_normal", False)
    n_before = X.shape[0]
    combined_mask = np.ones(n_before, dtype=bool)

    # Un échantillon sain ne doit pas subir le gating CD45/CD34 qui exclurait
    # les progéniteurs et cellules CD45-dim (hématogones, plasmablastes...)
    is_sain = mode_blastes_vs_normal and condition.lower() in (
        "sain",
        "normal",
        "healthy",
        "nbm",
        "moelle normale",
    )

    # Gate 1 – Débris
    if getattr(pregate_cfg, "viable", True):
        if mode == "auto":
            # Appliquer le sous-échantillonnage GMM configuré (pregate_advanced.gmm_max_samples)
            AutoGating.GMM_MAX_SAMPLES = getattr(
                pregate_cfg, "gmm_max_samples", AutoGating.GMM_MAX_SAMPLES
            )
            mask = AutoGating.auto_gate_debris(X, var_names)
        else:
            mask = PreGating.gate_viable_cells(
                X,
                var_names,
                min_percentile=getattr(pregate_cfg, "debris_min_percentile", 2.0),
                max_percentile=getattr(pregate_cfg, "debris_max_percentile", 99.0),
            )
        gating_logger.log(
            file_name, "G1_debris", n_before, int(mask.sum()), method=mode
        )
        combined_mask &= mask

    # Gate 2 – Singlets
    if getattr(pregate_cfg, "singlets", True):
        if mode == "auto":
            mask = AutoGating.auto_gate_singlets(
                X,
                var_names,
                r2_threshold=getattr(pregate_cfg, "ransac_r2_threshold", 0.85),
                mad_factor=getattr(pregate_cfg, "ransac_mad_factor", 3.0),
            )
        else:
            mask = PreGating.gate_singlets(
                X,
                var_names,
                ratio_min=getattr(pregate_cfg, "singlet_ratio_min", 0.6),
                ratio_max=getattr(pregate_cfg, "singlet_ratio_max", 1.5),
            )
        gating_logger.log(
            file_name,
            "G2_singlets",
            int(combined_mask.sum()),
            int((combined_mask & mask).sum()),
            method=mode,
        )
        combined_mask &= mask

    # Gate 3 – CD45 (ASYMÉTRIQUE : ignoré si échantillon sain)
    if getattr(pregate_cfg, "cd45", True):
        if is_sain:
            _logger.info(
                "%s [%s]: G3_cd45 ignoré (gating asymétrique — progéniteurs conservés)",
                file_name,
                condition,
            )
            gating_logger.log(
                file_name,
                "G3_cd45",
                int(combined_mask.sum()),
                int(combined_mask.sum()),
                method="skip_asymmetric",
            )
        else:
            if mode == "auto":
                mask = AutoGating.auto_gate_cd45(X, var_names)
            else:
                mask = PreGating.gate_cd45_positive(
                    X,
                    var_names,
                    threshold_percentile=getattr(
                        pregate_cfg, "cd45_min_percentile", 5.0
                    ),
                )
            gating_logger.log(
                file_name,
                "G3_cd45",
                int(combined_mask.sum()),
                int((combined_mask & mask).sum()),
                method=mode,
            )
            combined_mask &= mask

    # Gate 4 – CD34+ blastes (optionnel, ASYMÉTRIQUE : ignoré si échantillon sain)
    if getattr(pregate_cfg, "cd34", False):
        if is_sain:
            _logger.info(
                "%s [%s]: G4_cd34 ignoré (gating asymétrique)",
                file_name,
                condition,
            )
            gating_logger.log(
                file_name,
                "G4_cd34",
                int(combined_mask.sum()),
                int(combined_mask.sum()),
                method="skip_asymmetric",
            )
        else:
            if mode == "auto":
                mask = AutoGating.auto_gate_cd34(X, var_names)
            else:
                mask = PreGating.gate_cd34_blasts(
                    X,
                    var_names,
                    threshold_percentile=getattr(
                        pregate_cfg, "cd34_min_percentile", 70.0
                    ),
                    ssc_max_percentile=getattr(pregate_cfg, "cd34_ssc_max", 40.0),
                )
            gating_logger.log(
                file_name,
                "G4_cd34",
                int(combined_mask.sum()),
                int((combined_mask & mask).sum()),
                method=mode,
            )
            combined_mask &= mask

    X_gated = X[combined_mask]

    _logger.info(
        "%s [%s]: gating %s → %d/%d cellules conservées (%.1f%%)",
        file_name,
        condition,
        mode,
        X_gated.shape[0],
        n_before,
        X_gated.shape[0] / max(n_before, 1) * 100,
    )

    return X_gated, var_names, combined_mask


def preprocess_all_samples(
    samples: List[FlowSample],
    config: PipelineConfig,
    gating_logger: Optional[GatingLogger] = None,
) -> List[FlowSample]:
    """
    Prétraite une liste complète de FlowSample.

    Args:
        samples: Liste des échantillons à traiter.
        config: Configuration du pipeline.
        gating_logger: Logger de gating partagé (optionnel).

    Returns:
        Liste des échantillons prétraités (échecs ignorés).
    """
    if gating_logger is None:
        gating_logger = GatingLogger()

    processed: List[FlowSample] = []
    for sample in samples:
        result = preprocess_sample(sample, config, gating_logger=gating_logger)
        if result is not None:
            processed.append(result)

    _logger.info(
        "Prétraitement terminé: %d/%d échantillons",
        len(processed),
        len(samples),
    )
    return processed


# =============================================================================
# PRÉTRAITEMENT COMBINÉ — Reproduit fidèlement flowsom_pipeline.py
# =============================================================================
# Dans le monolithe, le gating est appliqué sur la CONCATÉNATION de tous les
# fichiers avant toute autre opération. Cela est crucial car :
#  - Le GMM CD45 est calibré sur la distribution combinée sain+patho :
#    les cellules NBM (CD45-high) servent d'étalon pour le seuil CD45+/CD45-
#    → les blastes AML (CD45-dim) ne sont pas éliminés par erreur.
#  - Le RANSAC singlets est exécuté par fichier sur l'ensemble combiné.
#  - Il n'y a AUCUN sous-échantillonnage avant le gating.
# =============================================================================


def preprocess_combined(
    samples: List[FlowSample],
    config: "PipelineConfig",
    gating_logger: Optional[GatingLogger] = None,
    gating_plot_dir: "Optional[Path]" = None,
) -> List[FlowSample]:
    """
    Reproduit fidèlement flowsom_pipeline.py.

    Contrairement à ``preprocess_all_samples`` (qui traite chaque fichier
    indépendamment), cette fonction:

    1. Calcule l'intersection des marqueurs communs (équivalent join='inner').
    2. Empile toutes les cellules brutes en une seule matrice.
    3. Applique le gating sur les données **combinées** :
       - Gate débris (GMM 2D FSC-A/SSC-A) sur toutes les cellules.
       - Gate singlets (RANSAC) par fichier sur les données combinées.
       - Gate CD45 (si activé) : GMM calibré par les cellules NBM → seuil
         correct pour CD45-dim (blastes AML). Appliqué **uniquement aux patho**
         si ``mode_blastes_vs_normal=True``.
       - Gate CD34 (si activé) : même logique asymétrique.
    4. Filtre les marqueurs -H redondants (keep_area_only).
    5. Applique la transformation et la normalisation sur l'ensemble.
    6. Reconstruit la liste de :class:`FlowSample` par fichier d'origine.
       **AUCUN sous-échantillonnage avant FlowSOM** (identique au monolithe).
       Le sous-échantillonnage bootstrap (20 000 cellules) n'est utilisé que
       dans la phase 2 de l'auto-clustering.

    Args:
        samples: Échantillons bruts (non prétraités).
        config: Configuration du pipeline.
        gating_logger: Logger de gating partagé (optionnel).

    Returns:
        Tuple (liste FlowSample prétraités, dict figures matplotlib gating).
    """
    import pandas as _pd

    if gating_logger is None:
        gating_logger = GatingLogger()

    if not samples:
        return [], {}

    # ── 1. Intersection des marqueurs communs ─────────────────────────────────
    common_markers: List[str] = list(samples[0].var_names)
    for s in samples[1:]:
        s_set = set(s.var_names)
        common_markers = [m for m in common_markers if m in s_set]

    if not common_markers:
        _logger.error("Aucun marqueur commun entre tous les fichiers !")
        return [], {}

    _logger.info("Panel commun (intersection): %d marqueurs", len(common_markers))
    missing_summary: dict = {}
    for s in samples:
        missing = [m for m in s.var_names if m not in common_markers]
        if missing:
            missing_summary[s.name] = missing
    if missing_summary:
        for fname, miss in missing_summary.items():
            _logger.warning(
                "%s: %d marqueur(s) exclus du panel commun: %s",
                fname,
                len(miss),
                miss,
            )

    # ── 2. Empilage des données brutes avec tracking ───────────────────────────
    all_X: List[np.ndarray] = []
    all_conditions: List[str] = []
    all_file_origins: List[str] = []
    # Ordre canonique des samples pour reconstruire les FlowSample après gating
    sample_order: List[FlowSample] = []

    for s in samples:
        var_s = s.var_names
        X_s = s.matrix
        col_idx = [var_s.index(m) for m in common_markers]
        X_sel = X_s[:, col_idx].astype(np.float64)
        n = X_sel.shape[0]
        all_X.append(X_sel)
        all_conditions.extend([s.condition] * n)
        all_file_origins.extend([s.name] * n)
        sample_order.append(s)
        _logger.debug("%s: %d cellules empilées", s.name, n)

    X_raw = np.vstack(all_X)
    conditions = np.array(all_conditions)
    file_origins = np.array(all_file_origins)
    n_total_raw = X_raw.shape[0]
    var_names: List[str] = list(common_markers)

    _logger.info(
        "Données combinées: %d cellules × %d marqueurs (%d fichiers)",
        n_total_raw,
        len(var_names),
        len(samples),
    )
    for s in sample_order:
        _logger.info(
            "  %s [%s]: %d cellules",
            s.name,
            s.condition,
            int((file_origins == s.name).sum()),
        )

    # ── QC NaN sur les données brutes combinées ───────────────────────────────
    n_nan = check_nan(X_raw)
    if n_nan > 0:
        _logger.warning("%d NaN détectés — imputation à 0", n_nan)
        X_raw = np.nan_to_num(X_raw, nan=0.0)

    # ── 3. Gating sur données combinées ───────────────────────────────────────
    pregate_cfg = config.pregate
    gate_masks: Dict[str, np.ndarray] = {}
    X_for_plot = X_raw.copy()  # Matrix brute avant gating (pour plots QC)
    var_for_plot: List[str] = list(var_names)
    conditions_for_plot = (
        conditions.copy()
    )  # Conditions avant gating (pour plots QC CD45)
    if getattr(pregate_cfg, "apply", True):
        X_raw, var_names, conditions, file_origins, gate_masks = _apply_gating_combined(
            X_raw,
            var_names,
            conditions,
            file_origins,
            pregate_cfg,
            gating_logger,
        )

    # ── Plots QC de gating (si plot_dir fourni) ───────────────────────────────
    gating_figures: Dict[str, Any] = {}
    if gating_plot_dir is not None and gate_masks:
        try:
            from flowsom_pipeline_pro.src.visualization.gating_plots import (
                generate_all_gating_plots,
            )

            gating_plot_dir = Path(gating_plot_dir)
            _fig_results = generate_all_gating_plots(
                X_for_plot,
                var_for_plot,
                gate_masks,
                output_dir=gating_plot_dir / "gating",
                sample_name="combined",
                conditions=conditions_for_plot,
            )
            # Renommer les figures selon les labels notebook
            _fig_map = {
                "01_overview": "fig_overview",
                "02_debris": "fig_gate_debris",
                "03_singlets": "fig_gate_singlets",
                "04_cd45": "fig_gate_cd45",
                "05_cd34": "fig_gate_cd34",
                "06_kde_debris": "fig_kde_debris",
                "07_kde_cd45": "fig_kde_cd45",
            }
            for _key, _fig in _fig_results.items():
                if _fig is not None:
                    label = _fig_map.get(_key, f"fig_gate_{_key}")
                    gating_figures[label] = _fig
            _logger.info(
                "Plots QC gating générés: %d figures dans %s",
                len(gating_figures),
                gating_plot_dir / "gating",
            )
        except Exception as _e:
            _logger.warning("Plots QC gating échoués (non bloquant): %s", _e)

    n_after_gate = X_raw.shape[0]
    _logger.info(
        "Après gating combiné: %d/%d cellules (%.1f%%)",
        n_after_gate,
        n_total_raw,
        n_after_gate / max(n_total_raw, 1) * 100,
    )

    # ── 4. Filtrage -A/-H (keep_area_only) ───────────────────────────────────
    keep_area = getattr(config.markers, "keep_area_only", False)
    if keep_area:
        X_raw, var_names = _filter_area_markers(X_raw, var_names)
        _logger.info("Filtrage -A uniquement: %d marqueurs conservés", len(var_names))

    # ── Save pré-transformation — pour export FCS Kaluza (valeurs linéaires) ─
    # Les données ici sont après gating + filtre -A/-H, mais AVANT toute
    # transformation (logicle/arcsinh) et normalisation. C'est exactement ce
    # qu'attend Kaluza pour afficher les données dans l'espace d'intensité.
    X_pre_transform = X_raw.copy()
    var_names_pre_transform: List[str] = list(var_names)

    # ── 5. Transformation cytométrique ───────────────────────────────────────
    transform_cfg = config.transform
    if transform_cfg.method != "none":
        X_raw = DataTransformer.apply(
            X_raw,
            method=transform_cfg.method,
            cofactor=transform_cfg.cofactor,
            var_names=var_names,
            apply_to_scatter=transform_cfg.apply_to_scatter,
        )

    # ── 6. Normalisation ─────────────────────────────────────────────────────
    normalize_cfg = config.normalize
    if normalize_cfg.method != "none":
        X_raw = DataNormalizer.apply(X_raw, method=normalize_cfg.method)

    # ── 7. Split par fichier — AUCUN sous-échantillonnage avant FlowSOM ────────
    # Identique au monolithe flowsom_pipeline.py : FlowSOM s'entraîne sur
    # L'INTÉGRALITÉ des cellules gatées. Le SOM (grille xdim×ydim) fait lui-même
    # le résumé en prototypes de nodes — aucun sous-échantillonnage a priori.
    # Le sous-échantillonnage bootstrap (20 000 cellules) n'est utilisé QUE
    # dans la phase 2 de l'auto-clustering (stability bootstrap).
    processed: List[FlowSample] = []
    for orig_sample in sample_order:
        file_mask = file_origins == orig_sample.name
        X_file = X_raw[file_mask]
        n_file = X_file.shape[0]

        if n_file < 100:
            _logger.warning(
                "%s: %d cellules après gating — ignoré",
                orig_sample.name,
                n_file,
            )
            gating_logger.log(
                orig_sample.name,
                "final_count",
                n_file,
                n_file,
                warning=f"Ignoré: seulement {n_file} cellules après gating",
            )
            continue

        n_raw_orig = orig_sample.n_cells_raw or n_file

        processed.append(
            FlowSample(
                name=orig_sample.name,
                path=orig_sample.path,
                condition=orig_sample.condition,
                data=_pd.DataFrame(X_file, columns=var_names),
                metadata={
                    **orig_sample.metadata,
                    "preprocessed": True,
                    "n_original_cells": n_raw_orig,
                    "n_after_gating": n_file,
                },
                n_cells_raw=n_raw_orig,
                raw_data=_pd.DataFrame(
                    X_pre_transform[file_mask], columns=var_names_pre_transform
                ),
            )
        )
        _logger.info(
            "%s [%s]: %d → %d cellules après gating (toutes conservées pour FlowSOM)",
            orig_sample.name,
            orig_sample.condition,
            n_raw_orig,
            n_file,
        )

    _logger.info(
        "Prétraitement combiné terminé: %d/%d fichiers",
        len(processed),
        len(samples),
    )
    return processed, gating_figures


def _apply_gating_combined(
    X: np.ndarray,
    var_names: List[str],
    conditions: np.ndarray,
    file_origins: np.ndarray,
    pregate_cfg,
    gating_logger: GatingLogger,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Gating sur les données combinées — reproduit exactement flowsom_pipeline.py.

    Chaque gate est calculé sur la TOTALITÉ de X (toutes conditions mélangées),
    puis les masques sont combinés avec la logique asymétrique
    sain/patho si ``mode_blastes_vs_normal=True``.

    Args:
        X: Matrice brute combinée (toutes cellules).
        var_names: Noms des marqueurs.
        conditions: Condition par cellule (ex: "Sain", "Pathologique").
        file_origins: Nom de fichier par cellule.
        pregate_cfg: Configuration du gating.
        gating_logger: Logger de gating.

    Returns:
        Tuple (X_filtré, var_names, conditions_filtré, file_origins_filtré).
    """
    mode = getattr(pregate_cfg, "mode", "auto")
    mode_blastes_vs_normal = getattr(pregate_cfg, "mode_blastes_vs_normal", False)
    n_before = X.shape[0]
    combined_mask = np.ones(n_before, dtype=bool)
    gate_masks: Dict[str, np.ndarray] = {}

    _sain_labels = {"sain", "normal", "healthy", "nbm", "moelle normale"}
    is_sain_vec = np.array([c.lower() in _sain_labels for c in conditions], dtype=bool)
    is_patho_vec = ~is_sain_vec

    # ── Gate 1 : Débris (GMM 2D FSC-A/SSC-A) ─────────────────────────────────
    # Calculé sur TOUTES les cellules combinées.
    if getattr(pregate_cfg, "viable", True):
        _logger.info("Gate 1 — Débris [%s] sur %d cellules combinées", mode, n_before)
        if mode == "auto":
            AutoGating.GMM_MAX_SAMPLES = getattr(
                pregate_cfg, "gmm_max_samples", AutoGating.GMM_MAX_SAMPLES
            )
            mask_debris = AutoGating.auto_gate_debris(X, var_names)
        else:
            mask_debris = PreGating.gate_viable_cells(
                X,
                var_names,
                min_percentile=getattr(pregate_cfg, "debris_min_percentile", 2.0),
                max_percentile=getattr(pregate_cfg, "debris_max_percentile", 99.0),
            )
        n_debris_kept = int(mask_debris.sum())
        _logger.info(
            "  G1_debris: %d/%d (%.1f%%)",
            n_debris_kept,
            n_before,
            n_debris_kept / max(n_before, 1) * 100,
        )
        gating_logger.log(
            "COMBINED",
            "G1_debris",
            n_before,
            n_debris_kept,
            method=mode,
        )
        combined_mask &= mask_debris
        gate_masks["G1_debris"] = mask_debris
    else:
        mask_debris = np.ones(n_before, dtype=bool)

    # ── Gate 2 : Singlets (RANSAC par fichier, sur données combinées) ─────────
    # Passage explicite de file_origins → RANSAC per_file comme dans le monolithe.
    if getattr(pregate_cfg, "singlets", True):
        n_after_g1 = int(combined_mask.sum())
        _logger.info(
            "Gate 2 — Singlets [%s] par fichier sur %d cellules", mode, n_after_g1
        )
        if mode == "auto":
            mask_singlets = AutoGating.auto_gate_singlets(
                X,
                var_names,
                file_origin=file_origins,
                per_file=True,
                r2_threshold=getattr(pregate_cfg, "ransac_r2_threshold", 0.85),
                mad_factor=getattr(pregate_cfg, "ransac_mad_factor", 3.0),
            )
        else:
            mask_singlets = PreGating.gate_singlets(
                X,
                var_names,
                ratio_min=getattr(pregate_cfg, "singlet_ratio_min", 0.6),
                ratio_max=getattr(pregate_cfg, "singlet_ratio_max", 1.5),
            )
        n_g1g2 = int((combined_mask & mask_singlets).sum())
        _logger.info(
            "  G2_singlets: %d/%d (%.1f%%)",
            n_g1g2,
            n_after_g1,
            n_g1g2 / max(n_after_g1, 1) * 100,
        )
        gating_logger.log(
            "COMBINED",
            "G2_singlets",
            n_after_g1,
            n_g1g2,
            method=mode,
        )
        combined_mask &= mask_singlets
        gate_masks["G2_singlets"] = mask_singlets
    else:
        mask_singlets = np.ones(n_before, dtype=bool)

    # ── Gate 3 : CD45 (asymétrique si mode_blastes_vs_normal) ────────────────
    # CRUCIAL : le GMM est entraîné sur les données COMBINÉES (sain + patho).
    # Les cellules NBM (CD45-high) étalonnent le seuil → les blastes AML
    # (CD45-dim) passent la porte, contrairement au cas per-file.
    if getattr(pregate_cfg, "cd45", True):
        n_after_g1g2 = int(combined_mask.sum())
        if mode_blastes_vs_normal:
            _logger.info(
                "Gate 3 — CD45 [%s] ASYMÉTRIQUE (GMM combiné, patho uniquement)", mode
            )
            if mode == "auto":
                mask_cd45_full = AutoGating.auto_gate_cd45(X, var_names)
            else:
                mask_cd45_full = PreGating.gate_cd45_positive(
                    X,
                    var_names,
                    threshold_percentile=getattr(
                        pregate_cfg, "cd45_min_percentile", 5.0
                    ),
                )
            # Appliquer CD45 UNIQUEMENT aux patho (identique au monolithe)
            mask_cd45 = np.ones(n_before, dtype=bool)
            mask_cd45[is_patho_vec] = mask_cd45_full[is_patho_vec]
        else:
            _logger.info("Gate 3 — CD45 [%s] sur toutes les cellules", mode)
            if mode == "auto":
                mask_cd45 = AutoGating.auto_gate_cd45(X, var_names)
            else:
                mask_cd45 = PreGating.gate_cd45_positive(
                    X,
                    var_names,
                    threshold_percentile=getattr(
                        pregate_cfg, "cd45_min_percentile", 5.0
                    ),
                )

        n_after_cd45 = int((combined_mask & mask_cd45).sum())
        _logger.info(
            "  G3_cd45: %d/%d (%.1f%%)",
            n_after_cd45,
            n_after_g1g2,
            n_after_cd45 / max(n_after_g1g2, 1) * 100,
        )
        if mode_blastes_vs_normal:
            n_patho_pre = int((combined_mask & is_patho_vec).sum())
            n_patho_post = int((combined_mask & mask_cd45 & is_patho_vec).sum())
            n_sain_pre = int((combined_mask & is_sain_vec).sum())
            _logger.info(
                "  Patho CD45+ conservés: %d/%d (%.1f%%) | Sain conservés (100%%): %d",
                n_patho_post,
                n_patho_pre,
                n_patho_post / max(n_patho_pre, 1) * 100,
                n_sain_pre,
            )
        gating_logger.log(
            "COMBINED",
            "G3_cd45",
            n_after_g1g2,
            n_after_cd45,
            method=mode,
        )
        combined_mask &= mask_cd45
        gate_masks["G3_cd45"] = mask_cd45

    # ── Gate 4 : CD34+ blastes (asymétrique) ──────────────────────────────────
    if getattr(pregate_cfg, "cd34", False):
        n_after_g123 = int(combined_mask.sum())
        if mode_blastes_vs_normal:
            _logger.info("Gate 4 — CD34 [%s] ASYMÉTRIQUE (patho uniquement)", mode)
            if mode == "auto":
                mask_cd34_full = AutoGating.auto_gate_cd34(X, var_names)
            else:
                mask_cd34_full = PreGating.gate_cd34_blasts(
                    X,
                    var_names,
                    threshold_percentile=getattr(
                        pregate_cfg, "cd34_min_percentile", 70.0
                    ),
                    ssc_max_percentile=getattr(pregate_cfg, "cd34_ssc_max", 40.0),
                )
            mask_cd34 = np.ones(n_before, dtype=bool)
            mask_cd34[is_patho_vec] = mask_cd34_full[is_patho_vec]
        else:
            _logger.info("Gate 4 — CD34 [%s] sur toutes les cellules", mode)
            if mode == "auto":
                mask_cd34 = AutoGating.auto_gate_cd34(X, var_names)
            else:
                mask_cd34 = PreGating.gate_cd34_blasts(
                    X,
                    var_names,
                    threshold_percentile=getattr(
                        pregate_cfg, "cd34_min_percentile", 70.0
                    ),
                    ssc_max_percentile=getattr(pregate_cfg, "cd34_ssc_max", 40.0),
                )
        n_after_cd34 = int((combined_mask & mask_cd34).sum())
        _logger.info(
            "  G4_cd34: %d/%d (%.1f%%)",
            n_after_cd34,
            n_after_g123,
            n_after_cd34 / max(n_after_g123, 1) * 100,
        )
        gating_logger.log(
            "COMBINED",
            "G4_cd34",
            n_after_g123,
            n_after_cd34,
            method=mode,
        )
        combined_mask &= mask_cd34
        gate_masks["G4_cd34"] = mask_cd34

    # ── Résumé par condition ───────────────────────────────────────────────────
    n_final = int(combined_mask.sum())
    _logger.info(
        "Gating combiné: %d/%d cellules totales (%.1f%%)",
        n_final,
        n_before,
        n_final / max(n_before, 1) * 100,
    )
    if mode_blastes_vs_normal:
        n_sain_final = int((combined_mask & is_sain_vec).sum())
        n_patho_final = int((combined_mask & is_patho_vec).sum())
        _logger.info(
            "  Sain: %d cellules conservées | Patho: %d cellules conservées",
            n_sain_final,
            n_patho_final,
        )
        if n_patho_final > 0 and n_sain_final > 0:
            ratio = n_sain_final / n_patho_final
            if ratio > 10:
                _logger.warning(
                    "Déséquilibre conditions: %d sain vs %d patho (ratio %.1f×) "
                    "— considérer balance_conditions=true dans la config",
                    n_sain_final,
                    n_patho_final,
                    ratio,
                )

    # ── Application du masque final ───────────────────────────────────────────
    X_filtered = X[combined_mask]
    conditions_filtered = conditions[combined_mask]
    file_origins_filtered = file_origins[combined_mask]

    return X_filtered, var_names, conditions_filtered, file_origins_filtered, gate_masks
