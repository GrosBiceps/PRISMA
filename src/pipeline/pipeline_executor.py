"""
pipeline_executor.py — Exécuteur principal du pipeline FlowSOM.

Orchestre le pipeline de bout en bout:
  1. Chargement des fichiers FCS (sain + pathologique)
  2. Prétraitement de chaque échantillon
  3. Clustering FlowSOM (avec auto-clustering optionnel)
  4. Construction du DataFrame cellulaire
  5. Exports (FCS, CSV, JSON, plots)
  6. Retour d'un PipelineResult structuré

Point d'entrée: FlowSOMPipeline.execute(config)
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from flowsom_pipeline_pro.config.pipeline_config import PipelineConfig
from flowsom_pipeline_pro.src.models.sample import FlowSample
from flowsom_pipeline_pro.src.models.pipeline_result import (
    PipelineResult,
    ClusteringMetrics,
)
from flowsom_pipeline_pro.src.io.fcs_reader import get_fcs_files, load_as_flow_samples
from flowsom_pipeline_pro.src.core.clustering import FlowSOMClusterer
from flowsom_pipeline_pro.src.services.preprocessing_service import (
    preprocess_all_samples,
    preprocess_combined,
)
from flowsom_pipeline_pro.src.services.clustering_service import (
    run_clustering,
    build_cells_dataframe,
    stack_samples,
    stack_raw_markers,
)
from flowsom_pipeline_pro.src.services.export_service import ExportService
from flowsom_pipeline_pro.src.utils.logger import GatingLogger, get_logger

_logger = get_logger("pipeline.executor")


def _safe_plot(name: str, figures: dict, key: str, fn, *args, **kwargs):
    """
    Appelle fn(*args, **kwargs) et stocke le résultat dans figures[key].
    En cas d'échec, log un warning non-bloquant. Retourne le résultat ou None.
    """
    try:
        result = fn(*args, **kwargs)
        if result is not None:
            figures[key] = result
        _logger.info("%s sauvegardé.", name)
        return result
    except Exception as _e:
        _logger.warning("%s échoué (non bloquant): %s", name, _e)
        return None


class FlowSOMPipeline:
    """
    Pipeline FlowSOM de niveau production.

    Encapsule l'exécution complète du pipeline en un seul objet.

    Usage:
        config = PipelineConfig.from_yaml("config.yaml")
        pipeline = FlowSOMPipeline(config)
        result = pipeline.execute()
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._gating_logger = GatingLogger()
        self._result: Optional[PipelineResult] = None

    @property
    def result(self) -> Optional[PipelineResult]:
        """Accès au résultat du dernier execute()."""
        return self._result

    def execute(self) -> PipelineResult:
        """
        Exécute le pipeline complet.

        Returns:
            PipelineResult avec tous les résultats et les chemins d'export.
        """
        start_time = time.time()
        config = self.config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        _logger.info("=" * 60)
        _logger.info("PIPELINE FLOWSOM — DÉMARRAGE")
        _logger.info("=" * 60)

        try:
            # ── Étape 1: Chargement des fichiers FCS ──────────────────────────
            _logger.info("Étape 1: Chargement des fichiers FCS...")
            samples = self._load_all_samples()

            if not samples:
                return PipelineResult.failure(
                    error="Aucun échantillon chargé",
                    config=config,
                )

            _logger.info("  %d échantillon(s) chargé(s)", len(samples))

            # Définir output_dir tôt (utilisé pour les plots de gating)
            output_dir = Path(config.paths.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # ── Étape 2: Prétraitement ───────────────────────────────────────────
            _logger.info("Étape 2: Prétraitement (gating combiné + transformation)...")
            # Utilise l'approche combinée (fidèle à flowsom_pipeline.py) :
            # gating sur les données brutes concaténées, sous-échantillonnage APRÈS.
            viz_cfg = getattr(config, "visualization", None)
            viz_save = getattr(viz_cfg, "save_plots", True)
            umap_enabled = getattr(viz_cfg, "umap_enabled", True)
            processed_samples, gating_figures = preprocess_combined(
                samples,
                config,
                gating_logger=self._gating_logger,
                gating_plot_dir=output_dir / "plots" if viz_save else None,
            )

            if not processed_samples:
                return PipelineResult.failure(
                    error="Aucun échantillon valide après prétraitement",
                    config=config,
                )

            _logger.info(
                "  %d/%d échantillon(s) après prétraitement",
                len(processed_samples),
                len(samples),
            )

            # ── Étape 3: Clustering FlowSOM ───────────────────────────────────
            _logger.info("Étape 3: Clustering FlowSOM...")
            metaclustering, clustering, clusterer, selected_markers = run_clustering(
                processed_samples, config
            )

            # Récupérer la matrice empilée complète
            X_stacked, obs = stack_samples(
                processed_samples,
                selected_markers,
                seed=config.flowsom.seed,
            )

            n_meta = (
                int(metaclustering.max()) + 1
                if metaclustering is not None and len(metaclustering) > 0
                else 0
            )
            _logger.info(
                "  FlowSOM terminé: %d cellules, %d marqueurs, %d métaclusters",
                X_stacked.shape[0],
                len(selected_markers),
                n_meta,
            )

            # ── Étape 4: Construction du DataFrame cellulaire ─────────────────
            _logger.info("Étape 4: Construction du DataFrame cellulaire...")
            df_cells = build_cells_dataframe(
                X_stacked, selected_markers, obs, metaclustering, clustering
            )

            # ── Étape 4b: Construction du DataFrame FCS complet ───────────────
            # Identique au monolithe flowsom_pipeline.py :
            # - Toutes les colonnes (données brutes pré-transformation)
            # - FlowSOM_metacluster, FlowSOM_cluster (+1 pour Kaluza ≥ 1)
            # - xGrid, yGrid, xNodes, yNodes avec jitter circulaire (style R)
            # - size (nb cellules par node), Condition_Num
            _logger.info("  Construction du DataFrame FCS complet (style Kaluza)...")
            df_fcs = self._build_fcs_dataframe(
                processed_samples,
                metaclustering,
                clustering,
                clusterer,
            )

            # Matrice MFI — convertie en DataFrame (marqueurs × metaclusters)
            mfi_raw = clusterer.get_mfi_matrix(X_stacked, selected_markers)
            unique_mc = np.unique(metaclustering)
            mfi_matrix = pd.DataFrame(
                mfi_raw,
                index=[f"MC{i}" for i in unique_mc],
                columns=selected_markers[: mfi_raw.shape[1]],
            )

            # ── Étape 5: Metrics de clustering ────────────────────────────────
            metrics = self._compute_metrics(X_stacked, metaclustering)
            # ── Accumulation des figures pour le rapport HTML ─────────────────
            # Clés calquées sur les noms du notebook de référence
            _mpl_figures: Dict[str, object] = dict(
                gating_figures
            )  # fig_overview, fig_gate_*
            _plotly_figures: Dict[str, object] = {}

            # ── Étape 5b: UMAP (si save_plots ET umap_enabled) ───────────────────
            if viz_save and umap_enabled:
                try:
                    from umap import UMAP
                    from flowsom_pipeline_pro.src.visualization.flowsom_plots import (
                        plot_umap,
                    )

                    _logger.info("Calcul UMAP...")
                    fig_cfg = getattr(viz_cfg, "figures", {}) or {}
                    umap_cfg = fig_cfg.get("umap", {})
                    umap_sample_size = min(
                        umap_cfg.get("n_sample", 100_000), X_stacked.shape[0]
                    )
                    rng_umap = np.random.default_rng(config.flowsom.seed)
                    idx_umap = rng_umap.choice(
                        X_stacked.shape[0], umap_sample_size, replace=False
                    )
                    umap_coords = UMAP(
                        n_components=2, random_state=config.flowsom.seed, n_jobs=1
                    ).fit_transform(X_stacked[idx_umap])
                    fig_umap = plot_umap(
                        umap_coords,
                        metaclustering[idx_umap],
                        output_dir / "plots" / f"umap_{timestamp}.png",
                        n_metaclusters=int(metaclustering.max()) + 1,
                        seed=config.flowsom.seed,
                    )
                    if fig_umap is not None:
                        _mpl_figures["fig_umap"] = fig_umap
                    _logger.info("UMAP sauvegardé.")
                except Exception as _e:
                    _logger.warning("UMAP échoué (non bloquant): %s", _e)

            # ── MST + SOM Plotly/Matplotlib ───────────────────────────────────
            if viz_save:
                from flowsom_pipeline_pro.src.visualization import flowsom_plots as _fp

                _safe_plot(
                    "MST statique", _mpl_figures, "fig_mst_static",
                    _fp.plot_mst_static,
                    clusterer, mfi_matrix, metaclustering,
                    output_dir / "plots" / f"mst_static_{timestamp}.png",
                )
                _safe_plot(
                    "MST Plotly", _plotly_figures, "fig_mst",
                    _fp.plot_mst_plotly,
                    clusterer, mfi_matrix, metaclustering,
                    output_dir / "plots" / f"mst_interactive_{timestamp}.html",
                )
                _safe_plot(
                    "SOM Grid Plotly", _plotly_figures, "fig_grid_mc",
                    _fp.plot_som_grid_plotly,
                    clustering, metaclustering, clusterer,
                    output_dir / "plots" / f"som_grid_{timestamp}.html",
                    seed=config.flowsom.seed,
                )

            # ── Star Chart FlowSOM (§13) ──────────────────────────────────────
            # fs.pl.plot_stars requiert un objet fs.FlowSOM natif (CPU uniquement).
            # Pour le backend GPU on utilise plot_star_chart_custom (matplotlib pur).
            if viz_save:
                try:
                    _fsom_native = getattr(clusterer, "_fsom_model", None)
                    _used_gpu = getattr(clusterer, "used_gpu_", False)

                    if (
                        _fsom_native is not None
                        and not _used_gpu
                        and hasattr(_fsom_native, "get_cluster_data")
                    ):
                        # ── Chemin CPU : fs.pl.plot_stars ────────────────────
                        from flowsom_pipeline_pro.src.visualization.flowsom_plots import (
                            plot_star_chart,
                        )

                        fig_star = plot_star_chart(
                            _fsom_native,
                            output_dir
                            / "plots"
                            / f"flowsom_star_chart_{timestamp}.png",
                        )
                        if fig_star is not None:
                            _mpl_figures["fig_star_chart"] = fig_star
                        _logger.info("Star Chart (CPU natif) sauvegardé.")
                    else:
                        # ── Chemin GPU : star chart custom matplotlib ─────────
                        from flowsom_pipeline_pro.src.visualization.flowsom_plots import (
                            plot_star_chart_custom,
                        )

                        _star_marker_names = (
                            list(processed_samples[0].markers)
                            if processed_samples
                            and hasattr(processed_samples[0], "markers")
                            else None
                        )
                        fig_star = plot_star_chart_custom(
                            clusterer,
                            output_dir
                            / "plots"
                            / f"flowsom_star_chart_{timestamp}.png",
                            marker_names=_star_marker_names,
                            title=f"FlowSOM Star Chart — GPU ({clusterer.xdim}×{clusterer.ydim})",
                        )
                        if fig_star is not None:
                            _mpl_figures["fig_star_chart"] = fig_star
                        _logger.info("Star Chart (GPU custom) sauvegardé.")
                except Exception as _e:
                    _logger.warning("Star Chart échoué (non bloquant): %s", _e)

            # ── Grille SOM statique PNG (§14) ────────────────────────────────
            if viz_save:
                try:
                    from flowsom_pipeline_pro.src.visualization.flowsom_plots import (
                        plot_som_grid_static,
                    )

                    _cond_labels_grid = (
                        df_cells["condition"].values
                        if "condition" in df_cells.columns
                        else None
                    )
                    # plot_som_grid_static attend un metaclustering par NODE
                    # (n_nodes,), PAS par cellule (n_cells,).
                    _mc_per_node = getattr(clusterer, "metacluster_map_", None)
                    if _mc_per_node is None:
                        _mc_per_node = np.array(
                            [
                                int(
                                    np.bincount(
                                        metaclustering[clustering == i]
                                    ).argmax()
                                )
                                if (clustering == i).any()
                                else 0
                                for i in range(clusterer.n_nodes)
                            ],
                            dtype=int,
                        )
                    fig_grid_s = plot_som_grid_static(
                        clustering,
                        _mc_per_node,
                        clusterer.get_grid_coords(),
                        _cond_labels_grid,
                        clusterer.xdim,
                        clusterer.ydim,
                        output_dir / "plots" / f"flowsom_som_grid_{timestamp}.png",
                        seed=config.flowsom.seed,
                    )
                    if fig_grid_s is not None:
                        _mpl_figures["fig_som_grid_static"] = fig_grid_s
                    _logger.info("Grille SOM statique sauvegardée.")
                except Exception as _e:
                    _logger.warning(
                        "Grille SOM statique échouée (non bloquant): %s", _e
                    )

            # ── Radar + bar charts + vue combinée ────────────────────────────
            if viz_save:
                from flowsom_pipeline_pro.src.visualization import flowsom_plots as _fp2

                _cond_labels = (
                    df_cells["condition"].values
                    if "condition" in df_cells.columns
                    else None
                )

                _safe_plot(
                    "Radar métaclusters", _plotly_figures, "fig_radar",
                    _fp2.plot_metacluster_radar,
                    mfi_matrix.values,
                    list(mfi_matrix.columns),
                    metaclustering,
                    output_dir / "plots" / f"metacluster_radar_{timestamp}.html",
                    n_metaclusters=n_meta,
                )

                # Clusters exclusifs (log uniquement, pas de figure)
                try:
                    if _cond_labels is not None:
                        _excl = _fp2.compute_exclusive_clusters(
                            metaclustering, _cond_labels, n_meta
                        )
                        for _line in _excl.get("summary_lines", []):
                            _logger.info("%s", _line)
                    else:
                        _logger.info(
                            "Clusters exclusifs ignorés (condition_labels non disponible)."
                        )
                except Exception as _e:
                    _logger.warning("Clusters exclusifs échoués (non bloquant): %s", _e)

                # Bar charts par métacluster
                if _cond_labels is not None:
                    _safe_plot(
                        "Bar chart % patho par cluster", _plotly_figures, "fig_patho_pct",
                        _fp2.plot_patho_pct_per_cluster,
                        metaclustering,
                        _cond_labels,
                        output_html=output_dir / "plots" / f"patho_pct_per_cluster_{timestamp}.html",
                        output_jpg=output_dir / "plots" / f"patho_pct_per_cluster_{timestamp}.jpg",
                    )
                else:
                    _logger.info("Bar chart %% patho ignoré (condition_labels non disponible).")

                _safe_plot(
                    "Bar chart % cellules par cluster", _plotly_figures, "fig_cells_pct",
                    _fp2.plot_cells_pct_per_cluster,
                    metaclustering,
                    output_html=output_dir / "plots" / f"cells_pct_per_cluster_{timestamp}.html",
                    output_jpg=output_dir / "plots" / f"cells_pct_per_cluster_{timestamp}.jpg",
                    condition_labels=_cond_labels,
                )

                # Bar charts par nœud SOM
                if _cond_labels is not None:
                    _safe_plot(
                        "Bar chart % patho par nœud SOM", _plotly_figures, "fig_patho_pct_som",
                        _fp2.plot_patho_pct_per_som_node,
                        clustering,
                        _cond_labels,
                        output_html=output_dir / "plots" / f"patho_pct_per_som_node_{timestamp}.html",
                        output_jpg=output_dir / "plots" / f"patho_pct_per_som_node_{timestamp}.jpg",
                    )
                else:
                    _logger.info("Bar chart %% patho SOM ignoré (condition_labels non disponible).")

                _safe_plot(
                    "Bar chart % cellules par nœud SOM", _plotly_figures, "fig_cells_pct_som",
                    _fp2.plot_cells_pct_per_som_node,
                    clustering,
                    output_html=output_dir / "plots" / f"cells_pct_per_som_node_{timestamp}.html",
                    output_jpg=output_dir / "plots" / f"cells_pct_per_som_node_{timestamp}.jpg",
                    condition_labels=_cond_labels,
                )

                # Vue combinée
                if _cond_labels is not None:
                    _safe_plot(
                        "Vue combinée nœuds SOM", _plotly_figures, "fig_som_combined",
                        _fp2.plot_combined_som_node_html,
                        clustering,
                        _cond_labels,
                        output_html=output_dir / "plots" / f"som_node_combined_{timestamp}.html",
                    )
                else:
                    _logger.info("Vue combinée SOM ignorée (condition_labels non disponible).")

            # ── Étape 6: Exports ───────────────────────────────────────────────────
            _logger.info("Étape 6: Exports...")
            exporter = ExportService(config, output_dir, timestamp=timestamp)

            input_files = [str(s.path) for s in samples]
            export_paths = exporter.export_all(
                df_cells=df_cells,
                df_fcs=df_fcs,
                mfi_matrix=mfi_matrix,
                metaclustering=metaclustering,
                selected_markers=selected_markers,
                input_files=input_files,
                gating_logger=self._gating_logger,
                clustering=clustering,
            )

            # ── Étape 6b: Distribution Sain/Patho par cluster SOM ────────────
            _cond_dist = (
                df_cells["condition"].values
                if "condition" in df_cells.columns
                else None
            )
            if _cond_dist is not None:
                try:
                    dist_paths = exporter.export_cluster_distribution(
                        clustering=clustering,
                        metaclustering=metaclustering,
                        condition_labels=_cond_dist,
                    )
                    export_paths.update(dist_paths)
                    _logger.info(
                        "Distribution Sain/Patho exportee: %d fichier(s)",
                        len(dist_paths),
                    )
                except Exception as _e:
                    _logger.warning(
                        "Export distribution clusters echoue (non bloquant): %s", _e
                    )

            # Plots (si activés dans la config)
            if viz_cfg is not None and viz_save:
                condition_labels = (
                    df_cells["condition"].values
                    if "condition" in df_cells.columns
                    else None
                )
                flowsom_figs = exporter.export_flowsom_plots(
                    mfi_matrix,
                    metaclustering,
                    n_metaclusters=int(metaclustering.max()) + 1,
                    condition_labels=condition_labels,
                )
                if isinstance(flowsom_figs, dict):
                    _mpl_figures.update(
                        {k: v for k, v in flowsom_figs.items() if v is not None}
                    )

                # ── Sankey gating ─────────────────────────────────────────
                try:
                    events = self._gating_logger.events
                    gate_map = {e.gate_name: e for e in events if e.file == "COMBINED"}
                    n_total = sum(s.matrix.shape[0] for s in samples)
                    gate_counts = {
                        "n_total": n_total,
                        "n_g1_pass": gate_map["G1_debris"].n_after
                        if "G1_debris" in gate_map
                        else n_total,
                        "n_g2_pass": gate_map["G2_singlets"].n_after
                        if "G2_singlets" in gate_map
                        else n_total,
                        "n_g3_pass": gate_map["G3_cd45"].n_after
                        if "G3_cd45" in gate_map
                        else n_total,
                        "n_final": int(metaclustering.shape[0]),
                    }
                    sankey_result = exporter.export_sankey(
                        gate_counts,
                        filter_blasts=getattr(config.pregate, "cd34", False),
                    )
                    if (
                        isinstance(sankey_result, dict)
                        and sankey_result.get("fig_sankey") is not None
                    ):
                        _plotly_figures["fig_sankey"] = sankey_result["fig_sankey"]
                    _logger.info("Sankey exporté.")
                except Exception as _e:
                    _logger.warning("Sankey échoué (non bloquant): %s", _e)

            # ── Étape 7: Mapping des populations (Section 10) ─────────────────
            population_mapping_result = None
            pop_map_cfg = getattr(config, "population_mapping", None)
            if pop_map_cfg is not None and getattr(pop_map_cfg, "enabled", False):
                _logger.info("Étape 7: Mapping populations via MFI (Section 10)...")
                try:
                    from flowsom_pipeline_pro.src.services.population_mapping_service import (
                        PopulationMappingService,
                    )

                    # Trouver le FCS exporté à l'étape 6
                    fcs_exported = None
                    for key in (
                        "fcs_complete",
                        "fcs_with_clusters",
                        "fcs_export",
                        "fcs",
                    ):
                        p = export_paths.get(key)
                        if p and Path(p).exists():
                            fcs_exported = Path(p)
                            break

                    if fcs_exported is None:
                        # Fallback : chercher le FCS le plus récent dans output_dir (récursif)
                        fcs_candidates = sorted(
                            output_dir.glob("**/*.fcs"),
                            key=lambda f: f.stat().st_mtime,
                            reverse=True,
                        )
                        if fcs_candidates:
                            fcs_exported = fcs_candidates[0]

                    if fcs_exported is not None:
                        _logger.info("  FCS source: %s", fcs_exported.name)
                        pm_service = PopulationMappingService(pop_map_cfg)
                        population_mapping_result = pm_service.run_full_mapping(
                            fcs_path=fcs_exported,
                            cluster_data=clusterer,
                            cell_data=None,  # AnnData non exposé ici — passer si disponible
                            output_dir=output_dir,
                            timestamp=timestamp,
                        )
                        _logger.info("Étape 7: Mapping populations OK.")
                    else:
                        _logger.warning(
                            "Étape 7: FCS exporté introuvable — mapping ignoré."
                        )

                except Exception as exc:
                    _logger.error("Étape 7 échouée (non bloquant): %s", exc)

            # Récupérer les figures Plotly de la cartographie de populations
            if population_mapping_result is not None:
                pop_figs = getattr(population_mapping_result, "figures_plotly", {})
                if pop_figs:
                    _plotly_figures.update(
                        {k: v for k, v in pop_figs.items() if v is not None}
                    )

            # ── Étape 8: Calcul MRD résiduelle ────────────────────────────────
            mrd_result = None
            try:
                from flowsom_pipeline_pro.src.analysis.mrd_calculator import (
                    load_mrd_config, compute_mrd,
                )
                mrd_cfg = load_mrd_config()

                if mrd_cfg.enabled and "condition" in df_cells.columns:
                    _logger.info("Étape 8: Calcul MRD résiduelle (nœuds SOM)...")
                    mrd_result = compute_mrd(df_cells, clustering, mrd_cfg)

                    # Visualisation MRD
                    if viz_save and mrd_result is not None:
                        from flowsom_pipeline_pro.src.visualization.flowsom_plots import (
                            plot_mrd_summary,
                        )
                        _safe_plot(
                            "MRD Résiduelle", _plotly_figures, "fig_mrd_summary",
                            plot_mrd_summary,
                            mrd_result,
                            output_html=output_dir / "plots" / f"mrd_summary_{timestamp}.html",
                            output_png=output_dir / "plots" / f"mrd_summary_{timestamp}.png",
                        )

                    # Export JSON des résultats MRD
                    if mrd_result is not None:
                        try:
                            import json
                            mrd_json_path = output_dir / f"mrd_results_{timestamp}.json"
                            mrd_json_path.write_text(
                                json.dumps(mrd_result.to_dict(), indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                            export_paths["mrd_results"] = str(mrd_json_path)
                            _logger.info("MRD JSON: %s", mrd_json_path.name)
                        except Exception as _je:
                            _logger.warning("Export JSON MRD échoué: %s", _je)
                else:
                    _logger.info("Étape 8: MRD désactivée ou pas de colonne condition.")
            except Exception as _mrd_exc:
                _logger.warning("Étape 8 MRD échouée (non bloquant): %s", _mrd_exc)

            # ── Rapport HTML self-contained (APRÈS étape 8) ──────────────────
            if viz_save:
                try:
                    _logger.info("Génération du rapport HTML...")
                    mc_table = [
                        {
                            "metacluster": f"MC{int(mc)}",
                            "n_cells": int((metaclustering == mc).sum()),
                            "pct": round(float((metaclustering == mc).mean() * 100), 2),
                            "top_markers": ", ".join(
                                mfi_matrix.loc[f"MC{int(mc)}"]
                                .nlargest(3)
                                .index.tolist()
                            )
                            if f"MC{int(mc)}" in mfi_matrix.index
                            else "",
                        }
                        for mc in np.unique(metaclustering)
                    ]

                    # ── Données par condition ──────────────────────────────
                    condition_data: List[Dict] = []
                    if "condition" in df_cells.columns:
                        cond_counts = df_cells["condition"].value_counts()
                        total_cells = len(df_cells)
                        for cond_name, n_c in cond_counts.items():
                            condition_data.append(
                                {
                                    "condition": str(cond_name),
                                    "n_cells": int(n_c),
                                    "pct": round(float(n_c / total_cells * 100), 1),
                                }
                            )

                    # ── Données par fichier ────────────────────────────────
                    files_data: List[Dict] = []
                    file_col = next(
                        (
                            c
                            for c in ("file_origin", "File_Origin")
                            if c in df_cells.columns
                        ),
                        None,
                    )
                    if file_col:
                        for fname, n_f in df_cells[file_col].value_counts().items():
                            files_data.append(
                                {
                                    "file": str(fname),
                                    "n_cells": int(n_f),
                                }
                            )

                    # ── Labels des figures ─────────────────────────────────
                    _FIGURE_LABELS = {
                        "fig_overview": "Vue d'ensemble (pré-gating)",
                        "fig_gate_debris": "QC Gate 1 — Exclusion Débris",
                        "fig_gate_singlets": "QC Gate 2 — Exclusion Doublets",
                        "fig_gate_cd45": "QC Gate 3 — Cellules CD45+",
                        "fig_gate_cd34": "QC Gate 4 — Blastes CD34+",
                        "fig_kde_debris": "QC Gate 1 — Densité KDE FSC-A (GMM vs KDE)",
                        "fig_kde_cd45": "QC Gate 3 — Densité KDE CD45 (GMM vs KDE)",
                        "fig_heatmap": "Heatmap MFI — Métaclusters × Marqueurs (Z-score)",
                        "fig_comp": "Distribution Cellules par Métacluster",
                        "fig_umap": "UMAP — Coloré par Métacluster FlowSOM",
                        "fig_mst_static": "MST Statique — Topologie FlowSOM",
                        "fig_sankey": "Diagramme Sankey — Flux de Gating",
                        "fig_mst": "MST Interactif — Populations FlowSOM",
                        "fig_grid_mc": "Grille SOM ScatterGL",
                        "fig_heatmap_clinical": "Expression Phénotypique — Référence vs Métaclusters",
                        "fig_barplots": "Blast Score — Profil Marqueurs",
                        "fig_radar": "Profils Blast — Radar Charts",
                        "fig_stars": "Blast Scores — Bar Chart",
                        "fig_patho_pct": "% Cellules Pathologiques par Cluster",
                        "fig_cells_pct": "% Cellules par Cluster (Distribution Globale)",
                        "fig_mrd_summary": "MRD Résiduelle — Nœuds SOM (JF / Flo + contrôles ELN)",
                    }

                    html_path = exporter.export_html_report(
                        analysis_params={
                            "Transformation": config.transform.method,
                            "Normalisation": config.normalize.method,
                            "Grille SOM": f"{config.flowsom.xdim}×{config.flowsom.ydim}",
                            "Métaclusters": n_meta,
                            "Seed": config.flowsom.seed,
                        },
                        summary_stats={
                            "n_cells": int(X_stacked.shape[0]),
                            "n_markers": len(selected_markers),
                            "n_files": len(samples),
                            "n_clusters": n_meta,
                        },
                        metacluster_table=mc_table,
                        markers=list(selected_markers),
                        matplotlib_figures=_mpl_figures,
                        plotly_figures=_plotly_figures,
                        figure_labels=_FIGURE_LABELS,
                        condition_data=condition_data,
                        files_data=files_data,
                        export_paths=export_paths,
                    )
                    if html_path:
                        export_paths["html_report"] = html_path
                        _logger.info("Rapport HTML: %s", html_path)
                except Exception as _e:
                    _logger.warning("Rapport HTML échoué (non bloquant): %s", _e)

            # ── Assemblage du résultat ────────────────────────────────────────
            elapsed = time.time() - start_time

            result = PipelineResult(
                data=df_cells,
                mfi_matrix=mfi_matrix,
                gating_report=[e.to_dict() for e in self._gating_logger.events],
                clustering_metrics=metrics,
                output_files=export_paths,
                config_snapshot=config.to_dict() if hasattr(config, "to_dict") else {},
                timestamp=timestamp,
                elapsed_seconds=elapsed,
                population_mapping=population_mapping_result,
            )

            _logger.info("=" * 60)
            _logger.info("PIPELINE TERMINÉ EN %.1fs", elapsed)
            _logger.info("  Cellules: %d", result.n_cells)
            _logger.info("  Marqueurs: %d", len(selected_markers))
            _logger.info("  Métaclusters: %d", result.n_metaclusters)
            _logger.info("=" * 60)

            self._result = result
            return result

        except Exception as exc:
            _logger.exception("Erreur critique dans le pipeline: %s", exc)
            # Sécurité : écrire l'erreur dans un fichier pour le mode frozen
            # (logging peut échouer si sys.stderr est None)
            try:
                import traceback as _tb
                err_path = Path(config.paths.output_dir) / "pipeline_error.log"
                err_path.parent.mkdir(parents=True, exist_ok=True)
                err_path.write_text(
                    f"{type(exc).__name__}: {exc}\n\n{''.join(_tb.format_exception(exc))}",
                    encoding="utf-8",
                )
            except Exception:
                pass
            return PipelineResult.failure(error=str(exc), config=config)

    def _load_all_samples(self) -> List[FlowSample]:
        """
        Charge tous les fichiers FCS depuis les dossiers configurés.

        Returns:
            Liste de FlowSample.
        """
        config = self.config
        samples: List[FlowSample] = []

        # Fichiers sains (NBM)
        healthy_folder = Path(getattr(config.paths, "healthy_folder", ""))
        if healthy_folder.exists():
            healthy_files = get_fcs_files(healthy_folder)
            healthy_samples = load_as_flow_samples(healthy_files, condition="Sain")
            samples.extend(healthy_samples)
            _logger.info("  Sain (NBM): %d fichiers", len(healthy_samples))

        # Fichiers pathologiques
        patho_folder = Path(getattr(config.paths, "patho_folder", ""))
        if patho_folder.exists():
            patho_files = get_fcs_files(patho_folder)
            patho_samples = load_as_flow_samples(patho_files, condition="Pathologique")
            samples.extend(patho_samples)
            _logger.info("  Pathologique: %d fichiers", len(patho_samples))

        # Dossier unique (mode simplifié)
        if not samples:
            data_folder = Path(getattr(config.paths, "data_folder", ""))
            if data_folder.exists():
                all_files = get_fcs_files(data_folder)
                all_samples = load_as_flow_samples(all_files, condition="Unknown")
                samples.extend(all_samples)
                _logger.info("  Dossier unique: %d fichiers", len(all_samples))

        return samples

    # ------------------------------------------------------------------
    # FCS export helpers (style monolithe flowsom_pipeline.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _circular_jitter(
        n_points: int,
        cluster_ids: np.ndarray,
        node_sizes: np.ndarray,
        max_radius: float = 0.45,
        min_radius: float = 0.1,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Jitter circulaire style FlowSOM R.

        Le rayon de chaque cellule dépend de la taille de son node SOM.
        On utilise sqrt(u) pour une distribution uniforme dans le disque.
        """
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2 * np.pi, n_points)
        u = rng.uniform(0, 1, n_points)
        max_size = node_sizes.max() if node_sizes.max() > 0 else 1.0
        radii = min_radius + (max_radius - min_radius) * np.sqrt(
            node_sizes[cluster_ids.astype(int)] / max_size
        )
        r = np.sqrt(u) * radii
        return (r * np.cos(theta)).astype(np.float32), (r * np.sin(theta)).astype(
            np.float32
        )

    def _build_fcs_dataframe(
        self,
        processed_samples: List[FlowSample],
        metaclustering: np.ndarray,
        clustering: np.ndarray,
        clusterer: "FlowSOMClusterer",
    ) -> pd.DataFrame:
        """
        Construit le DataFrame complet pour l'export FCS compatible Kaluza.

        Reproduit EXACTEMENT la logique de flowsom_pipeline.py :
          1. Données brutes (pré-transformation) pour TOUS les marqueurs
          2. FlowSOM_metacluster (1-based), FlowSOM_cluster (1-based)
          3. xGrid, yGrid avec jitter circulaire (grille SOM)
          4. xNodes, yNodes avec jitter circulaire (MST)
          5. size (nb cellules par node)
          6. Condition, Condition_Num, File_Origin

        Returns:
            DataFrame avec toutes les colonnes numériques prêtes pour fcswrite.
        """
        from flowsom_pipeline_pro.src.core.clustering import FlowSOMClusterer as _FSC

        # ── 1. Données brutes (toutes colonnes) ───────────────────────────────
        X_raw, raw_var_names, obs = stack_raw_markers(processed_samples)

        n_cells = X_raw.shape[0]
        n_nodes = clusterer.n_nodes

        # ── 2. Taille de chaque node ───────────────────────────────────────────
        node_sizes = clusterer.get_node_sizes()

        # ── 3. Coordonnées grille SOM avec jitter circulaire ──────────────────
        grid_coords = clusterer.get_grid_coords()  # (n_nodes, 2)

        cl_int = clustering.astype(int)
        xGrid_base = grid_coords[cl_int, 0].astype(np.float32)
        yGrid_base = grid_coords[cl_int, 1].astype(np.float32)

        jitter_x, jitter_y = self._circular_jitter(
            n_cells, cl_int, node_sizes, max_radius=0.45, min_radius=0.1
        )
        xGrid_j = xGrid_base + jitter_x
        yGrid_j = yGrid_base + jitter_y
        # Décaler pour que min = 1 (comme le monolithe)
        xGrid = xGrid_j - xGrid_j.min() + 1.0
        yGrid = yGrid_j - yGrid_j.min() + 1.0

        _logger.info("  xGrid: [%.3f – %.3f]", xGrid.min(), xGrid.max())
        _logger.info("  yGrid: [%.3f – %.3f]", yGrid.min(), yGrid.max())

        # ── 4. Coordonnées MST avec jitter circulaire ─────────────────────────
        layout_coords = clusterer.get_layout_coords()  # (n_nodes, 2)
        x_ptp = float(layout_coords[:, 0].max() - layout_coords[:, 0].min())
        y_ptp = float(layout_coords[:, 1].max() - layout_coords[:, 1].min())
        x_range = x_ptp if x_ptp > 0 else 1.0
        y_range = y_ptp if y_ptp > 0 else 1.0
        mst_scale = min(x_range, y_range) / (clusterer.xdim * 2)

        xN_base = layout_coords[cl_int, 0].astype(np.float32)
        yN_base = layout_coords[cl_int, 1].astype(np.float32)
        mst_jx, mst_jy = self._circular_jitter(
            n_cells,
            cl_int,
            node_sizes,
            max_radius=mst_scale * 0.8,
            min_radius=mst_scale * 0.2,
        )
        xNodes_j = xN_base + mst_jx
        yNodes_j = yN_base + mst_jy
        xNodes = xNodes_j - xNodes_j.min() + 1.0
        yNodes = yNodes_j - yNodes_j.min() + 1.0

        _logger.info("  xNodes: [%.3f – %.3f]", xNodes.min(), xNodes.max())
        _logger.info("  yNodes: [%.3f – %.3f]", yNodes.min(), yNodes.max())

        # ── 5. Assemblage ─────────────────────────────────────────────────────
        df = pd.DataFrame(X_raw, columns=raw_var_names)

        # Colonnes FlowSOM (+1 pour commencer à 1 dans Kaluza)
        df["FlowSOM_metacluster"] = (metaclustering + 1).astype(np.float32)
        df["FlowSOM_cluster"] = (clustering + 1).astype(np.float32)

        # Coordonnées SOM
        df["xGrid"] = xGrid.astype(np.float32)
        df["yGrid"] = yGrid.astype(np.float32)
        df["xNodes"] = xNodes.astype(np.float32)
        df["yNodes"] = yNodes.astype(np.float32)

        # Taille du node
        df["size"] = node_sizes[cl_int].astype(np.float32)

        # Métadonnées cellule
        df["Condition"] = obs["condition"].values
        df["Condition_Num"] = np.where(df["Condition"] == "Sain", 1.0, 2.0).astype(
            np.float32
        )
        df["File_Origin"] = obs["file_origin"].values

        _logger.info(
            "  DataFrame FCS complet: %d cellules × %d colonnes",
            df.shape[0],
            df.shape[1],
        )
        return df

    def _compute_metrics(
        self,
        X: np.ndarray,
        metaclustering: np.ndarray,
    ) -> ClusteringMetrics:
        """
        Calcule les métriques de qualité du clustering.

        Args:
            X: Matrice de données (n_cells, n_markers).
            metaclustering: Assignation par cellule.

        Returns:
            ClusteringMetrics.
        """
        n_metaclusters = int(metaclustering.max()) + 1
        silhouette = None

        try:
            from sklearn.metrics import silhouette_score

            # Calculer le silhouette sur un sous-échantillon (rapide)
            rng = np.random.default_rng(42)
            n_sample = min(10_000, X.shape[0])
            idx = rng.choice(X.shape[0], n_sample, replace=False)
            if len(np.unique(metaclustering[idx])) >= 2:
                silhouette = float(silhouette_score(X[idx], metaclustering[idx]))
                _logger.info("  Silhouette score: %.4f", silhouette)
        except ImportError:
            pass
        except Exception as e:
            _logger.warning("Silhouette score non calculé: %s", e)

        counts = np.bincount(metaclustering.astype(int), minlength=n_metaclusters)
        n_per_cluster = {i: int(counts[i]) for i in range(n_metaclusters)}

        return ClusteringMetrics(
            n_metaclusters=n_metaclusters,
            silhouette_score=silhouette,
            n_cells_per_cluster=n_per_cluster,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Fonction standalone — compatibilité avec flowsom_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────


def run_flowsom_pipeline(
    config: Optional[PipelineConfig] = None,
    **kwargs,
) -> PipelineResult:
    """
    Point d'entrée fonctionnel du pipeline FlowSOM.

    Équivalent à ``FlowSOMPipeline(config).execute()`` mais sous forme de
    fonction standalone, selon le style de ``flowsom_pipeline.py``.

    Exemples d'utilisation::

        # Depuis un PipelineConfig existant
        result = run_flowsom_pipeline(config)

        # Depuis un dict de paramètres kwargs (création implicite de config)
        result = run_flowsom_pipeline(
            healthy_folder="/data/NBM",
            patho_folder="/data/LAM",
            xdim=10, ydim=10,
            n_metaclusters=12,
        )

    Args:
        config: PipelineConfig complet. Si None, un config est construit
                depuis les kwargs fournis.
        **kwargs: Paramètres nommés transmis à PipelineConfig (ignorés si
                  config est fourni explicitement).

    Returns:
        PipelineResult avec ``success``, ``n_cells``, ``n_metaclusters``,
        ``output_dir`` et la méthode ``summary()``.

    Raises:
        ValueError: Si config est None et qu'aucun healthy_folder n'est fourni.
    """
    if config is None:
        if not kwargs:
            raise ValueError(
                "run_flowsom_pipeline() requiert soit un PipelineConfig, "
                "soit des kwargs de configuration (ex. healthy_folder=…)."
            )
        config = PipelineConfig.from_dict(kwargs)

    pipeline = FlowSOMPipeline(config=config)
    return pipeline.execute()
