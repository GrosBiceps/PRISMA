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
from flowsom_pipeline_pro.src.services.preprocessing_service import (
    preprocess_all_samples,
)
from flowsom_pipeline_pro.src.services.clustering_service import (
    run_clustering,
    build_cells_dataframe,
    stack_samples,
)
from flowsom_pipeline_pro.src.services.export_service import ExportService
from flowsom_pipeline_pro.src.utils.logger import GatingLogger, get_logger

_logger = get_logger("pipeline.executor")


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

            # ── Étape 2: Prétraitement ────────────────────────────────────────
            _logger.info("Étape 2: Prétraitement (gating + transformation)...")
            processed_samples = preprocess_all_samples(
                samples, config, gating_logger=self._gating_logger
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

            # ── Étape 6: Exports ──────────────────────────────────────────────
            _logger.info("Étape 6: Exports...")
            output_dir = Path(config.paths.output_dir)
            exporter = ExportService(config, output_dir, timestamp=timestamp)

            input_files = [str(s.path) for s in samples]
            export_paths = exporter.export_all(
                df_cells=df_cells,
                mfi_matrix=mfi_matrix,
                metaclustering=metaclustering,
                selected_markers=selected_markers,
                input_files=input_files,
                gating_logger=self._gating_logger,
            )

            # Plots (si activés dans la config)
            viz_config = getattr(config, "visualization", None)
            if viz_config is not None and getattr(viz_config, "enabled", True):
                condition_labels = (
                    df_cells["condition"].values
                    if "condition" in df_cells.columns
                    else None
                )
                exporter.export_flowsom_plots(
                    mfi_matrix,
                    metaclustering,
                    n_metaclusters=int(metaclustering.max()) + 1,
                    condition_labels=condition_labels,
                )

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

        n_per_cluster = {
            i: int((metaclustering == i).sum()) for i in range(n_metaclusters)
        }

        return ClusteringMetrics(
            n_metaclusters=n_metaclusters,
            silhouette_score=silhouette,
            n_cells_per_cluster=n_per_cluster,
        )
