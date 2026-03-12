# -*- coding: utf-8 -*-
"""
workers.py — QThread workers pour FlowSomAnalyzerPro.

Exécute le pipeline FlowSOM dans un thread séparé pour garder l'UI fluide.
Capture les logs via un handler logging dédié et les transmet en temps réel.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import QThread, pyqtSignal

# Carte de progression : (fragment de message log → % avancement)
_PIPELINE_PROGRESS_MAP: List[Tuple[str, int]] = [
    ("Étape 1:", 8),
    ("Sain (NBM):", 10),
    ("Pathologique:", 12),
    ("Étape 2:", 20),
    ("Gating combiné", 25),
    ("après prétraitement", 30),
    ("Étape 3:", 40),
    ("FlowSOM terminé:", 60),
    ("Étape 4:", 65),
    ("DataFrame FCS complet", 67),
    ("Calcul UMAP", 70),
    ("UMAP sauvegardé", 73),
    ("MST statique sauvegardé", 76),
    ("MST Plotly sauvegardé", 78),
    ("SOM Grid Plotly sauvegardé", 80),
    ("Star Chart", 82),
    ("Grille SOM statique sauvegardée", 84),
    ("Radar métaclusters sauvegardé", 85),
    ("Bar chart", 86),
    ("Vue combinée nœuds SOM", 87),
    ("Étape 6:", 88),
    ("Distribution Sain/Patho exportee", 90),
    ("Exports terminés:", 91),
    ("Étape 7:", 93),
    ("Mapping populations OK", 95),
    ("Génération du rapport HTML", 96),
    ("Rapport HTML:", 97),
    ("PIPELINE TERMINÉ", 98),
]


class LogCapture(logging.Handler):
    """Handler logging qui émet chaque message via un signal Qt et infère la progression."""

    def __init__(
        self, log_signal: pyqtSignal, progress_signal: Optional[pyqtSignal] = None
    ) -> None:
        super().__init__()
        self._log_signal = log_signal
        self._progress_signal = progress_signal
        self._last_progress = 0
        self.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._log_signal.emit(msg)
            # Infère la progression depuis le contenu du message
            if self._progress_signal is not None:
                raw_msg = record.getMessage()
                for pattern, pct in _PIPELINE_PROGRESS_MAP:
                    if pattern in raw_msg and pct > self._last_progress:
                        self._last_progress = pct
                        self._progress_signal.emit(pct)
                        break
        except Exception:
            pass


class PipelineWorker(QThread):
    """
    Thread dédié à l'exécution du pipeline FlowSOM complet.

    Signaux :
        log_message  — chaque ligne de log (str)
        progress     — pourcentage estimé 0–100 (int)
        finished     — résultat PipelineResult (ou None si échec)
        error        — message d'erreur (str)
    """

    log_message = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        config: Any,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._log_handler: Optional[LogCapture] = None

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Point d'entrée du thread — exécute FlowSOMPipeline.execute()."""
        from flowsom_pipeline_pro.config.pipeline_config import PipelineConfig
        from flowsom_pipeline_pro.src.pipeline.pipeline_executor import FlowSOMPipeline

        # Installe le handler de log pour capturer les messages ET la progression
        root_logger = logging.getLogger()
        self._log_handler = LogCapture(self.log_message, progress_signal=self.progress)
        self._log_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(self._log_handler)

        # Redirige stdout/stderr via logging pour capturer les print()
        stdout_logger = _StdoutToSignal(self.log_message)

        try:
            self.log_message.emit("═══ Démarrage du pipeline FlowSOM ═══")
            self.progress.emit(5)

            pipeline = FlowSOMPipeline(self._config)

            import sys

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_logger
            sys.stderr = stdout_logger

            try:
                result = pipeline.execute()
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            self.progress.emit(100)

            if result is not None and result.success:
                self.log_message.emit(
                    f"═══ Pipeline terminé — {result.n_cells:,} cellules, "
                    f"{result.n_metaclusters} métaclusters ═══"
                )
            else:
                self.log_message.emit(
                    "═══ Pipeline terminé avec des avertissements ═══"
                )

            self.finished.emit(result)

        except Exception as exc:
            tb = traceback.format_exc()
            self.log_message.emit(f"[ERREUR] {exc}")
            self.log_message.emit(tb)
            self.error.emit(str(exc))
            self.finished.emit(None)

        finally:
            if self._log_handler is not None:
                root_logger.removeHandler(self._log_handler)


class SpiderPlotWorker(QThread):
    """
    Thread pour générer un Spider Plot (radar) matplotlib pour un nœud SOM donné.

    Signaux :
        figure_ready — matplotlib.figure.Figure prête à afficher
        error        — message d'erreur (str)
    """

    figure_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        mfi_row: Any,  # ndarray ou Series — profil MFI du cluster
        marker_names: list,  # subset des marqueurs à afficher
        cluster_label: str,  # libellé pour le titre (ex. "Cluster 42")
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._mfi_row = mfi_row
        self._marker_names = marker_names
        self._cluster_label = cluster_label

    def run(self) -> None:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            markers = self._marker_names
            if not markers:
                self.error.emit("Aucun marqueur sélectionné pour le Spider Plot.")
                return

            # Extrait les valeurs des marqueurs sélectionnés
            import pandas as pd

            row = self._mfi_row
            if isinstance(row, pd.Series):
                values = np.array([row.get(m, 0.0) for m in markers], dtype=float)
            else:
                values = np.asarray(row, dtype=float)[: len(markers)]

            # Normalisation min-max
            vmin, vmax = values.min(), values.max()
            rng = vmax - vmin if vmax != vmin else 1.0
            values_norm = (values - vmin) / rng

            n = len(markers)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
            vals_plot = np.concatenate([values_norm, [values_norm[0]]])
            angles_plot = angles + angles[:1]

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
            fig.patch.set_facecolor("#1e1e2e")
            ax.set_facecolor("#1e1e2e")

            ax.plot(angles_plot, vals_plot, "o-", color="#a6e3a1", linewidth=2)
            ax.fill(angles_plot, vals_plot, alpha=0.25, color="#a6e3a1")

            ax.set_xticks(angles)
            ax.set_xticklabels(markers, size=8, color="#cdd6f4")
            ax.tick_params(axis="y", colors="#6c7086")
            ax.spines["polar"].set_color("#45475a")
            ax.set_ylim(0, 1)
            ax.set_title(
                self._cluster_label,
                color="#cdd6f4",
                fontsize=12,
                fontweight="bold",
                pad=20,
            )
            fig.tight_layout()
            self.figure_ready.emit(fig)
        except Exception as exc:
            self.error.emit(str(exc))


# ──────────────────────────────────────────────────────────────────────
# Utilitaire interne
# ──────────────────────────────────────────────────────────────────────


class _StdoutToSignal:
    """Redirige les appels write() vers un pyqtSignal(str)."""

    def __init__(self, signal: pyqtSignal) -> None:
        self._signal = signal
        self._buffer = ""

    def write(self, text: str) -> None:
        if not text:
            return
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._signal.emit(line)

    def flush(self) -> None:
        if self._buffer.strip():
            self._signal.emit(self._buffer.strip())
            self._buffer = ""

    def isatty(self) -> bool:
        return False
