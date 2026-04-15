# -*- coding: utf-8 -*-
"""
workers.py — QThread workers pour FlowSomAnalyzerPro.

Exécute le pipeline FlowSOM dans un thread séparé pour garder l'UI fluide.
Capture les logs via un QueueHandler thread-safe et les transmet en temps réel
sans détournement de sys.stdout ni scraping des messages de log.
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
import traceback
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from flowsom_pipeline_pro.config.pipeline_config import PipelineConfig

from PyQt5.QtCore import QThread, QTimer, pyqtSignal


class _QtLogHandler(logging.handlers.QueueHandler):
    """
    Handler thread-safe qui place les LogRecord dans une queue interne.
    Un QTimer dans le thread principal draine la queue et émet les signaux Qt.

    Avantages par rapport à l'approche directe :
    - Aucun appel Qt depuis un thread secondaire (évite segfaults).
    - Aucun blocage sur l'Event Loop Qt en cas de burst de messages.
    - QueueHandler est conçu exactement pour cet usage (stdlib Python).
    """

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        super().__init__(self._queue)
        self.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(message)s",
                datefmt="%H:%M:%S",
            )
        )


class LogCapture:
    """
    Capture de logs thread-safe pour l'UI PyQt5.

    Installe un QueueHandler sur le logger root depuis le thread secondaire,
    puis draine la queue via un QTimer dans le thread principal (sans appel
    Qt cross-thread).

    Usage :
        capture = LogCapture(log_signal)
        capture.install()          # depuis le thread worker (avant pipeline)
        capture.start_drain()      # depuis le thread principal (après démarrage worker)
        ...
        capture.stop_drain()       # depuis le thread principal (après fin worker)
        capture.uninstall()        # depuis le thread principal (cleanup)
    """

    # Loggers tiers verbeux à silencer — produisent des milliers de messages
    # DEBUG lors de la compilation JIT (Numba) ou du calcul KNN (pynndescent),
    # ce qui sature la SimpleQueue et freeze l'UI.
    _NOISY_LOGGERS = ("numba", "numba.core", "pynndescent", "umap")

    def __init__(self, log_signal: pyqtSignal) -> None:
        self._log_signal = log_signal
        self._handler = _QtLogHandler()
        self._handler.setLevel(logging.INFO)
        self._timer: Optional[QTimer] = None
        self._root_logger = logging.getLogger()

    # ── Gestion du handler ─────────────────────────────────────────────────

    def install(self) -> None:
        """Ajoute le handler au logger root. Appelé depuis le thread worker."""
        # Ne pas forcer DEBUG sur le root — cela ouvre les vannes de tous les
        # loggers tiers (numba, pynndescent…) et sature la queue.
        if self._root_logger.level == logging.NOTSET:
            self._root_logger.setLevel(logging.INFO)
        for name in self._NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
        self._root_logger.addHandler(self._handler)

    def uninstall(self) -> None:
        """Retire le handler. Appelé depuis le thread principal après la fin."""
        self._root_logger.removeHandler(self._handler)
        self._drain_once()  # Vide les derniers messages

    # ── Drainage de la queue vers l'UI ────────────────────────────────────

    def start_drain(self) -> None:
        """
        Démarre un QTimer dans le thread principal qui draine la queue toutes
        les 100 ms. Doit être appelé depuis le thread principal.
        """
        self._timer = QTimer()
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._drain_once)
        self._timer.start()

    def stop_drain(self) -> None:
        """Arrête le timer de drainage. Doit être appelé depuis le thread principal."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._drain_once()  # Dernier vidage

    def _drain_once(self) -> None:
        """Draine au plus 50 messages par tick pour ne pas bloquer l'UI."""
        q = self._handler._queue
        for _ in range(50):
            try:
                record = q.get_nowait()
                msg = self._handler.format(record)
                self._log_signal.emit(msg)
            except Exception:
                break


class PipelineWorker(QThread):
    """
    Thread dédié à l'exécution du pipeline FlowSOM complet.

    La progression est émise via un progress_callback propre transmis à
    FlowSOMPipeline.execute(), sans scraping des messages de log.

    Signaux :
        log_message  — chaque ligne de log (str)
        progress     — pourcentage estimé 0–100 (int)
        gating_done  — résultats du pre-gating prêts pour validation UI
                       (dict avec clés : n_kept, n_total, pct_kept, fallbacks)
        finished     — résultat PipelineResult (ou None si échec)
        error        — message d'erreur (str)
    """

    log_message = pyqtSignal(str)
    progress = pyqtSignal(int)
    gating_done = pyqtSignal(dict)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        config: "PipelineConfig",
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._log_capture = LogCapture(self.log_message)

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Point d'entrée du thread — exécute FlowSOMPipeline.execute()."""
        from flowsom_pipeline_pro.src.pipeline.pipeline_executor import (
            FlowSOMPipeline,
            PipelineStep,
        )

        # Installe le handler de log (thread-safe via QueueHandler)
        self._log_capture.install()

        try:
            self.log_message.emit("═══ Démarrage du pipeline FlowSOM ═══")

            pipeline = FlowSOMPipeline(self._config)
            _gating_emitted = False

            def _on_progress(step: PipelineStep, pct: int) -> None:
                """Callback appelé par le pipeline à chaque étape majeure."""
                nonlocal _gating_emitted
                self.progress.emit(pct)
                # Émettre le résumé de gating une seule fois dès que GATING_DONE
                if not _gating_emitted and pct >= PipelineStep.GATING_DONE:
                    _gating_emitted = True
                    try:
                        from flowsom_pipeline_pro.src.models.gate_result import gating_reports

                        # Aligner le résumé UI sur la logique Sankey: chaîne COMBINED
                        # (G1 -> G2 -> G3 -> G4), sans sommer les gates entre elles.
                        events = pipeline._gating_logger.events
                        gate_map = {e.gate_name: e for e in events if e.file == "COMBINED"}

                        gate_order = ("G4_cd34", "G3_cd45", "G2_singlets", "G1_debris")
                        terminal_event = next(
                            (gate_map[g] for g in gate_order if g in gate_map), None
                        )
                        first_event = gate_map.get("G1_debris", terminal_event)

                        if terminal_event is not None and first_event is not None:
                            n_kept = int(terminal_event.n_after)
                            n_total = int(first_event.n_before)
                            n_gates = len(gate_map)
                        elif gating_reports:
                            # Fallback défensif en cas d'événements COMBINED absents
                            n_kept = int(gating_reports[-1].n_kept)
                            n_total = int(gating_reports[0].n_total)
                            n_gates = len(gating_reports)
                        else:
                            n_kept = 0
                            n_total = 0
                            n_gates = 0

                        fallbacks = [
                            g.gate_name
                            for g in gating_reports
                            if g.warnings or (g.method and "fallback" in g.method.lower())
                        ]
                        self.gating_done.emit(
                            {
                                "n_kept": n_kept,
                                "n_total": n_total,
                                "pct_kept": round(n_kept / max(n_total, 1) * 100, 1),
                                "n_gates": n_gates,
                                "fallbacks": fallbacks,
                            }
                        )
                    except Exception:
                        pass  # Non bloquant

            result = pipeline.execute(progress_callback=_on_progress)

            self.progress.emit(100)

            if result is not None and result.success:
                self.log_message.emit(
                    f"═══ Pipeline terminé — {result.n_cells:,} cellules, "
                    f"{result.n_metaclusters} métaclusters ═══"
                )
            else:
                self.log_message.emit("═══ Pipeline terminé avec des avertissements ═══")

            self.finished.emit(result)

        except Exception as exc:
            tb = traceback.format_exc()
            self.log_message.emit(f"[ERREUR] {exc}")
            self.log_message.emit(tb)
            self.error.emit(str(exc))
            self.finished.emit(None)

        finally:
            self._log_capture.uninstall()


class BatchWorker(QThread):
    """
    Thread dédié au traitement par lots (BatchPipeline).

    Signaux :
        log_message     — chaque ligne de log (str)
        progress        — pourcentage estimé 0–100 (int)
        file_started    — (index: int, total: int, filename: str)
        file_finished   — (stem: str, success: bool)
        finished        — dict {"results": [...], "excel": str|None}
        error           — message d'erreur (str)
    """

    log_message = pyqtSignal(str)
    progress = pyqtSignal(int)
    file_started = pyqtSignal(int, int, str)
    file_finished = pyqtSignal(str, bool)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        config: "PipelineConfig",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._log_capture = LogCapture(self.log_message)

    def run(self) -> None:
        """Point d'entrée du thread batch."""
        from flowsom_pipeline_pro.src.pipeline.batch_pipeline import BatchPipeline

        self._log_capture.install()

        try:
            self.log_message.emit("═══ Démarrage du mode Batch ═══")
            self.progress.emit(2)

            batch = BatchPipeline(self._config)
            summary = batch.execute(progress_callback=self._on_file_progress)

            self.progress.emit(100)
            n_ok = sum(1 for _, r in summary.get("results", []) if r is not None and r.success)
            n_total = len(summary.get("results", []))
            self.log_message.emit(f"═══ Batch terminé — {n_ok}/{n_total} fichier(s) réussis ═══")
            self.finished.emit(summary)

        except Exception as exc:
            tb = traceback.format_exc()
            self.log_message.emit(f"[ERREUR BATCH] {exc}")
            self.log_message.emit(tb)
            self.error.emit(str(exc))
            self.finished.emit(None)

        finally:
            self._log_capture.uninstall()

    def _on_file_progress(self, current: int, total: int, filename: str) -> None:
        """Appelé par BatchPipeline entre chaque fichier."""
        self.file_started.emit(current, total, filename)
        if total > 0:
            pct = int(current / total * 95)
            self.progress.emit(max(2, pct))


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
