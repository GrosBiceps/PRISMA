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
        # Forcer INFO sur le root — en mode frozen certaines libs imposent WARNING
        # au démarrage, ce qui silencerait tous les messages INFO du pipeline.
        self._root_logger.setLevel(logging.INFO)
        for name in self._NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

        # Retirer tous les StreamHandlers qui écrivent sur stdout/stderr
        # (installés par logging.basicConfig ou les libs tierces) pour éviter
        # les exceptions silencieuses sur streams None en mode frozen console=False.
        import sys as _sys
        dead_streams = {None, getattr(_sys, "__stdout__", None), getattr(_sys, "__stderr__", None)}
        for h in list(self._root_logger.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.handlers.QueueHandler
            ):
                stream = getattr(h, "stream", None)
                if stream is None or stream in dead_streams:
                    self._root_logger.removeHandler(h)

        self._root_logger.addHandler(self._handler)

    def uninstall(self) -> None:
        """Retire le handler. Appelé depuis le thread principal après la fin."""
        self._root_logger.removeHandler(self._handler)
        self._drain_once()  # Vide les derniers messages

    # ── Drainage de la queue vers l'UI ────────────────────────────────────

    def start_drain(self, parent=None) -> None:
        """
        Démarre un QTimer dans le thread principal qui draine la queue toutes
        les 100 ms. Doit être appelé depuis le thread principal.

        parent — QObject optionnel propriétaire du timer (évite les fuites
                 mémoire et ancre le timer à la boucle d'événements Qt même
                 en mode frozen console=False).
        """
        self._timer = QTimer(parent)
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
        """
        Draine au plus 50 messages par tick pour ne pas bloquer l'UI.

        Si la queue contient plus de 200 messages (burst post-GIL après compilation
        Numba), on vide en plusieurs batches différés via QTimer.singleShot pour
        éviter de bloquer l'event loop Qt le temps du traitement complet.
        """
        q = self._handler._queue
        batch: List[str] = []
        for _ in range(50):
            try:
                record = q.get_nowait()
                msg = self._handler.format(record)
                batch.append(msg)
            except Exception:
                break

        if batch:
            self._log_signal.emit("\n".join(batch))


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
    prescreening_done = pyqtSignal(dict)
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
    @staticmethod
    def _warmup_numba() -> None:
        """
        Déclenche la compilation JIT Numba/pynndescent sur un micro-dataset avant
        le vrai pipeline.

        Sans ce warmup, Numba compile ses kernels LLVM lors du premier appel
        FlowSOM réel (20–90 s sur CPU) tout en tenant le GIL Python, ce qui
        gèle l'UI Qt et marque le processus "Pas de réponse" dans le Gestionnaire
        des tâches. Ce warmup s'exécute dans le QThread, hors thread principal.
        """
        try:
            import numpy as _np
            import warnings as _w

            # Micro-SOM : 500 cellules × 4 marqueurs — assez pour déclencher le JIT,
            # trop petit pour durer plus de 2–3 s.
            _rng = _np.random.default_rng(0)
            _X_tiny = _rng.random((500, 4), dtype=_np.float32)

            with _w.catch_warnings():
                _w.simplefilter("ignore")
                import anndata as _ad
                import flowsom as _fs

                _adata = _ad.AnnData(_X_tiny)
                _fs.FlowSOM(
                    _adata,
                    cols_to_use=list(range(4)),
                    xdim=3, ydim=3,
                    n_clusters=2,
                    rlen=5,
                    seed=0,
                )
        except Exception:
            pass  # Non bloquant — si le warmup échoue, le vrai pipeline prend le relais

    def run(self) -> None:
        """Point d'entrée du thread — exécute FlowSOMPipeline.execute()."""
        from flowsom_pipeline_pro.src.pipeline.pipeline_executor import (
            FlowSOMPipeline,
            PipelineStep,
        )

        # Installe le handler de log (thread-safe via QueueHandler)
        self._log_capture.install()

        # Logger racine local pour passer par la queue (jamais d'emit direct cross-thread)
        _log = logging.getLogger("flowsom_pipeline_pro.worker")

        try:
            _log.info("═══ Démarrage du pipeline FlowSOM ═══")

            # Préchauffage Numba/pynndescent JIT avant le vrai pipeline.
            # Évite le freeze "Pas de réponse" dû à la compilation LLVM au premier appel.
            _log.info("[Init] Compilation JIT Numba (première exécution)…")
            self._warmup_numba()
            _log.info("[Init] JIT prêt.")

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

            # Émettre le résultat du pré-screening si disponible
            _ps = getattr(result, "prescreening_result", None) if result else None
            if _ps is not None:
                try:
                    self.prescreening_done.emit(
                        {
                            "n_cd34_pos": int(_ps.n_cd34_pos),
                            "n_cd34_neg": int(_ps.n_cd34_neg),
                            "n_cd45dim": int(_ps.n_cd45dim),
                            "ratio_pct": float(_ps.ratio_pct),
                            "gmm_ratio_pct": float(_ps.gmm_ratio_pct),
                            "kde_ratio_pct": float(_ps.kde_ratio_pct),
                            "alert_level": str(_ps.alert_level),
                            "alert_message": str(_ps.alert_message),
                            "method_used": str(_ps.method_used),
                            "laip_tracking_recommended": bool(_ps.laip_tracking_recommended),
                            "interpretation_warning": str(_ps.interpretation_warning),
                        }
                    )
                except Exception:
                    pass  # Non bloquant

            if result is not None and result.success:
                _log.info(
                    "═══ Pipeline terminé — %s cellules, %s métaclusters ═══",
                    f"{result.n_cells:,}",
                    result.n_metaclusters,
                )
            else:
                _log.info("═══ Pipeline terminé avec des avertissements ═══")

            self.finished.emit(result)

        except Exception as exc:
            tb = traceback.format_exc()
            _log.error("[ERREUR] %s", exc)
            _log.error("%s", tb)
            # Flush queue avant d'émettre les signaux de fin
            self._log_capture._drain_once()
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

        _log = logging.getLogger("flowsom_pipeline_pro.batch_worker")

        try:
            _log.info("═══ Démarrage du mode Batch ═══")
            self.progress.emit(2)

            batch = BatchPipeline(self._config)
            summary = batch.execute(progress_callback=self._on_file_progress)

            self.progress.emit(100)
            n_ok = sum(1 for _, r in summary.get("results", []) if r is not None and r.success)
            n_total = len(summary.get("results", []))
            _log.info("═══ Batch terminé — %d/%d fichier(s) réussis ═══", n_ok, n_total)
            self.finished.emit(summary)

        except Exception as exc:
            tb = traceback.format_exc()
            _log.error("[ERREUR BATCH] %s", exc)
            _log.error("%s", tb)
            self._log_capture._drain_once()
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


class FcsLoaderWorker(QThread):
    """
    Charge un fichier FCS hors du thread principal pour éviter tout gel de l'UI.

    Tente 4 backends dans l'ordre :
      1. flowsom.io.read_FCS
      2. flowio + anndata
      3. fcsparser
      4. lecture binaire brute (_read_fcs_binary du main_window)

    Signaux :
        loaded  — AnnData résultant (objet Python)
        error   — message d'erreur (str) si tous les backends échouent
        log     — messages de progression (str)
    """

    loaded = pyqtSignal(object)   # AnnData
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, file_path: str, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._file_path = file_path

    def run(self) -> None:
        import numpy as np

        fp = self._file_path
        adata = None
        last_error: Optional[Exception] = None

        # ── Backend 1 : flowsom ───────────────────────────────────────────
        try:
            import flowsom as fs
            adata = fs.io.read_FCS(fp)
            self.log.emit("Chargé avec flowsom")
        except Exception as e1:
            self.log.emit(f"flowsom échoué : {str(e1)[:60]}")
            last_error = e1

        # ── Backend 2 : flowio + anndata ─────────────────────────────────
        if adata is None:
            try:
                import flowio
                import anndata as ad

                fcs_data = flowio.FlowData(fp)
                events = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))
                n_ev, n_ch = events.shape
                if n_ev > 0 and n_ch > 0:
                    ch_names: List[str] = []
                    for i in range(1, n_ch + 1):
                        name = None
                        for key in (f"$P{i}N", f"P{i}N", f"$P{i}S", f"P{i}S"):
                            if key in fcs_data.text:
                                name = fcs_data.text[key]
                                break
                        ch_names.append(str(name) if name else f"Channel_{i}")
                    adata = ad.AnnData(events.astype(np.float32))
                    adata.var_names = ch_names
                    self.log.emit("Chargé avec flowio")
                else:
                    raise ValueError(f"Fichier vide : {n_ev} events, {n_ch} canaux")
            except ImportError:
                self.log.emit("flowio non installé")
            except Exception as e2:
                self.log.emit(f"flowio échoué : {str(e2)[:60]}")
                last_error = e2

        # ── Backend 3 : fcsparser ─────────────────────────────────────────
        if adata is None:
            try:
                import fcsparser
                import anndata as ad

                for naming in ("$PnS", "$PnN"):
                    try:
                        meta, data = fcsparser.parse(
                            fp,
                            meta_data_only=False,
                            reformat_meta=False,
                            channel_naming=naming,
                        )
                        adata = ad.AnnData(data.values.astype(np.float32))
                        adata.var_names = list(data.columns)
                        self.log.emit(f"Chargé avec fcsparser ({naming})")
                        break
                    except Exception:
                        pass
            except ImportError:
                self.log.emit("fcsparser non installé")
            except Exception as e3:
                self.log.emit(f"fcsparser échoué : {str(e3)[:60]}")
                last_error = e3

        # ── Backend 4 : lecture binaire brute ────────────────────────────
        if adata is None:
            try:
                adata = self._read_fcs_binary(fp)
                self.log.emit("Chargé avec lecture binaire directe")
            except Exception as e4:
                self.log.emit(f"Binaire échoué : {str(e4)[:60]}")
                last_error = e4

        if adata is None:
            self.error.emit(
                f"Impossible de charger le FCS. Dernière erreur : {last_error}"
            )
            return

        self.loaded.emit(adata)

    def _read_fcs_binary(self, file_path: str):
        """Lecture binaire FCS brute — fallback ultime."""
        import struct
        import numpy as np
        import anndata as ad

        with open(file_path, "rb") as f:
            header = f.read(58)
            if len(header) < 58:
                raise ValueError("Fichier FCS trop court")
            version = header[:6].decode("ascii", errors="replace").strip()
            if not version.startswith("FCS"):
                raise ValueError(f"Version FCS non reconnue : {version}")
            text_start = int(header[10:18].strip())
            text_end = int(header[18:26].strip())
            data_start = int(header[26:34].strip())
            data_end = int(header[34:42].strip())
            f.seek(text_start)
            text_raw = f.read(text_end - text_start + 1).decode("latin-1", errors="replace")
        if not text_raw:
            raise ValueError("Segment TEXT vide")
        delim = text_raw[0]
        parts = text_raw.split(delim)
        kv: dict = {}
        for i in range(1, len(parts) - 1, 2):
            kv[parts[i].strip().upper()] = parts[i + 1].strip() if i + 1 < len(parts) else ""
        n_params = int(kv.get("$PAR", 0))
        n_events = int(kv.get("$TOT", 0))
        data_type = kv.get("$DATATYPE", "F").upper()
        byte_order = kv.get("$BYTEORD", "1,2,3,4")
        endian = "<" if byte_order.startswith("1") else ">"
        fmt_char = {"F": "f", "D": "d", "I": "i"}.get(data_type, "f")
        dtype = np.dtype(f"{endian}{fmt_char}")
        with open(file_path, "rb") as f:
            f.seek(data_start)
            n_bytes = (data_end - data_start + 1) if data_end > data_start else (
                n_events * n_params * dtype.itemsize
            )
            raw = f.read(n_bytes)
        arr = np.frombuffer(raw, dtype=dtype).reshape(n_events, n_params).astype(np.float32)
        channel_names = []
        for i in range(1, n_params + 1):
            name = kv.get(f"$P{i}N") or kv.get(f"$P{i}S") or f"Channel_{i}"
            channel_names.append(name)
        adata = ad.AnnData(arr)
        adata.var_names = channel_names
        return adata
