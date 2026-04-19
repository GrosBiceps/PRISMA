"""
launch_gui.py — Point d'entrée pour la compilation PyInstaller (GUI uniquement).

Usage:
    python launch_gui.py       (développement)
    pyinstaller flowsom_gui.spec  (production → .exe)
"""

import logging as _logging
import multiprocessing
import os
import sys
from pathlib import Path

# freeze_support() doit être le tout premier appel en mode frozen pour intercepter
# les sous-processus spawn déclenchés par Numba/joblib/UMAP avant tout import lourd.
if getattr(sys, "frozen", False):
    multiprocessing.freeze_support()


def _enable_windows_crisp_rendering() -> None:
    """Request per-monitor DPI awareness to avoid blurry bitmap-scaled UI on Windows."""
    if os.name != "nt":
        return

    # Qt DPI env hints must be set before QApplication is created.
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "RoundPreferFloor")

    try:
        import ctypes

        # PER_MONITOR_AWARE_V2 gives the sharpest text on mixed-DPI displays.
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass

    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass

    try:
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# ── Mode windowed (console=False) : sys.stderr/stdout sont None → crash logging ─
# On redirige vers un flux nul AVANT tout import pour éviter les
# AttributeError: 'NoneType' object has no attribute 'write'.
if getattr(sys, "frozen", False):
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")


def _install_crash_guard() -> None:
    """
    Installe sys.excepthook (thread principal) ET threading.excepthook (tous les
    QThread/threads Python) pour intercepter toute exception non catchée et afficher
    une boîte de dialogue Qt au lieu d'une mort silencieuse en mode frozen console=False.
    Un fichier crash.log est écrit à côté du .exe pour diagnostic post-mortem.
    """
    import traceback as _tb
    import threading as _threading

    def _write_crash_log(msg: str) -> None:
        try:
            _log_path = Path(sys.executable).parent / "crash.log"
            with open(_log_path, "a", encoding="utf-8") as _f:
                _f.write(msg)
        except Exception:
            pass

    def _show_crash_dialog(title: str, exc_type, exc_value, msg: str) -> None:
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance()
            if app is not None:
                box = QMessageBox()
                box.setWindowTitle(title)
                box.setIcon(QMessageBox.Critical)
                box.setText(f"<b>{exc_type.__name__}</b>: {exc_value}")
                box.setDetailedText(msg)
                box.exec_()
        except Exception:
            pass

    # ── Hook thread principal ──────────────────────────────────────────────────
    def _main_excepthook(exc_type, exc_value, exc_traceback):
        msg = "".join(_tb.format_exception(exc_type, exc_value, exc_traceback))
        _write_crash_log(f"\n[Main thread crash]\n{msg}\n")
        _show_crash_dialog("Erreur inattendue — PRISMA", exc_type, exc_value, msg)

    sys.excepthook = _main_excepthook

    # ── Hook tous les threads secondaires (QThread inclus) — Python ≥ 3.8 ─────
    def _thread_excepthook(args) -> None:
        msg = "".join(_tb.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        thread_name = getattr(args.thread, "name", str(args.thread))
        _write_crash_log(f"\n[Thread crash — {thread_name}]\n{msg}\n")
        _show_crash_dialog(
            f"Erreur dans un thread — PRISMA ({thread_name})",
            args.exc_type,
            args.exc_value,
            msg,
        )

    _threading.excepthook = _thread_excepthook


_install_crash_guard()

# ── Résolution du chemin vers les dépendances dans le .exe ─────────────────────
# En mode onedir, sys._MEIPASS = sous-dossier _internal à côté de l'exe.
if getattr(sys, "frozen", False):
    _BASE_DIR = Path(sys._MEIPASS)
else:
    _BASE_DIR = Path(__file__).resolve().parent

# Rendre le package flowsom_pipeline_pro importable (mode frozen ou dev)
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

# Idem pour le dossier parent (mode dev sans pip install)
_PARENT = _BASE_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

# ── Suppression des logs verbeux de kaleido (Chromium/CDP) ─────────────────────
# kaleido utilise le logger root Python — on le filtre à WARNING pour ne garder
# que les vraies erreurs, pas les traces internes Chromium/CDP.
for _noisy in (
    "kaleido",
    "kaleido.scopes",
    "kaleido.scopes.base",
    "kaleido.scopes.chromium",
    "chromote",
    "pyppeteer",
    "matplotlib",
    "matplotlib.font_manager",
):
    _logging.getLogger(_noisy).setLevel(_logging.WARNING)

# Important: must run before importing Qt modules / creating QApplication.
_enable_windows_crisp_rendering()

from flowsom_pipeline_pro.gui.main_window import main

if __name__ == "__main__":
    main()
