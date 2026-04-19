# -*- coding: utf-8 -*-
"""
main_window.py — Interface graphique FlowSomAnalyzerPro v3 (Wizard / Stepper).

Architecture UX en 5 étapes (QStackedWidget) :
    Étape 1 — Accueil   : Landing page de démarrage avec CTA
    Étape 2 — Import    : Drag & Drop fichiers FCS + sélection dossiers
    Étape 3 — Paramétrage : Grille SOM, MRD, gating, options — avec validation visuelle
    Étape 4 — Exécution : Console log + barre de progression par étape
    Étape 5 — Résultats : Onglets MRD / Visualisation / Clusters / Représentations

Design :
  - Sidebar de navigation (StepSidebar) avec indicateurs d'état colorés
  - Boutons avec icônes qtawesome (fa5s)
    - Flat Design PRISMA v2 (styles.py)
  - Police Segoe UI / Inter / Roboto
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from flowsom_pipeline_pro.config.pipeline_config import PipelineConfig
    from flowsom_pipeline_pro.src.models.pipeline_result import PipelineResult

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QProgressBar,
    QStatusBar,
    QMessageBox,
    QTabWidget,
    QScrollArea,
    QFrame,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGridLayout,
    QTextEdit,
    QSizePolicy,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QLineEdit,
)
from PyQt5.QtCore import Qt, QSize, QMimeData, QUrl, QByteArray
from PyQt5.QtGui import QFont, QColor, QDragEnterEvent, QDropEvent, QIcon, QFontDatabase, QFontInfo

try:
    from PyQt5.QtSvg import QSvgWidget

    _SVG_AVAILABLE = True
except ImportError:
    _SVG_AVAILABLE = False

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib

matplotlib.use("Qt5Agg")

from flowsom_pipeline_pro.gui.styles import STYLESHEET, COLORS
from flowsom_pipeline_pro.gui.prisma_icons import get_prisma_icon


def _asset_path(filename: str) -> Path:
    """Resolve asset path whether running from source or PyInstaller bundle."""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path(__file__).parent.parent
    return base / "assets" / filename


def _register_embedded_fonts() -> None:
    """Register project fonts (assets/fonts) for deterministic rendering."""
    # Variable font covers all requested Outfit weights (200..900).
    font_files = [
        "fonts/Outfit[wght].ttf",
    ]
    for rel_path in font_files:
        font_path = _asset_path(rel_path)
        if not font_path.exists():
            continue
        QFontDatabase.addApplicationFont(str(font_path))


from flowsom_pipeline_pro.gui.workers import PipelineWorker, SpiderPlotWorker
from flowsom_pipeline_pro.gui.tabs.home_tab import HomeTab
from flowsom_pipeline_pro.gui.widgets.log_console import LogConsole
from flowsom_pipeline_pro.gui.widgets.toggle_switch import ToggleSwitch
from flowsom_pipeline_pro.gui.widgets.settings_card import SettingsCard

# qtawesome — icônes vectorielles Font Awesome 5
try:
    import qtawesome as qta

    _QTA = True
except ImportError:
    _QTA = False

_WEBENGINE = False
_WEBENGINE_ACTIVE = False

# Chemin YAML
if getattr(sys, "frozen", False):
    _DEFAULT_CONFIG_PATH = Path(sys.executable).parent / "config" / "default_config.yaml"
    _MRD_CONFIG_PATH = Path(sys.executable).parent / "config" / "mrd_config.yaml"
else:
    _DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"
    _MRD_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "mrd_config.yaml"


# ══════════════════════════════════════════════════════════════════════
# Utilitaires
# ══════════════════════════════════════════════════════════════════════


def _icon(name: str, color: str = "#EEF2F7", size: int = 16) -> Any:
    """Renvoie un QIcon PRISMA, sinon qtawesome, sinon None."""
    if name.startswith("prisma."):
        custom = get_prisma_icon(name.split(".", 1)[1], size=size, color=color)
        if custom is not None:
            return custom

    if _QTA:
        try:
            return qta.icon(name, color=color)
        except Exception:
            pass
    return None


class DarkComboBox(QComboBox):
    """QComboBox avec popup sans cadre/ombre natif Windows (fix fond blanc)."""

    def showPopup(self) -> None:  # noqa: N802
        popup = self.view().window()
        popup.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        super().showPopup()


class MatplotlibCanvas(FigureCanvas):
    """Canvas Matplotlib intégré dans PyQt5 avec thème sombre."""

    def __init__(self, parent: Optional[QWidget] = None, width: int = 8, height: int = 6) -> None:
        self.fig = Figure(figsize=(width, height), dpi=100)
        self.fig.patch.set_facecolor(COLORS["surface"])
        self.axes = self.fig.add_subplot(111)
        self._style_axes(self.axes)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _style_axes(self, ax: Any) -> None:
        ax.set_facecolor(COLORS["surface"])
        ax.tick_params(colors=COLORS["paper"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["raised"])

    def clear_and_reset(self) -> None:
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self._style_axes(self.axes)
        self.draw()

    def display_figure(self, fig: Figure) -> None:
        import matplotlib.pyplot as plt

        old_fig = self.fig
        self.fig = fig
        self.figure = fig
        self.fig.set_canvas(self)
        self.fig.patch.set_facecolor(COLORS["surface"])
        dpi = self.fig.get_dpi() or 100
        w_px, h_px = max(1, self.width()), max(1, self.height())
        self.fig.set_size_inches(w_px / dpi, h_px / dpi)
        self.draw()
        plt.close(old_fig)


# ══════════════════════════════════════════════════════════════════════
# Zone Drag & Drop (Étape 1)
# ══════════════════════════════════════════════════════════════════════


class DropZoneLabel(QLabel):
    """Label qui accepte le glisser-déposer de dossiers."""

    def __init__(self, placeholder: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(placeholder, parent)
        self._path: Optional[str] = None
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(72)
        self.setWordWrap(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("dragOver", True)
            self.style().unpolish(self)
            self.style().polish(self)

    def dragLeaveEvent(self, event: Any) -> None:
        self.setProperty("dragOver", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event: QDropEvent) -> None:
        self.setProperty("dragOver", False)
        self.style().unpolish(self)
        self.style().polish(self)
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).is_dir():
                self.set_path(path)
                return
        # Fichier FCS → on prend le parent
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).is_file():
                self.set_path(str(Path(path).parent))
                return

    def set_path(self, path: str) -> None:
        self._path = path
        name = Path(path).name
        self.setText(f"  {name}\n  {path}")
        self.setObjectName("dropZoneOk")
        self.style().unpolish(self)
        self.style().polish(self)
        # Notifier la fenêtre principale pour rafraîchir la prévisualisation FCS
        mw = self._find_main_window()
        if mw and hasattr(mw, "_refresh_fcs_preview"):
            mw._refresh_fcs_preview()

    def _find_main_window(self) -> Optional[Any]:
        """Remonte la hiérarchie de parents pour trouver FlowSomAnalyzerPro."""
        p = self.parent()
        while p is not None:
            # Vérification par nom de classe pour éviter l'import circulaire
            if type(p).__name__ == "FlowSomAnalyzerPro":
                return p
            p = p.parent() if hasattr(p, "parent") else None
        return None

    @property
    def path(self) -> Optional[str]:
        return self._path


# ══════════════════════════════════════════════════════════════════════
# Sidebar Stepper
# ══════════════════════════════════════════════════════════════════════

_STEPS = [
    ("1", "Accueil", "Démarrage"),
    ("2", "Import", "Dossiers FCS"),
    ("3", "Paramétrage", "SOM · MRD · Gating"),
    ("4", "Exécution", "Lancement & logs"),
    ("5", "Résultats", "MRD · Visualisation"),
]

_STEP_ICONS = [
    "prisma.dot-plot",
    "prisma.fcs-file",
    "prisma.som-grid",
    "prisma.batch-cohort",
    "prisma.heatmap",
]


class StepSidebar(QWidget):
    """Barre latérale de navigation entre les 5 étapes du wizard."""

    # État : 0=pending, 1=active, 2=done, 3=error
    STATE_PENDING = 0
    STATE_ACTIVE = 1
    STATE_DONE = 2
    STATE_ERROR = 3

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("stepSidebar")
        self.setFixedWidth(220)
        self._buttons: List[QPushButton] = []
        self._states: List[int] = [self.STATE_PENDING] * len(_STEPS)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header logo ──
        header = QWidget()
        header.setObjectName("sidebarHeader")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(12, 16, 12, 14)
        h_layout.setSpacing(4)

        _LOGO_SVG = b"""<svg width="196" height="56" viewBox="0 0 196 56" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="lg1s" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="white" stop-opacity=".5"/>
      <stop offset="100%" stop-color="white" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <polygon points="26,4 48,50 4,50" fill="rgba(255,255,255,.025)" stroke="rgba(255,255,255,.18)" stroke-width="1.1"/>
  <circle cx="26" cy="27" r="2" fill="white" opacity=".35"/>
  <circle cx="26" cy="27" r="5.5" fill="url(#lg1s)"/>
  <line x1="26" y1="27" x2="56" y2="10" stroke="#7B52FF" stroke-width="1" opacity=".7"/>
  <line x1="26" y1="27" x2="56" y2="19" stroke="#5BAAFF" stroke-width=".9" opacity=".55"/>
  <line x1="26" y1="27" x2="56" y2="27" stroke="#39FF8A" stroke-width="1.3" opacity=".8"/>
  <line x1="26" y1="27" x2="56" y2="36" stroke="#FF9B3D" stroke-width=".9" opacity=".6"/>
  <line x1="26" y1="27" x2="56" y2="45" stroke="#FF3D6E" stroke-width="1" opacity=".7"/>
  <circle cx="58" cy="9" r="2.5" fill="#7B52FF"/><circle cx="65" cy="6" r="1.8" fill="#7B52FF" opacity=".55"/>
  <circle cx="61" cy="15" r="1.5" fill="#7B52FF" opacity=".4"/>
  <circle cx="58" cy="18" r="2.2" fill="#5BAAFF" opacity=".8"/><circle cx="65" cy="14" r="1.5" fill="#5BAAFF" opacity=".45"/>
  <circle cx="59" cy="26" r="3" fill="#39FF8A"/><circle cx="67" cy="23" r="1.8" fill="#39FF8A" opacity=".6"/>
  <circle cx="65" cy="30" r="1.5" fill="#39FF8A" opacity=".5"/>
  <circle cx="58" cy="36" r="2.5" fill="#FF9B3D" opacity=".88"/><circle cx="66" cy="32" r="1.5" fill="#FF9B3D" opacity=".5"/>
  <circle cx="64" cy="40" r="1.8" fill="#FF9B3D" opacity=".45"/>
  <circle cx="58" cy="44" r="2.2" fill="#FF3D6E"/><circle cx="65" cy="41" r="1.2" fill="#FF3D6E" opacity=".5"/>
  <text x="76" y="34" font-family="Segoe UI, sans-serif" font-weight="900" font-size="26" fill="white" letter-spacing="-1.5">PRISM</text>
  <text x="162" y="34" font-family="Segoe UI, sans-serif" font-weight="900" font-size="26" fill="none" stroke="#39FF8A" stroke-width="1.5" letter-spacing="-1.5">A</text>
</svg>"""

        if _SVG_AVAILABLE:
            logo_widget = QSvgWidget()
            logo_widget.load(QByteArray(_LOGO_SVG))
            logo_widget.setFixedSize(196, 56)
            logo_widget.setStyleSheet("background: transparent;")
            h_layout.addWidget(logo_widget, alignment=Qt.AlignLeft)
        else:
            lbl_app = QLabel("PRISMA")
            lbl_app.setObjectName("brandTitle")
            h_layout.addWidget(lbl_app)

        lbl_sub = QLabel("CYTOMETRY · ANALYSIS SUITE")
        lbl_sub.setObjectName("brandSubtitle")
        h_layout.addWidget(lbl_sub)

        sep = QFrame()
        sep.setObjectName("sidebarSeparator")
        sep.setFrameShape(QFrame.HLine)
        h_layout.addWidget(sep)

        root.addWidget(header)

        # Spacer top
        root.addSpacing(8)

        # Étapes
        for i, (num, title, sub) in enumerate(_STEPS):
            btn = QPushButton()
            btn.setObjectName("stepBtn")
            btn.setCheckable(False)
            btn.setFlat(True)
            self._set_button_content(btn, i)
            btn.clicked.connect(lambda checked, idx=i: self._on_click(idx))
            self._buttons.append(btn)
            root.addWidget(btn)

        root.addStretch()

        # Version
        lbl_ver = QLabel("V3.0 · MAGNE FLORIAN")
        lbl_ver.setObjectName("versionLabel")
        lbl_ver.setAlignment(Qt.AlignLeft)
        root.addWidget(lbl_ver)

    def _set_button_content(self, btn: QPushButton, idx: int) -> None:
        num, title, sub = _STEPS[idx]
        state = self._states[idx]

        # Icône (qtawesome)
        ic_name = _STEP_ICONS[idx]
        if state == self.STATE_DONE:
            ic_color = "#39FF8A"
            ic_name = "prisma.check-circle"
        elif state == self.STATE_ERROR:
            ic_color = "#FF3D6E"
            ic_name = "prisma.alert-triangle"
        elif state == self.STATE_ACTIVE:
            ic_color = "#5BAAFF"
        else:
            ic_color = "#2A3342"

        ico = _icon(ic_name, ic_color, 24)
        if ico:
            btn.setIcon(ico)
            btn.setIconSize(QSize(24, 24))

        btn.setText(f"  {title}\n  {sub}")
        btn.setFont(QFont("Segoe UI", 9))

        if state == self.STATE_ACTIVE:
            btn.setObjectName("stepBtnActive")
        elif state == self.STATE_DONE:
            btn.setObjectName("stepBtnDone")
        else:
            btn.setObjectName("stepBtn")

        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def set_active(self, idx: int) -> None:
        for i, state in enumerate(self._states):
            if state == self.STATE_ACTIVE:
                self._states[i] = self.STATE_PENDING
        self._states[idx] = self.STATE_ACTIVE
        self._refresh()

    def set_done(self, idx: int) -> None:
        self._states[idx] = self.STATE_DONE
        self._refresh()

    def set_error(self, idx: int) -> None:
        self._states[idx] = self.STATE_ERROR
        self._refresh()

    def _refresh(self) -> None:
        for i, btn in enumerate(self._buttons):
            self._set_button_content(btn, i)

    def _on_click(self, idx: int) -> None:
        # Délègue au parent (FlowSomAnalyzerPro) via signal simulé
        mw = self._find_main_window()
        if mw:
            mw._navigate_to_step(idx)

    def _find_main_window(self) -> Optional["FlowSomAnalyzerPro"]:
        p = self.parent()
        while p:
            if isinstance(p, FlowSomAnalyzerPro):
                return p
            p = p.parent() if hasattr(p, "parent") else None
        return None


# ══════════════════════════════════════════════════════════════════════
# Fenêtre principale — Wizard 5 étapes
# ══════════════════════════════════════════════════════════════════════


class FlowSomAnalyzerPro(QMainWindow):
    """Application GUI FlowSOM Analysis Pro — architecture Wizard (5 étapes)."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PRISMA")
        self.setMinimumSize(1280, 820)
        self.resize(1600, 960)

        # App icon — works both from source and PyInstaller bundle
        _ico = _asset_path("prisma_logo.ico")
        if _ico.exists():
            self.setWindowIcon(QIcon(str(_ico)))

        self.setStyleSheet(STYLESHEET)

        # État interne
        self._config: Optional["PipelineConfig"] = None
        self._mrd_raw: Dict[str, Any] = {}
        self._result: Optional["PipelineResult"] = None
        self._worker: Optional[PipelineWorker] = None
        self._spider_worker: Optional[SpiderPlotWorker] = None
        self._cluster_mfi: Optional[Any] = None
        self._all_markers: List[str] = []
        self._output_dir: Optional[Path] = None
        self._output_plot_paths: Dict[str, str] = {}
        self._gate_plot_paths: Dict[str, str] = {}
        self._combined_html_path: Optional[str] = None
        self.current_fcs_adata: Optional[Any] = None
        self._patho_fcs_path: Optional[str] = None  # FCS patho auto-chargé après pipeline
        self._pending_prescreening: Optional[Dict] = None
        self._per_file_rename_rules: Dict[str, List] = {}

        self._current_step = 0

        self._init_ui()
        self._load_default_config()
        self._restore_session()
        self.statusBar().showMessage("Étape 1 / 5 — Accueil")

    # ------------------------------------------------------------------
    # Construction UI
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        central = QWidget()
        central.setObjectName("wizardShell")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        self._sidebar = StepSidebar(self)
        root.addWidget(self._sidebar)

        # Stack des 5 étapes
        self._step_stack = QStackedWidget()
        self._step_stack.setObjectName("stepContent")
        root.addWidget(self._step_stack, 1)

        # Étape 1 — Accueil
        self._step_stack.addWidget(self._build_step0_welcome())
        # Étape 2 — Import
        self._step_stack.addWidget(self._build_step1_import())
        # Étape 3 — Paramétrage
        self._step_stack.addWidget(self._build_step2_params())
        # Étape 4 — Exécution
        self._step_stack.addWidget(self._build_step3_run())
        # Étape 5 — Résultats
        self._step_stack.addWidget(self._build_step4_results())

        self._navigate_to_step(0)

    # ──────────────────────────────────────────────────────────────────
    # Navigation
    # ──────────────────────────────────────────────────────────────────

    def _navigate_to_step(self, idx: int) -> None:
        self._current_step = idx
        self._step_stack.setCurrentIndex(idx)
        self._sidebar.set_active(idx)
        labels = [
            "Étape 1 / 5 — Accueil",
            "Étape 2 / 5 — Importation des données FCS",
            "Étape 3 / 5 — Paramétrage du pipeline",
            "Étape 4 / 5 — Exécution du pipeline",
            "Étape 5 / 5 — Résultats & exports",
        ]
        self.statusBar().showMessage(labels[idx])

    # ══════════════════════════════════════════════════════════════════
    # ÉTAPE 1 — Accueil (Landing)
    # ══════════════════════════════════════════════════════════════════

    def _build_step0_welcome(self) -> QWidget:
        page = QWidget()
        page.setObjectName("welcomePage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(52, 28, 52, 28)
        layout.setSpacing(18)
        layout.setAlignment(Qt.AlignCenter)

        card = QWidget()
        card.setObjectName("welcomeCard")
        card.setMinimumWidth(880)
        card.setMaximumWidth(1040)

        c = QVBoxLayout(card)
        c.setContentsMargins(36, 28, 36, 24)
        c.setSpacing(14)

        top_spectrum = QFrame()
        top_spectrum.setObjectName("welcomeSpectrum")
        top_spectrum.setFixedHeight(1)
        c.addWidget(top_spectrum)

        badge = QLabel("CYTOMETRY · UNSUPERVISED ANALYSIS · RESEARCH TOOL")
        badge.setObjectName("welcomeEyebrow")
        badge.setAlignment(Qt.AlignCenter)
        c.addWidget(badge, alignment=Qt.AlignHCenter)

        title = QLabel(
            "<span style='color:#EEF2F7;'>PRISM</span><span style='color:#39FF8A;'>A</span>"
        )
        title.setObjectName("welcomeHeroTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setTextFormat(Qt.RichText)
        c.addWidget(title)

        expansion = QLabel(
            "<b>P</b>ipeline for <b>R</b>apid <b>I</b>dentification, <b>S</b>ingle-cell "
            "<b>M</b>apping &amp; <b>A</b>nalysis"
        )
        expansion.setObjectName("welcomeExpansion")
        expansion.setAlignment(Qt.AlignCenter)
        expansion.setTextFormat(Qt.RichText)
        c.addWidget(expansion)

        sub = QLabel('"Resolve the unseen." — unsupervised cytometry pipeline in 3 steps.')
        sub.setObjectName("welcomeSub")
        sub.setAlignment(Qt.AlignCenter)
        c.addWidget(sub)

        ch_row = QWidget()
        ch_row.setObjectName("welcomeChannelRow")
        ch_h = QHBoxLayout(ch_row)
        ch_h.setContentsMargins(0, 2, 0, 4)
        ch_h.setSpacing(8)
        ch_h.addStretch()

        for channel, label in (
            ("fitc", "FITC · MONO"),
            ("pe", "PE · GRANULO"),
            ("apc", "APC · MRD"),
            ("v450", "V450 · LYMPHO"),
        ):
            tag = QLabel(label)
            tag.setObjectName("welcomeChannelBadge")
            tag.setProperty("channel", channel)
            tag.setAlignment(Qt.AlignCenter)
            ch_h.addWidget(tag)

        ch_h.addStretch()
        c.addWidget(ch_row)

        body = QWidget()
        body.setObjectName("welcomeBody")
        body_h = QHBoxLayout(body)
        body_h.setContentsMargins(0, 0, 0, 0)
        body_h.setSpacing(2)

        col_left = QWidget()
        col_left.setObjectName("welcomeColLeft")
        l = QVBoxLayout(col_left)
        l.setContentsMargins(22, 20, 22, 20)
        l.setSpacing(10)

        l_title = QLabel("GUIDED PATH")
        l_title.setObjectName("welcomeColTitle")
        l.addWidget(l_title)
        for idx, item in [
            ("01", "Import FCS folders"),
            ("02", "Configure pipeline"),
            ("03", "Execute & visualize"),
        ]:
            lbl = QLabel(
                f"<span style='color:#39FF8A;'>{idx}</span>"
                f" <span style='color:#EEF2F7;'>· {item}</span>"
            )
            lbl.setObjectName("welcomeColItem")
            lbl.setProperty("lane", "left")
            lbl.setTextFormat(Qt.RichText)
            lbl.setWordWrap(True)
            l.addWidget(lbl)
        l.addStretch()

        col_right = QWidget()
        col_right.setObjectName("welcomeColRight")
        r = QVBoxLayout(col_right)
        r.setContentsMargins(22, 20, 22, 20)
        r.setSpacing(10)

        r_title = QLabel("AVAILABLE OUTPUTS")
        r_title.setObjectName("welcomeColTitle")
        r.addWidget(r_title)
        for item in [
            "▹ MRD gauges (JF / Flo / ELN)",
            "▹ Synthetic clinical conclusion",
            "▹ Clusters, visualizations & exports",
        ]:
            lbl = QLabel(f"<span style='color:#5BAAFF;'>▹</span> {item[2:]}")
            lbl.setObjectName("welcomeColItem")
            lbl.setProperty("lane", "right")
            lbl.setTextFormat(Qt.RichText)
            lbl.setWordWrap(True)
            r.addWidget(lbl)
        r.addStretch()

        body_h.addWidget(col_left, 1)
        body_h.addWidget(col_right, 1)
        c.addWidget(body)

        info = QLabel(
            "After execution: MRD results, visualizations, clusters and HTML/CSV/FCS exports."
        )
        info.setObjectName("welcomeInfo")
        info.setAlignment(Qt.AlignCenter)
        info.setWordWrap(True)
        c.addWidget(info)

        stats = QWidget()
        stats.setObjectName("welcomeStatsRow")
        stats_h = QHBoxLayout(stats)
        stats_h.setContentsMargins(0, 0, 0, 0)
        stats_h.setSpacing(2)

        def _stat_block(value: str, key: str, tone: str) -> QWidget:
            w = QWidget()
            w.setObjectName("welcomeStatCard")
            v = QVBoxLayout(w)
            v.setContentsMargins(14, 12, 14, 10)
            v.setSpacing(3)

            lbl_v = QLabel(value)
            lbl_v.setObjectName("welcomeStatValue")
            lbl_v.setProperty("tone", tone)
            v.addWidget(lbl_v)

            lbl_k = QLabel(key)
            lbl_k.setObjectName("welcomeStatKey")
            v.addWidget(lbl_k)
            return w

        stats_h.addWidget(_stat_block("<0.1%", "MRD sensitivity", "accent"), 1)
        stats_h.addWidget(_stat_block("No limit", "Events / sample", "info"), 1)
        stats_h.addWidget(_stat_block("No limit", "Max SOM grid", "brand"), 1)
        stats_h.addWidget(_stat_block("0 ms", "Manual gating", "warm"), 1)
        c.addWidget(stats)

        c.addSpacing(2)
        btn_start = QPushButton("  START FILE SELECTION  →")
        btn_start.setObjectName("welcomeCta")
        btn_start.setMinimumHeight(46)
        btn_start.setMinimumWidth(320)
        ico_next = _icon("prisma.arrow-right", "#FFFFFF")
        if ico_next:
            btn_start.setIcon(ico_next)
            btn_start.setIconSize(QSize(16, 16))
        btn_start.clicked.connect(lambda: self._navigate_to_step(1))
        c.addWidget(btn_start, alignment=Qt.AlignHCenter)

        hint = QLabel("Tip · drag-and-drop folders to speed up FCS import.")
        hint.setObjectName("welcomeHint")
        hint.setAlignment(Qt.AlignCenter)
        c.addWidget(hint)

        bottom_spectrum = QFrame()
        bottom_spectrum.setObjectName("welcomeSpectrum")
        bottom_spectrum.setFixedHeight(1)
        c.addWidget(bottom_spectrum)

        layout.addWidget(card, alignment=Qt.AlignCenter)
        return page

    # ══════════════════════════════════════════════════════════════════
    # ÉTAPE 1 — Import (Drag & Drop)
    # ══════════════════════════════════════════════════════════════════

    def _build_step1_import(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 32, 40, 32)
        layout.setSpacing(24)

        # Titre
        title = QLabel("Importation des données")
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        sub = QLabel("Glissez-déposez les dossiers FCS ou utilisez les boutons Parcourir.")
        sub.setObjectName("subtitleLabel")
        layout.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)

        # ── 3 zones de drop ──────────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(16)

        def _folder_row(label_text: str, placeholder: str, browse_slot) -> DropZoneLabel:
            lbl_cat = QLabel(label_text)
            lbl_cat.setObjectName("cardLabel")
            grid.addWidget(lbl_cat, grid.rowCount(), 0, 1, 2)

            drop = DropZoneLabel(
                f"  {placeholder}\n  Glissez un dossier ici ou cliquez sur Parcourir…"
            )
            row = grid.rowCount()
            grid.addWidget(drop, row, 0)

            btn = QPushButton("  Parcourir…")
            btn.setObjectName("ghostBtn")
            btn.setMinimumWidth(130)
            btn.setMaximumWidth(150)
            ico = _icon("prisma.folder-open", "#5BAAFF")
            if ico:
                btn.setIcon(ico)
                btn.setIconSize(QSize(16, 16))
            btn.clicked.connect(browse_slot)
            grid.addWidget(btn, row, 1, Qt.AlignTop)
            return drop

        self.drop_healthy = _folder_row(
            "DOSSIER NBM / MOELLE SAINE",
            "Dossiers .fcs témoins (contrôle normal)",
            self._select_healthy_folder,
        )
        self.drop_patho = _folder_row(
            "DOSSIER PATHOLOGIQUE",
            "Dossiers .fcs patient(s)",
            self._select_patho_folder,
        )
        self.drop_output = _folder_row(
            "DOSSIER DE SORTIE",
            "Destination des résultats (plots, CSV, FCS, rapport)",
            self._select_output_folder,
        )

        layout.addLayout(grid)

        # ── Boutons d'actions rapides (Aperçu FCS + Renommage) ────────
        actions_sep = QFrame()
        actions_sep.setFrameShape(QFrame.HLine)
        layout.addWidget(actions_sep)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(12)

        # Bouton Aperçu FCS
        self.btn_open_preview = QPushButton("  Aperçu fichiers FCS")
        self.btn_open_preview.setObjectName("ghostBtn")
        self.btn_open_preview.setMinimumHeight(44)
        ico_preview = _icon("prisma.dot-plot", "#5BAAFF", 18)
        if ico_preview:
            self.btn_open_preview.setIcon(ico_preview)
            self.btn_open_preview.setIconSize(QSize(18, 18))
        self.btn_open_preview.setToolTip(
            "Ouvre une fenêtre avec la liste complète des fichiers FCS détectés\n"
            "dans les dossiers sélectionnés (nom, condition, cellules, marqueurs)."
        )
        self.btn_open_preview.clicked.connect(self._open_preview_dialog)
        actions_row.addWidget(self.btn_open_preview, 1)

        # Bouton Renommage colonnes
        self.btn_open_rename = QPushButton("  Renommer colonnes FCS  ·  Kaluza")
        self.btn_open_rename.setObjectName("ghostBtn")
        self.btn_open_rename.setMinimumHeight(44)
        ico_rename = _icon("prisma.gate-strategy", "#cba6f7", 18)
        if ico_rename:
            self.btn_open_rename.setIcon(ico_rename)
            self.btn_open_rename.setIconSize(QSize(18, 18))
        self.btn_open_rename.setToolTip(
            "Ouvre l'éditeur de renommage de colonnes FCS.\n"
            "Permet de mapper les noms bruts (ex: 'CD45 KO') vers les noms\n"
            "canoniques Kaluza (ex: 'CD45') avant l'analyse."
        )
        self.btn_open_rename.clicked.connect(self._open_rename_dialog)
        actions_row.addWidget(self.btn_open_rename, 1)

        layout.addLayout(actions_row)

        # Badge de résumé (mis à jour après chaque sélection de dossier)
        self.lbl_preview_summary = QLabel(
            "Sélectionnez les dossiers FCS ci-dessus, puis cliquez sur «Aperçu» pour vérifier les fichiers."
        )
        self.lbl_preview_summary.setWordWrap(True)
        self.lbl_preview_summary.setObjectName("summaryLabel")
        layout.addWidget(self.lbl_preview_summary)

        # Badge renommage (nombre de règles actives)
        self.lbl_rename_summary = QLabel("Renommage colonnes : aucune règle configurée.")
        self.lbl_rename_summary.setWordWrap(True)
        self.lbl_rename_summary.setObjectName("summaryLabel")
        layout.addWidget(self.lbl_rename_summary)

        # Table de renommage (cachée — stockage interne uniquement)
        self.rename_table = QTableWidget()
        self.rename_table.setColumnCount(2)
        self.rename_table.setHorizontalHeaderLabels(["Colonne FCS (brute)", "Nom cible (Kaluza)"])
        self.rename_table.hide()

        # Bouton suivant
        layout.addStretch()
        nav = QHBoxLayout()

        nav.addStretch()
        btn_next = QPushButton("  Paramétrage")
        btn_next.setObjectName("primaryBtn")
        btn_next.setMinimumHeight(42)
        ico_next = _icon("prisma.arrow-right", "#11111b")
        if ico_next:
            btn_next.setIcon(ico_next)
            btn_next.setIconSize(QSize(16, 16))
        btn_next.clicked.connect(lambda: self._navigate_to_step(2))
        nav.addWidget(btn_next)
        layout.addLayout(nav)

        return page

    # ══════════════════════════════════════════════════════════════════
    # ÉTAPE 2 — Paramétrage
    # ══════════════════════════════════════════════════════════════════

    def _build_step2_params(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Titre fixe
        title_bar = QWidget()
        title_bar.setObjectName("titleBar")
        tbl = QHBoxLayout(title_bar)
        tbl.setContentsMargins(40, 20, 40, 20)
        tl = QLabel("Paramétrage du pipeline")
        tl.setObjectName("titleLabel")
        tbl.addWidget(tl)
        tbl.addStretch()
        outer.addWidget(title_bar)

        # Scroll pour le contenu
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(16)

        # Grille 3 colonnes
        col_layout = QHBoxLayout()
        col_layout.setSpacing(20)

        col1 = QVBoxLayout()
        col2 = QVBoxLayout()
        col3 = QVBoxLayout()

        # ── Colonne 1 : FlowSOM + Transformation ─────────────────────
        col1.addWidget(self._build_som_group())
        col1.addWidget(self._build_transform_group())
        col1.addStretch()

        # ── Colonne 2 : Gating + Options ─────────────────────────────
        col2.addWidget(self._build_gating_group())
        col2.addWidget(self._build_markers_group())
        col2.addWidget(self._build_options_group())
        col2.addStretch()

        # ── Colonne 3 : MRD + Harmony + Stratified DS ────────────────
        col3.addWidget(self._build_mrd_group())
        col3.addWidget(self._build_harmony_group())
        col3.addWidget(self._build_stratified_ds_group())
        col3.addStretch()

        col_layout.addLayout(col1)
        col_layout.addLayout(col2)
        col_layout.addLayout(col3)
        layout.addLayout(col_layout)

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # Barre de navigation
        nav_bar = QWidget()
        nav_bar.setObjectName("navBar")
        nbl = QHBoxLayout(nav_bar)
        nbl.setContentsMargins(40, 14, 40, 14)

        btn_back = QPushButton("  Import")
        btn_back.setObjectName("ghostBtn")
        ico_back = _icon("prisma.arrow-left", "#5BAAFF")
        if ico_back:
            btn_back.setIcon(ico_back)
            btn_back.setIconSize(QSize(16, 16))
        btn_back.clicked.connect(lambda: self._navigate_to_step(1))
        nbl.addWidget(btn_back)

        nbl.addStretch()

        btn_launch = QPushButton("  Lancer le Pipeline")
        btn_launch.setObjectName("primaryBtn")
        btn_launch.setMinimumHeight(42)
        btn_launch.setMinimumWidth(180)
        ico_play = _icon("prisma.play", "#EEF2F7")
        if ico_play:
            btn_launch.setIcon(ico_play)
            btn_launch.setIconSize(QSize(16, 16))
        btn_launch.clicked.connect(self._run_pipeline)
        nbl.addWidget(btn_launch)

        outer.addWidget(nav_bar)
        return page

    def _build_som_group(self) -> QGroupBox:
        group = QGroupBox("Paramètres FlowSOM")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        grid.addWidget(QLabel("Grille X (xdim) :"), 0, 0)
        self.spin_xdim = QSpinBox()
        self.spin_xdim.setRange(3, 50)
        self.spin_xdim.setValue(10)
        self.spin_xdim.setToolTip("Dimension X de la grille SOM (défaut : 10)")
        grid.addWidget(self.spin_xdim, 0, 1)

        grid.addWidget(QLabel("Grille Y (ydim) :"), 1, 0)
        self.spin_ydim = QSpinBox()
        self.spin_ydim.setRange(3, 50)
        self.spin_ydim.setValue(10)
        grid.addWidget(self.spin_ydim, 1, 1)

        grid.addWidget(QLabel("Métaclusters :"), 2, 0)
        self.spin_metaclusters = QSpinBox()
        self.spin_metaclusters.setRange(2, 50)
        self.spin_metaclusters.setValue(8)
        grid.addWidget(self.spin_metaclusters, 2, 1)

        grid.addWidget(QLabel("Seed :"), 3, 0)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_seed.setValue(42)
        grid.addWidget(self.spin_seed, 3, 1)

        grid.addWidget(QLabel("Learning rate :"), 4, 0)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.001, 1.0)
        self.spin_lr.setSingleStep(0.01)
        self.spin_lr.setValue(0.05)
        self.spin_lr.setDecimals(3)
        grid.addWidget(self.spin_lr, 4, 1)

        grid.addWidget(QLabel("Sigma voisinage :"), 5, 0)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 10.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setValue(1.5)
        self.spin_sigma.setDecimals(1)
        grid.addWidget(self.spin_sigma, 5, 1)

        self.chk_auto_clustering = ToggleSwitch("Auto-sélection clusters (bootstrap)")
        grid.addWidget(self.chk_auto_clustering, 6, 0, 1, 2)

        return group

    def _build_transform_group(self) -> QGroupBox:
        group = QGroupBox("Transformation & Normalisation")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        grid.addWidget(QLabel("Transformation :"), 0, 0)
        self.combo_transform = DarkComboBox()
        self.combo_transform.addItems(["logicle", "arcsinh", "log10", "none"])
        grid.addWidget(self.combo_transform, 0, 1)

        grid.addWidget(QLabel("Cofacteur (arcsinh) :"), 1, 0)
        self.spin_cofactor = QDoubleSpinBox()
        self.spin_cofactor.setRange(1.0, 500.0)
        self.spin_cofactor.setValue(5.0)
        self.spin_cofactor.setDecimals(1)
        grid.addWidget(self.spin_cofactor, 1, 1)

        grid.addWidget(QLabel("Normalisation :"), 2, 0)
        self.combo_normalize = DarkComboBox()
        self.combo_normalize.addItems(["zscore", "minmax", "none"])
        grid.addWidget(self.combo_normalize, 2, 1)

        return group

    def _build_markers_group(self) -> QGroupBox:
        group = QGroupBox("Marqueurs & Scatter")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        # Exclude scatter (FSC/SSC)
        self.chk_exclude_scatter = ToggleSwitch("Exclure scatter (FSC/SSC)", checked=True)
        self.chk_exclude_scatter.setToolTip(
            "Si activé, les colonnes FSC et SSC sont exclues du clustering FlowSOM.\n"
            "Recommandé pour ne garder que les marqueurs immunophénotypiques."
        )
        grid.addWidget(self.chk_exclude_scatter, 0, 0, 1, 2)

        # Keep Area only (A) vs garder les deux (A + H)
        self.chk_keep_area_only = ToggleSwitch(
            "Garder Area seulement (-A, exclure -H)", checked=True
        )
        self.chk_keep_area_only.setToolTip(
            "Si activé, les colonnes -H (Height) sont supprimées quand le doublon\n"
            "-A (Area) existe. Réduit la colinéarité et accélère le SOM.\n"
            "Désactivez pour garder les deux (-A et -H) dans le clustering."
        )
        grid.addWidget(self.chk_keep_area_only, 1, 0, 1, 2)

        # Colonnes supplémentaires à exclure
        grid.addWidget(QLabel("Colonnes à exclure (séparées par ,) :"), 2, 0, 1, 2)
        self.edit_exclude_cols = QLineEdit()
        self.edit_exclude_cols.setPlaceholderText("ex: Time, Width, Event_length")
        self.edit_exclude_cols.setToolTip(
            "Liste de colonnes supplémentaires à exclure du clustering,\nséparées par des virgules."
        )
        grid.addWidget(self.edit_exclude_cols, 3, 0, 1, 2)

        return group

    def _build_harmony_group(self) -> QGroupBox:
        group = QGroupBox("Harmony (Correction batch)")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        # Toggle principal
        self.chk_harmony = ToggleSwitch("Activer Harmony (harmonypy)", checked=True)
        self.chk_harmony.setToolTip(
            "Active la correction d'effet batch inter-fichiers via harmonypy.\n"
            "Recommandé quand les fichiers NBM proviennent de sessions différentes."
        )
        grid.addWidget(self.chk_harmony, 0, 0, 1, 2)

        # Marqueurs à aligner
        grid.addWidget(QLabel("Marqueurs à aligner (vide = tous) :"), 1, 0, 1, 2)
        self.edit_harmony_markers = QLineEdit()
        self.edit_harmony_markers.setPlaceholderText(
            "ex: FSC-A, SSC-A  (vide = tous les marqueurs)"
        )
        self.edit_harmony_markers.setToolTip(
            "Liste de marqueurs à corriger avec Harmony (séparés par virgules).\n"
            "Vide = tous les marqueurs du clustering. Recommandé : FSC-A, SSC-A."
        )
        grid.addWidget(self.edit_harmony_markers, 2, 0, 1, 2)

        # Paramètres Harmony
        grid.addWidget(QLabel("Sigma :"), 3, 0)
        self.spin_harmony_sigma = QDoubleSpinBox()
        self.spin_harmony_sigma.setRange(0.001, 1.0)
        self.spin_harmony_sigma.setSingleStep(0.01)
        self.spin_harmony_sigma.setValue(0.05)
        self.spin_harmony_sigma.setDecimals(3)
        self.spin_harmony_sigma.setToolTip(
            "Paramètre de largeur de la distribution de Harmony.\n"
            "Plus petit = correction plus agressive. Défaut : 0.05"
        )
        grid.addWidget(self.spin_harmony_sigma, 3, 1)

        grid.addWidget(QLabel("nclust (0=auto) :"), 4, 0)
        self.spin_harmony_nclust = QSpinBox()
        self.spin_harmony_nclust.setRange(0, 200)
        self.spin_harmony_nclust.setValue(30)
        self.spin_harmony_nclust.setToolTip(
            "Nombre de clusters Harmony internes.\n"
            "0 = auto (N/30, très lent sur grands datasets). Défaut : 30"
        )
        grid.addWidget(self.spin_harmony_nclust, 4, 1)

        grid.addWidget(QLabel("Max itérations :"), 5, 0)
        self.spin_harmony_max_iter = QSpinBox()
        self.spin_harmony_max_iter.setRange(1, 100)
        self.spin_harmony_max_iter.setValue(10)
        self.spin_harmony_max_iter.setToolTip("Nombre max d'itérations Harmony. Défaut : 10")
        grid.addWidget(self.spin_harmony_max_iter, 5, 1)

        grid.addWidget(QLabel("Block size :"), 6, 0)
        self.spin_harmony_block = QDoubleSpinBox()
        self.spin_harmony_block.setRange(0.01, 1.0)
        self.spin_harmony_block.setSingleStep(0.05)
        self.spin_harmony_block.setValue(0.20)
        self.spin_harmony_block.setDecimals(2)
        self.spin_harmony_block.setToolTip(
            "Fraction de cellules par bloc pour les mises à jour Harmony.\n"
            "Défaut : 0.20 (20% = 5 blocs)"
        )
        grid.addWidget(self.spin_harmony_block, 6, 1)

        return group

    def _build_gating_group(self) -> QGroupBox:
        group = QGroupBox("Pré-gating automatique")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)

        self.chk_pregate = ToggleSwitch("Activer le pré-gating", checked=True)
        vbox.addWidget(self.chk_pregate)

        grid = QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(QLabel("Mode :"), 0, 0)
        self.combo_gate_mode = DarkComboBox()
        self.combo_gate_mode.addItems(["auto", "manual"])
        grid.addWidget(self.combo_gate_mode, 0, 1)

        self.chk_viable = ToggleSwitch("Débris (FSC/SSC)", checked=True)
        grid.addWidget(self.chk_viable, 1, 0)

        self.chk_singlets = ToggleSwitch("Doublets (FSC-H/FSC-A)", checked=True)
        grid.addWidget(self.chk_singlets, 1, 1)

        self.chk_cd45 = ToggleSwitch("CD45 dim")
        grid.addWidget(self.chk_cd45, 2, 0)

        self.chk_cd34 = ToggleSwitch("CD34+ blastes")
        grid.addWidget(self.chk_cd34, 2, 1)

        self.chk_mode_blastes = ToggleSwitch(
            "Gating CD45 asymétrique (patho seulement)", checked=True
        )
        grid.addWidget(self.chk_mode_blastes, 3, 0, 1, 2)

        grid.addWidget(QLabel("Dénominateur MRD :"), 4, 0)
        self.combo_cd45_autogating_mode = DarkComboBox()
        self.combo_cd45_autogating_mode.addItems(
            [
                "none",  # Toutes cellules patho (comportement historique)
                "cd45",  # Cellules patho CD45+ standard
                "cd45_dim",  # Cellules patho CD45+ (inclut blastes CD45-dim)
            ]
        )
        self.combo_cd45_autogating_mode.setToolTip(
            "none     → MRD % = blastes / toutes cellules patho\n"
            "cd45     → MRD % = blastes / cellules patho CD45+\n"
            "cd45_dim → MRD % = blastes / cellules patho CD45+ (blastes dim inclus)"
        )
        grid.addWidget(self.combo_cd45_autogating_mode, 4, 1)

        # ── Paramètres méthode de densité (Tri initial) ───────────────
        density_lbl = QLabel("── Méthode Tri Initial ──")
        density_lbl.setObjectName("subtitleLabel")
        grid.addWidget(density_lbl, 5, 0, 1, 2)

        grid.addWidget(QLabel("Méthode (viable) :"), 6, 0)
        self.combo_density_method = DarkComboBox()
        self.combo_density_method.addItems(["GMM", "KDE"])
        self.combo_density_method.setToolTip(
            "GMM (Gaussian Mixture Model) : robuste, recommandé\n"
            "KDE (Kernel Density Estimation) : plus léger, bon pour CD45"
        )
        grid.addWidget(self.combo_density_method, 6, 1)

        grid.addWidget(QLabel("Composantes GMM :"), 7, 0)
        self.spin_gmm_components = QSpinBox()
        self.spin_gmm_components.setRange(1, 10)
        self.spin_gmm_components.setValue(3)
        self.spin_gmm_components.setToolTip(
            "Nombre de composantes gaussiennes pour le gating débris/viables.\n"
            "3 = debris + transitoire + cellules viables (recommandé)"
        )
        grid.addWidget(self.spin_gmm_components, 7, 1)

        grid.addWidget(QLabel("Type covariance GMM :"), 8, 0)
        self.combo_gmm_cov = DarkComboBox()
        self.combo_gmm_cov.addItems(["full", "tied", "diag", "spherical"])
        self.combo_gmm_cov.setToolTip(
            "full      : chaque composante a sa propre matrice de covariance (défaut)\n"
            "tied      : toutes partagent la même matrice\n"
            "diag      : matrices diagonales (moins de paramètres)\n"
            "spherical : variances scalaires (le plus simple)"
        )
        grid.addWidget(self.combo_gmm_cov, 8, 1)

        # ── Paramètres KDE CD45 ───────────────────────────────────────
        kde_lbl = QLabel("── Paramètres KDE CD45 ──")
        kde_lbl.setObjectName("subtitleLabel")
        grid.addWidget(kde_lbl, 9, 0, 1, 2)

        grid.addWidget(QLabel("Finesse bandwidth :"), 10, 0)
        self.spin_kde_finesse = QDoubleSpinBox()
        self.spin_kde_finesse.setRange(0.1, 2.0)
        self.spin_kde_finesse.setSingleStep(0.05)
        self.spin_kde_finesse.setValue(0.6)
        self.spin_kde_finesse.setDecimals(2)
        self.spin_kde_finesse.setToolTip(
            "Facteur de bandwidth Silverman pour KDE CD45.\n"
            "< 1 = plus fin, > 1 = plus lissé. Défaut : 0.6"
        )
        grid.addWidget(self.spin_kde_finesse, 10, 1)

        grid.addWidget(QLabel("Sigma lissage :"), 11, 0)
        self.spin_kde_sigma = QSpinBox()
        self.spin_kde_sigma.setRange(1, 50)
        self.spin_kde_sigma.setValue(10)
        self.spin_kde_sigma.setToolTip(
            "Lissage gaussien post-KDE (sigma en points de grille).\n"
            "Réduit les faux-creux. Défaut : 10"
        )
        grid.addWidget(self.spin_kde_sigma, 11, 1)

        grid.addWidget(QLabel("Seuil relatif CD45 :"), 12, 0)
        self.spin_kde_seuil = QDoubleSpinBox()
        self.spin_kde_seuil.setRange(0.01, 0.5)
        self.spin_kde_seuil.setSingleStep(0.01)
        self.spin_kde_seuil.setValue(0.05)
        self.spin_kde_seuil.setDecimals(3)
        self.spin_kde_seuil.setToolTip(
            "Fraction du pic max pour détecter le 'pied' du pic CD45.\n"
            "Plus petit = seuil plus bas (inclut plus de cellules CD45-dim). Défaut : 0.05"
        )
        grid.addWidget(self.spin_kde_seuil, 12, 1)

        vbox.addLayout(grid)
        return group

    def _build_options_group(self) -> QGroupBox:
        group = QGroupBox("Options")
        grid = QGridLayout(group)
        grid.setSpacing(6)

        self.chk_umap = ToggleSwitch("Calculer UMAP")
        grid.addWidget(self.chk_umap, 0, 0)

        self.chk_gpu = ToggleSwitch("GPU (CUDA)", checked=True)
        grid.addWidget(self.chk_gpu, 0, 1)

        self.chk_compare = ToggleSwitch("Mode comparaison Sain vs Patho", checked=True)
        grid.addWidget(self.chk_compare, 1, 0, 1, 2)

        self.chk_pop_mapping = ToggleSwitch("Mapping populations (Ref MFI)")
        grid.addWidget(self.chk_pop_mapping, 2, 0, 1, 2)

        self.chk_downsampling = ToggleSwitch("Downsampling")
        grid.addWidget(self.chk_downsampling, 3, 0)

        self.spin_max_cells = QSpinBox()
        self.spin_max_cells.setRange(1000, 5_000_000)
        self.spin_max_cells.setSingleStep(10000)
        self.spin_max_cells.setValue(50000)
        self.spin_max_cells.setSuffix(" cell./fichier")
        grid.addWidget(self.spin_max_cells, 3, 1)

        self.chk_batch = ToggleSwitch("Mode Batch (tous les fichiers patho)")
        grid.addWidget(self.chk_batch, 4, 0, 1, 2)

        grid.addWidget(QLabel("Mode export :"), 5, 0)
        self.combo_export_mode = DarkComboBox()
        self.combo_export_mode.addItems(["standard", "compact"])
        grid.addWidget(self.combo_export_mode, 5, 1)

        return group

    def _build_mrd_group(self) -> QGroupBox:
        group = QGroupBox("Paramètres MRD")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        grid.addWidget(QLabel("Méthode MRD :"), 0, 0)
        self.combo_mrd_method = DarkComboBox()
        self.combo_mrd_method.addItems(["all", "flo", "jf", "eln"])
        grid.addWidget(self.combo_mrd_method, 0, 1)

        grid.addWidget(QLabel("Méthode FCS export :"), 1, 0)
        self.combo_mrd_fcs_method = DarkComboBox()
        self.combo_mrd_fcs_method.addItems(["flo", "jf"])
        grid.addWidget(self.combo_mrd_fcs_method, 1, 1)

        # ELN
        eln_lbl = QLabel("── ELN ──")
        eln_lbl.setObjectName("subtitleLabel")
        grid.addWidget(eln_lbl, 2, 0, 1, 2)

        grid.addWidget(QLabel("Min events/nœud (LOQ) :"), 3, 0)
        self.spin_eln_min_events = QSpinBox()
        self.spin_eln_min_events.setRange(1, 500)
        self.spin_eln_min_events.setValue(50)
        grid.addWidget(self.spin_eln_min_events, 3, 1)

        grid.addWidget(QLabel("Seuil positivité ELN (%) :"), 4, 0)
        self.spin_eln_positivity = QDoubleSpinBox()
        self.spin_eln_positivity.setRange(0.01, 10.0)
        self.spin_eln_positivity.setSingleStep(0.05)
        self.spin_eln_positivity.setValue(0.1)
        self.spin_eln_positivity.setDecimals(2)
        grid.addWidget(self.spin_eln_positivity, 4, 1)

        # Flo
        flo_lbl = QLabel("── Méthode Flo ──")
        flo_lbl.setObjectName("subtitleLabel")
        grid.addWidget(flo_lbl, 5, 0, 1, 2)

        grid.addWidget(QLabel("Multiplicateur normal :"), 6, 0)
        self.spin_flo_multiplier = QDoubleSpinBox()
        self.spin_flo_multiplier.setRange(0.5, 20.0)
        self.spin_flo_multiplier.setSingleStep(0.5)
        self.spin_flo_multiplier.setValue(2.0)
        self.spin_flo_multiplier.setDecimals(1)
        grid.addWidget(self.spin_flo_multiplier, 6, 1)

        # JF
        jf_lbl = QLabel("── Méthode JF ──")
        jf_lbl.setObjectName("subtitleLabel")
        grid.addWidget(jf_lbl, 7, 0, 1, 2)

        grid.addWidget(QLabel("Max % moelle normale :"), 8, 0)
        self.spin_jf_max_normal = QDoubleSpinBox()
        self.spin_jf_max_normal.setRange(0.01, 10.0)
        self.spin_jf_max_normal.setSingleStep(0.05)
        self.spin_jf_max_normal.setValue(0.1)
        self.spin_jf_max_normal.setDecimals(2)
        grid.addWidget(self.spin_jf_max_normal, 8, 1)

        grid.addWidget(QLabel("Min % cellules patho :"), 9, 0)
        self.spin_jf_min_patho = QDoubleSpinBox()
        self.spin_jf_min_patho.setRange(0.1, 100.0)
        self.spin_jf_min_patho.setSingleStep(1.0)
        self.spin_jf_min_patho.setValue(10.0)
        self.spin_jf_min_patho.setDecimals(1)
        grid.addWidget(self.spin_jf_min_patho, 9, 1)

        # Filtre phénotypique hybride
        hybrid_lbl = QLabel("── Filtre Phénotypique (Hybride) ──")
        hybrid_lbl.setObjectName("subtitleLabel")
        grid.addWidget(hybrid_lbl, 10, 0, 1, 2)

        self.chk_blast_filter = QCheckBox("Porte biologique ELN 2022")
        self.chk_blast_filter.setChecked(False)
        self.chk_blast_filter.setToolTip(
            "Active le filtre hybride à deux portes :\n"
            "  1. Porte Topologique  : critère mathématique JF / Flo / ELN\n"
            "  2. Porte Biologique   : blast_score ELN 2022 (BLAST_HIGH ou BLAST_MODERATE)\n\n"
            "Un nœud ne passe qu'en satisfaisant les DEUX portes.\n"
            "Réduit fortement les faux positifs liés à l'effet batch."
        )
        grid.addWidget(self.chk_blast_filter, 11, 0, 1, 2)

        return group

    def _build_stratified_ds_group(self) -> QGroupBox:
        group = QGroupBox("Déséquilibre Maîtrisé")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        self.chk_balance_conditions = QCheckBox("Rééquilibrage sain/patho")
        self.chk_balance_conditions.setChecked(True)
        grid.addWidget(self.chk_balance_conditions, 0, 0, 1, 2)

        grid.addWidget(QLabel("Ratio sain / patho :"), 1, 0)
        self.spin_imbalance_ratio = QDoubleSpinBox()
        self.spin_imbalance_ratio.setRange(0.5, 10.0)
        self.spin_imbalance_ratio.setSingleStep(0.5)
        self.spin_imbalance_ratio.setValue(2.0)
        self.spin_imbalance_ratio.setDecimals(1)
        grid.addWidget(self.spin_imbalance_ratio, 1, 1)

        self.chk_allow_oversampling = QCheckBox("Oversampling NBM si quota non atteint")
        self.chk_allow_oversampling.setChecked(False)
        self.chk_allow_oversampling.setToolTip(
            "Si activé, les fichiers NBM sont rééchantillonnés avec remplacement\n"
            "pour atteindre le ratio cible quand les cellules disponibles sont\n"
            "insuffisantes. Garantit le ratio exact mais introduit des doublons."
        )
        grid.addWidget(self.chk_allow_oversampling, 2, 0, 1, 2)

        return group

    # ══════════════════════════════════════════════════════════════════
    # ÉTAPE 3 — Exécution (Console + Progress)
    # ══════════════════════════════════════════════════════════════════

    def _build_step3_run(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 32, 40, 24)
        layout.setSpacing(16)

        # Titre
        title = QLabel("Exécution du Pipeline")
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        # ── Indicateur d'étape textuel ────────────────────────────────
        self.lbl_pipeline_step = QLabel("En attente du lancement…")
        self.lbl_pipeline_step.setObjectName("pipelineStepLabel")
        layout.addWidget(self.lbl_pipeline_step)

        # ── Barre de progression ──────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("pipelineProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimumHeight(20)
        self.progress_bar.setMaximumHeight(20)
        layout.addWidget(self.progress_bar)

        # ── Console logs ──────────────────────────────────────────────
        self.log_output = LogConsole()
        layout.addWidget(self.log_output, 1)

        # ── Boutons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        btn_back2 = QPushButton("  Paramétrage")
        btn_back2.setObjectName("ghostBtn")
        ico_back = _icon("prisma.arrow-left", "#5BAAFF")
        if ico_back:
            btn_back2.setIcon(ico_back)
            btn_back2.setIconSize(QSize(16, 16))
        btn_back2.clicked.connect(lambda: self._navigate_to_step(2))
        btn_row.addWidget(btn_back2)

        btn_clear_log = QPushButton("  Effacer")
        btn_clear_log.setObjectName("ghostBtn")
        ico_clear = _icon("prisma.trash", "#a6adc8")
        if ico_clear:
            btn_clear_log.setIcon(ico_clear)
            btn_clear_log.setIconSize(QSize(16, 16))
        btn_clear_log.clicked.connect(lambda: self.log_output.clear())
        btn_row.addWidget(btn_clear_log)

        btn_copy_log = QPushButton("  Copier")
        btn_copy_log.setObjectName("ghostBtn")
        ico_copy = _icon("prisma.copy", "#a6adc8")
        if ico_copy:
            btn_copy_log.setIcon(ico_copy)
            btn_copy_log.setIconSize(QSize(16, 16))
        btn_copy_log.clicked.connect(
            lambda: QApplication.clipboard().setText(self.log_output.toPlainText())
        )
        btn_row.addWidget(btn_copy_log)

        btn_row.addStretch()

        # Bouton STOP
        self.btn_stop = QPushButton("  Arrêter")
        self.btn_stop.setObjectName("dangerBtn")
        self.btn_stop.setEnabled(False)
        ico_stop = _icon("prisma.stop-circle", "#11111b")
        if ico_stop:
            self.btn_stop.setIcon(ico_stop)
            self.btn_stop.setIconSize(QSize(16, 16))
        self.btn_stop.clicked.connect(self._stop_pipeline)
        btn_row.addWidget(self.btn_stop)

        # Bouton RUN (disponible ici aussi)
        self.btn_run_step3 = QPushButton("  Lancer le Pipeline")
        self.btn_run_step3.setObjectName("primaryBtn")
        self.btn_run_step3.setMinimumHeight(40)
        ico_play = _icon("prisma.play", "#EEF2F7")
        if ico_play:
            self.btn_run_step3.setIcon(ico_play)
            self.btn_run_step3.setIconSize(QSize(16, 16))
        self.btn_run_step3.clicked.connect(self._run_pipeline)
        btn_row.addWidget(self.btn_run_step3)

        layout.addLayout(btn_row)
        return page

    # ══════════════════════════════════════════════════════════════════
    # ÉTAPE 4 — Résultats (onglets)
    # ══════════════════════════════════════════════════════════════════

    def _build_step4_results(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Barre d'actions en haut
        action_bar = QWidget()
        action_bar.setObjectName("actionBar")
        abl = QHBoxLayout(action_bar)
        abl.setContentsMargins(24, 12, 24, 12)
        abl.setSpacing(8)

        btn_fcs = QPushButton("  Export FCS")
        btn_fcs.setObjectName("exportBtn")
        ico_fcs = _icon("prisma.fcs-file", "#cba6f7", 16)
        if ico_fcs:
            btn_fcs.setIcon(ico_fcs)
            btn_fcs.setIconSize(QSize(16, 16))
        btn_fcs.clicked.connect(self._export_fcs)
        abl.addWidget(btn_fcs)

        btn_csv = QPushButton("  Export CSV")
        btn_csv.setObjectName("exportBtn")
        ico_csv = _icon("prisma.export-fig", "#cba6f7", 16)
        if ico_csv:
            btn_csv.setIcon(ico_csv)
            btn_csv.setIconSize(QSize(16, 16))
        btn_csv.clicked.connect(self._export_csv)
        abl.addWidget(btn_csv)

        btn_report = QPushButton("  Rapport HTML")
        btn_report.setObjectName("exportBtn")
        ico_rep = _icon("prisma.export-fig", "#cba6f7", 16)
        if ico_rep:
            btn_report.setIcon(ico_rep)
            btn_report.setIconSize(QSize(16, 16))
        btn_report.clicked.connect(self._open_html_report)
        abl.addWidget(btn_report)

        btn_folder = QPushButton("  Ouvrir dossier")
        ico_fol = _icon("prisma.folder-open", "#EEF2F7")
        if ico_fol:
            btn_folder.setIcon(ico_fol)
            btn_folder.setIconSize(QSize(16, 16))
        btn_folder.clicked.connect(self._open_output_folder)
        abl.addWidget(btn_folder)

        abl.addStretch()

        btn_back3 = QPushButton("  Logs")
        btn_back3.setObjectName("ghostBtn")
        btn_back3.clicked.connect(lambda: self._navigate_to_step(3))
        abl.addWidget(btn_back3)

        layout.addWidget(action_bar)

        # ── Bandeau avertissement clinique (P3.6) ─────────────────────
        clinical_warning = QWidget()
        clinical_warning.setObjectName("clinicalWarningBanner")
        cw_layout = QHBoxLayout(clinical_warning)
        cw_layout.setContentsMargins(20, 9, 20, 9)
        cw_layout.setSpacing(12)

        ico_warn = _icon("prisma.alert-triangle", "#FF3D6E")
        if ico_warn:
            lbl_ico = QLabel()
            lbl_ico.setPixmap(ico_warn.pixmap(14, 14))
            cw_layout.addWidget(lbl_ico)

        lbl_warn = QLabel(
            "<b style='color:#FF3D6E; letter-spacing:0.12em;'>RESEARCH TOOL · NOT FOR CLINICAL USE</b>"
            "  <span style='color: #EEF2F7;'>Aide à l'analyse et à la visualisation."
            " Ne remplace pas l'expert biologiste/médecin ni les procédures AQ du laboratoire."
            " Les seuils de scoring blastique sont des heuristiques non validées cliniquement.</span>"
        )
        lbl_warn.setObjectName("clinicalWarningText")
        lbl_warn.setTextFormat(Qt.RichText)
        lbl_warn.setWordWrap(True)
        cw_layout.addWidget(lbl_warn, 1)

        layout.addWidget(clinical_warning)

        # Onglets résultats
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        self.tabs.tabBar().setExpanding(False)
        self.tabs.setIconSize(QSize(16, 16))
        self._build_home_tab()  # 0 — Accueil MRD
        self._build_viz_tab()  # 1 — Visualisation
        self._build_pregate_tab()  # 2 — Représentations
        self._build_clusters_tab()  # 3 — Clusters
        self._build_results_tab()  # 4 — Résultats clusters
        self._build_fcs_viewer_tab()  # 5 — Visualisation FCS
        layout.addWidget(self.tabs, 1)

        return page

    # ── Onglets (identiques à l'ancienne version, sans onglet Logs séparé) ──

    def _build_home_tab(self) -> None:
        self._home_tab = HomeTab()
        self._home_tab.open_html_requested.connect(self._open_html_report)
        self._home_tab.curation_changed.connect(self._on_curation_changed)
        self._home_tab.verification_commit_requested.connect(self._on_verification_commit_requested)
        _ico_home = _icon("prisma.dot-plot", "#39FF8A", 16)
        self.tabs.addTab(self._home_tab, _ico_home or QIcon(), "ACCUEIL MRD")

    def _build_viz_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Afficher :"))
        self.combo_plot = DarkComboBox()
        self.combo_plot.addItems(
            [
                "Heatmap MFI",
                "Distribution Métaclusters",
                "UMAP",
                "Star Chart FlowSOM",
                "Grille SOM statique",
                "MST Statique",
                "Sankey Gating",
                "MST Interactif",
                "Grille SOM interactive",
                "Radar Métaclusters",
                "% Cellules Patho / Cluster",
                "% Cellules / Cluster",
                "% Patho / Nœud SOM",
                "% Cellules / Nœud SOM",
                "Vue Combinée Nœuds SOM",
            ]
        )
        self.combo_plot.currentIndexChanged.connect(self._on_plot_selection_changed)
        selector_layout.addWidget(self.combo_plot, 1)

        btn_refresh = QPushButton("  Rafraîchir")
        ico_ref = _icon("prisma.mrd-kinetics", "#EEF2F7", 16)
        if ico_ref:
            btn_refresh.setIcon(ico_ref)
            btn_refresh.setIconSize(QSize(16, 16))
        btn_refresh.clicked.connect(self._refresh_current_plot)
        selector_layout.addWidget(btn_refresh)

        btn_browser = QPushButton("  Navigateur")
        btn_browser.setObjectName("successBtn")
        ico_nav = _icon("prisma.external-link", "#11111b")
        if ico_nav:
            btn_browser.setIcon(ico_nav)
            btn_browser.setIconSize(QSize(16, 16))
        btn_browser.clicked.connect(self._open_current_plot_browser)
        selector_layout.addWidget(btn_browser)

        layout.addLayout(selector_layout)

        self._viz_stack = QStackedWidget()
        png_widget = QWidget()
        png_layout = QVBoxLayout(png_widget)
        png_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MatplotlibCanvas(tab, width=10, height=7)
        self.toolbar = NavigationToolbar(self.canvas, tab)
        self.toolbar.setObjectName("matplotlibToolbar")
        png_layout.addWidget(self.toolbar)
        png_layout.addWidget(self.canvas, 1)
        self._viz_stack.addWidget(png_widget)

        html_placeholder = QLabel(
            "Figures interactives (.html)\nCliquez sur  'Navigateur'  pour les afficher."
        )
        html_placeholder.setAlignment(Qt.AlignCenter)
        html_placeholder.setObjectName("subtitleLabel")
        self._web_view = None
        self._viz_stack.addWidget(html_placeholder)

        layout.addWidget(self._viz_stack, 1)
        _ico_viz = _icon("prisma.umap-embed", "#5BAAFF", 16)
        self.tabs.addTab(tab, _ico_viz or QIcon(), "VISUALISATION")

    def _build_pregate_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Figure :"))
        self.combo_gate_plot = DarkComboBox()
        self.combo_gate_plot.currentIndexChanged.connect(self._on_gate_plot_changed)
        selector_row.addWidget(self.combo_gate_plot, 1)

        btn_gate_browser = QPushButton("  Navigateur")
        btn_gate_browser.setObjectName("successBtn")
        ico_nav = _icon("prisma.external-link", "#11111b")
        if ico_nav:
            btn_gate_browser.setIcon(ico_nav)
            btn_gate_browser.setIconSize(QSize(16, 16))
        btn_gate_browser.clicked.connect(self._open_current_repr_browser)
        selector_row.addWidget(btn_gate_browser)
        layout.addLayout(selector_row)

        self.gate_canvas = MatplotlibCanvas(tab, width=10, height=6)
        gate_toolbar = NavigationToolbar(self.gate_canvas, tab)
        gate_toolbar.setObjectName("matplotlibToolbar")
        layout.addWidget(gate_toolbar)
        layout.addWidget(self.gate_canvas, 1)

        lbl_gate = QLabel("Rapport de prégating")
        lbl_gate.setObjectName("subtitleLabel")
        layout.addWidget(lbl_gate)

        self.gate_table = QTableWidget()
        self.gate_table.setColumnCount(6)
        self.gate_table.setHorizontalHeaderLabels(
            [
                "Gate",
                "Fichier",
                "Cellules avant",
                "Cellules après",
                "% conservé",
                "Mode",
            ]
        )
        self.gate_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.gate_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.gate_table.setMaximumHeight(180)
        layout.addWidget(self.gate_table)

        self._gate_plot_keys = {
            "Prégating — Vue d'ensemble": ["fig_overview", "overview"],
            "Prégating — Débris": ["fig_gate_debris", "gate_debris", "debris"],
            "Prégating — Doublets": ["fig_gate_singlets", "gate_singlets", "singlets"],
            "Prégating — CD45": ["fig_gate_cd45", "gate_cd45", "cd45"],
            "Prégating — CD34+": ["fig_gate_cd34", "gate_cd34", "cd34"],
            "Heatmap MFI": ["mfi_heatmap"],
            "Distribution Métaclusters": ["metacluster_distribution"],
            "UMAP": ["umap"],
            "Star Chart FlowSOM": ["flowsom_star_chart"],
            "Grille SOM statique": ["flowsom_som_grid"],
            "MST Statique": ["mst_static"],
            "Sankey Gating": ["sankey_global"],
            "Radar Métaclusters": ["metacluster_radar"],
            "% Cellules Patho / Cluster": ["patho_pct_per_cluster"],
            "% Cellules / Cluster": ["cells_pct_per_cluster"],
            "% Patho / Nœud SOM": ["patho_pct_per_som_node"],
            "% Cellules / Nœud SOM": ["cells_pct_per_som_node"],
            "Vue Combinée Nœuds SOM": ["som_node_combined"],
        }
        self.combo_gate_plot.addItems(list(self._gate_plot_keys.keys()))
        _ico_rep = _icon("prisma.gate-strategy", "#7B52FF", 16)
        self.tabs.addTab(tab, _ico_rep or QIcon(), "REPRÉSENTATIONS")

    def _build_clusters_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        left = QVBoxLayout()
        left.setSpacing(6)

        lbl_clusters = QLabel("Clusters (nœuds SOM)")
        lbl_clusters.setObjectName("sectionLabel")
        left.addWidget(lbl_clusters)

        self.cluster_list = QListWidget()
        self.cluster_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.cluster_list.currentRowChanged.connect(self._on_cluster_selected)
        self.cluster_list.setMaximumWidth(240)
        left.addWidget(self.cluster_list, 2)

        lbl_markers = QLabel("Marqueurs Spider Plot")
        lbl_markers.setObjectName("subtitleLabel")
        left.addWidget(lbl_markers)

        self.marker_list = QListWidget()
        self.marker_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.marker_list.setMaximumWidth(240)
        left.addWidget(self.marker_list, 2)

        btn_all_markers = QPushButton("Tout sélectionner")
        btn_all_markers.clicked.connect(lambda: self.marker_list.selectAll())
        left.addWidget(btn_all_markers)

        btn_clear_markers = QPushButton("Tout désélectionner")
        btn_clear_markers.clicked.connect(lambda: self.marker_list.clearSelection())
        left.addWidget(btn_clear_markers)

        btn_spider = QPushButton("  Générer Spider Plot")
        btn_spider.setObjectName("primaryBtn")
        ico_sp = _icon("prisma.metacluster", "#11111b", 16)
        if ico_sp:
            btn_spider.setIcon(ico_sp)
            btn_spider.setIconSize(QSize(16, 16))
        btn_spider.clicked.connect(self._generate_spider_plot)
        left.addWidget(btn_spider)

        layout.addLayout(left)
        self.star_canvas = MatplotlibCanvas(tab, width=7, height=7)
        layout.addWidget(self.star_canvas, 1)
        _ico_clust = _icon("prisma.metacluster", "#FF9B3D", 16)
        self.tabs.addTab(tab, _ico_clust or QIcon(), "CLUSTERS")

    def _build_results_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        hdr = QHBoxLayout()
        lbl = QLabel("Statistiques par Cluster (nœuds SOM)")
        lbl.setObjectName("sectionLabel")
        hdr.addWidget(lbl)
        hdr.addStretch()
        btn_export_txt = QPushButton("  Exporter .txt")
        btn_export_txt.setObjectName("exportBtn")
        ico_exp = _icon("prisma.export-fig", "#cba6f7")
        if ico_exp:
            btn_export_txt.setIcon(ico_exp)
            btn_export_txt.setIconSize(QSize(16, 16))
        btn_export_txt.clicked.connect(self._export_cluster_txt)
        hdr.addWidget(btn_export_txt)
        layout.addLayout(hdr)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Cluster (nœud)", "Métacluster", "Cellules", "% Total", "% Pathologique"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setAlternatingRowColors(False)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setMaximumHeight(280)
        layout.addWidget(self.results_table)

        # ── Barre Vue Combinée : label + bouton interactif sur une seule ligne ──
        combined_bar = QWidget()
        combined_bar.setObjectName("combinedBar")
        combined_bar.setStyleSheet("""
            QWidget#combinedBar {
                background: rgba(16,24,37,0.92);
                border: 1px solid rgba(255,255,255,0.055);
                border-top: 1px solid rgba(123,82,255,0.35);
            }
        """)
        combined_bar_layout = QHBoxLayout(combined_bar)
        combined_bar_layout.setContentsMargins(14, 10, 14, 10)
        combined_bar_layout.setSpacing(12)

        lbl_combined_icon = QLabel("⬡")
        lbl_combined_icon.setFont(QFont("Segoe UI", 13))
        lbl_combined_icon.setStyleSheet("color: #7B52FF; background: transparent;")
        combined_bar_layout.addWidget(lbl_combined_icon)

        lbl2 = QLabel("Vue Combinée Nœuds SOM")
        lbl2.setObjectName("subtitleLabel")
        lbl2.setStyleSheet(
            "color: #EEF2F7; font-size: 10pt; font-weight: 600; background: transparent;"
        )
        combined_bar_layout.addWidget(lbl2)
        combined_bar_layout.addStretch()

        self.btn_open_combined = QPushButton("  Ouvrir interactif")
        self.btn_open_combined.setObjectName("successBtn")
        self.btn_open_combined.setEnabled(False)
        self.btn_open_combined.setFixedHeight(32)
        ico_oc = _icon("prisma.external-link", "#11111b")
        if ico_oc:
            self.btn_open_combined.setIcon(ico_oc)
            self.btn_open_combined.setIconSize(QSize(16, 16))
        self.btn_open_combined.clicked.connect(self._open_combined_html)
        combined_bar_layout.addWidget(self.btn_open_combined)
        layout.addWidget(combined_bar)

        self._results_web = None
        # Canvas factice (jamais visible) — conservé pour compatibilité _populate_results
        self._combined_canvas = MatplotlibCanvas(tab, width=1, height=1)
        self._combined_canvas.hide()
        self._combined_toolbar = NavigationToolbar(self._combined_canvas, tab)
        self._combined_toolbar.hide()

        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setPlaceholderText("Le résumé de l'analyse apparaîtra ici…")
        layout.addWidget(self.txt_summary, 1)

        _ico_res = _icon("prisma.heatmap", "#FF3D6E", 16)
        self.tabs.addTab(tab, _ico_res or QIcon(), "RÉSULTATS")

    def _build_fcs_viewer_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(8)

        # Bouton recharger le FCS de sortie (patho auto-chargé)
        self.btn_reload_patho_fcs = QPushButton("  Recharger FCS sortie")
        self.btn_reload_patho_fcs.setObjectName("ghostBtn")
        self.btn_reload_patho_fcs.setEnabled(False)
        ico_reload = _icon("prisma.sync", "#39FF8A")
        if ico_reload:
            self.btn_reload_patho_fcs.setIcon(ico_reload)
            self.btn_reload_patho_fcs.setIconSize(QSize(16, 16))
        self.btn_reload_patho_fcs.clicked.connect(self._reload_patho_fcs)
        ctrl_layout.addWidget(self.btn_reload_patho_fcs)

        # Bouton parcourir un autre FCS
        self.btn_load_fcs_viz = QPushButton("  Parcourir…")
        self.btn_load_fcs_viz.setObjectName("ghostBtn")
        ico_fcs = _icon("prisma.fcs-file", "#5BAAFF")
        if ico_fcs:
            self.btn_load_fcs_viz.setIcon(ico_fcs)
            self.btn_load_fcs_viz.setIconSize(QSize(16, 16))
        self.btn_load_fcs_viz.clicked.connect(lambda: self._load_fcs_for_visualization())
        ctrl_layout.addWidget(self.btn_load_fcs_viz)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: rgba(255,255,255,0.10);")
        ctrl_layout.addWidget(sep)

        ctrl_layout.addWidget(QLabel("Axe X:"))
        self.combo_fcs_x = DarkComboBox()
        self.combo_fcs_x.setMinimumWidth(120)
        self.combo_fcs_x.currentIndexChanged.connect(self._update_fcs_plot)
        ctrl_layout.addWidget(self.combo_fcs_x)

        ctrl_layout.addWidget(QLabel("Axe Y:"))
        self.combo_fcs_y = DarkComboBox()
        self.combo_fcs_y.setMinimumWidth(120)
        self.combo_fcs_y.currentIndexChanged.connect(self._update_fcs_plot)
        ctrl_layout.addWidget(self.combo_fcs_y)

        ctrl_layout.addWidget(QLabel("Type:"))
        self.combo_fcs_plot_type = DarkComboBox()
        self.combo_fcs_plot_type.addItems(["Scatter", "Densite", "Contour"])
        self.combo_fcs_plot_type.currentIndexChanged.connect(self._update_fcs_plot)
        ctrl_layout.addWidget(self.combo_fcs_plot_type)

        ctrl_layout.addWidget(QLabel("Couleur:"))
        self.combo_fcs_color = DarkComboBox()
        self.combo_fcs_color.addItems(
            ["Aucune", "FlowSOM_cluster", "FlowSOM_metacluster", "Condition"]
        )
        self.combo_fcs_color.currentIndexChanged.connect(self._update_fcs_plot)
        ctrl_layout.addWidget(self.combo_fcs_color)

        ctrl_layout.addWidget(QLabel("Cellules:"))
        self.spin_fcs_cells = QSpinBox()
        self.spin_fcs_cells.setRange(1000, 500000)
        self.spin_fcs_cells.setValue(10000)
        self.spin_fcs_cells.setSingleStep(5000)
        ctrl_layout.addWidget(self.spin_fcs_cells)

        self.chk_fcs_all_cells = QCheckBox("Toutes")
        self.chk_fcs_all_cells.stateChanged.connect(self._toggle_fcs_all_cells)
        ctrl_layout.addWidget(self.chk_fcs_all_cells)

        self.chk_fcs_jitter = QCheckBox("Jitter")
        self.chk_fcs_jitter.setChecked(False)
        self.chk_fcs_jitter.stateChanged.connect(self._update_fcs_plot)
        ctrl_layout.addWidget(self.chk_fcs_jitter)

        btn_refresh = QPushButton("  Rafraichir")
        ico_ref = _icon("prisma.sync", "#EEF2F7")
        if ico_ref:
            btn_refresh.setIcon(ico_ref)
            btn_refresh.setIconSize(QSize(16, 16))
        btn_refresh.clicked.connect(self._update_fcs_plot)
        ctrl_layout.addWidget(btn_refresh)

        ctrl_layout.addStretch()
        layout.addWidget(ctrl)

        self.fcs_viz_canvas = MatplotlibCanvas(tab, width=10, height=8)
        self.fcs_viz_canvas.setMinimumHeight(480)
        fcs_toolbar = NavigationToolbar(self.fcs_viz_canvas, tab)
        fcs_toolbar.setObjectName("matplotlibToolbar")
        layout.addWidget(fcs_toolbar)
        layout.addWidget(self.fcs_viz_canvas)

        self.lbl_fcs_info = QLabel("Chargez un fichier FCS pour visualiser")
        self.lbl_fcs_info.setObjectName("fcsInfoLabel")
        layout.addWidget(self.lbl_fcs_info)

        _ico_fcs = _icon("prisma.fcs-file", "#7EC8E3", 16)
        self.tabs.addTab(tab, _ico_fcs or QIcon(), "VIEWER FCS")

    # ==================================================================
    # LOGIQUE : Chargement config
    # ==================================================================

    def _load_default_config(self) -> None:
        try:
            from flowsom_pipeline_pro.config.pipeline_config import PipelineConfig

            if _DEFAULT_CONFIG_PATH.exists():
                self._config = PipelineConfig.from_yaml(str(_DEFAULT_CONFIG_PATH))
                self._sync_config_to_ui()
                self._log(f" Config chargée : {_DEFAULT_CONFIG_PATH.name}")
            else:
                self._config = PipelineConfig()
        except Exception as e:
            self._log(f" Erreur chargement config : {e}")

        try:
            import yaml

            if _MRD_CONFIG_PATH.exists():
                with open(_MRD_CONFIG_PATH, "r", encoding="utf-8") as f:
                    self._mrd_raw = yaml.safe_load(f) or {}
                self._sync_mrd_config_to_ui()
        except Exception as e:
            self._log(f" Avertissement MRD config : {e}")

    def _sync_config_to_ui(self) -> None:
        c = self._config
        if c is None:
            return

        if hasattr(c, "paths"):
            if c.paths.healthy_folder:
                self.drop_healthy.set_path(str(c.paths.healthy_folder))
            if c.paths.patho_folder:
                self.drop_patho.set_path(str(c.paths.patho_folder))
            if c.paths.output_dir:
                self.drop_output.set_path(str(c.paths.output_dir))

        if hasattr(c, "flowsom"):
            self.spin_xdim.setValue(c.flowsom.xdim)
            self.spin_ydim.setValue(c.flowsom.ydim)
            self.spin_metaclusters.setValue(c.flowsom.n_metaclusters)
            self.spin_seed.setValue(c.flowsom.seed)
            self.spin_lr.setValue(c.flowsom.learning_rate)
            self.spin_sigma.setValue(c.flowsom.sigma)

        if hasattr(c, "transform"):
            idx = self.combo_transform.findText(c.transform.method)
            if idx >= 0:
                self.combo_transform.setCurrentIndex(idx)
            self.spin_cofactor.setValue(c.transform.cofactor)

        if hasattr(c, "normalize"):
            idx = self.combo_normalize.findText(c.normalize.method)
            if idx >= 0:
                self.combo_normalize.setCurrentIndex(idx)

        if hasattr(c, "pregate"):
            self.chk_pregate.setChecked(c.pregate.apply)
            idx = self.combo_gate_mode.findText(c.pregate.mode)
            if idx >= 0:
                self.combo_gate_mode.setCurrentIndex(idx)
            self.chk_viable.setChecked(c.pregate.viable)
            self.chk_singlets.setChecked(c.pregate.singlets)
            self.chk_cd45.setChecked(c.pregate.cd45)
            self.chk_cd34.setChecked(c.pregate.cd34)
            if hasattr(c.pregate, "mode_blastes_vs_normal"):
                self.chk_mode_blastes.setChecked(c.pregate.mode_blastes_vs_normal)
            if hasattr(c.pregate, "cd45_autogating_mode"):
                _cd45_mode_idx = self.combo_cd45_autogating_mode.findText(
                    c.pregate.cd45_autogating_mode
                )
                if _cd45_mode_idx >= 0:
                    self.combo_cd45_autogating_mode.setCurrentIndex(_cd45_mode_idx)

        if hasattr(c, "visualization"):
            self.chk_umap.setChecked(c.visualization.umap_enabled)
        if hasattr(c, "gpu"):
            self.chk_gpu.setChecked(c.gpu.enabled)
        if hasattr(c, "batch"):
            self.chk_batch.setChecked(c.batch.enabled)
        if hasattr(c, "analysis"):
            self.chk_compare.setChecked(c.analysis.compare_mode)
        if hasattr(c, "auto_clustering"):
            self.chk_auto_clustering.setChecked(c.auto_clustering.enabled)
        if hasattr(c, "population_mapping"):
            self.chk_pop_mapping.setChecked(c.population_mapping.enabled)
        if hasattr(c, "downsampling"):
            self.chk_downsampling.setChecked(c.downsampling.enabled)
            self.spin_max_cells.setValue(c.downsampling.max_cells_per_file)
        if hasattr(c, "export_mode"):
            idx = self.combo_export_mode.findText(getattr(c.export_mode, "mode", "standard"))
            if idx >= 0:
                self.combo_export_mode.setCurrentIndex(idx)
        if hasattr(c, "patho_fcs_export"):
            idx = self.combo_mrd_fcs_method.findText(
                getattr(c.patho_fcs_export, "mrd_method", "flo")
            )
            if idx >= 0:
                self.combo_mrd_fcs_method.setCurrentIndex(idx)
        if hasattr(c, "stratified_downsampling"):
            self.chk_balance_conditions.setChecked(
                getattr(c.stratified_downsampling, "balance_conditions", True)
            )
            self.spin_imbalance_ratio.setValue(
                getattr(c.stratified_downsampling, "imbalance_ratio", 2.0)
            )
            self.chk_allow_oversampling.setChecked(
                getattr(c.stratified_downsampling, "allow_oversampling", False)
            )

        # ── Marqueurs & Scatter ───────────────────────────────────────
        if hasattr(c, "markers"):
            self.chk_exclude_scatter.setChecked(getattr(c.markers, "exclude_scatter", True))
            self.chk_keep_area_only.setChecked(getattr(c.markers, "keep_area_only", True))
            excl = getattr(c.markers, "exclude_additional", [])
            self.edit_exclude_cols.setText(", ".join(excl) if excl else "")

        # ── Harmony ───────────────────────────────────────────────────
        if hasattr(c, "data_integration"):
            self.chk_harmony.setChecked(getattr(c.data_integration, "enabled", True))
            hp = getattr(c.data_integration, "harmony_params", None)
            if hp is not None:
                self.spin_harmony_sigma.setValue(getattr(hp, "sigma", 0.05))
                nclust = getattr(hp, "nclust", 30)
                self.spin_harmony_nclust.setValue(nclust if nclust is not None else 0)
                self.spin_harmony_max_iter.setValue(getattr(hp, "max_iter", 10))
                self.spin_harmony_block.setValue(getattr(hp, "block_size", 0.20))
                markers_align = getattr(hp, "markers_to_align", [])
                self.edit_harmony_markers.setText(", ".join(markers_align) if markers_align else "")

        # ── Paramètres GMM / KDE ──────────────────────────────────────
        if hasattr(c, "pregate"):
            self.combo_density_method.setCurrentIndex(
                0 if getattr(c.pregate, "density_method", "GMM") == "GMM" else 1
            )
            self.spin_gmm_components.setValue(getattr(c.pregate, "gmm_n_components_debris", 3))
            idx_cov = self.combo_gmm_cov.findText(getattr(c.pregate, "gmm_covariance_type", "full"))
            if idx_cov >= 0:
                self.combo_gmm_cov.setCurrentIndex(idx_cov)
            self.spin_kde_finesse.setValue(getattr(c.pregate, "kde_cd45_finesse", 0.6))
            self.spin_kde_sigma.setValue(getattr(c.pregate, "kde_cd45_sigma_smooth", 10))
            self.spin_kde_seuil.setValue(getattr(c.pregate, "kde_cd45_seuil_relatif", 0.05))

    def _sync_ui_to_config(self) -> None:
        c = self._config
        if c is None:
            return

        healthy = self.drop_healthy.path
        if healthy:
            c.paths.healthy_folder = healthy
        patho = self.drop_patho.path
        if patho:
            c.paths.patho_folder = patho
        output = self.drop_output.path
        if output:
            c.paths.output_dir = output

        c.flowsom.xdim = self.spin_xdim.value()
        c.flowsom.ydim = self.spin_ydim.value()
        c.flowsom.n_metaclusters = self.spin_metaclusters.value()
        c.flowsom.seed = self.spin_seed.value()
        c.flowsom.learning_rate = self.spin_lr.value()
        c.flowsom.sigma = self.spin_sigma.value()

        c.transform.method = self.combo_transform.currentText()
        c.transform.cofactor = self.spin_cofactor.value()
        c.normalize.method = self.combo_normalize.currentText()

        c.pregate.apply = self.chk_pregate.isChecked()
        c.pregate.mode = self.combo_gate_mode.currentText()
        c.pregate.viable = self.chk_viable.isChecked()
        c.pregate.singlets = self.chk_singlets.isChecked()
        c.pregate.cd45 = self.chk_cd45.isChecked()
        c.pregate.cd34 = self.chk_cd34.isChecked()
        if hasattr(c.pregate, "mode_blastes_vs_normal"):
            c.pregate.mode_blastes_vs_normal = self.chk_mode_blastes.isChecked()
        if hasattr(c.pregate, "cd45_autogating_mode"):
            c.pregate.cd45_autogating_mode = self.combo_cd45_autogating_mode.currentText()

        c.visualization.umap_enabled = self.chk_umap.isChecked()
        c.gpu.enabled = self.chk_gpu.isChecked()
        c.batch.enabled = self.chk_batch.isChecked()
        c.analysis.compare_mode = self.chk_compare.isChecked()
        c.auto_clustering.enabled = self.chk_auto_clustering.isChecked()
        c.population_mapping.enabled = self.chk_pop_mapping.isChecked()
        c.downsampling.enabled = self.chk_downsampling.isChecked()
        c.downsampling.max_cells_per_file = self.spin_max_cells.value()
        if hasattr(c, "export_mode"):
            c.export_mode.mode = self.combo_export_mode.currentText()
        if hasattr(c, "patho_fcs_export"):
            c.patho_fcs_export.mrd_method = self.combo_mrd_fcs_method.currentText()
        if hasattr(c, "stratified_downsampling"):
            c.stratified_downsampling.balance_conditions = self.chk_balance_conditions.isChecked()
            c.stratified_downsampling.imbalance_ratio = self.spin_imbalance_ratio.value()
            c.stratified_downsampling.allow_oversampling = self.chk_allow_oversampling.isChecked()

        # ── Marqueurs & Scatter ───────────────────────────────────────
        if hasattr(c, "markers"):
            c.markers.exclude_scatter = self.chk_exclude_scatter.isChecked()
            c.markers.keep_area_only = self.chk_keep_area_only.isChecked()
            raw_excl = self.edit_exclude_cols.text().strip()
            c.markers.exclude_additional = (
                [s.strip() for s in raw_excl.split(",") if s.strip()] if raw_excl else []
            )

        # ── Harmony ───────────────────────────────────────────────────
        if hasattr(c, "data_integration"):
            c.data_integration.enabled = self.chk_harmony.isChecked()
            hp = c.data_integration.harmony_params
            hp.sigma = self.spin_harmony_sigma.value()
            nclust_val = self.spin_harmony_nclust.value()
            hp.nclust = nclust_val if nclust_val > 0 else None
            hp.max_iter = self.spin_harmony_max_iter.value()
            hp.block_size = self.spin_harmony_block.value()
            raw_markers = self.edit_harmony_markers.text().strip()
            hp.markers_to_align = (
                [s.strip() for s in raw_markers.split(",") if s.strip()] if raw_markers else []
            )

        # ── Paramètres GMM / KDE ──────────────────────────────────────
        if hasattr(c, "pregate"):
            c.pregate.density_method = self.combo_density_method.currentText()
            c.pregate.gmm_n_components_debris = self.spin_gmm_components.value()
            c.pregate.gmm_covariance_type = self.combo_gmm_cov.currentText()
            c.pregate.kde_cd45_finesse = self.spin_kde_finesse.value()
            c.pregate.kde_cd45_sigma_smooth = self.spin_kde_sigma.value()
            c.pregate.kde_cd45_seuil_relatif = self.spin_kde_seuil.value()

        # ── Mapping colonnes FCS (injecté dans config._extra) ─────────
        rename_map = self._get_column_rename_map()
        if rename_map:
            c._extra["column_rename_map"] = rename_map
        else:
            c._extra.pop("column_rename_map", None)

        self._sync_ui_to_mrd_config()

    def _sync_mrd_config_to_ui(self) -> None:
        mrd = getattr(self, "_mrd_raw", {}) or {}
        params = mrd.get("mrd_parameters", {})
        if not params:
            return
        method = params.get("method", "all")
        idx = self.combo_mrd_method.findText(method)
        if idx >= 0:
            self.combo_mrd_method.setCurrentIndex(idx)
        eln = params.get("eln_standards", {})
        self.spin_eln_min_events.setValue(int(eln.get("min_cluster_events", 50)))
        self.spin_eln_positivity.setValue(float(eln.get("clinical_positivity_pct", 0.1)))
        flo = params.get("method_flo", {})
        self.spin_flo_multiplier.setValue(float(flo.get("normal_marrow_multiplier", 2.0)))
        jf = params.get("method_jf", {})
        self.spin_jf_max_normal.setValue(float(jf.get("max_normal_marrow_pct", 0.1)))
        self.spin_jf_min_patho.setValue(float(jf.get("min_patho_cells_pct", 10.0)))
        bpf = params.get("blast_phenotype_filter", {})
        self.chk_blast_filter.setChecked(bool(bpf.get("enabled", False)))

    def _sync_ui_to_mrd_config(self) -> None:
        if not hasattr(self, "_mrd_raw"):
            self._mrd_raw = {}
        params = self._mrd_raw.setdefault("mrd_parameters", {})
        params["method"] = self.combo_mrd_method.currentText()
        params.setdefault("eln_standards", {})["min_cluster_events"] = (
            self.spin_eln_min_events.value()
        )
        params["eln_standards"]["clinical_positivity_pct"] = self.spin_eln_positivity.value()
        params.setdefault("method_flo", {})["normal_marrow_multiplier"] = (
            self.spin_flo_multiplier.value()
        )
        params.setdefault("method_jf", {})["max_normal_marrow_pct"] = (
            self.spin_jf_max_normal.value()
        )
        params["method_jf"]["min_patho_cells_pct"] = self.spin_jf_min_patho.value()
        params.setdefault("blast_phenotype_filter", {})["enabled"] = (
            self.chk_blast_filter.isChecked()
        )
        try:
            import yaml

            with open(_MRD_CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(self._mrd_raw, f, allow_unicode=True, default_flow_style=False)
        except Exception as e:
            self._log(f" Avertissement sauvegarde MRD config : {e}")

    # ==================================================================
    # Sélection dossiers (via boutons Parcourir)
    # ==================================================================

    def _select_healthy_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier NBM / Sain")
        if path:
            self.drop_healthy.set_path(path)
            self._refresh_fcs_preview()

    def _select_patho_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier Pathologique")
        if path:
            self.drop_patho.set_path(path)
            self._refresh_fcs_preview()

    def _select_output_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier de sortie")
        if path:
            self.drop_output.set_path(path)

    def _refresh_fcs_preview(self) -> None:
        """Met à jour le badge de résumé après sélection d'un dossier."""
        folder_conditions = []
        if self.drop_healthy.path and Path(self.drop_healthy.path).is_dir():
            folder_conditions.append((self.drop_healthy.path, "Sain"))
        if self.drop_patho.path and Path(self.drop_patho.path).is_dir():
            folder_conditions.append((self.drop_patho.path, "Pathologique"))

        if not folder_conditions:
            self.lbl_preview_summary.setText(
                "Sélectionnez les dossiers FCS ci-dessus, puis cliquez sur «Aperçu» pour vérifier les fichiers."
            )
            return

        # Comptage rapide uniquement pour le badge
        n_sain = (
            sum(1 for p in Path(folder_conditions[0][0]).iterdir() if p.suffix.lower() == ".fcs")
            if folder_conditions
            else 0
        )
        n_patho = (
            sum(1 for p in Path(folder_conditions[-1][0]).iterdir() if p.suffix.lower() == ".fcs")
            if len(folder_conditions) > 1
            else 0
        )

        parts = []
        if folder_conditions[0][1] == "Sain" and n_sain:
            parts.append(f"NBM : {n_sain} fichier(s)")
        if n_patho:
            parts.append(f"Patho : {n_patho} fichier(s)")
        self.lbl_preview_summary.setText(
            "  ·  ".join(parts) + "   —   cliquez sur «Aperçu» pour les détails."
            if parts
            else "Aucun fichier FCS trouvé."
        )

    def _open_preview_dialog(self) -> None:
        """Ouvre une fenêtre modale avec la liste complète des fichiers FCS."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton

        folder_conditions = []
        if self.drop_healthy.path and Path(self.drop_healthy.path).is_dir():
            folder_conditions.append((self.drop_healthy.path, "Sain"))
        if self.drop_patho.path and Path(self.drop_patho.path).is_dir():
            folder_conditions.append((self.drop_patho.path, "Pathologique"))

        if not folder_conditions:
            QMessageBox.information(
                self,
                "Aperçu FCS",
                "Aucun dossier sélectionné.\nVeuillez d'abord choisir un dossier FCS.",
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Aperçu des fichiers FCS détectés")
        dlg.resize(900, 520)
        dlg.setStyleSheet(self.styleSheet())
        vbox = QVBoxLayout(dlg)
        vbox.setContentsMargins(16, 16, 16, 12)
        vbox.setSpacing(10)

        # En-tête
        hdr = QHBoxLayout()
        lbl_title = QLabel("Fichiers FCS détectés")
        lbl_title.setObjectName("dialogTitle")
        hdr.addWidget(lbl_title)
        hdr.addStretch()
        btn_refresh = QPushButton("  Actualiser")
        btn_refresh.setObjectName("ghostBtn")
        ico_r = _icon("prisma.sync", "#5BAAFF")
        if ico_r:
            btn_refresh.setIcon(ico_r)
            btn_refresh.setIconSize(QSize(16, 16))
        hdr.addWidget(btn_refresh)
        vbox.addLayout(hdr)

        # Table
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(
            ["Fichier", "Condition", "Cellules", "Canaux", "Marqueurs ($PnS)"]
        )
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setAlternatingRowColors(False)
        table.setWordWrap(True)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        vbox.addWidget(table, 1)

        # Label résumé
        lbl_sum = QLabel("")
        lbl_sum.setObjectName("dialogCount")
        vbox.addWidget(lbl_sum)

        def _populate():
            try:
                _populate_inner()
            except Exception as _exc:
                import traceback

                lbl_sum.setText(f"Erreur aperçu : {_exc}")
                self._log(f"Erreur aperçu FCS : {traceback.format_exc()}")

        def _populate_inner():
            rows = []
            for folder, condition in folder_conditions:
                fcs_files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() == ".fcs")
                for fcs_path in fcs_files:
                    n_ev, n_ch, marker_names = self._read_fcs_header_full(fcs_path)
                    rows.append((fcs_path.name, condition, n_ev, n_ch, marker_names))

            table.setRowCount(len(rows))
            for i, (fname, cond, n_ev, n_ch, markers) in enumerate(rows):
                table.setItem(i, 0, QTableWidgetItem(fname))
                cond_item = QTableWidgetItem(cond)
                cond_item.setForeground(QColor("#39FF8A") if cond == "Sain" else QColor("#FF3D6E"))
                table.setItem(i, 1, cond_item)

                ev_item = QTableWidgetItem(f"{n_ev:,}" if isinstance(n_ev, int) else str(n_ev))
                ev_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(i, 2, ev_item)

                ch_item = QTableWidgetItem(str(n_ch))
                ch_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(i, 3, ch_item)

                mk_str = ", ".join(m for m in markers if m) if markers else "—"
                mk_item = QTableWidgetItem(mk_str)
                mk_item.setToolTip(mk_str)
                table.setItem(i, 4, mk_item)

            total_sain = sum(r[2] for r in rows if r[1] == "Sain" and isinstance(r[2], int))
            total_patho = sum(
                r[2] for r in rows if r[1] == "Pathologique" and isinstance(r[2], int)
            )
            n_sf = sum(1 for r in rows if r[1] == "Sain")
            n_pf = sum(1 for r in rows if r[1] == "Pathologique")
            parts = []
            if n_sf:
                parts.append(f"NBM : {n_sf} fichier(s) — {total_sain:,} cellules")
            if n_pf:
                parts.append(f"Patho : {n_pf} fichier(s) — {total_patho:,} cellules")
            lbl_sum.setText("  |  ".join(parts) if parts else "Aucun fichier trouvé.")

        btn_refresh.clicked.connect(_populate)
        _populate()

        # Boutons bas
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_close = QPushButton("  Fermer")
        btn_close.setObjectName("primaryBtn")
        btn_close.setMinimumHeight(38)
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        vbox.addLayout(btn_row)

        dlg.exec_()

    def _open_rename_dialog(self) -> None:
        """Ouvre l'éditeur complet de renommage des colonnes FCS."""
        import re
        from PyQt5.QtWidgets import (
            QDialog,
            QVBoxLayout,
            QHBoxLayout,
            QPushButton,
            QLabel as _QL,
            QTabWidget,
            QComboBox as _QCombo,
            QGroupBox,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Renommage des colonnes FCS — harmonisation Kaluza")
        dlg.resize(900, 640)
        dlg.setStyleSheet(self.styleSheet())
        vbox = QVBoxLayout(dlg)
        vbox.setContentsMargins(16, 16, 16, 12)
        vbox.setSpacing(10)

        # Titre + description
        lbl_title = _QL("Renommage colonnes FCS  →  Kaluza")
        lbl_title.setObjectName("dialogTitle")
        vbox.addWidget(lbl_title)

        lbl_desc = _QL(
            "Deux modes disponibles : «Fichier par fichier» pour des règles spécifiques à chaque FCS, "
            "«Homogénéisation» pour appliquer les mêmes règles à tous les fichiers."
        )
        lbl_desc.setWordWrap(True)
        lbl_desc.setObjectName("dialogDesc")
        vbox.addWidget(lbl_desc)

        # ── Onglets Fichier-par-fichier vs Homogénéisation ──────────────
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(
            "QTabBar::tab { padding: 6px 18px; font-size: 9pt; }"
            "QTabBar::tab:selected { color: #5BAAFF; border-bottom: 2px solid #5BAAFF; }"
        )

        # ── Onglet 1 : Fichier par fichier ─────────────────────────────
        tab_per_file = QWidget()
        tab_per_file.setStyleSheet("background: transparent;")
        per_vbox = QVBoxLayout(tab_per_file)
        per_vbox.setContentsMargins(8, 8, 8, 8)
        per_vbox.setSpacing(8)

        # Sélecteur de fichier FCS
        file_sel_row = QHBoxLayout()
        lbl_file_sel = _QL("Fichier FCS :")
        lbl_file_sel.setObjectName("dialogDesc")
        file_sel_row.addWidget(lbl_file_sel)

        combo_files = _QCombo()
        combo_files.setMinimumWidth(360)
        combo_files.setStyleSheet(
            "QComboBox { background: #101825; color: #EEF2F7; border: 1px solid rgba(255,255,255,0.12);"
            " border-radius: 4px; padding: 4px 10px; } "
            "QComboBox QAbstractItemView { background: #101825; color: #EEF2F7; }"
        )
        file_sel_row.addWidget(combo_files, 1)

        # Collecte tous les FCS disponibles (sain + patho)
        _all_fcs: List[Path] = []
        for folder in (self.drop_healthy.path, self.drop_patho.path):
            if folder and Path(folder).is_dir():
                _all_fcs += sorted(p for p in Path(folder).iterdir() if p.suffix.lower() == ".fcs")
        for p in _all_fcs:
            combo_files.addItem(p.name, str(p))
        if not _all_fcs:
            combo_files.addItem("Aucun fichier FCS trouvé")

        per_vbox.addLayout(file_sel_row)

        # Table renommage par fichier
        per_table = QTableWidget()
        per_table.setColumnCount(2)
        per_table.setHorizontalHeaderLabels(["Colonne FCS brute (source)", "Nom cible Kaluza"])
        per_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        per_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        per_table.setSelectionBehavior(QTableWidget.SelectRows)
        per_table.setAlternatingRowColors(False)
        per_table.verticalHeader().setVisible(False)
        per_table.setMinimumHeight(260)
        per_vbox.addWidget(per_table, 1)

        # Stockage par fichier : dict {fcs_path_str: [(src, dst), ...]}
        _per_file_rules: Dict[str, List] = {}

        def _load_per_file(fcs_path_str: str) -> None:
            """Charge les règles existantes pour ce fichier dans per_table."""
            per_table.blockSignals(True)
            per_table.setRowCount(0)
            rules = _per_file_rules.get(fcs_path_str, [])
            for src, dst in rules:
                r = per_table.rowCount()
                per_table.insertRow(r)
                per_table.setItem(r, 0, QTableWidgetItem(src))
                per_table.setItem(r, 1, QTableWidgetItem(dst))
            per_table.blockSignals(False)

        def _save_per_file(fcs_path_str: str) -> None:
            """Sauvegarde les règles de per_table pour ce fichier."""
            rules = []
            for r in range(per_table.rowCount()):
                s = per_table.item(r, 0)
                d = per_table.item(r, 1)
                src = s.text().strip() if s else ""
                dst = d.text().strip() if d else ""
                if src:
                    rules.append((src, dst))
            _per_file_rules[fcs_path_str] = rules

        _current_per_file: List[str] = [""]

        def _on_file_changed(idx: int) -> None:
            if _current_per_file[0]:
                _save_per_file(_current_per_file[0])
            fcs_str = combo_files.currentData() or ""
            _current_per_file[0] = fcs_str
            _load_per_file(fcs_str)

        combo_files.currentIndexChanged.connect(_on_file_changed)
        if _all_fcs:
            _current_per_file[0] = str(_all_fcs[0])

        # Boutons per-file
        per_btn_row = QHBoxLayout()

        def _mk_btn(label, icon_name, color="#5BAAFF", tbl=None):
            b = QPushButton(f"  {label}")
            b.setObjectName("ghostBtn")
            b.setMinimumHeight(34)
            ico = _icon(icon_name, color)
            if ico:
                b.setIcon(ico)
                b.setIconSize(QSize(16, 16))
            return b

        btn_per_detect = _mk_btn("Détecter colonnes", "prisma.search", "#5BAAFF")
        btn_per_add = _mk_btn("Ajouter ligne", "prisma.plus", "#39FF8A")
        btn_per_del = _mk_btn("Supprimer sélection", "prisma.trash", "#FF3D6E")

        per_btn_row.addWidget(btn_per_detect)
        per_btn_row.addWidget(btn_per_add)
        per_btn_row.addWidget(btn_per_del)
        per_btn_row.addStretch()
        per_vbox.addLayout(per_btn_row)

        def _per_detect():
            fcs_str = combo_files.currentData() or ""
            if not fcs_str or not Path(fcs_str).exists():
                QMessageBox.information(dlg, "Détecter", "Sélectionnez d'abord un fichier FCS.")
                return
            _, _, col_names = self._read_fcs_header_full(Path(fcs_str))
            existing = set()
            for r in range(per_table.rowCount()):
                it = per_table.item(r, 0)
                if it:
                    existing.add(it.text().strip())
            new_cols = [c for c in col_names if c and c not in existing]
            for col in new_cols:
                short = re.sub(
                    r"\s+(KO|FITC|PE|APC|BV\d+|Cy\d+|PerCP|EF\d+|BUV\d+|"
                    r"BB\d+|R\d+|AF\d+|V\d+|Pacific[- ]Blue|AlexaFluor\d*"
                    r"|Pacific\s*Orange|BrilliantViolet\d*)\b.*",
                    "",
                    col,
                    flags=re.IGNORECASE,
                ).strip()
                r = per_table.rowCount()
                per_table.insertRow(r)
                per_table.setItem(r, 0, QTableWidgetItem(col))
                per_table.setItem(r, 1, QTableWidgetItem(short))
            if new_cols:
                QMessageBox.information(
                    dlg,
                    "Colonnes détectées",
                    f"{len(new_cols)} colonne(s) ajoutée(s) depuis\n{Path(fcs_str).name}",
                )

        def _per_add():
            r = per_table.rowCount()
            per_table.insertRow(r)
            per_table.setItem(r, 0, QTableWidgetItem(""))
            per_table.setItem(r, 1, QTableWidgetItem(""))
            per_table.editItem(per_table.item(r, 0))

        def _per_del():
            rows = sorted({idx.row() for idx in per_table.selectedIndexes()}, reverse=True)
            for r in rows:
                per_table.removeRow(r)

        btn_per_detect.clicked.connect(_per_detect)
        btn_per_add.clicked.connect(_per_add)
        btn_per_del.clicked.connect(_per_del)

        tab_widget.addTab(tab_per_file, "  Fichier par fichier  ")

        # ── Onglet 2 : Homogénéisation globale ─────────────────────────
        tab_global = QWidget()
        tab_global.setStyleSheet("background: transparent;")
        glob_vbox = QVBoxLayout(tab_global)
        glob_vbox.setContentsMargins(8, 8, 8, 8)
        glob_vbox.setSpacing(8)

        lbl_glob = _QL(
            "Ces règles s'appliquent à TOUS les fichiers FCS, avant les règles fichier-par-fichier."
        )
        lbl_glob.setWordWrap(True)
        lbl_glob.setObjectName("dialogDesc")
        glob_vbox.addWidget(lbl_glob)

        local_table = QTableWidget()
        local_table.setColumnCount(2)
        local_table.setHorizontalHeaderLabels(["Colonne FCS brute (source)", "Nom cible Kaluza"])
        local_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        local_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        local_table.setSelectionBehavior(QTableWidget.SelectRows)
        local_table.setAlternatingRowColors(False)
        local_table.verticalHeader().setVisible(False)
        local_table.setMinimumHeight(260)
        glob_vbox.addWidget(local_table, 1)

        lbl_count = _QL("0 règle(s)")
        lbl_count.setObjectName("dialogCount")

        def _update_count():
            n = local_table.rowCount()
            active = sum(
                1
                for r in range(n)
                if (local_table.item(r, 0) and local_table.item(r, 0).text().strip())
                and (local_table.item(r, 1) and local_table.item(r, 1).text().strip())
                and local_table.item(r, 0).text().strip() != local_table.item(r, 1).text().strip()
            )
            lbl_count.setText(f"{active} règle(s) active(s)")

        local_table.itemChanged.connect(lambda _: _update_count())

        # Charger les règles globales existantes depuis self.rename_table
        for r in range(self.rename_table.rowCount()):
            src_item = self.rename_table.item(r, 0)
            dst_item = self.rename_table.item(r, 1)
            src = src_item.text() if src_item else ""
            dst = dst_item.text() if dst_item else ""
            row = local_table.rowCount()
            local_table.insertRow(row)
            local_table.setItem(row, 0, QTableWidgetItem(src))
            local_table.setItem(row, 1, QTableWidgetItem(dst))
        _update_count()

        glob_btn_row = QHBoxLayout()
        btn_detect = _mk_btn("Détecter colonnes FCS", "prisma.search", "#5BAAFF")
        btn_add = _mk_btn("Ajouter ligne", "prisma.plus", "#39FF8A")
        btn_del = _mk_btn("Supprimer sélection", "prisma.trash", "#FF3D6E")
        btn_clear = _mk_btn("Tout effacer", "prisma.eraser", "#f9e2af")
        glob_btn_row.addWidget(btn_detect)
        glob_btn_row.addWidget(btn_add)
        glob_btn_row.addWidget(btn_del)
        glob_btn_row.addWidget(btn_clear)
        glob_btn_row.addStretch()
        glob_btn_row.addWidget(lbl_count)
        glob_vbox.addLayout(glob_btn_row)

        def _detect():
            fcs_path: Optional[Path] = None
            for folder in (self.drop_healthy.path, self.drop_patho.path):
                if folder and Path(folder).is_dir():
                    for p in sorted(Path(folder).iterdir()):
                        if p.suffix.lower() == ".fcs":
                            fcs_path = p
                            break
                if fcs_path:
                    break
            if fcs_path is None:
                QMessageBox.information(
                    dlg,
                    "Détecter colonnes",
                    "Aucun fichier FCS trouvé.\nSélectionnez d'abord les dossiers dans l'onglet Import.",
                )
                return
            _, _, col_names = self._read_fcs_header_full(fcs_path)
            existing_srcs = set()
            for r in range(local_table.rowCount()):
                item = local_table.item(r, 0)
                if item:
                    existing_srcs.add(item.text().strip())
            new_cols = [c for c in col_names if c and c not in existing_srcs]
            for col in new_cols:
                short = re.sub(
                    r"\s+(KO|FITC|PE|APC|BV\d+|Cy\d+|PerCP|EF\d+|BUV\d+|"
                    r"BB\d+|R\d+|AF\d+|V\d+|Pacific[- ]Blue|AlexaFluor\d*"
                    r"|Pacific\s*Orange|BrilliantViolet\d*)\b.*",
                    "",
                    col,
                    flags=re.IGNORECASE,
                ).strip()
                row = local_table.rowCount()
                local_table.insertRow(row)
                local_table.setItem(row, 0, QTableWidgetItem(col))
                local_table.setItem(row, 1, QTableWidgetItem(short))
            _update_count()
            if new_cols:
                QMessageBox.information(
                    dlg,
                    "Colonnes détectées",
                    f"{len(new_cols)} colonne(s) ajoutée(s) depuis\n{fcs_path.name}",
                )

        def _add_row():
            row = local_table.rowCount()
            local_table.insertRow(row)
            local_table.setItem(row, 0, QTableWidgetItem(""))
            local_table.setItem(row, 1, QTableWidgetItem(""))
            local_table.editItem(local_table.item(row, 0))
            _update_count()

        def _del_rows():
            rows = sorted({idx.row() for idx in local_table.selectedIndexes()}, reverse=True)
            for r in rows:
                local_table.removeRow(r)
            _update_count()

        def _clear():
            reply = QMessageBox.question(
                dlg,
                "Effacer",
                "Supprimer toutes les règles de renommage ?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                local_table.setRowCount(0)
                _update_count()

        btn_detect.clicked.connect(_detect)
        btn_add.clicked.connect(_add_row)
        btn_del.clicked.connect(_del_rows)
        btn_clear.clicked.connect(_clear)

        tab_widget.addTab(tab_global, "  Homogénéisation (tous fichiers)  ")
        vbox.addWidget(tab_widget, 1)

        # ── Boutons bas ─────────────────────────────────────────────────
        btn_row_layout = QHBoxLayout()
        btn_cancel = QPushButton("  Annuler")
        btn_cancel.setObjectName("ghostBtn")
        btn_cancel.setMinimumHeight(38)
        btn_cancel.clicked.connect(dlg.reject)
        btn_row_layout.addWidget(btn_cancel)
        btn_row_layout.addStretch()

        btn_apply = QPushButton("  Appliquer")
        btn_apply.setObjectName("primaryBtn")
        btn_apply.setMinimumHeight(38)
        btn_apply.setMinimumWidth(140)
        ico_ok = _icon("prisma.check", "#11111b")
        if ico_ok:
            btn_apply.setIcon(ico_ok)
            btn_apply.setIconSize(QSize(16, 16))
        btn_row_layout.addWidget(btn_apply)
        vbox.addLayout(btn_row_layout)

        def _apply():
            # Sauvegarder l'état courant du tab per-file
            if _current_per_file[0]:
                _save_per_file(_current_per_file[0])

            # Sauvegarder règles globales dans self.rename_table
            self.rename_table.setRowCount(0)
            for r in range(local_table.rowCount()):
                src_item = local_table.item(r, 0)
                dst_item = local_table.item(r, 1)
                src = src_item.text().strip() if src_item else ""
                dst = dst_item.text().strip() if dst_item else ""
                if src:
                    row = self.rename_table.rowCount()
                    self.rename_table.insertRow(row)
                    self.rename_table.setItem(row, 0, QTableWidgetItem(src))
                    self.rename_table.setItem(row, 1, QTableWidgetItem(dst))

            # Stocker les règles per-file dans l'objet pour usage pipeline
            self._per_file_rename_rules = _per_file_rules

            # Mettre à jour le badge
            active_global = sum(
                1
                for r in range(self.rename_table.rowCount())
                if (self.rename_table.item(r, 0) and self.rename_table.item(r, 0).text().strip())
                and (self.rename_table.item(r, 1) and self.rename_table.item(r, 1).text().strip())
                and self.rename_table.item(r, 0).text().strip()
                != self.rename_table.item(r, 1).text().strip()
            )
            active_per = sum(len(v) for v in _per_file_rules.values())
            active = active_global + active_per
            if active:
                self.lbl_rename_summary.setText(
                    f"Renommage colonnes : {active_global} règle(s) globale(s), "
                    f"{active_per} règle(s) par fichier. "
                    f"Cliquez sur «Renommer colonnes» pour modifier."
                )
                self.lbl_rename_summary.setObjectName("summaryLabelActive")
            else:
                self.lbl_rename_summary.setText("Renommage colonnes : aucune règle configurée.")
                self.lbl_rename_summary.setObjectName("summaryLabel")
            self.lbl_rename_summary.style().unpolish(self.lbl_rename_summary)
            self.lbl_rename_summary.style().polish(self.lbl_rename_summary)
            dlg.accept()

        btn_apply.clicked.connect(_apply)
        dlg.exec_()

    @staticmethod
    def _read_fcs_header(fcs_path: Path) -> tuple:
        """
        Lit uniquement le header FCS pour obtenir le nombre d'événements
        et le nombre de paramètres. Ne charge pas les données en mémoire.

        Returns:
            (n_events, n_markers) ou ("?", "?") en cas d'erreur.
        """
        n_ev, n_ch, _ = FlowSomAnalyzerPro._read_fcs_header_full(fcs_path)
        return n_ev, n_ch

    @staticmethod
    def _read_fcs_header_full(fcs_path: Path) -> tuple:
        """
        Lit le header FCS et extrait le nombre d'événements, le nombre de
        canaux et la liste des noms de canaux ($PnS en priorité, puis $PnN).

        flowio stocke les clés du TEXT segment en minuscules — on normalise
        en cherchant toutes les variantes ($PnS, $pns, pns, pnS, etc.).

        Returns:
            (n_events, n_channels, channel_names: List[str])
            En cas d'erreur : ("?", "?", [])
        """
        # ── Parsing binaire du TEXT segment (lecture header seul, rapide) ──
        try:
            with open(fcs_path, "rb") as f:
                raw_hdr = f.read(58)
                text_start = int(raw_hdr[10:18].strip())
                text_end = int(raw_hdr[18:26].strip())
                f.seek(text_start)
                text_raw = f.read(text_end - text_start + 1).decode("latin-1", errors="replace")

            delimiter = text_raw[0] if text_raw else "/"
            parts = text_raw[1:].split(delimiter)
            meta_upper: Dict[str, str] = {}
            for i in range(0, len(parts) - 1, 2):
                k = parts[i].strip().upper()
                v = parts[i + 1].strip() if i + 1 < len(parts) else ""
                meta_upper[k] = v
                if k.startswith("$"):
                    meta_upper[k[1:]] = v

            n_events_str = meta_upper.get("$TOT", meta_upper.get("TOT", ""))
            n_events = int(n_events_str) if n_events_str.isdigit() else 0
            n_par_str = meta_upper.get("$PAR", meta_upper.get("PAR", ""))
            n_par = int(n_par_str) if n_par_str.isdigit() else 0

            names = []
            for i in range(1, n_par + 1):
                name = ""
                for key in (f"$P{i}S", f"P{i}S", f"$P{i}N", f"P{i}N"):
                    val = meta_upper.get(key, "").strip()
                    if val:
                        name = val
                        break
                names.append(name if name else f"Channel_{i}")

            if n_par > 0:
                return n_events if n_events > 0 else "?", n_par, names
        except Exception:
            pass

        # ── Fallback flowio (lit le fichier complet — plus lent) ─────
        try:
            import flowio

            fcs = flowio.FlowData(str(fcs_path))
            n_events = int(fcs.event_count)
            n_ch = int(fcs.channel_count)
            text_lower = {k.lower(): v for k, v in fcs.text.items()}
            names = []
            for i in range(1, n_ch + 1):
                name = ""
                for key in (f"$p{i}s", f"p{i}s", f"$p{i}n", f"p{i}n"):
                    val = str(text_lower.get(key, "")).strip()
                    if val:
                        name = val
                        break
                names.append(name if name else f"Channel_{i}")
            return n_events, n_ch, names
        except Exception:
            return "?", "?", []

    # ==================================================================
    # Renommage colonnes FCS
    # ==================================================================

    def _add_rename_row(self) -> None:
        """Ajoute une ligne vide dans la table de renommage."""
        row = self.rename_table.rowCount()
        self.rename_table.insertRow(row)
        self.rename_table.setItem(row, 0, QTableWidgetItem(""))
        self.rename_table.setItem(row, 1, QTableWidgetItem(""))
        self.rename_table.editItem(self.rename_table.item(row, 0))

    def _remove_rename_row(self) -> None:
        """Supprime la ligne sélectionnée dans la table de renommage."""
        rows = sorted({idx.row() for idx in self.rename_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.rename_table.removeRow(r)

    def _get_column_rename_map(self) -> Dict[str, str]:
        """Retourne le mapping {col_brute: col_cible} depuis la table de renommage."""
        rename: Dict[str, str] = {}
        for r in range(self.rename_table.rowCount()):
            src_item = self.rename_table.item(r, 0)
            dst_item = self.rename_table.item(r, 1)
            if src_item and dst_item:
                src = src_item.text().strip()
                dst = dst_item.text().strip()
                if src and dst and src != dst:
                    rename[src] = dst
        return rename

    # ==================================================================
    # Exécution du pipeline
    # ==================================================================

    def _run_pipeline(self) -> None:
        if self._config is None:
            QMessageBox.warning(self, "Erreur", "Aucune configuration chargée.")
            return

        healthy = self.drop_healthy.path
        patho = self.drop_patho.path

        if not healthy:
            QMessageBox.warning(self, "Erreur", "Veuillez sélectionner le dossier NBM / Sain.")
            return
        if not Path(healthy).is_dir():
            QMessageBox.warning(self, "Erreur", f"Dossier NBM introuvable :\n{healthy}")
            return

        if self.chk_compare.isChecked():
            if not patho:
                QMessageBox.warning(
                    self, "Erreur", "Mode comparaison : sélectionnez le dossier Patho."
                )
                return
            if not Path(patho).is_dir():
                QMessageBox.warning(self, "Erreur", f"Dossier Patho introuvable :\n{patho}")
                return

        self._sync_ui_to_config()

        # Passer à l'étape Exécution
        self._navigate_to_step(3)
        self._sidebar.set_active(3)

        self.btn_run_step3.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_output.clear()

        self._log("═══════════════════════════════════════════════")
        self._log(f"Pipeline FlowSOM Pro — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Grille : {self._config.flowsom.xdim}×{self._config.flowsom.ydim}")
        self._log(f"Métaclusters : {self._config.flowsom.n_metaclusters}")
        self._log(f"Transformation : {self._config.transform.method}")
        self._log(f"GPU : {'Oui' if self._config.gpu.enabled else 'Non'}")
        if self._config.batch.enabled:
            self._log("Mode : Batch (traitement par lots)")
        self._log("═══════════════════════════════════════════════")

        if self._config.batch.enabled:
            from flowsom_pipeline_pro.gui.workers import BatchWorker

            self._worker = BatchWorker(self._config, parent=self)
            self._worker.log_message.connect(self._on_log_message)
            self._worker.progress.connect(self._on_progress)
            self._worker.file_started.connect(self._on_batch_file_started)
            self._worker.file_finished.connect(self._on_batch_file_finished)
            self._worker.finished.connect(self._on_batch_finished)
            self._worker.error.connect(self._on_pipeline_error)
            self._worker.start()
            # Démarre le drainage de la queue de logs depuis le thread principal
            self._worker._log_capture.start_drain()
            self.statusBar().showMessage(" Batch en cours d'exécution…")
        else:
            self._worker = PipelineWorker(self._config, parent=self)
            self._worker.log_message.connect(self._on_log_message)
            self._worker.progress.connect(self._on_progress)
            self._worker.gating_done.connect(self._on_gating_done)
            self._worker.prescreening_done.connect(self._on_prescreening_done)
            self._worker.finished.connect(self._on_pipeline_finished)
            self._worker.error.connect(self._on_pipeline_error)
            self._worker.start()
            # Démarre le drainage de la queue de logs depuis le thread principal
            self._worker._log_capture.start_drain()
            self.statusBar().showMessage(" Pipeline en cours d'exécution…")

    def _stop_pipeline(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirmation",
                "Voulez-vous interrompre le pipeline ?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._worker.terminate()
                self._worker.wait(3000)
                self._log(" Pipeline interrompu par l'utilisateur")
                self.btn_run_step3.setEnabled(True)
                self.btn_stop.setEnabled(False)
                self.statusBar().showMessage("Pipeline interrompu")
                self._sidebar.set_error(3)

    def _auto_load_patho_fcs(self, result: Any) -> None:
        """Charge automatiquement le FCS pathologique exporté dans le Viewer FCS."""
        try:
            patho_fcs = self._resolve_patho_fcs_path(result)
            if patho_fcs and Path(patho_fcs).exists():
                self._patho_fcs_path = patho_fcs
                if hasattr(self, "btn_reload_patho_fcs"):
                    self.btn_reload_patho_fcs.setEnabled(True)
                    self.btn_reload_patho_fcs.setToolTip(f"Recharger : {Path(patho_fcs).name}")
                self._log(f"[Viewer FCS] Chargement automatique : {Path(patho_fcs).name}")
                self._load_fcs_for_visualization(file_path=patho_fcs)
        except Exception as e:
            self._log(f"[Viewer FCS] Auto-chargement ignoré : {e}")

    def _resolve_patho_fcs_path(self, result: Any = None) -> str:
        """Résout le chemin du FCS pathologique MRD avec fallback robuste."""
        source_result = result if result is not None else self._result
        output_files = getattr(source_result, "output_files", {}) or {}

        # 1) Chemin explicite exporté par le pipeline.
        patho_fcs = output_files.get("fcs_patho_mrd", "")
        if patho_fcs and Path(patho_fcs).exists():
            return str(patho_fcs)

        # 2) Dernier chemin déjà détecté par le viewer.
        if getattr(self, "_patho_fcs_path", None) and Path(self._patho_fcs_path).exists():
            return str(self._patho_fcs_path)

        # 3) Fallback scan output/fcs/patho_mrd_*.fcs
        out_dir = getattr(self.drop_output, "path", "") or ""
        if out_dir:
            fcs_dir = Path(out_dir) / "fcs"
            candidates = sorted(fcs_dir.glob("patho_mrd_*.fcs"), reverse=True)
            if candidates:
                return str(candidates[0])

        return ""

    def _resolve_full_fcs_path(self, result: Any = None) -> str:
        """Résout le chemin du FCS complet exporté par le pipeline."""
        source_result = result if result is not None else self._result
        output_files = getattr(source_result, "output_files", {}) or {}
        full_fcs = output_files.get("fcs_kaluza") or output_files.get("fcs") or ""
        if full_fcs and Path(full_fcs).exists():
            return str(full_fcs)
        return ""

    def _reload_patho_fcs(self) -> None:
        """Recharge le FCS de sortie patho auto-détecté après pipeline."""
        if self._patho_fcs_path and Path(self._patho_fcs_path).exists():
            self._load_fcs_for_visualization(file_path=self._patho_fcs_path)
        else:
            QMessageBox.information(
                self, "Viewer FCS", "Aucun FCS de sortie disponible.\nLancez d'abord le pipeline."
            )

    def _show_pending_prescreening(self) -> None:
        """Affiche le popup pré-screening différé, après que HomeTab soit rendu."""
        ps = getattr(self, "_pending_prescreening", None)
        if ps is None:
            return
        self._pending_prescreening = None
        from PyQt5.QtCore import QTimer

        def _show():
            msg = QMessageBox(self)
            msg.setWindowTitle(ps["title"])
            msg.setIcon(ps["icon"])
            msg.setText(ps["text"])
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setTextFormat(Qt.RichText)
            msg.setWindowModality(Qt.NonModal)
            msg.show()

        QTimer.singleShot(400, _show)

    # ── Slots worker ───────────────────────────────────────────────────

    def _on_log_message(self, msg: str) -> None:
        self.log_output.append(msg)
        sb = self.log_output.verticalScrollBar()
        sb.setValue(sb.maximum())
        # Mise à jour de l'indicateur d'étape
        if "Étape" in msg:
            self.lbl_pipeline_step.setText(msg.strip())

    def _on_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    def _on_gating_done(self, info: dict) -> None:
        """Affiche un résumé de pré-gating non bloquant après l'étape de gating."""
        n_kept = info.get("n_kept", 0)
        n_total = info.get("n_total", 0)
        pct = info.get("pct_kept", 0.0)
        n_gates = info.get("n_gates", 0)
        fallbacks = info.get("fallbacks", [])

        lines = [
            f"<b>Pré-gating terminé</b> ({n_gates} gate(s))",
            f"Cellules conservées : <b>{n_kept:,} / {n_total:,}</b> ({pct:.1f} %)",
        ]
        if fallbacks:
            names = ", ".join(fallbacks[:3])
            if len(fallbacks) > 3:
                names += f" +{len(fallbacks) - 3}"
            lines.append(f"<span style='color:#FF3D6E;'>⚠ Fallbacks : {names}</span>")

        msg = QMessageBox(self)
        msg.setWindowTitle("Validation pré-gating")
        msg.setIcon(QMessageBox.Information)
        msg.setText("<br>".join(lines))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setWindowModality(Qt.NonModal)
        msg.show()

        # Log également dans le panneau de logs
        self._log(
            f"[GATING] {n_kept:,}/{n_total:,} cellules ({pct:.1f} %)"
            + (f" — fallbacks: {', '.join(fallbacks)}" if fallbacks else "")
        )

    def _on_prescreening_done(self, info: dict) -> None:
        """Affiche un popup de pré-screening CD34+/CD45dim en fin de pipeline."""
        alert_level = info.get("alert_level", "none")
        ratio_pct = info.get("ratio_pct", 0.0)
        n_cd34_pos = info.get("n_cd34_pos", 0)
        n_cd34_neg = info.get("n_cd34_neg", 0)
        n_cd45dim = info.get("n_cd45dim", 0)
        gmm_pct = info.get("gmm_ratio_pct", 0.0)
        kde_pct = info.get("kde_ratio_pct", 0.0)
        method_used = info.get("method_used", "KDE")
        laip = info.get("laip_tracking_recommended", False)
        interpretation = info.get("interpretation_warning", "")

        # Couleur et icône selon le niveau d'alerte
        if alert_level == "high":
            icon = QMessageBox.Warning
            color_ratio = "#FF3D6E"
            title = "⚠ Pré-screening CD34+/CD45dim — ALERTE"
        elif alert_level == "moderate":
            icon = QMessageBox.Warning
            color_ratio = "#F59E0B"
            title = "⚠ Pré-screening CD34+/CD45dim — Rapport élevé"
        else:
            icon = QMessageBox.Information
            color_ratio = "#86efac"
            title = "✓ Pré-screening CD34+/CD45dim — Normal"

        lines = [
            f"<b>Pré-screening CD34+ / CD45dim</b><br>",
            f"<b>Méthode de référence :</b> {method_used}<br>",
            f"<table style='border-collapse:collapse; width:100%;'>",
            f"<tr><td style='padding:3px 8px;'><b>Cellules CD45dim :</b></td>"
            f"<td style='padding:3px 8px;'>{n_cd45dim:,}</td></tr>",
            f"<tr><td style='padding:3px 8px;'><b>CD34+ dans CD45dim :</b></td>"
            f"<td style='padding:3px 8px;'>{n_cd34_pos:,}</td></tr>",
            f"<tr><td style='padding:3px 8px;'><b>CD34− dans CD45dim :</b></td>"
            f"<td style='padding:3px 8px;'>{n_cd34_neg:,}</td></tr>",
            f"<tr><td style='padding:3px 8px;'><b>Ratio CD34+/CD45dim :</b></td>"
            f"<td style='padding:3px 8px; color:{color_ratio};'>"
            f"<b>{ratio_pct:.1f}%</b></td></tr>",
            f"<tr><td style='padding:3px 8px; color:#94a3b8;'>GMM :</td>"
            f"<td style='padding:3px 8px; color:#94a3b8;'>{gmm_pct:.1f}%</td></tr>",
            f"<tr><td style='padding:3px 8px; color:#94a3b8;'>KDE :</td>"
            f"<td style='padding:3px 8px; color:#94a3b8;'>{kde_pct:.1f}%</td></tr>",
            f"</table><br>",
        ]
        if interpretation:
            lines.append(f"<i style='color:{color_ratio};'>{interpretation}</i><br>")
        if laip:
            lines.append(
                "<br><b style='color:#F59E0B;'>→ LAIP Tracking classique recommandé<br>"
                "→ Rapport CD34+/CD45dim élevé — attention pour l'interprétation de la MRD</b>"
            )

        # Stocker pour affichage différé (après chargement HomeTab)
        self._pending_prescreening = {
            "title": title,
            "icon": icon,
            "text": "".join(lines),
            "log": (
                f"[PRESCREENING] CD34+/CD45dim={ratio_pct:.1f}% "
                f"(GMM={gmm_pct:.1f}%, KDE={kde_pct:.1f}%) "
                f"— alerte={alert_level}" + (" — LAIP recommandé" if laip else "")
            ),
        }

        # Log immédiat dans le panneau
        self._log(
            f"[PRESCREENING] CD34+/CD45dim={ratio_pct:.1f}% "
            f"(GMM={gmm_pct:.1f}%, KDE={kde_pct:.1f}%) "
            f"— alerte={alert_level}" + (" — LAIP recommandé" if laip else "")
        )

    def _on_pipeline_finished(self, result: Any) -> None:
        self.btn_run_step3.setEnabled(True)
        self.btn_stop.setEnabled(False)
        # Arrête le drainage de la queue de logs et vide les derniers messages
        if (
            self._worker is not None
            and hasattr(self._worker, "_log_capture")
            and self._worker._log_capture is not None
        ):
            self._worker._log_capture.stop_drain()
        self._result = result

        if result is not None and result.success:
            self.progress_bar.setValue(100)
            self._sidebar.set_done(3)
            elapsed = f"{result.elapsed_seconds:.1f}s" if hasattr(result, "elapsed_seconds") else ""
            self.statusBar().showMessage(
                f" Terminé — {result.n_cells:,} cellules, "
                f"{result.n_metaclusters} métaclusters  {elapsed}"
            )
            self.lbl_pipeline_step.setText(
                f"Pipeline terminé — {result.n_cells:,} cellules en {elapsed}"
            )
            self._populate_results(result)
            self._populate_cluster_list(result)
            self._populate_pregate_tab(result)
            self._load_output_plots(result)
            method_used = self.combo_mrd_method.currentText()
            self._home_tab.load_result(result, method_used)
            # Afficher/masquer la barre ELN selon l'état du checkbox
            eln_active = self.chk_blast_filter.isChecked()
            self._home_tab.show_eln_html_bar(eln_active)
            # Connecter le signal de curation experte → patch HTML temps réel
            try:
                self._home_tab.curation_changed.disconnect()
            except Exception:
                pass
            self._home_tab.curation_changed.connect(self._on_curation_changed)
            try:
                self._home_tab.verification_commit_requested.disconnect()
            except Exception:
                pass
            self._home_tab.verification_commit_requested.connect(
                self._on_verification_commit_requested
            )
            # Aller automatiquement aux résultats
            self._navigate_to_step(4)
            self._sidebar.set_done(4)
            self.tabs.setCurrentIndex(0)
            # Charger automatiquement le FCS patho dans le Viewer FCS
            self._auto_load_patho_fcs(result)
            # Afficher le popup pré-screening APRÈS le chargement de HomeTab
            self._show_pending_prescreening()
        else:
            self._sidebar.set_error(3)
            self.statusBar().showMessage(" Pipeline terminé avec des erreurs")
            self._log("═══ Pipeline terminé avec des erreurs — vérifiez les logs ═══")

    def _on_pipeline_error(self, msg: str) -> None:
        if (
            self._worker is not None
            and hasattr(self._worker, "_log_capture")
            and self._worker._log_capture is not None
        ):
            self._worker._log_capture.stop_drain()
        self._sidebar.set_error(3)
        self.statusBar().showMessage(f" Erreur : {msg[:80]}")

    def _on_batch_file_started(self, current: int, total: int, filename: str) -> None:
        if total > 0:
            self.progress_bar.setValue(max(2, int(current / total * 95)))
        self.statusBar().showMessage(f" Batch [{current}/{total}] : {filename}…")
        self._log(f"══ Batch [{current + 1}/{total}] : {filename} ══")

    def _on_batch_file_finished(self, stem: str, success: bool) -> None:
        self._log(f"  → {stem} : {'OK' if success else 'ERREUR'}")

    def _on_batch_finished(self, summary: Any) -> None:
        self.btn_run_step3.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        if (
            self._worker is not None
            and hasattr(self._worker, "_log_capture")
            and self._worker._log_capture is not None
        ):
            self._worker._log_capture.stop_drain()

        if summary is None:
            self._sidebar.set_error(3)
            self.statusBar().showMessage(" Batch terminé avec des erreurs")
            return

        results = summary.get("results", [])
        excel = summary.get("excel")
        n_ok = sum(1 for _, r in results if r is not None and r.success)
        n_total = len(results)
        self._sidebar.set_done(3)
        self.statusBar().showMessage(f" Batch terminé — {n_ok}/{n_total} fichier(s)")
        self._log(f"BATCH TERMINÉ : {n_ok}/{n_total} fichier(s) réussis")
        if excel:
            self._log(f"Excel de synthèse : {excel}")

        for stem, result in reversed(results):
            if result is not None and result.success:
                self._populate_results(result)
                self._populate_cluster_list(result)
                self._result = result
                break

        if excel and Path(excel).exists():
            reply = QMessageBox.question(
                self,
                "Batch terminé",
                f"{n_ok}/{n_total} fichier(s).\n\nOuvrir l'Excel de synthèse ?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                os.startfile(excel)

    # ==================================================================
    # LOGIQUE : Affichage des résultats (identique à v2)
    # ==================================================================

    def _populate_results(self, result: Any) -> None:
        try:
            import platform
            import psutil

            lines = []

            # ── En-tête pipeline ──
            lines.append("═" * 64)
            lines.append("  RÉSUMÉ PIPELINE — FlowSOM MRD Analyzer Pro")
            lines.append("═" * 64)

            ts = getattr(result, "timestamp", "")
            if ts:
                lines.append(f"  Timestamp     : {ts}")

            elapsed = getattr(result, "elapsed_seconds", 0.0)
            if elapsed:
                m, s = divmod(int(elapsed), 60)
                lines.append(f"  Durée run     : {m}m {s:02d}s ({elapsed:.1f}s)")

            # ── Comptages cellulaires ──
            lines.append("")
            lines.append("  CELLULES")
            lines.append("  " + "─" * 40)
            n_cells = getattr(result, "n_cells", 0)
            lines.append(f"  Total analysées          : {n_cells:,}")

            df = getattr(result, "data", None)
            if df is not None and "condition" in df.columns:
                import numpy as np

                mask_patho = (
                    df["condition"].str.lower().str.contains("patho|pathologique", na=False)
                )
                n_patho = int(mask_patho.sum())
                n_sain = int((~mask_patho).sum())
                pct_patho = n_patho / n_cells * 100 if n_cells > 0 else 0
                pct_sain = n_sain / n_cells * 100 if n_cells > 0 else 0
                lines.append(f"  Cellules pathologiques   : {n_patho:,}  ({pct_patho:.1f}%)")
                lines.append(f"  Cellules saines (NBM)    : {n_sain:,}  ({pct_sain:.1f}%)")

                # x3 : par fichier si file_origin présent
                if "file_origin" in df.columns:
                    lines.append("")
                    lines.append("  Détail par fichier :")
                    for fname, grp in df.groupby("file_origin"):
                        m_p = (
                            grp["condition"]
                            .str.lower()
                            .str.contains("patho|pathologique", na=False)
                        )
                        np_ = int(m_p.sum())
                        ns_ = int((~m_p).sum())
                        lines.append(f"    {str(fname)[:40]:40s}  patho={np_:,}  sain={ns_:,}")

            # ── MRD ──
            mrd = getattr(result, "mrd_result", None)
            if mrd is not None:
                lines.append("")
                lines.append("  MRD")
                lines.append("  " + "─" * 40)
                for attr in ("mrd_percent_jf", "mrd_percent_flo", "mrd_percent_eln"):
                    val = getattr(mrd, attr, None)
                    method = attr.replace("mrd_percent_", "").upper()
                    if val is not None:
                        lines.append(f"  MRD {method:5s}               : {val:.4f}%")

            # ── Métaclusters ──
            n_mc = getattr(result, "n_metaclusters", None)
            if n_mc:
                lines.append("")
                lines.append(f"  Métaclusters             : {n_mc}")

            # ── CPU / RAM ──
            lines.append("")
            lines.append("  RESSOURCES SYSTÈME")
            lines.append("  " + "─" * 40)
            try:
                cpu_pct = psutil.cpu_percent(interval=0)
                ram = psutil.virtual_memory()
                lines.append(f"  CPU utilisation          : {cpu_pct:.1f}%")
                lines.append(
                    f"  RAM utilisée             : {ram.used / 1e9:.1f} Go / {ram.total / 1e9:.1f} Go"
                )
                lines.append(f"  Plateforme               : {platform.processor()[:60]}")
            except Exception:
                pass

            # ── GPU si disponible ──
            try:
                import subprocess

                gpu_out = (
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        timeout=3,
                        stderr=subprocess.DEVNULL,
                    )
                    .decode(errors="ignore")
                    .strip()
                )
                if gpu_out:
                    lines.append("")
                    lines.append("  GPU (NVIDIA)")
                    for gpu_line in gpu_out.split("\n"):
                        parts = [p.strip() for p in gpu_line.split(",")]
                        if len(parts) >= 4:
                            name, mem_used, mem_total, util = parts[:4]
                            lines.append(
                                f"  {name[:36]:36s}  {mem_used} / {mem_total} Mo  util={util}%"
                            )
            except Exception:
                pass

            # ── Avertissements ──
            warns = getattr(result, "warnings", [])
            if warns:
                lines.append("")
                lines.append("  AVERTISSEMENTS")
                lines.append("  " + "─" * 40)
                for w in warns[:10]:
                    lines.append(f"  ⚠ {str(w)[:80]}")

            lines.append("")
            lines.append("═" * 64)
            self.txt_summary.setPlainText("\n".join(lines))
            self.txt_summary.setMaximumHeight(16777215)  # libère la hauteur
        except Exception as exc:
            try:
                self.txt_summary.setPlainText(result.summary())
            except Exception:
                self.txt_summary.setPlainText(f"Cellules : {result.n_cells:,}")

        try:
            import pandas as pd
            import numpy as np

            df = result.data
            if df is None:
                return

            has_cluster = "FlowSOM_cluster" in df.columns
            has_mc = "FlowSOM_metacluster" in df.columns
            has_cond = "condition" in df.columns

            if not has_cluster and not has_mc:
                return

            group_col = "FlowSOM_cluster" if has_cluster else "FlowSOM_metacluster"
            total = len(df)
            counts = df[group_col].value_counts().sort_index()

            self.results_table.setRowCount(len(counts))
            for i, (cl_id, count) in enumerate(counts.items()):
                self.results_table.setItem(i, 0, QTableWidgetItem(str(int(cl_id))))
                if has_mc and has_cluster:
                    mc_mode = df[df[group_col] == cl_id]["FlowSOM_metacluster"].mode()
                    mc_val = str(int(mc_mode.iloc[0])) if len(mc_mode) > 0 else "?"
                else:
                    mc_val = str(int(cl_id))
                self.results_table.setItem(i, 1, QTableWidgetItem(mc_val))
                self.results_table.setItem(i, 2, QTableWidgetItem(f"{count:,}"))
                pct = count / total * 100 if total > 0 else 0
                self.results_table.setItem(i, 3, QTableWidgetItem(f"{pct:.1f}%"))
                if has_cond:
                    sub = df[df[group_col] == cl_id]
                    n_patho = (
                        sub["condition"].str.lower().str.contains("patho|pathologique", na=False)
                    ).sum()
                    pct_patho = n_patho / count * 100 if count > 0 else 0
                    self.results_table.setItem(i, 4, QTableWidgetItem(f"{pct_patho:.1f}%"))
                else:
                    self.results_table.setItem(i, 4, QTableWidgetItem("N/A"))
        except Exception as e:
            self._log(f"Erreur tableau résultats : {e}")

    def _export_cluster_txt(self) -> None:
        if self._result is None or not self._result.success:
            QMessageBox.information(self, "Info", "Aucun résultat disponible.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter clusters", "clusters_stats.txt", "Text files (*.txt)"
        )
        if not path:
            return
        try:
            import numpy as np

            df = self._result.data
            lines = [
                "=" * 70,
                "STATISTIQUES DES CLUSTERS — FlowSOM Analyzer Pro",
                "=" * 70,
                f"  Analyse  : {self._result.timestamp}",
                f"  Cellules : {self._result.n_cells:,}",
                "",
            ]
            group_col = (
                "FlowSOM_cluster" if "FlowSOM_cluster" in df.columns else "FlowSOM_metacluster"
            )
            has_mc = "FlowSOM_metacluster" in df.columns and group_col == "FlowSOM_cluster"
            has_cond = "condition" in df.columns
            total = len(df)
            lines.append(
                f"{'Cluster':>10}  {'Métacluster':>12}  {'Cellules':>10}  {'% total':>8}  {'% patho':>8}"
            )
            lines.append("-" * 70)
            for cl_id, count in df[group_col].value_counts().sort_index().items():
                pct = count / total * 100 if total > 0 else 0
                mc_val = "?"
                if has_mc:
                    mc_mode = df[df[group_col] == cl_id]["FlowSOM_metacluster"].mode()
                    mc_val = str(int(mc_mode.iloc[0])) if len(mc_mode) > 0 else "?"
                patho_str = "N/A"
                if has_cond:
                    sub = df[df[group_col] == cl_id]
                    n_p = (
                        sub["condition"].str.lower().str.contains("patho|pathologique", na=False)
                    ).sum()
                    patho_str = f"{n_p / count * 100:.1f}%"
                lines.append(
                    f"{int(cl_id):>10}  {mc_val:>12}  {count:>10,}  {pct:>7.1f}%  {patho_str:>8}"
                )
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self._log(f"Clusters exportés : {path}")
            QMessageBox.information(self, "Export réussi", f"Fichier exporté :\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Erreur export", str(e))

    def _open_combined_html(self) -> None:
        if self._combined_html_path and Path(self._combined_html_path).exists():
            webbrowser.open(str(Path(self._combined_html_path).resolve()))
        else:
            QMessageBox.information(self, "Info", "Vue combinée non disponible.")

    def _populate_cluster_list(self, result: Any) -> None:
        self.cluster_list.clear()
        self.marker_list.clear()
        self._cluster_mfi = None
        self._all_markers = []

        if result is None or result.data is None:
            return

        df = result.data
        try:
            import numpy as np

            if "FlowSOM_cluster" not in df.columns:
                return

            _meta_cols = {
                "FlowSOM_cluster",
                "FlowSOM_metacluster",
                "condition",
                "file_origin",
                "xGrid",
                "yGrid",
                "xNodes",
                "yNodes",
                "size",
                "Condition_Num",
                "Condition",
                "Timepoint",
                "Timepoint_Num",
            }
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            marker_cols = [c for c in numeric_cols if c not in _meta_cols]
            self._all_markers = marker_cols

            for m in marker_cols:
                self.marker_list.addItem(QListWidgetItem(m))
            self.marker_list.selectAll()

            mc_col = "FlowSOM_metacluster" if "FlowSOM_metacluster" in df.columns else None
            mfi_df = df.groupby("FlowSOM_cluster")[marker_cols].mean()
            mfi_df["n_cells"] = df.groupby("FlowSOM_cluster").size()
            if mc_col:
                mfi_df["metacluster"] = (
                    df.groupby("FlowSOM_cluster")[mc_col]
                    .agg(lambda x: int(x.mode().iloc[0]) if len(x) > 0 else -1)
                    .astype(int)
                )

            self._cluster_mfi = mfi_df
            total_cells = len(df)
            for cl_id, row_data in mfi_df.iterrows():
                mc = int(row_data["metacluster"]) if "metacluster" in row_data.index else -1
                n = int(row_data["n_cells"])
                pct = n / total_cells * 100 if total_cells > 0 else 0
                label_text = f"Cluster {cl_id}  (MC{mc})  — {n:,} cell. ({pct:.1f}%)"
                self.cluster_list.addItem(QListWidgetItem(label_text))

            self.cluster_list.update()
            if self.cluster_list.count() > 0:
                self.cluster_list.setCurrentRow(0)

        except Exception as e:
            import traceback

            self._log(f"Erreur peuplement clusters : {e}\n{traceback.format_exc()}")

    def _on_cluster_selected(self, row: int) -> None:
        if row >= 0 and self._cluster_mfi is not None:
            self._generate_spider_plot()

    def _generate_spider_plot(self) -> None:
        if self._spider_worker is not None and self._spider_worker.isRunning():
            self._spider_worker.terminate()
            self._spider_worker.wait(200)

        row = self.cluster_list.currentRow()
        if row < 0 or self._cluster_mfi is None:
            QMessageBox.information(self, "Info", "Sélectionnez un cluster.")
            return

        selected_markers = [
            self.marker_list.item(i).text()
            for i in range(self.marker_list.count())
            if self.marker_list.item(i).isSelected()
        ]
        if not selected_markers:
            QMessageBox.warning(self, "Avertissement", "Sélectionnez au moins un marqueur.")
            return
        if len(selected_markers) < 3:
            QMessageBox.warning(self, "Avertissement", "Sélectionnez au moins 3 marqueurs.")
            return

        cluster_ids = list(self._cluster_mfi.index)
        if row >= len(cluster_ids):
            return
        cl_id = cluster_ids[row]
        mfi_row = self._cluster_mfi.loc[cl_id]
        mc = int(mfi_row.get("metacluster", -1)) if "metacluster" in mfi_row else -1
        n = int(mfi_row.get("n_cells", 0))
        label = f"Cluster {cl_id}  (MC{mc})  — {n:,} cellules"

        self._spider_worker = SpiderPlotWorker(
            mfi_row=mfi_row[selected_markers],
            marker_names=selected_markers,
            cluster_label=label,
            parent=self,
        )
        self._spider_worker.figure_ready.connect(self._on_spider_ready)
        self._spider_worker.error.connect(lambda msg: self._log(f"Spider erreur : {msg}"))
        self._spider_worker.start()

    def _on_spider_ready(self, fig: Any) -> None:
        self.star_canvas.display_figure(fig)

    # ==================================================================
    # Plots output
    # ==================================================================

    _PLOT_FILENAME_MAP = [
        ("mfi_heatmap", "Heatmap MFI"),
        ("metacluster_distribution", "Distribution Métaclusters"),
        ("umap", "UMAP"),
        ("flowsom_star_chart", "Star Chart FlowSOM"),
        ("flowsom_som_grid", "Grille SOM statique"),
        ("mst_static", "MST Statique"),
        ("sankey_global", "Sankey Gating"),
        ("mst_interactive", "MST Interactif"),
        ("som_grid", "Grille SOM interactive"),
        ("metacluster_radar", "Radar Métaclusters"),
        ("patho_pct_per_cluster", "% Cellules Patho / Cluster"),
        ("cells_pct_per_cluster", "% Cellules / Cluster"),
        ("patho_pct_per_som_node", "% Patho / Nœud SOM"),
        ("cells_pct_per_som_node", "% Cellules / Nœud SOM"),
        ("som_node_combined", "Vue Combinée Nœuds SOM"),
    ]

    def _find_output_dir(self, result: Any) -> Optional[Path]:
        if result is None or not result.output_files:
            return None
        for v in result.output_files.values():
            if v and Path(v).exists():
                p = Path(v)
                candidate = p.parent.parent
                if (candidate / "plots").is_dir():
                    return candidate
                if p.parent.is_dir():
                    return p.parent
        return None

    def _load_output_plots(self, result: Any) -> None:
        self._output_plot_paths = {}
        self._gate_plot_paths = {}
        self._combined_html_path = None

        output_dir = self._find_output_dir(result)
        if output_dir is None:
            self._refresh_current_plot()
            return

        self._output_dir = output_dir
        plots_dir = output_dir / "plots"
        if not plots_dir.is_dir():
            self._refresh_current_plot()
            return

        all_files = list(plots_dir.rglob("*.png")) + list(plots_dir.rglob("*.html"))
        for file_path in all_files:
            fname = file_path.name.lower()
            if "per_file" in str(file_path).lower():
                continue
            if "gating" in str(file_path.parent).lower():
                for label, keys in self._gate_plot_keys.items():
                    if not label.startswith("Prégating"):
                        continue
                    if any(
                        k.replace("fig_", "").replace("_", "") in fname.replace("_", "")
                        for k in keys
                    ):
                        if label not in self._gate_plot_paths:
                            self._gate_plot_paths[label] = str(file_path)
                        break
                continue
            for fragment, label in self._PLOT_FILENAME_MAP:
                if fragment in fname:
                    if label not in self._output_plot_paths:
                        self._output_plot_paths[label] = str(file_path)
                    if "som_node_combined" in fname:
                        self._combined_html_path = str(file_path)
                    break

        if hasattr(self, "btn_open_combined"):
            self.btn_open_combined.setEnabled(bool(self._combined_html_path))

        combined_png = self._output_plot_paths.get("Vue Combinée Nœuds SOM")
        if combined_png and Path(combined_png).exists() and combined_png.lower().endswith(".png"):
            try:
                import matplotlib.image as mpimg

                self._combined_canvas.fig.clear()
                ax = self._combined_canvas.fig.add_subplot(111)
                ax.imshow(mpimg.imread(combined_png))
                ax.axis("off")
                self._combined_canvas.fig.patch.set_facecolor(COLORS["surface"])
                self._combined_canvas.fig.tight_layout(pad=0.3)
                self._combined_canvas.draw()
            except Exception as e:
                self._log(f"Avertissement vue combinée PNG : {e}")

        _repr_mapping = {
            k: k
            for k in [
                "Heatmap MFI",
                "Distribution Métaclusters",
                "UMAP",
                "Star Chart FlowSOM",
                "Grille SOM statique",
                "MST Statique",
                "Sankey Gating",
                "Radar Métaclusters",
                "% Cellules Patho / Cluster",
                "% Cellules / Cluster",
                "% Patho / Nœud SOM",
                "% Cellules / Nœud SOM",
                "Vue Combinée Nœuds SOM",
            ]
        }
        for repr_label, plot_label in _repr_mapping.items():
            if plot_label in self._output_plot_paths and repr_label not in self._gate_plot_paths:
                self._gate_plot_paths[repr_label] = self._output_plot_paths[plot_label]

        self._on_gate_plot_changed(self.combo_gate_plot.currentIndex())
        self._refresh_current_plot()

    def _on_plot_selection_changed(self, index: int) -> None:
        self._refresh_current_plot()

    def _refresh_current_plot(self) -> None:
        label = self.combo_plot.currentText()
        if not self._output_plot_paths:
            self._show_placeholder("Lancez le pipeline pour générer les figures")
            return
        path = self._output_plot_paths.get(label)
        if not path or not Path(path).exists():
            self._show_placeholder(f"'{label}' non disponible")
            return
        if path.lower().endswith(".html"):
            self._show_html_plot(path)
        else:
            self._show_png_plot(path)

    def _show_png_plot(self, path: str) -> None:
        self._viz_stack.setCurrentIndex(0)
        try:
            import matplotlib.image as mpimg

            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot(111)
            ax.imshow(mpimg.imread(path))
            ax.axis("off")
            self.canvas.fig.patch.set_facecolor(COLORS["surface"])
            self.canvas.fig.tight_layout(pad=0.5)
            self.canvas.draw()
        except Exception as e:
            self._show_placeholder(f"Erreur affichage : {e}")

    def _show_html_plot(self, path: str) -> None:
        webbrowser.open(str(Path(path).resolve()))
        self._viz_stack.setCurrentIndex(1)

    def _show_placeholder(self, text: str) -> None:
        self._viz_stack.setCurrentIndex(0)
        self.canvas.clear_and_reset()
        self.canvas.axes.text(
            0.5,
            0.5,
            text,
            transform=self.canvas.axes.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            color=COLORS["paper"],
            style="italic",
            wrap=True,
        )
        self.canvas.axes.set_xlim(0, 1)
        self.canvas.axes.set_ylim(0, 1)
        self.canvas.axes.axis("off")
        self.canvas.draw()

    def _open_current_plot_browser(self) -> None:
        label = self.combo_plot.currentText()
        path = self._output_plot_paths.get(label)
        if path and Path(path).exists():
            webbrowser.open(str(Path(path).resolve()))
        else:
            QMessageBox.information(self, "Info", f"Figure '{label}' non disponible.")

    # ── Prégating ──────────────────────────────────────────────────────

    def _populate_pregate_tab(self, result: Any) -> None:
        if result is None:
            return
        events = result.gating_report or []
        self.gate_table.setRowCount(len(events))
        for i, ev in enumerate(events):
            gate_name = ev.get("gate_name", ev.get("gate", "?"))
            file_name = ev.get("file", "COMBINED")
            n_before = ev.get("n_before", ev.get("n_total", 0))
            n_after = ev.get("n_after", ev.get("n_kept", 0))
            pct = ev.get("pct_kept", (n_after / n_before * 100) if n_before > 0 else 0)
            mode = ev.get("method", ev.get("mode", "auto"))
            self.gate_table.setItem(i, 0, QTableWidgetItem(str(gate_name)))
            self.gate_table.setItem(i, 1, QTableWidgetItem(str(file_name)))
            self.gate_table.setItem(i, 2, QTableWidgetItem(f"{int(n_before):,}"))
            self.gate_table.setItem(i, 3, QTableWidgetItem(f"{int(n_after):,}"))
            self.gate_table.setItem(i, 4, QTableWidgetItem(f"{float(pct):.1f}%"))
            self.gate_table.setItem(i, 5, QTableWidgetItem(str(mode)))
        self.tabs.setTabText(2, f"  Représentations ({len(events)} gates)")

    def _on_gate_plot_changed(self, index: int) -> None:
        label = self.combo_gate_plot.currentText()
        path = self._gate_plot_paths.get(label) if hasattr(self, "_gate_plot_paths") else None
        if path and Path(path).exists():
            if path.lower().endswith(".html"):
                webbrowser.open(str(Path(path).resolve()))
                self.gate_canvas.clear_and_reset()
                self.gate_canvas.axes.text(
                    0.5,
                    0.5,
                    f"Figure interactive\nOuverture dans le navigateur…\n\n{label}",
                    transform=self.gate_canvas.axes.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=COLORS["paper"],
                    style="italic",
                )
                self.gate_canvas.axes.axis("off")
                self.gate_canvas.draw()
            else:
                try:
                    import matplotlib.image as mpimg

                    self.gate_canvas.fig.clear()
                    ax = self.gate_canvas.fig.add_subplot(111)
                    ax.imshow(mpimg.imread(path))
                    ax.axis("off")
                    self.gate_canvas.fig.patch.set_facecolor(COLORS["surface"])
                    self.gate_canvas.fig.tight_layout(pad=0.3)
                    self.gate_canvas.draw()
                except Exception as e:
                    self._log(f"Erreur affichage représentation : {e}")
        else:
            self.gate_canvas.clear_and_reset()
            self.gate_canvas.axes.text(
                0.5,
                0.5,
                f"'{label}'\nnon disponible pour cette analyse",
                transform=self.gate_canvas.axes.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color=COLORS["paper"],
                style="italic",
            )
            self.gate_canvas.axes.axis("off")
            self.gate_canvas.draw()

    def _open_current_repr_browser(self) -> None:
        label = self.combo_gate_plot.currentText()
        path = self._gate_plot_paths.get(label) if hasattr(self, "_gate_plot_paths") else None
        if path and Path(path).exists():
            webbrowser.open(str(Path(path).resolve()))
        else:
            QMessageBox.information(self, "Info", f"'{label}' non disponible.")

    # ==================================================================
    # Exports
    # ==================================================================

    # ------------------------------------------------------------------
    # Validation experte — injection avant tout export
    # ------------------------------------------------------------------

    def _inject_human_curation(self) -> None:
        """
        Lit les décisions de validation experte depuis MRDNodeTable et les injecte
        dans self._result avant tout appel d'export.

        Ne fait rien si :
          - _result est None
          - la grille de validation n'a aucune carte (aucun nœud MRD visible)

        Important : si le biologiste a écarté tous les nœuds, on injecte
        quand même curated_nodes=[] et curated_mrd_percent=0 pour que le
        bandeau HTML reflète la décision experte (MRD négatif après validation).
        """
        if self._result is None:
            return
        node_table = getattr(self._home_tab, "_node_table", None)
        if node_table is None:
            return

        # Aucun nœud chargé = pas d'analyse MRD, on n'écrase rien
        if not getattr(node_table, "_nodes", None):
            return

        curated_nodes = node_table.get_human_curated_results()
        total_mrd_cells = sum(n.get("n_patho", 0) for n in curated_nodes)

        # Dénominateur : même logique que dans _update_node_table
        mrd = self._result.mrd_result
        n_pre = getattr(mrd, "n_patho_pre_cd45", 0) if mrd else 0
        total_patho = n_pre if n_pre > 0 else (getattr(mrd, "total_cells_patho", 0) if mrd else 0)
        denom = max(total_patho, 1)

        curated_pct = round(total_mrd_cells / denom * 100.0, 6)

        self._result.curated_mrd_percent = curated_pct
        self._result.curated_mrd_cells = total_mrd_cells
        self._result.curated_nodes = curated_nodes

    def _export_fcs(self) -> None:
        self._inject_human_curation()
        if self._result is None or not self._result.success:
            QMessageBox.information(self, "Info", "Aucun résultat à exporter.")
            return
        output_files = self._result.output_files or {}
        source_fcs = (
            output_files.get("fcs_kaluza")
            or output_files.get("fcs")
            or self._resolve_patho_fcs_path(self._result)
        )
        if not source_fcs or not Path(source_fcs).exists():
            QMessageBox.warning(
                self,
                "Export FCS",
                "Aucun fichier FCS source disponible.\n"
                "Vérifie les options d'export FCS puis relance le pipeline.",
            )
            self._log("[Export FCS] Annulé : aucun FCS source trouvé.")
            return

        # IMPORTANT: le FCS source exporté doit refléter la curation manuelle.
        if not self._patch_is_mrd_in_fcs_file(source_fcs, log_prefix="[FCS export]"):
            QMessageBox.warning(
                self,
                "Export FCS",
                "Échec de mise à jour de Is_MRD selon la validation manuelle.\n"
                "Export annulé pour éviter un FCS incohérent.\n"
                "Consulte les logs [FCS export] pour le détail.",
            )
            return

        try:
            default_name = Path(source_fcs).name
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Exporter FCS",
                str(Path(self.drop_output.path or "") / default_name),
                "FCS Files (*.fcs)",
            )
            if not path:
                return
            target = Path(path)
            if target.suffix.lower() != ".fcs":
                target = target.with_suffix(".fcs")
            shutil.copy2(source_fcs, target)
            self._log(f"[Export FCS] FCS exporté : {target}")
            QMessageBox.information(self, "Export FCS", f"FCS exporté :\n{target}")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def _export_csv(self) -> None:
        self._inject_human_curation()
        # Réécriture Is_MRD demandée explicitement uniquement à l'export.
        self._patch_is_mrd_in_patho_fcs()
        if self._result is None or not self._result.success:
            QMessageBox.information(self, "Info", "Aucun résultat à exporter.")
            return
        output_files = self._result.output_files or {}
        csv_path = output_files.get("cells_csv") or output_files.get("csv")
        if csv_path and Path(csv_path).exists():
            QMessageBox.information(self, "Export CSV", f"Fichier CSV déjà exporté :\n{csv_path}")
        else:
            try:
                path, _ = QFileDialog.getSaveFileName(self, "Exporter CSV", "", "CSV Files (*.csv)")
                if path and self._result.data is not None:
                    self._result.data.to_csv(path, index=False)
                    self._log(f" CSV exporté : {path}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", str(e))

        # ── Dashboard MRD (toujours déclenché, append-safe en mode batch) ──
        self._export_mrd_dashboard()

    def _export_mrd_dashboard(self) -> None:
        """
        Exporte dashboard_metrics.csv avec les valeurs algorithmiques + validées.
        Silencieux en cas d'erreur (ne doit pas bloquer l'utilisateur).
        """
        if self._result is None or self._output_dir is None:
            return
        try:
            from flowsom_pipeline_pro.src.services.export_service import ExportService

            exporter = ExportService(
                config=self._config,
                output_dir=self._output_dir,
                timestamp=getattr(self._result, "timestamp", "")[:15]
                .replace(":", "")
                .replace("-", ""),
                patho_name=getattr(self._result, "patho_stem", None),
                patho_date=getattr(self._result, "patho_date", None),
            )
            # Récupère les gauges calculées par HomeTab si disponibles
            gauges = getattr(self._home_tab, "_last_gauges_data", None) or []
            dash_path = exporter.export_mrd_dashboard_csv(self._result, gauges=gauges)
            if dash_path:
                self._log(f" Dashboard MRD exporté : {dash_path}")
        except Exception as exc:
            _logger.warning("_export_mrd_dashboard: %s", exc)

    def _on_curation_changed(self) -> None:
        """
        Slot connecté à HomeTab.curation_changed.
        Appelé après chaque KEEP/DISCARD ou ajout/suppression de nœud expert.
        Met à jour _result et patche silencieusement le HTML existant.
        """
        self._inject_human_curation()
        if self._result is None or self._result.curated_nodes is None:
            return
        output_files = self._result.output_files or {}
        html_path = output_files.get("html_report")
        if not (html_path and Path(html_path).exists()):
            return
        try:
            from flowsom_pipeline_pro.src.visualization.html_report import (
                patch_curated_banner_in_html,
            )

            gauges = getattr(self._home_tab, "_last_gauges_data", None) or []
            ok = patch_curated_banner_in_html(
                html_path,
                curated_mrd_percent=self._result.curated_mrd_percent or 0.0,
                curated_mrd_cells=self._result.curated_mrd_cells or 0,
                curated_nodes=self._result.curated_nodes,
                algo_gauges=gauges,
            )
            if ok:
                self._log(" HTML mis à jour (validation experte).")
        except Exception as _e:
            _logger.warning("_on_curation_changed patch HTML: %s", _e)

        # IMPORTANT perf UI : ne pas réécrire le FCS ici.
        # Cette opération disque coûteuse est déclenchée uniquement via les actions d'export.

    def _on_verification_commit_requested(self, filter_label: str) -> None:
        """
        Validation explicite depuis le bouton UI : applique immédiatement
        la curation au HTML et au FCS pathologique Is_MRD.
        """
        self._log(f"[Validation experte] Demande explicite reçue (filtre: {filter_label}).")
        self._inject_human_curation()
        self._on_curation_changed()  # patch HTML
        patho_ok = self._patch_is_mrd_in_patho_fcs()
        full_fcs = self._resolve_full_fcs_path(self._result)
        full_ok = False
        if full_fcs and Path(full_fcs).exists():
            patho_fcs = self._resolve_patho_fcs_path(self._result)
            if not patho_fcs or Path(full_fcs).resolve() != Path(patho_fcs).resolve():
                full_ok = self._patch_is_mrd_in_fcs_file(full_fcs, log_prefix="[FCS complet]")
        fcs_ok = patho_ok or full_ok

        method_label = (
            self.combo_mrd_fcs_method.currentText()
            if hasattr(self, "combo_mrd_fcs_method")
            else "flo"
        )
        if hasattr(self, "_home_tab") and self._home_tab is not None:
            try:
                self._home_tab.set_validation_status(method_label, filter_label)
            except Exception:
                pass
        if fcs_ok:
            # Recharger explicitement le FCS patho patché dans le viewer et forcer la coloration MRD.
            try:
                _patched_path = self._resolve_patho_fcs_path(self._result)
                if _patched_path and Path(_patched_path).exists():
                    self._load_fcs_for_visualization(file_path=_patched_path)
                    self._set_fcs_viewer_color_to_mrd()
            except Exception as _viewer_e:
                self._log(f"[Viewer FCS] Synchronisation post-validation ignorée : {_viewer_e}")
            self._log(
                " Validation explicite appliquée — HTML + FCS mis à jour "
                f"(méthode FCS: {method_label.upper()}, filtre: {filter_label})."
            )
            QMessageBox.information(
                self,
                "Validation experte",
                "Validation appliquée avec succès.\n"
                "Le HTML et le FCS pathologique ont été mis à jour.",
            )
        else:
            self._log(
                "[Validation experte] HTML mis à jour mais FCS non réécrit; voir logs [FCS patho]."
            )
            QMessageBox.warning(
                self,
                "Validation experte",
                "Validation appliquée, mais la réécriture FCS a échoué/été ignorée.\n"
                "Consulte les logs [FCS patho] et [FCS complet] pour le détail.",
            )

    def _set_fcs_viewer_color_to_mrd(self) -> None:
        """Force la coloration du viewer sur la colonne MRD si disponible."""
        try:
            candidates = (
                "is_mrd",
                "ismrd",
                "is_mrd_flo",
                "is_mrd_jf",
                "mrd+/mrd",
            )
            target_idx = -1
            for i in range(self.combo_fcs_color.count()):
                txt = (self.combo_fcs_color.itemText(i) or "").lower().replace(" ", "_")
                data = self.combo_fcs_color.itemData(i)
                data_col = ""
                if isinstance(data, tuple) and data:
                    data_col = str(data[0]).lower().replace(" ", "_")
                if any(c in txt for c in candidates) or any(c in data_col for c in candidates):
                    target_idx = i
                    break
            if target_idx >= 0:
                self.combo_fcs_color.setCurrentIndex(target_idx)
                self._update_fcs_plot()
                self._log("[Viewer FCS] Coloration synchronisée sur MRD.")
        except Exception as _e:
            self._log(f"[Viewer FCS] Impossible de forcer la coloration MRD : {_e}")

    def _patch_is_mrd_in_patho_fcs(self) -> bool:
        """Met à jour la/les colonne(s) Is_MRD du FCS patho selon curated_nodes."""
        if self._result is None:
            self._log("[FCS patho] Patch Is_MRD ignoré : aucun résultat pipeline en mémoire.")
            return False
        patho_fcs = self._resolve_patho_fcs_path(self._result)
        if not patho_fcs or not Path(patho_fcs).exists():
            self._log(
                "[FCS patho] Patch Is_MRD ignoré : fichier pathologique introuvable "
                "(fcs_patho_mrd et fallback viewer/output)."
            )
            self._log("[FCS patho] Vérifie patho_fcs_export.enabled=true puis relance le pipeline.")
            return False
        return self._patch_is_mrd_in_fcs_file(patho_fcs, log_prefix="[FCS patho]")

    def _patch_is_mrd_in_fcs_file(self, fcs_path: str, log_prefix: str = "[FCS]") -> bool:
        """Applique la curation manuelle au champ Is_MRD d'un fichier FCS donné."""
        curated_nodes = getattr(self._result, "curated_nodes", None)
        if curated_nodes is None:
            self._log(
                f"{log_prefix} Patch Is_MRD ignoré : aucune curation détectée (curated_nodes=None)."
            )
            return False
        try:
            import numpy as np

            # Ensemble des node_id curatés comme MRD (0-based côté MRD/per_node)
            mrd_node_ids: set = set()
            curated_cells_expected = 0
            for nd in curated_nodes:
                nid = nd.get("node_id") if isinstance(nd, dict) else getattr(nd, "node_id", None)
                n_patho = (
                    nd.get("n_patho", 0) if isinstance(nd, dict) else getattr(nd, "n_patho", 0)
                )
                if nid is not None:
                    mrd_node_ids.add(int(nid))
                try:
                    curated_cells_expected += int(n_patho)
                except Exception:
                    pass

            self._log(f"{log_prefix} Fichier cible patch Is_MRD : {fcs_path}")
            events = None
            col_names: List[str] = []

            # 1) Lecture flowio (prioritaire)
            try:
                import flowio

                fcs_data = flowio.FlowData(fcs_path)
                events = np.reshape(fcs_data.events, (-1, fcs_data.channel_count)).copy()
                n_ch = fcs_data.channel_count
                text = {k.lower(): v for k, v in fcs_data.text.items()}
                for i in range(1, n_ch + 1):
                    for key in (f"$p{i}s", f"p{i}s", f"$p{i}n", f"p{i}n"):
                        if key in text:
                            col_names.append(str(text[key]))
                            break
                    else:
                        col_names.append(f"Channel_{i}")
            except Exception as _e_flowio:
                self._log(f"{log_prefix} lecture flowio indisponible: {str(_e_flowio)[:120]}")

            # 2) Fallback fcsparser
            if events is None:
                try:
                    import fcsparser

                    for naming in ("$PnS", "$PnN"):
                        try:
                            _, data = fcsparser.parse(
                                fcs_path,
                                meta_data_only=False,
                                reformat_meta=False,
                                channel_naming=naming,
                            )
                            events = data.values.astype(np.float32)
                            col_names = [str(c) for c in data.columns]
                            break
                        except Exception:
                            continue
                except Exception as _e_fcsparser:
                    self._log(
                        f"{log_prefix} lecture fcsparser indisponible: {str(_e_fcsparser)[:120]}"
                    )

            # 3) Fallback lecteur binaire interne
            if events is None:
                try:
                    adata = self._read_fcs_binary(fcs_path)
                    X = adata.X
                    if hasattr(X, "toarray"):
                        X = X.toarray()
                    events = np.asarray(X, dtype=np.float32)
                    col_names = [str(c) for c in list(adata.var_names)]
                except Exception as _e_bin:
                    self._log(f"{log_prefix} lecture binaire indisponible: {str(_e_bin)[:120]}")

            if events is None or len(col_names) == 0:
                self._log(f"{log_prefix} Impossible de lire le FCS pour patch Is_MRD.")
                return False

            col_lower = [str(c).lower().strip() for c in col_names]
            col_norm = [c.replace(" ", "_").replace("-", "_") for c in col_lower]

            # Cluster: priorité à FlowSOM_cluster puis fallback cluster non-meta.
            cluster_idx = None
            if "flowsom_cluster" in col_norm:
                cluster_idx = col_norm.index("flowsom_cluster")
            else:
                for i, c in enumerate(col_norm):
                    if c.startswith("flowsom_cluster"):
                        cluster_idx = i
                        break
            if cluster_idx is None and "cluster" in col_norm:
                cluster_idx = col_norm.index("cluster")
            if cluster_idx is None:
                for i, c in enumerate(col_norm):
                    if c.endswith("cluster") and "meta" not in c:
                        cluster_idx = i
                        break
            if cluster_idx is None:
                self._log(f"{log_prefix} Colonne FlowSOM_cluster introuvable; patch Is_MRD ignoré.")
                return False

            # Réécrire Is_MRD en fonction des nœuds curatés.
            # IMPORTANT: la curation manuelle devient la vérité terrain finale.
            # Le codage des clusters peut être 0-based ou 1-based selon l'export.
            clusters_raw = np.rint(events[:, cluster_idx]).astype(np.int32)
            mask_0b = np.array([int(c) in mrd_node_ids for c in clusters_raw], dtype=bool)
            clusters_1b = np.clip(clusters_raw - 1, 0, None)
            mask_1b = np.array([int(c) in mrd_node_ids for c in clusters_1b], dtype=bool)

            n0 = int(mask_0b.sum())
            n1 = int(mask_1b.sum())
            if curated_cells_expected > 0:
                use_1b = abs(n1 - curated_cells_expected) < abs(n0 - curated_cells_expected)
            else:
                use_1b = n1 > n0
            is_mrd_new = (mask_1b if use_1b else mask_0b).astype(np.float32)
            self._log(
                f"{log_prefix} Mapping cluster retenu: {'1-based→0-based' if use_1b else '0-based direct'} "
                f"(MRD={int(is_mrd_new.sum()):,}, attendu={curated_cells_expected:,})"
            )

            # Met à jour toutes les variantes Is_MRD existantes (Is_MRD, Is_MRD_FLO, Is_MRD_JF...).
            mrd_indices = [
                i
                for i, c in enumerate(col_norm)
                if c == "is_mrd" or c == "ismrd" or c.startswith("is_mrd") or c.startswith("ismrd")
            ]
            patched_cols: List[str] = []
            for idx in mrd_indices:
                events[:, idx] = is_mrd_new
                patched_cols.append(col_names[idx])

            # Si aucune colonne MRD n'existe, on ajoute une colonne canonique Is_MRD.
            if not mrd_indices:
                events = np.column_stack([events, is_mrd_new.astype(np.float32)])
                col_names.append("Is_MRD")
                patched_cols.append("Is_MRD")

            # FCS ne supporte pas NaN/Inf.
            events = np.nan_to_num(events, nan=0.0, posinf=1e6, neginf=0.0).astype(np.float32)

            # Réécriture atomique via fichier temporaire.
            try:
                import fcswrite

                target_path = Path(fcs_path)
                tmp_path = str(target_path.with_name(target_path.stem + ".tmp_patch.fcs"))

                fcswrite.write_fcs(
                    filename=tmp_path,
                    chn_names=col_names,
                    data=events,
                    compat_chn_names=True,
                    compat_percent=False,
                    endianness="big",
                )
                os.replace(tmp_path, fcs_path)
            except ImportError:
                self._log(f"{log_prefix} fcswrite non installé : tentative via export_to_fcs.")
                try:
                    import pandas as pd
                    from flowsom_pipeline_pro.src.io.fcs_writer import export_to_fcs

                    target_path = Path(fcs_path)
                    tmp_path = str(target_path.with_name(target_path.stem + ".tmp_patch.fcs"))
                    df_tmp = pd.DataFrame(events, columns=col_names)
                    ok = export_to_fcs(df_tmp, tmp_path, compat_chn_names=True)
                    if not ok or not Path(tmp_path).exists():
                        self._log(f"{log_prefix} fallback export_to_fcs a échoué.")
                        return False
                    os.replace(tmp_path, fcs_path)
                except Exception as _fallback_e:
                    self._log(f"{log_prefix} fallback export_to_fcs impossible : {_fallback_e}")
                    return False
            except Exception as _write_e:
                self._log(f"{log_prefix} écriture fcswrite échouée : {_write_e}")
                try:
                    import pandas as pd
                    from flowsom_pipeline_pro.src.io.fcs_writer import export_to_fcs

                    target_path = Path(fcs_path)
                    tmp_path = str(target_path.with_name(target_path.stem + ".tmp_patch.fcs"))
                    df_tmp = pd.DataFrame(events, columns=col_names)
                    ok = export_to_fcs(df_tmp, tmp_path, compat_chn_names=True)
                    if not ok or not Path(tmp_path).exists():
                        self._log(
                            f"{log_prefix} fallback export_to_fcs a échoué après erreur fcswrite."
                        )
                        return False
                    os.replace(tmp_path, fcs_path)
                except Exception as _fallback_e2:
                    self._log(f"{log_prefix} fallback export_to_fcs impossible : {_fallback_e2}")
                    return False

            n_mrd = int((is_mrd_new > 0).sum())
            self._log(
                f" {log_prefix} Is_MRD manuel appliqué : {n_mrd:,} cellules MRD / {len(events):,}"
            )
            self._log(f" {log_prefix} Colonnes MRD écrasées: {', '.join(patched_cols)}")
            self._log(
                f" {log_prefix} [AUDIT MRD EXPORT] "
                "Nœuds curés=%d | Cellules attendues=%d | Cellules Is_MRD exportées=%d"
                % (len(mrd_node_ids), curated_cells_expected, n_mrd)
            )
            if curated_cells_expected != n_mrd:
                self._log(
                    f" {log_prefix} [AUDIT MRD EXPORT][ATTENTION] "
                    "Écart attendu/exporté = %d cellule(s)." % (n_mrd - curated_cells_expected)
                )

            # Recharger le viewer FCS si c'est ce fichier qui est affiché
            info_text = self.lbl_fcs_info.text()
            if Path(fcs_path).name in info_text:
                self._load_fcs_for_visualization(file_path=fcs_path)
                self._set_fcs_viewer_color_to_mrd()

            return True

        except Exception as _e:
            self._log(f"{log_prefix}[ERREUR] Patch Is_MRD échoué : {_e}")
            try:
                _logger.warning("_patch_is_mrd_in_fcs_file: %s", _e)
            except Exception:
                pass
            return False

    def _open_html_report(self) -> None:
        self._inject_human_curation()
        if self._result is None:
            QMessageBox.information(self, "Info", "Aucun résultat disponible.")
            return
        output_files = self._result.output_files or {}
        html_path = output_files.get("html_report")
        if not (html_path and Path(html_path).exists()):
            QMessageBox.information(self, "Info", "Rapport HTML non trouvé.")
            return

        # ── Patch validation experte dans le HTML existant ───────────────
        # Dès que le biologiste a interagi avec la grille (curated_nodes is not
        # None, même liste vide = tous écartés), on met à jour le bandeau MRD
        # dans le fichier HTML sans régénérer les figures.
        if self._result.curated_nodes is not None:
            try:
                from flowsom_pipeline_pro.src.visualization.html_report import (
                    patch_curated_banner_in_html,
                )

                gauges = getattr(self._home_tab, "_last_gauges_data", None) or []
                ok = patch_curated_banner_in_html(
                    html_path,
                    curated_mrd_percent=self._result.curated_mrd_percent or 0.0,
                    curated_mrd_cells=self._result.curated_mrd_cells or 0,
                    curated_nodes=self._result.curated_nodes,
                    algo_gauges=gauges,
                )
                if ok:
                    self._log(" Rapport HTML mis à jour avec la validation experte.")
                else:
                    _logger.warning("_open_html_report: patch validation experte échoué")
            except Exception as _patch_err:
                _logger.warning("_open_html_report patch: %s", _patch_err)

        webbrowser.open(str(Path(html_path).resolve()))

    def _open_output_folder(self) -> None:
        output = self.drop_output.path
        if output and Path(output).is_dir():
            os.startfile(output)
        elif self._result and self._result.output_files:
            for v in self._result.output_files.values():
                if v and Path(v).exists():
                    os.startfile(str(Path(v).parent))
                    return
            QMessageBox.information(self, "Info", "Dossier de sortie non trouvé.")
        else:
            QMessageBox.information(self, "Info", "Aucun dossier de sortie configuré.")

    # ==================================================================
    # Visualisation FCS (identique à v2)
    # ==================================================================

    def _toggle_fcs_all_cells(self, state: int) -> None:
        self.spin_fcs_cells.setEnabled(state != Qt.Checked)
        self._update_fcs_plot()

    def _extract_fcs_names(self, file_path: str, n_channels: int) -> List[str]:
        try:
            import flowio

            text = flowio.FlowData(file_path).text
            norm: Dict[str, str] = {}
            for k, v in text.items():
                raw = k.strip().upper()
                if not raw.startswith("$"):
                    raw = "$" + raw
                norm[raw] = str(v).strip()
            names: List[str] = []
            for i in range(1, n_channels + 1):
                name = ""
                for key in (f"$P{i}S", f"$P{i}N"):
                    val = norm.get(key, "").strip()
                    if val:
                        name = val
                        break
                names.append(name if name else f"Channel_{i}")
            return names
        except Exception:
            pass

        try:
            with open(file_path, "rb") as f:
                header = f.read(58)
                text_start = int(header[10:18].decode("ascii").strip())
                text_end = int(header[18:26].decode("ascii").strip())
                f.seek(text_start)
                raw_seg = f.read(text_end - text_start + 1)
            try:
                text_str = raw_seg.decode("latin-1")
            except Exception:
                text_str = raw_seg.decode("utf-8", errors="replace")
            delim = text_str[0]
            parts = text_str[1:].split(delim)
            td: Dict[str, str] = {}
            for j in range(0, len(parts) - 1, 2):
                td[parts[j].strip().upper()] = parts[j + 1].strip() if j + 1 < len(parts) else ""
            names = []
            for i in range(1, n_channels + 1):
                name = ""
                for key in (f"$P{i}S", f"P{i}S", f"$P{i}N", f"P{i}N"):
                    val = td.get(key, "").strip()
                    if val:
                        name = val
                        break
                names.append(name if name else f"Channel_{i}")
            return names
        except Exception:
            pass

        return [f"Channel_{i}" for i in range(1, n_channels + 1)]

    def _read_fcs_binary(self, file_path: str) -> Any:
        import struct
        import numpy as np

        try:
            import anndata as ad
        except ImportError:
            raise ImportError("anndata requis pour la lecture FCS")

        with open(file_path, "rb") as f:
            header = f.read(58)
            text_start = int(header[10:18].decode("ascii").strip())
            text_end = int(header[18:26].decode("ascii").strip())
            data_start = int(header[26:34].decode("ascii").strip())
            data_end = int(header[34:42].decode("ascii").strip())
            f.seek(text_start)
            text_segment = f.read(text_end - text_start + 1)
            try:
                text_str = text_segment.decode("latin-1")
            except Exception:
                text_str = text_segment.decode("utf-8", errors="replace")
            delimiter = text_str[0]
            parts = text_str[1:].split(delimiter)
            text_dict: Dict[str, str] = {}
            for i in range(0, len(parts) - 1, 2):
                text_dict[parts[i].strip().upper()] = (
                    parts[i + 1].strip() if i + 1 < len(parts) else ""
                )
            n_params = int(text_dict.get("$PAR", text_dict.get("PAR", 0)))
            n_events = int(text_dict.get("$TOT", text_dict.get("TOT", 0)))
            datatype = text_dict.get("$DATATYPE", text_dict.get("DATATYPE", "F")).upper()
            byteord = text_dict.get("$BYTEORD", text_dict.get("BYTEORD", "1,2,3,4"))
            if n_params == 0 or n_events == 0:
                raise ValueError(f"Paramètres invalides : {n_params} params, {n_events} events")
            endian = "<" if byteord in ("1,2,3,4", "1,2") else ">"
            channel_names: List[str] = []
            for i in range(1, n_params + 1):
                name = None
                for key in (f"$P{i}S", f"P{i}S", f"$P{i}N", f"P{i}N"):
                    if key in text_dict:
                        name = text_dict[key]
                        break
                channel_names.append(name or f"Channel_{i}")
            if datatype == "F":
                fmt = f"{endian}{n_params}f"
                bpe = n_params * 4
            elif datatype == "D":
                fmt = f"{endian}{n_params}d"
                bpe = n_params * 8
            else:
                bits = int(text_dict.get("$P1B", text_dict.get("P1B", 16)))
                fmt = f"{endian}{n_params}{'H' if bits == 16 else 'I'}"
                bpe = n_params * (2 if bits == 16 else 4)
            f.seek(data_start)
            data_bytes = f.read(data_end - data_start + 1)

        events = []
        for i in range(n_events):
            offset = i * bpe
            if offset + bpe <= len(data_bytes):
                try:
                    events.append(struct.unpack(fmt, data_bytes[offset : offset + bpe]))
                except Exception:
                    break
        if not events:
            raise ValueError("Aucun event lu depuis le fichier FCS")
        data_array = np.array(events, dtype=np.float32)
        adata = ad.AnnData(data_array)
        adata.var_names = channel_names
        self._log(f"Lecture binaire FCS : {len(events)} events, {n_params} canaux")
        return adata

    def _load_fcs_for_visualization(self, file_path: Optional[str] = None) -> None:
        import numpy as np

        if file_path is None:
            dlg = QFileDialog(self, "Charger un fichier FCS")
            dlg.setFileMode(QFileDialog.ExistingFile)
            dlg.setNameFilter("FCS Files (*.fcs *.FCS);;All Files (*)")
            dlg.setWindowModality(Qt.ApplicationModal)
            dlg.raise_()
            dlg.activateWindow()
            if dlg.exec_() == QFileDialog.Accepted:
                selected = dlg.selectedFiles()
                file_path = selected[0] if selected else ""
            else:
                file_path = ""
        if not file_path:
            return
        try:
            self._log(f"Chargement FCS : {Path(file_path).name}")
            adata = None
            last_error: Optional[Exception] = None

            try:
                import flowsom as fs

                adata = fs.io.read_FCS(file_path)
                self._log("Chargé avec flowsom")
            except Exception as e1:
                self._log(f"flowsom échoué : {str(e1)[:60]}")
                last_error = e1

            if adata is None:
                try:
                    import flowio
                    import anndata as ad

                    fcs_data = flowio.FlowData(file_path)
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
                        self._log("Chargé avec flowio")
                    else:
                        raise ValueError(f"Fichier vide : {n_ev} events, {n_ch} canaux")
                except ImportError:
                    self._log("flowio non installé")
                except Exception as e2:
                    self._log(f"flowio échoué : {str(e2)[:60]}")
                    last_error = e2

            if adata is None:
                try:
                    import fcsparser
                    import anndata as ad

                    for naming in ("$PnS", "$PnN"):
                        try:
                            meta, data = fcsparser.parse(
                                file_path,
                                meta_data_only=False,
                                reformat_meta=False,
                                channel_naming=naming,
                            )
                            adata = ad.AnnData(data.values.astype(np.float32))
                            adata.var_names = list(data.columns)
                            self._log(f"Chargé avec fcsparser ({naming})")
                            break
                        except Exception:
                            pass
                except ImportError:
                    self._log("fcsparser non installé")
                except Exception as e3:
                    self._log(f"fcsparser échoué : {str(e3)[:60]}")
                    last_error = e3

            if adata is None:
                try:
                    adata = self._read_fcs_binary(file_path)
                    self._log("Chargé avec lecture binaire directe")
                except Exception as e4:
                    self._log(f"Binaire échoué : {str(e4)[:60]}")
                    last_error = e4

            if adata is None:
                raise RuntimeError(f"Impossible de charger le FCS. Dernière erreur : {last_error}")

            real_names = self._extract_fcs_names(file_path, adata.shape[1])
            try:
                adata.var_names = real_names
            except Exception:
                pass

            self.current_fcs_adata = adata
            markers = list(adata.var_names)

            for combo in (self.combo_fcs_x, self.combo_fcs_y):
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(markers)
                combo.blockSignals(False)

            self.combo_fcs_color.blockSignals(True)
            self.combo_fcs_color.clear()
            self.combo_fcs_color.addItem("Aucune")

            # Colonnes classiques FlowSOM
            color_patterns = (
                "flowsom_cluster",
                "flowsom_metacluster",
                "condition",
                "cluster",
                "metacluster",
                "flowsom",
            )
            # Colonnes d'annotation spéciales — traitement prioritaire avec palettes dédiées
            # Inclut les variantes avec underscore ET avec espace (fcswrite compat_chn_names=True
            # remplace _ par espace : "Is_MRD" → "Is MRD", "CD45_Status" → "CD45 Status", etc.)
            # Format clé: nom normalisé (lowercase, underscores) → (label_display, clé_palette)
            _SPECIAL_COLS = {
                # Is_MRD — variantes fcswrite
                "is_mrd": ("Is_MRD", "__palette_is_mrd__"),
                "ismrd": ("Is_MRD", "__palette_is_mrd__"),
                "is mrd": ("Is_MRD", "__palette_is_mrd__"),
                "is_mrd_flo": ("Is_MRD", "__palette_is_mrd__"),
                "is_mrd_jf": ("Is_MRD", "__palette_is_mrd__"),
                "is mrd flo": ("Is_MRD", "__palette_is_mrd__"),
                "is mrd jf": ("Is_MRD", "__palette_is_mrd__"),
                # CD45_Status — CD45+ / CD45dim / CD45-
                "cd45_status": ("CD45_Status", "__palette_cd45__"),
                "cd45 status": ("CD45_Status", "__palette_cd45__"),
                "cd45status": ("CD45_Status", "__palette_cd45__"),
                # CD34_Status — CD34+ / CD34-
                "cd34_status": ("CD34_Status", "__palette_cd34__"),
                "cd34 status": ("CD34_Status", "__palette_cd34__"),
                "cd34status": ("CD34_Status", "__palette_cd34__"),
                # Debris_Flag — Débris oui/non
                "debris_flag": ("Debris_Flag", "__palette_debris__"),
                "debris flag": ("Debris_Flag", "__palette_debris__"),
                "debrisflag": ("Debris_Flag", "__palette_debris__"),
                # Doublet_Flag — Doublets oui/non
                "doublet_flag": ("Doublet_Flag", "__palette_doublet__"),
                "doublet flag": ("Doublet_Flag", "__palette_doublet__"),
                "doubletflag": ("Doublet_Flag", "__palette_doublet__"),
            }
            # Labels d'affichage explicites dans le combo
            _SPECIAL_DISPLAY = {
                "Is_MRD": "MRD+/MRD−  (Is_MRD)",
                "CD45_Status": "CD45+ / CD45dim / CD45−",
                "CD34_Status": "CD34+ / CD34−",
                "Debris_Flag": "Débris oui/non",
                "Doublet_Flag": "Doublets oui/non",
            }
            _special_found: dict = {}  # label_interne → (col_name_réelle, palette_key)

            for m in markers:
                # Normaliser : lowercase + remplacer espaces par underscores pour lookup
                ml = m.lower().replace(" ", "_")
                if ml in _SPECIAL_COLS:
                    label, palette_key = _SPECIAL_COLS[ml]
                    _special_found[label] = (m, palette_key)
                    continue
                if any(p in m.lower() for p in color_patterns):
                    self.combo_fcs_color.addItem(m)

            # Ajouter les colonnes spéciales avec leur palette encodée dans userData
            _auto_select_col: Optional[int] = None
            for label, (col_name, palette_key) in _special_found.items():
                display = _SPECIAL_DISPLAY.get(label, label)
                self.combo_fcs_color.addItem(display, (col_name, palette_key))
                if label == "Is_MRD":
                    _auto_select_col = self.combo_fcs_color.count() - 1

            # Sélection automatique Is_MRD si présent, sinon CD45_Status
            if _auto_select_col is not None:
                self.combo_fcs_color.setCurrentIndex(_auto_select_col)
            elif "CD45_Status" in _special_found:
                idx = next(
                    (
                        i
                        for i in range(self.combo_fcs_color.count())
                        if "CD45" in (self.combo_fcs_color.itemText(i) or "")
                    ),
                    -1,
                )
                if idx >= 0:
                    self.combo_fcs_color.setCurrentIndex(idx)

            self.combo_fcs_color.blockSignals(False)

            fsc_idx = next((i for i, m in enumerate(markers) if "FSC" in m.upper()), 0)
            ssc_idx = next(
                (i for i, m in enumerate(markers) if "SSC" in m.upper()),
                min(1, len(markers) - 1),
            )
            self.combo_fcs_x.setCurrentIndex(fsc_idx)
            self.combo_fcs_y.setCurrentIndex(ssc_idx)

            self.lbl_fcs_info.setText(
                f"{Path(file_path).name}  |  {adata.shape[0]:,} cellules  |  {adata.shape[1]} paramètres"
            )
            self._update_fcs_plot()
            self._log(f"FCS chargé : {adata.shape[0]:,} cellules, {adata.shape[1]} paramètres")

        except Exception as e:
            QMessageBox.critical(self, "Erreur chargement FCS", str(e))
            self._log(f"Erreur chargement FCS : {e}")

    def _update_fcs_plot(self) -> None:
        if self.current_fcs_adata is None:
            return
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        try:
            x_marker = self.combo_fcs_x.currentText()
            y_marker = self.combo_fcs_y.currentText()
            plot_type = self.combo_fcs_plot_type.currentText()
            # currentData() = (col_name, palette_key) pour colonnes spéciales, None sinon
            color_by_data = self.combo_fcs_color.currentData()
            _palette_key: Optional[str] = None
            if isinstance(color_by_data, tuple):
                color_by, _palette_key = color_by_data  # (col_name, "__palette_xxx__")
            else:
                color_by = self.combo_fcs_color.currentText()
            show_all = self.chk_fcs_all_cells.isChecked()
            max_cells = float("inf") if show_all else self.spin_fcs_cells.value()

            if not x_marker or not y_marker:
                return

            X = self.current_fcs_adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            var_names = list(self.current_fcs_adata.var_names)
            x_data = X[:, var_names.index(x_marker)].copy()
            y_data = X[:, var_names.index(y_marker)].copy()

            # Détecter si coloré par Is_MRD pour une colormap dédiée (rétro-compat texte brut)
            _is_mrd_coloring = _palette_key == "__palette_is_mrd__" or (
                not _palette_key
                and isinstance(color_by, str)
                and color_by.lower().replace(" ", "_") in ("is_mrd", "ismrd")
            )
            color_data = None
            # Normalisation robuste : ignore underscore vs espace, casse
            # (fcswrite compat_chn_names convertit "Is_MRD" → "Is MRD")
            _var_names_norm = [v.lower().replace(" ", "_").replace("-", "_") for v in var_names]
            if _palette_key and isinstance(color_by, str):
                # Colonnes spéciales : chercher par nom normalisé dans var_names
                _ck = color_by.lower().replace(" ", "_").replace("-", "_")
                if color_by in var_names:
                    color_data = X[:, var_names.index(color_by)].copy()
                elif _ck in _var_names_norm:
                    color_data = X[:, _var_names_norm.index(_ck)].copy()
                else:
                    # Dernier recours : correspondance partielle sur le radical du nom
                    _radical = _ck.split("_")[0]  # ex: "is", "cd45", "cd34", "debris", "doublet"
                    for _vi, _vn in enumerate(_var_names_norm):
                        if _vn.startswith(_radical) and len(_vn) <= len(_ck) + 4:
                            color_data = X[:, _vi].copy()
                            break
            elif color_by != "Aucune" and isinstance(color_by, str):
                # Colonnes classiques FlowSOM_cluster etc.
                _color_key = color_by.lower().replace(" ", "_").replace("-", "_")
                if color_by in var_names:
                    color_data = X[:, var_names.index(color_by)].copy()
                elif _color_key in _var_names_norm:
                    _ci = _var_names_norm.index(_color_key)
                    color_data = X[:, _ci].copy()

            som_grid = {"xgrid", "ygrid"}
            som_nodes = {"xnodes", "ynodes"}
            is_grid_x = x_marker.lower() in som_grid
            is_grid_y = y_marker.lower() in som_grid
            is_nodes_x = x_marker.lower() in som_nodes
            is_nodes_y = y_marker.lower() in som_nodes
            is_som_x = is_grid_x or is_nodes_x
            is_som_y = is_grid_y or is_nodes_y

            apply_jitter = self.chk_fcs_jitter.isChecked()
            if apply_jitter and (is_som_x or is_som_y):
                n = len(x_data)
                r = np.sqrt(np.random.uniform(0, 1, n))
                theta = np.random.uniform(0, 2 * np.pi, n)
                radius = 0.35 if (is_grid_x or is_grid_y) else 20.0
                if is_som_x:
                    x_data = x_data + r * np.cos(theta) * radius
                if is_som_y:
                    y_data = y_data + r * np.sin(theta) * radius

            dim_cols = {
                "tsne1",
                "tsne2",
                "umap1",
                "umap2",
                "tSNE1",
                "tSNE2",
                "UMAP1",
                "UMAP2",
            }
            is_dim_x = x_marker in dim_cols
            is_dim_y = y_marker in dim_cols
            if is_dim_x or is_dim_y:
                mask = (
                    np.isfinite(x_data)
                    & (x_data != -999.0)
                    & np.isfinite(y_data)
                    & (y_data != -999.0)
                )
            else:
                mask = np.isfinite(x_data) & np.isfinite(y_data)

            x_data = x_data[mask]
            y_data = y_data[mask]
            if color_data is not None:
                color_data = color_data[mask]

            n_total = len(x_data)
            if not show_all and n_total > max_cells:
                idx = np.random.choice(n_total, int(max_cells), replace=False)
                x_data = x_data[idx]
                y_data = y_data[idx]
                if color_data is not None:
                    color_data = color_data[idx]

            n_shown = len(x_data)
            self.lbl_fcs_info.setText(
                f"Affichage : {n_shown:,} / {self.current_fcs_adata.shape[0]:,} cellules"
            )

            self.fcs_viz_canvas.clear_and_reset()
            ax = self.fcs_viz_canvas.axes
            scatter_colors = COLORS["info"]
            legend_handles = None

            if color_data is not None and plot_type == "Scatter":
                from matplotlib.patches import Patch

                if _is_mrd_coloring or _palette_key == "__palette_is_mrd__":
                    # MRD+ rouge néon / Non-MRD bleu transparent
                    _n = len(color_data)
                    _c = np.zeros((_n, 4), dtype=float)
                    _mrd_mask = color_data > 0.5
                    _c[_mrd_mask] = [1.0, 0.239, 0.431, 0.92]  # #FF3D6E  MRD+
                    _c[~_mrd_mask] = [0.357, 0.667, 1.0, 0.30]  # #5BAAFF  Non-MRD
                    scatter_colors = _c
                    _n_mrd = int(_mrd_mask.sum())
                    _n_non = _n - _n_mrd
                    legend_handles = [
                        Patch(
                            facecolor="#5BAAFF",
                            edgecolor="none",
                            label=f"Non-MRD  ({_n_non:,} cellules)",
                        ),
                        Patch(
                            facecolor="#FF3D6E",
                            edgecolor="none",
                            label=f"MRD positif  ({_n_mrd:,} cellules)",
                        ),
                    ]
                elif _palette_key == "__palette_cd45__":
                    # CD45_Status : 0=N/D  1=CD45+bright  2=CD45dim  3=CD45−
                    _cd45_map = {
                        0.0: (0.55, 0.55, 0.65, 0.20),  # N/D — gris discret
                        1.0: (0.224, 1.0, 0.541, 0.88),  # CD45+ bright — vert #39FF8A
                        2.0: (1.0, 0.608, 0.239, 0.88),  # CD45dim     — orange #FF9B3D
                        3.0: (1.0, 0.239, 0.431, 0.88),  # CD45−       — rouge #FF3D6E
                    }
                    _c = np.zeros((len(color_data), 4), dtype=float)
                    _counts_cd45 = {}
                    for _val, _rgba in _cd45_map.items():
                        _mask_v = np.abs(color_data - _val) < 0.5
                        _c[_mask_v] = _rgba
                        _counts_cd45[_val] = int(_mask_v.sum())
                    scatter_colors = _c
                    legend_handles = [
                        Patch(
                            facecolor="#39FF8A",
                            edgecolor="none",
                            label=f"CD45+  bright  ({_counts_cd45.get(1.0, 0):,})",
                        ),
                        Patch(
                            facecolor="#FF9B3D",
                            edgecolor="none",
                            label=f"CD45  dim     ({_counts_cd45.get(2.0, 0):,})",
                        ),
                        Patch(
                            facecolor="#FF3D6E",
                            edgecolor="none",
                            label=f"CD45−  négatif ({_counts_cd45.get(3.0, 0):,})",
                        ),
                        Patch(
                            facecolor=(0.55, 0.55, 0.65, 0.5),
                            edgecolor="none",
                            label=f"N/D  ({_counts_cd45.get(0.0, 0):,})",
                        ),
                    ]
                elif _palette_key == "__palette_cd34__":
                    # CD34_Status : 1=CD34+  0=CD34−
                    _n = len(color_data)
                    _c = np.zeros((_n, 4), dtype=float)
                    _cd34_pos = color_data > 0.5
                    _c[_cd34_pos] = [0.357, 0.667, 1.0, 0.92]  # #5BAAFF  CD34+
                    _c[~_cd34_pos] = [0.55, 0.55, 0.65, 0.18]  # gris discret CD34−
                    scatter_colors = _c
                    _n_pos = int(_cd34_pos.sum())
                    legend_handles = [
                        Patch(
                            facecolor="#5BAAFF",
                            edgecolor="none",
                            label=f"CD34+  ({_n_pos:,} cellules)",
                        ),
                        Patch(
                            facecolor=(0.55, 0.55, 0.65, 0.5),
                            edgecolor="none",
                            label=f"CD34−  ({_n - _n_pos:,} cellules)",
                        ),
                    ]
                elif _palette_key == "__palette_debris__":
                    # Debris_Flag : 1=débris  0=cellule valide
                    _n = len(color_data)
                    _c = np.zeros((_n, 4), dtype=float)
                    _debris_mask = color_data > 0.5
                    _c[_debris_mask] = [1.0, 0.239, 0.431, 0.88]  # #FF3D6E  débris
                    _c[~_debris_mask] = [0.357, 0.667, 1.0, 0.22]  # #5BAAFF  valide
                    scatter_colors = _c
                    _n_deb = int(_debris_mask.sum())
                    legend_handles = [
                        Patch(
                            facecolor="#5BAAFF",
                            edgecolor="none",
                            label=f"Cellule valide  ({_n - _n_deb:,})",
                        ),
                        Patch(facecolor="#FF3D6E", edgecolor="none", label=f"Débris  ({_n_deb:,})"),
                    ]
                elif _palette_key == "__palette_doublet__":
                    # Doublet_Flag : 1=doublet  0=singlet
                    _n = len(color_data)
                    _c = np.zeros((_n, 4), dtype=float)
                    _doub_mask = color_data > 0.5
                    _c[_doub_mask] = [1.0, 0.608, 0.239, 0.88]  # #FF9B3D  doublet
                    _c[~_doub_mask] = [0.357, 0.667, 1.0, 0.22]  # #5BAAFF  singlet
                    scatter_colors = _c
                    _n_dbl = int(_doub_mask.sum())
                    legend_handles = [
                        Patch(
                            facecolor="#5BAAFF",
                            edgecolor="none",
                            label=f"Singlet  ({_n - _n_dbl:,})",
                        ),
                        Patch(
                            facecolor="#FF9B3D", edgecolor="none", label=f"Doublet  ({_n_dbl:,})"
                        ),
                    ]
                else:
                    unique_vals = np.unique(color_data[np.isfinite(color_data)])
                    n_c = len(unique_vals)
                    cmap = (
                        plt.cm.tab20
                        if n_c <= 20
                        else (plt.cm.tab20b if n_c <= 40 else plt.cm.turbo)
                    )
                    indices = np.searchsorted(unique_vals, color_data)
                    scatter_colors = cmap(indices / max(n_c - 1, 1))
                    if n_c <= 20:
                        legend_handles = [
                            Patch(
                                facecolor=cmap(i / max(n_c - 1, 1)),
                                edgecolor="white",
                                label=f"{color_by.replace('FlowSOM_', '')} {int(v)}",
                            )
                            for i, v in enumerate(unique_vals)
                        ]

            if plot_type == "Scatter":
                ax.scatter(
                    x_data,
                    y_data,
                    s=3,
                    alpha=0.6,
                    c=scatter_colors,
                    edgecolors="none",
                    rasterized=True,
                )
                if legend_handles:
                    ax.legend(
                        handles=legend_handles,
                        loc="upper right",
                        fontsize=7,
                        facecolor="#313244",
                        labelcolor="#EEF2F7",
                        edgecolor="#45475a",
                        framealpha=0.9,
                        ncol=2 if len(legend_handles) > 10 else 1,
                    )
            elif plot_type == "Densite":
                h = ax.hist2d(x_data, y_data, bins=100, cmap="viridis", norm=mcolors.LogNorm())
                cb = self.fcs_viz_canvas.fig.colorbar(h[3], ax=ax, label="Densité")
                cb.ax.tick_params(colors=COLORS["paper"])
                cb.ax.yaxis.label.set_color(COLORS["paper"])
            elif plot_type == "Contour":
                from scipy import stats as sp_stats

                try:
                    n_kde = min(5000, len(x_data))
                    kde_idx = np.random.choice(len(x_data), n_kde, replace=False)
                    xmin, xmax = x_data.min(), x_data.max()
                    ymin, ymax = y_data.min(), y_data.max()
                    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    kernel = sp_stats.gaussian_kde(np.vstack([x_data[kde_idx], y_data[kde_idx]]))
                    f = np.reshape(kernel(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)
                    ax.contourf(xx, yy, f, levels=20, cmap="viridis")
                    ax.contour(xx, yy, f, levels=10, colors="white", linewidths=0.3, alpha=0.5)
                except Exception:
                    ax.scatter(
                        x_data,
                        y_data,
                        s=2,
                        alpha=0.5,
                        c=COLORS["info"],
                        edgecolors="none",
                        rasterized=True,
                    )

            from matplotlib.ticker import FuncFormatter

            def _fmt(v, _):
                if abs(v) >= 1e6:
                    return f"{v / 1e6:.1f}M"
                if abs(v) >= 1e3:
                    return f"{v / 1e3:.0f}K"
                return f"{v:.0f}"

            ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
            x_margin = max((x_data.max() - x_data.min()) * 0.02, 1)
            y_margin = max((y_data.max() - y_data.min()) * 0.02, 1)
            ax.set_xlim(x_data.min() - x_margin, x_data.max() + x_margin)
            ax.set_ylim(y_data.min() - y_margin, y_data.max() + y_margin)

            subtitle = f"{n_shown:,} cellules"
            if apply_jitter and (is_som_x or is_som_y):
                subtitle += " | jitter"
            if color_by != "Aucune":
                subtitle += f" | couleur : {color_by.replace('FlowSOM_', '')}"
            ax.set_title(
                f"{x_marker} vs {y_marker}\n{subtitle}",
                fontsize=12,
                color=COLORS["paper"],
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel(x_marker, color=COLORS["paper"], fontsize=11, fontweight="bold")
            ax.set_ylabel(y_marker, color=COLORS["paper"], fontsize=11, fontweight="bold")
            self.fcs_viz_canvas.fig.tight_layout(pad=1.5)
            self.fcs_viz_canvas.draw()

        except Exception as e:
            self._log(f"Erreur plot FCS : {e}")

    # ==================================================================
    # Persistance de session (P3.4)
    # ==================================================================

    _SESSION_FILE = Path.home() / ".flowsom_session.json"

    def _save_session(self) -> None:
        """Sauvegarde les chemins et paramètres UI dans ~/.flowsom_session.json."""
        try:
            data: Dict[str, Any] = {
                # Chemins
                "healthy_folder": self.drop_healthy.path or "",
                "patho_folder": self.drop_patho.path or "",
                "output_folder": self.drop_output.path or "",
                # SOM
                "xdim": self.spin_xdim.value(),
                "ydim": self.spin_ydim.value(),
                "metaclusters": self.spin_metaclusters.value(),
                "seed": self.spin_seed.value(),
                "lr": self.spin_lr.value(),
                "sigma": self.spin_sigma.value(),
                "auto_clustering": self.chk_auto_clustering.isChecked(),
                # Preprocessing
                "cofactor": self.spin_cofactor.value(),
                # Gating
                "pregate": self.chk_pregate.isChecked(),
                "viable": self.chk_viable.isChecked(),
                "singlets": self.chk_singlets.isChecked(),
                "mode_blastes": self.chk_mode_blastes.isChecked(),
                # Options
                "umap": self.chk_umap.isChecked(),
                "compare": self.chk_compare.isChecked(),
                "downsampling": self.chk_downsampling.isChecked(),
                "max_cells": self.spin_max_cells.value(),
                "batch": self.chk_batch.isChecked(),
                "balance_conditions": self.chk_balance_conditions.isChecked(),
                "imbalance_ratio": self.spin_imbalance_ratio.value(),
                "allow_oversampling": self.chk_allow_oversampling.isChecked(),
                # MRD
                "mrd_method": self.combo_mrd_method.currentText(),
                "eln_min_events": self.spin_eln_min_events.value(),
                "eln_positivity": self.spin_eln_positivity.value(),
                "flo_multiplier": self.spin_flo_multiplier.value(),
                "jf_max_normal": self.spin_jf_max_normal.value(),
                "jf_min_patho": self.spin_jf_min_patho.value(),
                "blast_filter": self.chk_blast_filter.isChecked(),
            }
            with open(self._SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Session non critique — ne jamais bloquer la fermeture

    def _restore_session(self) -> None:
        """Restaure les chemins et paramètres UI depuis ~/.flowsom_session.json."""
        if not self._SESSION_FILE.exists():
            return
        try:
            with open(self._SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        # Chemins (uniquement si les dossiers existent encore)
        healthy = data.get("healthy_folder", "")
        if healthy and Path(healthy).is_dir():
            self.drop_healthy.set_path(healthy)

        patho = data.get("patho_folder", "")
        if patho and Path(patho).is_dir():
            self.drop_patho.set_path(patho)

        output = data.get("output_folder", "")
        if output and Path(output).is_dir():
            self.drop_output.set_path(output)

        # Rafraîchit l'aperçu FCS si au moins un dossier est présent
        if healthy or patho:
            self._refresh_fcs_preview()

        # Spinboxes / ComboBox / CheckBoxes — chaque widget protégé par try
        _w = {
            "xdim": (self.spin_xdim, "setValue"),
            "ydim": (self.spin_ydim, "setValue"),
            "metaclusters": (self.spin_metaclusters, "setValue"),
            "seed": (self.spin_seed, "setValue"),
            "lr": (self.spin_lr, "setValue"),
            "sigma": (self.spin_sigma, "setValue"),
            "cofactor": (self.spin_cofactor, "setValue"),
            "max_cells": (self.spin_max_cells, "setValue"),
            "eln_min_events": (self.spin_eln_min_events, "setValue"),
            "eln_positivity": (self.spin_eln_positivity, "setValue"),
            "flo_multiplier": (self.spin_flo_multiplier, "setValue"),
            "jf_max_normal": (self.spin_jf_max_normal, "setValue"),
            "jf_min_patho": (self.spin_jf_min_patho, "setValue"),
            "imbalance_ratio": (self.spin_imbalance_ratio, "setValue"),
        }
        for key, (widget, method) in _w.items():
            if key in data:
                try:
                    getattr(widget, method)(data[key])
                except Exception:
                    pass

        _chk = {
            "auto_clustering": self.chk_auto_clustering,
            "pregate": self.chk_pregate,
            "viable": self.chk_viable,
            "singlets": self.chk_singlets,
            "mode_blastes": self.chk_mode_blastes,
            "umap": self.chk_umap,
            "compare": self.chk_compare,
            "downsampling": self.chk_downsampling,
            "batch": self.chk_batch,
            "balance_conditions": self.chk_balance_conditions,
            "allow_oversampling": self.chk_allow_oversampling,
            "blast_filter": self.chk_blast_filter,
        }
        for key, widget in _chk.items():
            if key in data:
                try:
                    widget.setChecked(bool(data[key]))
                except Exception:
                    pass

        if "mrd_method" in data:
            try:
                idx = self.combo_mrd_method.findText(data["mrd_method"])
                if idx >= 0:
                    self.combo_mrd_method.setCurrentIndex(idx)
            except Exception:
                pass

    # ==================================================================
    # Événements fenêtre
    # ==================================================================

    def closeEvent(self, event: Any) -> None:  # type: ignore[override]
        """Sauvegarde la session avant fermeture."""
        self._save_session()
        super().closeEvent(event)

    # ==================================================================
    # Utilitaires
    # ==================================================================

    def _log(self, msg: str) -> None:
        # log_output créé à l'étape 3 — protection au cas où appelé avant
        if hasattr(self, "log_output"):
            self.log_output.append(msg)
            sb = self.log_output.verticalScrollBar()
            sb.setValue(sb.maximum())


# ══════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    # Apply before QApplication() so Qt renders in native DPI instead of bitmap scaling.
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(QApplication, "setHighDpiScaleFactorRoundingPolicy") and hasattr(
        Qt, "HighDpiScaleFactorRoundingPolicy"
    ):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.RoundPreferFloor
        )

    app = QApplication(sys.argv)
    _register_embedded_fonts()

    # Force Segoe UI as app-wide font for crisp native rendering on Windows
    _app_font = QFont("Segoe UI", 10)
    _app_font.setHintingPreference(QFont.PreferFullHinting)
    _app_font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(_app_font)

    app.setStyle("Fusion")
    app.setApplicationName("PRISMA")
    app.setOrganizationName("Magne Florian")
    _ico = _asset_path("prisma_logo.ico")
    if _ico.exists():
        app.setWindowIcon(QIcon(str(_ico)))
    window = FlowSomAnalyzerPro()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
