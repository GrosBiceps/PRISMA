# -*- coding: utf-8 -*-
"""
main_window.py — Interface graphique FlowSomAnalyzerPro (PyQt5).

Wrapper visuel complet sur la librairie flowsom_pipeline_pro :
  • Panneau de contrôle (gauche) : dossiers, paramètres SOM, transformation, gating
  • Visualisation (centre) : MatplotlibCanvas, heatmaps, UMAP, star charts
  • Logs (onglet) : sortie console pipeline en temps réel
  • Export : FCS Kaluza, PDF, CSV
"""

from __future__ import annotations

import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    QGraphicsDropShadowEffect,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QInputDialog,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib

matplotlib.use("Qt5Agg")

from flowsom_pipeline_pro.gui.styles import STYLESHEET, COLORS
from flowsom_pipeline_pro.gui.workers import PipelineWorker, SpiderPlotWorker

# QWebEngineView pour les visualisations Plotly HTML interactives
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView

    _WEBENGINE = True
except ImportError:
    _WEBENGINE = False

# Chemin par défaut du YAML
_DEFAULT_CONFIG_PATH = Path(
    r"C:\Users\Florian Travail\Documents\FlowSom\Perplexity"
    r"\flowsom_pipeline_pro\config\default_config.yaml"
)


# ══════════════════════════════════════════════════════════════════════
# QComboBox thème sombre (corrige le cadre blanc Windows)
# ══════════════════════════════════════════════════════════════════════


class DarkComboBox(QComboBox):
    """QComboBox avec popup sans cadre/ombre natif Windows (corrige le bug du fond blanc)."""

    def showPopup(self) -> None:  # noqa: N802
        # Appliquer les flags avant tout affichage
        popup = self.view().window()
        popup.setWindowFlags(
            Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint
        )
        super().showPopup()


# ══════════════════════════════════════════════════════════════════════
# Canvas Matplotlib réutilisable
# ══════════════════════════════════════════════════════════════════════


class MatplotlibCanvas(FigureCanvas):
    """
    Canvas Matplotlib intégré dans un widget PyQt5 avec thème sombre."""

    def __init__(
        self, parent: Optional[QWidget] = None, width: int = 8, height: int = 6
    ) -> None:
        self.fig = Figure(figsize=(width, height), dpi=100)
        self.fig.patch.set_facecolor(COLORS["base"])
        self.axes = self.fig.add_subplot(111)
        self._style_axes(self.axes)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _style_axes(self, ax: Any) -> None:
        ax.set_facecolor(COLORS["base"])
        ax.tick_params(colors=COLORS["subtext"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["surface1"])

    def clear_and_reset(self) -> None:
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self._style_axes(self.axes)
        self.draw()

    def display_figure(self, fig: Figure) -> None:
        """Remplace le contenu par une Figure externe, ajustée à la taille du widget."""
        import matplotlib.pyplot as plt

        old_fig = self.fig
        self.fig = fig
        self.figure = fig  # FigureCanvas.draw() utilise self.figure
        self.fig.set_canvas(self)
        self.fig.patch.set_facecolor(COLORS["base"])
        # Adapte la taille de la figure à celle du widget pour éviter les artefacts
        # (zone non recouverte conservant l'ancien rendu)
        dpi = self.fig.get_dpi() or 100
        w_px = max(1, self.width())
        h_px = max(1, self.height())
        self.fig.set_size_inches(w_px / dpi, h_px / dpi)
        self.draw()
        plt.close(old_fig)


# ══════════════════════════════════════════════════════════════════════
# Fenêtre principale
# ══════════════════════════════════════════════════════════════════════


class FlowSomAnalyzerPro(QMainWindow):
    """Application GUI complète pour piloter le pipeline FlowSOM Analysis Pro.

    Charge la config YAML, propose les réglages visuels, exécute le pipeline
    dans un QThread et affiche les résultats (figures, logs, exports).
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FlowSOM Analyzer Pro — MRD Pipeline ELN 2022")
        self.setMinimumSize(1440, 900)
        self.resize(1600, 1000)
        self.setStyleSheet(STYLESHEET)

        # État interne
        self._config: Optional[Any] = None
        self._result: Optional[Any] = None
        self._worker: Optional[PipelineWorker] = None
        self._spider_worker: Optional[SpiderPlotWorker] = None
        self._cluster_mfi: Optional[Any] = None  # DataFrame cluster × markers
        self._all_markers: List[str] = []  # tous les marqueurs disponibles
        self._output_dir: Optional[Path] = None  # dossier de sortie du dernier run
        self._output_plot_paths: Dict[str, str] = {}  # label → chemin fichier
        self.current_fcs_adata: Optional[Any] = (
            None  # AnnData chargé pour visualisation FCS
        )

        self._init_ui()
        self._load_default_config()
        self.statusBar().showMessage("Prêt — Chargez les données et lancez l'analyse")

    # ------------------------------------------------------------------
    # Construction de l'UI
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Horizontal, central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

        # ── Panneau gauche (contrôles) ─────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(380)
        left_scroll.setMaximumWidth(480)

        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(12)

        self._build_header(left_layout)
        self._build_folder_group(left_layout)
        self._build_som_group(left_layout)
        self._build_transform_group(left_layout)
        self._build_gating_group(left_layout)
        self._build_options_group(left_layout)
        self._build_run_section(left_layout)
        self._build_export_section(left_layout)

        left_layout.addStretch()
        left_scroll.setWidget(left_panel)
        splitter.addWidget(left_scroll)

        # ── Panneau droit (visualisation + logs) ───────────────────────
        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        self.tabs = QTabWidget()
        self._build_viz_tab()
        self._build_pregate_tab()
        self._build_clusters_tab()
        self._build_results_tab()
        self._build_logs_tab()
        self._build_fcs_viewer_tab()
        right_layout.addWidget(self.tabs)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ── Header ─────────────────────────────────────────────────────────

    def _build_header(self, layout: QVBoxLayout) -> None:
        title = QLabel("FlowSOM Analyzer Pro")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("MRD Pipeline — ELN 2022 · Cytométrie multi-paramétrique")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: rgba(137,180,250,0.2);")
        layout.addWidget(line)

    # ── Dossiers ───────────────────────────────────────────────────────

    def _build_folder_group(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Données FCS")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        # NBM / Sain
        grid.addWidget(QLabel("Dossier NBM / Sain :"), 0, 0)
        self.lbl_healthy = QLabel("Non sélectionné")
        self.lbl_healthy.setObjectName("fileLabel")
        self.lbl_healthy.setWordWrap(True)
        grid.addWidget(self.lbl_healthy, 1, 0, 1, 2)

        btn_healthy = QPushButton("Parcourir…")
        btn_healthy.clicked.connect(self._select_healthy_folder)
        grid.addWidget(btn_healthy, 0, 1)

        # Patho
        grid.addWidget(QLabel("Dossier Pathologique :"), 2, 0)
        self.lbl_patho = QLabel("Non sélectionné")
        self.lbl_patho.setObjectName("fileLabel")
        self.lbl_patho.setWordWrap(True)
        grid.addWidget(self.lbl_patho, 3, 0, 1, 2)

        btn_patho = QPushButton("Parcourir…")
        btn_patho.clicked.connect(self._select_patho_folder)
        grid.addWidget(btn_patho, 2, 1)

        # Output
        grid.addWidget(QLabel("Dossier de sortie :"), 4, 0)
        self.lbl_output = QLabel("Non sélectionné")
        self.lbl_output.setObjectName("fileLabel")
        self.lbl_output.setWordWrap(True)
        grid.addWidget(self.lbl_output, 5, 0, 1, 2)

        btn_output = QPushButton("Parcourir…")
        btn_output.clicked.connect(self._select_output_folder)
        grid.addWidget(btn_output, 4, 1)

        layout.addWidget(group)

    # ── Paramètres FlowSOM ─────────────────────────────────────────────

    def _build_som_group(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Paramètres FlowSOM")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        # xdim
        grid.addWidget(QLabel("Grille X (xdim) :"), 0, 0)
        self.spin_xdim = QSpinBox()
        self.spin_xdim.setRange(3, 50)
        self.spin_xdim.setValue(10)
        self.spin_xdim.setToolTip("Dimension X de la grille SOM (défaut : 10)")
        grid.addWidget(self.spin_xdim, 0, 1)

        # ydim
        grid.addWidget(QLabel("Grille Y (ydim) :"), 1, 0)
        self.spin_ydim = QSpinBox()
        self.spin_ydim.setRange(3, 50)
        self.spin_ydim.setValue(10)
        self.spin_ydim.setToolTip("Dimension Y de la grille SOM (défaut : 10)")
        grid.addWidget(self.spin_ydim, 1, 1)

        # n_metaclusters
        grid.addWidget(QLabel("Métaclusters :"), 2, 0)
        self.spin_metaclusters = QSpinBox()
        self.spin_metaclusters.setRange(2, 50)
        self.spin_metaclusters.setValue(8)
        self.spin_metaclusters.setToolTip("Nombre de métaclusters (défaut : 8)")
        grid.addWidget(self.spin_metaclusters, 2, 1)

        # seed
        grid.addWidget(QLabel("Seed :"), 3, 0)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip("Graine aléatoire (reproductibilité)")
        grid.addWidget(self.spin_seed, 3, 1)

        # learning rate
        grid.addWidget(QLabel("Learning rate :"), 4, 0)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.001, 1.0)
        self.spin_lr.setSingleStep(0.01)
        self.spin_lr.setValue(0.05)
        self.spin_lr.setDecimals(3)
        grid.addWidget(self.spin_lr, 4, 1)

        # sigma
        grid.addWidget(QLabel("Sigma voisinage :"), 5, 0)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 10.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setValue(1.5)
        self.spin_sigma.setDecimals(1)
        grid.addWidget(self.spin_sigma, 5, 1)

        # auto-clustering
        self.chk_auto_clustering = QCheckBox("Auto-sélection n° clusters (bootstrap)")
        self.chk_auto_clustering.setToolTip(
            "Sélection automatique du nombre optimal de métaclusters\n"
            "par stabilité bootstrap + silhouette"
        )
        grid.addWidget(self.chk_auto_clustering, 6, 0, 1, 2)

        layout.addWidget(group)

    # ── Transformation ─────────────────────────────────────────────────

    def _build_transform_group(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Transformation & Normalisation")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        grid.addWidget(QLabel("Transformation :"), 0, 0)
        self.combo_transform = DarkComboBox()
        self.combo_transform.addItems(["logicle", "arcsinh", "log10", "none"])
        self.combo_transform.setToolTip(
            "Logicle (recommandé ELN) · Arcsinh · Log10 · Aucune"
        )
        grid.addWidget(self.combo_transform, 0, 1)

        grid.addWidget(QLabel("Cofacteur (arcsinh) :"), 1, 0)
        self.spin_cofactor = QDoubleSpinBox()
        self.spin_cofactor.setRange(1.0, 500.0)
        self.spin_cofactor.setValue(5.0)
        self.spin_cofactor.setDecimals(1)
        self.spin_cofactor.setToolTip("Cofacteur arcsinh (5 pour flow, 150 pour CyTOF)")
        grid.addWidget(self.spin_cofactor, 1, 1)

        grid.addWidget(QLabel("Normalisation :"), 2, 0)
        self.combo_normalize = DarkComboBox()
        self.combo_normalize.addItems(["zscore", "minmax", "none"])
        grid.addWidget(self.combo_normalize, 2, 1)

        layout.addWidget(group)

    # ── Gating ─────────────────────────────────────────────────────────

    def _build_gating_group(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Pré-gating automatique")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)

        self.chk_pregate = QCheckBox("Activer le pré-gating")
        self.chk_pregate.setChecked(True)
        vbox.addWidget(self.chk_pregate)

        grid = QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(QLabel("Mode :"), 0, 0)
        self.combo_gate_mode = DarkComboBox()
        self.combo_gate_mode.addItems(["auto", "manual"])
        self.combo_gate_mode.setToolTip("auto = GMM/RANSAC · manual = percentiles")
        grid.addWidget(self.combo_gate_mode, 0, 1)

        self.chk_viable = QCheckBox("Débris (FSC/SSC)")
        self.chk_viable.setChecked(True)
        grid.addWidget(self.chk_viable, 1, 0)

        self.chk_singlets = QCheckBox("Doublets (FSC-H/FSC-A)")
        self.chk_singlets.setChecked(True)
        grid.addWidget(self.chk_singlets, 1, 1)

        self.chk_cd45 = QCheckBox("CD45 dim")
        grid.addWidget(self.chk_cd45, 2, 0)

        self.chk_cd34 = QCheckBox("CD34+ blastes")
        grid.addWidget(self.chk_cd34, 2, 1)

        vbox.addLayout(grid)
        layout.addWidget(group)

    # ── Options supplémentaires ────────────────────────────────────────

    def _build_options_group(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Options")
        grid = QGridLayout(group)
        grid.setSpacing(6)

        self.chk_umap = QCheckBox("Calculer UMAP")
        self.chk_umap.setToolTip("Réduction dimensionnelle UMAP après clustering")
        grid.addWidget(self.chk_umap, 0, 0)

        self.chk_gpu = QCheckBox("GPU (CUDA)")
        self.chk_gpu.setChecked(True)
        self.chk_gpu.setToolTip("Utiliser GPUFlowSOMEstimator si disponible")
        grid.addWidget(self.chk_gpu, 0, 1)

        self.chk_compare = QCheckBox("Mode comparaison Sain vs Patho")
        self.chk_compare.setChecked(True)
        grid.addWidget(self.chk_compare, 1, 0, 1, 2)

        self.chk_pop_mapping = QCheckBox("Mapping populations (Ref MFI)")
        self.chk_pop_mapping.setToolTip("Mapping automatique via MFI de référence ELN")
        grid.addWidget(self.chk_pop_mapping, 2, 0, 1, 2)

        self.chk_downsampling = QCheckBox("Downsampling")
        self.chk_downsampling.setToolTip("Limiter le nombre de cellules analysées")
        grid.addWidget(self.chk_downsampling, 3, 0)

        self.spin_max_cells = QSpinBox()
        self.spin_max_cells.setRange(1000, 5_000_000)
        self.spin_max_cells.setSingleStep(10000)
        self.spin_max_cells.setValue(50000)
        self.spin_max_cells.setSuffix(" cellules/fichier")
        grid.addWidget(self.spin_max_cells, 3, 1)

        layout.addWidget(group)

    # ── Section Run ────────────────────────────────────────────────────

    def _build_run_section(self, layout: QVBoxLayout) -> None:
        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% — Pipeline FlowSOM")
        layout.addWidget(self.progress_bar)

        # Bouton Run
        self.btn_run = QPushButton("Lancer le Pipeline")
        self.btn_run.setObjectName("primaryBtn")
        self.btn_run.setMinimumHeight(50)
        self.btn_run.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.btn_run.clicked.connect(self._run_pipeline)
        # Ombre portée
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(137, 180, 250, 80))
        shadow.setOffset(0, 4)
        self.btn_run.setGraphicsEffect(shadow)
        layout.addWidget(self.btn_run)

        # Bouton Stop
        self.btn_stop = QPushButton("Arrêter")
        self.btn_stop.setObjectName("dangerBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_pipeline)
        layout.addWidget(self.btn_stop)

    # ── Section Export ─────────────────────────────────────────────────

    def _build_export_section(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Exports")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)

        btn_fcs = QPushButton("Export FCS (Kaluza)")
        btn_fcs.setObjectName("exportBtn")
        btn_fcs.clicked.connect(self._export_fcs)
        btn_fcs.setToolTip("Exporte le fichier FCS enrichi compatible Kaluza/FlowJo")
        vbox.addWidget(btn_fcs)

        btn_csv = QPushButton("Export CSV")
        btn_csv.setObjectName("exportBtn")
        btn_csv.clicked.connect(self._export_csv)
        vbox.addWidget(btn_csv)

        btn_report = QPushButton("Rapport HTML")
        btn_report.setObjectName("successBtn")
        btn_report.clicked.connect(self._open_html_report)
        btn_report.setToolTip("Ouvre le rapport HTML interactif généré par le pipeline")
        vbox.addWidget(btn_report)

        btn_folder = QPushButton("Ouvrir dossier résultats")
        btn_folder.clicked.connect(self._open_output_folder)
        vbox.addWidget(btn_folder)

        layout.addWidget(group)

    # ── Onglet Visualisation ───────────────────────────────────────────

    def _build_viz_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        # Sélecteur de figure + boutons
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

        btn_refresh = QPushButton("Rafraîchir")
        btn_refresh.clicked.connect(self._refresh_current_plot)
        selector_layout.addWidget(btn_refresh)

        btn_browser = QPushButton("Ouvrir dans le navigateur")
        btn_browser.setObjectName("successBtn")
        btn_browser.clicked.connect(self._open_current_plot_browser)
        selector_layout.addWidget(btn_browser)

        layout.addLayout(selector_layout)

        # Zone empilée : canvas matplotlib / QWebEngineView
        self._viz_stack = QStackedWidget()

        # Page 0 : canvas PNG + toolbar
        png_widget = QWidget()
        png_layout = QVBoxLayout(png_widget)
        png_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MatplotlibCanvas(tab, width=10, height=7)
        self.toolbar = NavigationToolbar(self.canvas, tab)
        self.toolbar.setStyleSheet(
            "background: rgba(49,50,68,0.8); border-radius: 8px; padding: 4px;"
        )
        png_layout.addWidget(self.toolbar)
        png_layout.addWidget(self.canvas, 1)
        self._viz_stack.addWidget(png_widget)

        # Page 1 : QWebEngineView ou placeholder
        if _WEBENGINE:
            self._web_view = QWebEngineView()
            # Corrige le fond blanc/transparent dû à QWidget{background:transparent} dans le QSS
            self._web_view.page().setBackgroundColor(QColor(30, 30, 46))  # #1e1e2e
            self._viz_stack.addWidget(self._web_view)
        else:
            html_placeholder = QLabel(
                "QWebEngineView non disponible.\n"
                "Installez PyQtWebEngine : pip install PyQtWebEngine\n"
                "Utilisez le bouton 'Ouvrir dans le navigateur' pour les figures interactives."
            )
            html_placeholder.setAlignment(Qt.AlignCenter)
            html_placeholder.setObjectName("subtitleLabel")
            self._web_view = None
            self._viz_stack.addWidget(html_placeholder)

        layout.addWidget(self._viz_stack, 1)
        self.tabs.addTab(tab, "Visualisation")

    # ── Onglet Prégating ───────────────────────────────────────────────

    def _build_pregate_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        # Tableau des évènements de gating
        lbl = QLabel("Rapport de Prégating")
        lbl.setObjectName("sectionLabel")
        layout.addWidget(lbl)

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
        self.gate_table.setMaximumHeight(250)
        layout.addWidget(self.gate_table)

        # Sélecteur de figure de gating
        gate_selector = QHBoxLayout()
        gate_selector.addWidget(QLabel("Figure de gating :"))
        self.combo_gate_plot = DarkComboBox()
        self.combo_gate_plot.addItems(
            [
                "Vue d'ensemble",
                "Gate Débris",
                "Gate Doublets",
                "Gate CD45",
                "Gate CD34+",
            ]
        )
        self.combo_gate_plot.currentIndexChanged.connect(self._on_gate_plot_changed)
        gate_selector.addWidget(self.combo_gate_plot, 1)
        layout.addLayout(gate_selector)

        # Canvas pour les figures de gating
        self.gate_canvas = MatplotlibCanvas(tab, width=10, height=5)
        gate_toolbar = NavigationToolbar(self.gate_canvas, tab)
        gate_toolbar.setStyleSheet(
            "background: rgba(49,50,68,0.8); border-radius: 8px; padding: 4px;"
        )
        layout.addWidget(gate_toolbar)
        layout.addWidget(self.gate_canvas, 1)

        # Mapping nom combo → clé fichier attendue
        self._gate_plot_keys = {
            "Vue d'ensemble": ["fig_overview", "overview"],
            "Gate Débris": ["fig_gate_debris", "gate_debris", "debris"],
            "Gate Doublets": ["fig_gate_singlets", "gate_singlets", "singlets"],
            "Gate CD45": ["fig_gate_cd45", "gate_cd45", "cd45"],
            "Gate CD34+": ["fig_gate_cd34", "gate_cd34", "cd34"],
        }
        self._gate_plot_paths: Dict[str, str] = {}

        self.tabs.addTab(tab, "Prégating")

    # ── Onglet Clusters ────────────────────────────────────────────────

    def _build_clusters_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        # Panneau gauche : liste clusters + sélection marqueurs
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

        # Sélection des marqueurs pour le spider
        lbl_markers = QLabel("Marqueurs pour Spider Plot")
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

        btn_spider = QPushButton("Générer Spider Plot")
        btn_spider.setObjectName("primaryBtn")
        btn_spider.clicked.connect(self._generate_spider_plot)
        left.addWidget(btn_spider)

        layout.addLayout(left)

        # Canvas pour le spider plot
        self.star_canvas = MatplotlibCanvas(tab, width=7, height=7)
        layout.addWidget(self.star_canvas, 1)

        self.tabs.addTab(tab, "Clusters")

    # ── Onglet Résultats ───────────────────────────────────────────────

    def _build_results_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        # Tableau des clusters (SOM nodes)
        hdr = QHBoxLayout()
        lbl = QLabel("Statistiques par Cluster (nœuds SOM)")
        lbl.setObjectName("sectionLabel")
        hdr.addWidget(lbl)
        hdr.addStretch()
        btn_export_txt = QPushButton("Exporter Clusters .txt")
        btn_export_txt.setObjectName("exportBtn")
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

        # Vue combinée nœuds SOM interactive
        hdr2 = QHBoxLayout()
        lbl2 = QLabel("Vue Combinée Nœuds SOM (interactive)")
        lbl2.setObjectName("subtitleLabel")
        hdr2.addWidget(lbl2)
        hdr2.addStretch()
        self.btn_open_combined = QPushButton("Ouvrir dans navigateur")
        self.btn_open_combined.setObjectName("successBtn")
        self.btn_open_combined.setEnabled(False)
        self.btn_open_combined.clicked.connect(self._open_combined_html)
        hdr2.addWidget(self.btn_open_combined)
        layout.addLayout(hdr2)

        if _WEBENGINE:
            self._results_web = QWebEngineView()
            self._results_web.page().setBackgroundColor(QColor(30, 30, 46))  # #1e1e2e
            layout.addWidget(self._results_web, 1)
        else:
            self._results_web = None
            placeholder = QLabel(
                "Installez PyQtWebEngine pour afficher la vue interactive ici.\n"
                "Utilisez le bouton ci-dessus pour l'ouvrir dans le navigateur."
            )
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setObjectName("subtitleLabel")
            layout.addWidget(placeholder, 1)

        # Résumé textuel en bas
        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setMaximumHeight(150)
        self.txt_summary.setPlaceholderText(
            "Le résumé de l'analyse apparaîtra ici après exécution…"
        )
        layout.addWidget(self.txt_summary)

        self._combined_html_path: Optional[str] = None

        self.tabs.addTab(tab, "Résultats")

    # ── Onglet Logs ────────────────────────────────────────────────────

    def _build_logs_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(5, 5, 5, 5)

        header = QHBoxLayout()
        lbl = QLabel("Console Pipeline")
        lbl.setObjectName("sectionLabel")
        header.addWidget(lbl)
        header.addStretch()

        btn_clear = QPushButton("Effacer")
        btn_clear.clicked.connect(lambda: self.log_output.clear())
        header.addWidget(btn_clear)

        btn_copy = QPushButton("Copier")
        btn_copy.clicked.connect(
            lambda: QApplication.clipboard().setText(self.log_output.toPlainText())
        )
        header.addWidget(btn_copy)

        layout.addLayout(header)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Les logs du pipeline apparaîtront ici…")
        layout.addWidget(self.log_output)

        self.tabs.addTab(tab, "Logs")

    def _build_fcs_viewer_tab(self) -> None:
        """Onglet Visualisation FCS — style Kaluza (scatter / densité / contour)."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Barre de contrôle ─────────────────────────────────────────
        ctrl = QWidget()
        ctrl_layout = QHBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(8)

        self.btn_load_fcs_viz = QPushButton("Charger FCS")
        self.btn_load_fcs_viz.setObjectName("primaryBtn")
        self.btn_load_fcs_viz.setCursor(Qt.PointingHandCursor)
        self.btn_load_fcs_viz.clicked.connect(self._load_fcs_for_visualization)
        ctrl_layout.addWidget(self.btn_load_fcs_viz)

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
        self.combo_fcs_color.setToolTip(
            "Colorier les points par cluster ou métacluster"
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
        self.chk_fcs_all_cells.setToolTip(
            "Afficher toutes les cellules (peut être lent)"
        )
        self.chk_fcs_all_cells.stateChanged.connect(self._toggle_fcs_all_cells)
        ctrl_layout.addWidget(self.chk_fcs_all_cells)

        self.chk_fcs_jitter = QCheckBox("Jitter")
        self.chk_fcs_jitter.setToolTip(
            "Dispersion circulaire pour coordonnées SOM (xGrid/yGrid/xNodes/yNodes)"
        )
        self.chk_fcs_jitter.setChecked(True)
        self.chk_fcs_jitter.stateChanged.connect(self._update_fcs_plot)
        ctrl_layout.addWidget(self.chk_fcs_jitter)

        btn_refresh = QPushButton("Rafraichir")
        btn_refresh.setCursor(Qt.PointingHandCursor)
        btn_refresh.clicked.connect(self._update_fcs_plot)
        ctrl_layout.addWidget(btn_refresh)

        ctrl_layout.addStretch()
        layout.addWidget(ctrl)

        # ── Canvas matplotlib + barre de navigation ───────────────────
        self.fcs_viz_canvas = MatplotlibCanvas(tab, width=10, height=8)
        self.fcs_viz_canvas.setMinimumHeight(480)
        fcs_toolbar = NavigationToolbar(self.fcs_viz_canvas, tab)
        layout.addWidget(fcs_toolbar)
        layout.addWidget(self.fcs_viz_canvas)

        # ── Info cellules ─────────────────────────────────────────────
        self.lbl_fcs_info = QLabel("Chargez un fichier FCS pour visualiser")
        self.lbl_fcs_info.setStyleSheet("color: #a6adc8; padding: 4px;")
        layout.addWidget(self.lbl_fcs_info)

        self.tabs.addTab(tab, "Visualisation FCS")

    # ==================================================================
    # LOGIQUE : Chargement config
    # ==================================================================

    def _load_default_config(self) -> None:
        """
        Charge le YAML par défaut et synchronise les widgets."""
        try:
            from flowsom_pipeline_pro.config.pipeline_config import PipelineConfig

            if _DEFAULT_CONFIG_PATH.exists():
                self._config = PipelineConfig.from_yaml(str(_DEFAULT_CONFIG_PATH))
                self._sync_config_to_ui()
                self._log(f" Config chargée : {_DEFAULT_CONFIG_PATH.name}")
            else:
                self._config = PipelineConfig()
                self._log(
                    " Config par défaut introuvable, utilisation des valeurs par défaut"
                )
        except Exception as e:
            self._log(f" Erreur chargement config : {e}")

    def _sync_config_to_ui(self) -> None:
        """
        Pousse les valeurs de PipelineConfig vers les widgets."""
        c = self._config
        if c is None:
            return

        # Paths
        if hasattr(c, "paths"):
            if c.paths.healthy_folder:
                self.lbl_healthy.setText(str(c.paths.healthy_folder))
            if c.paths.patho_folder:
                self.lbl_patho.setText(str(c.paths.patho_folder))
            if c.paths.output_dir:
                self.lbl_output.setText(str(c.paths.output_dir))

        # FlowSOM
        if hasattr(c, "flowsom"):
            self.spin_xdim.setValue(c.flowsom.xdim)
            self.spin_ydim.setValue(c.flowsom.ydim)
            self.spin_metaclusters.setValue(c.flowsom.n_metaclusters)
            self.spin_seed.setValue(c.flowsom.seed)
            self.spin_lr.setValue(c.flowsom.learning_rate)
            self.spin_sigma.setValue(c.flowsom.sigma)

        # Transform
        if hasattr(c, "transform"):
            idx = self.combo_transform.findText(c.transform.method)
            if idx >= 0:
                self.combo_transform.setCurrentIndex(idx)
            self.spin_cofactor.setValue(c.transform.cofactor)

        # Normalize
        if hasattr(c, "normalize"):
            idx = self.combo_normalize.findText(c.normalize.method)
            if idx >= 0:
                self.combo_normalize.setCurrentIndex(idx)

        # Pregate
        if hasattr(c, "pregate"):
            self.chk_pregate.setChecked(c.pregate.apply)
            idx = self.combo_gate_mode.findText(c.pregate.mode)
            if idx >= 0:
                self.combo_gate_mode.setCurrentIndex(idx)
            self.chk_viable.setChecked(c.pregate.viable)
            self.chk_singlets.setChecked(c.pregate.singlets)
            self.chk_cd45.setChecked(c.pregate.cd45)
            self.chk_cd34.setChecked(c.pregate.cd34)

        # Options
        if hasattr(c, "visualization"):
            self.chk_umap.setChecked(c.visualization.umap_enabled)
        if hasattr(c, "gpu"):
            self.chk_gpu.setChecked(c.gpu.enabled)
        if hasattr(c, "analysis"):
            self.chk_compare.setChecked(c.analysis.compare_mode)
        if hasattr(c, "auto_clustering"):
            self.chk_auto_clustering.setChecked(c.auto_clustering.enabled)
        if hasattr(c, "population_mapping"):
            self.chk_pop_mapping.setChecked(c.population_mapping.enabled)
        if hasattr(c, "downsampling"):
            self.chk_downsampling.setChecked(c.downsampling.enabled)
            self.spin_max_cells.setValue(c.downsampling.max_cells_per_file)

    def _sync_ui_to_config(self) -> None:
        """
        Pousse les valeurs des widgets vers PipelineConfig."""
        c = self._config
        if c is None:
            return

        # Paths
        healthy = self.lbl_healthy.text()
        if healthy and healthy != "Non sélectionné":
            c.paths.healthy_folder = healthy
        patho = self.lbl_patho.text()
        if patho and patho != "Non sélectionné":
            c.paths.patho_folder = patho
        output = self.lbl_output.text()
        if output and output != "Non sélectionné":
            c.paths.output_dir = output

        # FlowSOM
        c.flowsom.xdim = self.spin_xdim.value()
        c.flowsom.ydim = self.spin_ydim.value()
        c.flowsom.n_metaclusters = self.spin_metaclusters.value()
        c.flowsom.seed = self.spin_seed.value()
        c.flowsom.learning_rate = self.spin_lr.value()
        c.flowsom.sigma = self.spin_sigma.value()

        # Transform
        c.transform.method = self.combo_transform.currentText()
        c.transform.cofactor = self.spin_cofactor.value()

        # Normalize
        c.normalize.method = self.combo_normalize.currentText()

        # Pregate
        c.pregate.apply = self.chk_pregate.isChecked()
        c.pregate.mode = self.combo_gate_mode.currentText()
        c.pregate.viable = self.chk_viable.isChecked()
        c.pregate.singlets = self.chk_singlets.isChecked()
        c.pregate.cd45 = self.chk_cd45.isChecked()
        c.pregate.cd34 = self.chk_cd34.isChecked()

        # Options
        c.visualization.umap_enabled = self.chk_umap.isChecked()
        c.gpu.enabled = self.chk_gpu.isChecked()
        c.analysis.compare_mode = self.chk_compare.isChecked()
        c.auto_clustering.enabled = self.chk_auto_clustering.isChecked()
        c.population_mapping.enabled = self.chk_pop_mapping.isChecked()
        c.downsampling.enabled = self.chk_downsampling.isChecked()
        c.downsampling.max_cells_per_file = self.spin_max_cells.value()

    # ==================================================================
    # LOGIQUE : Sélection dossiers
    # ==================================================================

    def _select_healthy_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Sélectionner le dossier NBM / Sain"
        )
        if path:
            self.lbl_healthy.setText(path)

    def _select_patho_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Sélectionner le dossier Pathologique"
        )
        if path:
            self.lbl_patho.setText(path)

    def _select_output_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Sélectionner le dossier de sortie"
        )
        if path:
            self.lbl_output.setText(path)

    # ==================================================================
    # LOGIQUE : Exécution du pipeline
    # ==================================================================

    def _run_pipeline(self) -> None:
        """
        Lance le pipeline dans un QThread."""
        if self._config is None:
            QMessageBox.warning(self, "Erreur", "Aucune configuration chargée.")
            return

        # Vérifie les dossiers
        healthy = self.lbl_healthy.text()
        patho = self.lbl_patho.text()
        if not healthy or healthy == "Non sélectionné":
            QMessageBox.warning(
                self, "Erreur", "Veuillez sélectionner le dossier NBM / Sain."
            )
            return
        if not Path(healthy).is_dir():
            QMessageBox.warning(self, "Erreur", f"Dossier NBM introuvable :\n{healthy}")
            return

        if self.chk_compare.isChecked():
            if not patho or patho == "Non sélectionné":
                QMessageBox.warning(
                    self,
                    "Erreur",
                    "Mode comparaison actif : sélectionnez le dossier Patho.",
                )
                return
            if not Path(patho).is_dir():
                QMessageBox.warning(
                    self, "Erreur", f"Dossier Patho introuvable :\n{patho}"
                )
                return

        # Synchronise UI → config
        self._sync_ui_to_config()

        # Prépare l'UI
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_output.clear()
        self.tabs.setCurrentIndex(4)  # Onglet Logs (tab 4 après ajout prégating)

        self._log("═══════════════════════════════════════════════")
        self._log(
            f"Pipeline FlowSOM Pro — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._log(f"Grille : {self._config.flowsom.xdim}×{self._config.flowsom.ydim}")
        self._log(f"Métaclusters : {self._config.flowsom.n_metaclusters}")
        self._log(f"Transformation : {self._config.transform.method}")
        self._log(f"GPU : {'Oui' if self._config.gpu.enabled else 'Non'}")
        self._log("═══════════════════════════════════════════════")

        # Lance le worker
        self._worker = PipelineWorker(self._config, parent=self)
        self._worker.log_message.connect(self._on_log_message)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.start()
        self.statusBar().showMessage(" Pipeline en cours d'exécution…")

    def _stop_pipeline(self) -> None:
        """
        Demande l'arrêt du thread pipeline."""
        if self._worker is not None and self._worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirmation",
                "Voulez-vous interrompre le pipeline en cours ?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._worker.terminate()
                self._worker.wait(3000)
                self._log(" Pipeline interrompu par l'utilisateur")
                self.btn_run.setEnabled(True)
                self.btn_stop.setEnabled(False)
                self.statusBar().showMessage("Pipeline interrompu")

    # ── Slots du worker ────────────────────────────────────────────────

    def _on_log_message(self, msg: str) -> None:
        self.log_output.append(msg)
        # Auto-scroll vers le bas
        sb = self.log_output.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    def _on_pipeline_finished(self, result: Any) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._result = result

        if result is not None and result.success:
            self.progress_bar.setValue(100)
            self.statusBar().showMessage(
                f" Terminé — {result.n_cells:,} cellules, "
                f"{result.n_metaclusters} métaclusters, "
                f"{result.elapsed_seconds:.1f}s"
            )
            self._populate_results(result)
            self._populate_cluster_list(result)
            self._populate_pregate_tab(result)
            self._load_output_plots(result)
            self.tabs.setCurrentIndex(0)  # Onglet Visualisation
        else:
            self.statusBar().showMessage(" Pipeline terminé avec des erreurs")
            self._log("═══ Pipeline terminé avec des erreurs — vérifiez les logs ═══")

    def _on_pipeline_error(self, msg: str) -> None:
        self.statusBar().showMessage(f" Erreur : {msg[:80]}")

    # ==================================================================
    # LOGIQUE : Affichage des résultats
    # ==================================================================

    def _populate_results(self, result: Any) -> None:
        """
        Remplit le tableau des clusters (nœuds SOM) et le résumé."""
        # Résumé textuel
        try:
            self.txt_summary.setPlainText(result.summary())
        except Exception:
            self.txt_summary.setPlainText(f"Cellules : {result.n_cells:,}")

        # Tableau cluster (SOM nodes)
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

                # Métacluster associé (mode)
                if has_mc and has_cluster:
                    mc_mode = df[df[group_col] == cl_id]["FlowSOM_metacluster"].mode()
                    mc_val = str(int(mc_mode.iloc[0])) if len(mc_mode) > 0 else "?"
                else:
                    mc_val = str(int(cl_id))
                self.results_table.setItem(i, 1, QTableWidgetItem(mc_val))

                self.results_table.setItem(i, 2, QTableWidgetItem(f"{count:,}"))
                pct = count / total * 100 if total > 0 else 0
                self.results_table.setItem(i, 3, QTableWidgetItem(f"{pct:.1f}%"))

                # % Pathologique
                if has_cond:
                    sub = df[df[group_col] == cl_id]
                    n_patho = (
                        sub["condition"]
                        .str.lower()
                        .str.contains("patho|pathologique", na=False)
                    ).sum()
                    pct_patho = n_patho / count * 100 if count > 0 else 0
                    self.results_table.setItem(
                        i, 4, QTableWidgetItem(f"{pct_patho:.1f}%")
                    )
                else:
                    self.results_table.setItem(i, 4, QTableWidgetItem("N/A"))

        except Exception as e:
            self._log(f"Erreur tableau résultats : {e}")

    def _export_cluster_txt(self) -> None:
        """
        Exporte les statistiques des clusters en fichier .txt."""
        if self._result is None or not self._result.success:
            QMessageBox.information(
                self, "Info", "Aucun résultat disponible. Lancez le pipeline."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Exporter clusters en .txt",
            "clusters_stats.txt",
            "Text files (*.txt)",
        )
        if not path:
            return

        try:
            import pandas as pd
            import numpy as np

            df = self._result.data
            lines = []
            lines.append("=" * 70)
            lines.append("STATISTIQUES DES CLUSTERS (NŒUDS SOM) — FlowSOM Analyzer Pro")
            lines.append("=" * 70)
            lines.append(f"  Analyse  : {self._result.timestamp}")
            lines.append(f"  Cellules : {self._result.n_cells:,}")
            lines.append(f"  Métriques clustering :")
            if self._result.clustering_metrics.silhouette_score is not None:
                lines.append(
                    f"    Silhouette : {self._result.clustering_metrics.silhouette_score:.4f}"
                )
            lines.append("")

            group_col = (
                "FlowSOM_cluster"
                if "FlowSOM_cluster" in df.columns
                else "FlowSOM_metacluster"
            )
            has_mc = (
                "FlowSOM_metacluster" in df.columns and group_col == "FlowSOM_cluster"
            )
            has_cond = "condition" in df.columns
            total = len(df)

            lines.append(
                f"{'Cluster':>10}  {'Métacluster':>12}  {'Cellules':>10}  {'% total':>8}  {'% patho':>8}"
            )
            lines.append("-" * 70)

            counts = df[group_col].value_counts().sort_index()
            for cl_id, count in counts.items():
                pct = count / total * 100 if total > 0 else 0

                if has_mc:
                    mc_mode = df[df[group_col] == cl_id]["FlowSOM_metacluster"].mode()
                    mc_val = str(int(mc_mode.iloc[0])) if len(mc_mode) > 0 else "?"
                else:
                    mc_val = str(int(cl_id))

                if has_cond:
                    sub = df[df[group_col] == cl_id]
                    n_patho = (
                        sub["condition"]
                        .str.lower()
                        .str.contains("patho|pathologique", na=False)
                    ).sum()
                    pct_patho = n_patho / count * 100 if count > 0 else 0
                    patho_str = f"{pct_patho:.1f}%"
                else:
                    patho_str = "N/A"

                lines.append(
                    f"{int(cl_id):>10}  {mc_val:>12}  {count:>10,}  {pct:>7.1f}%  {patho_str:>8}"
                )

            lines.append("")
            if self._result.gating_report:
                lines.append("GATING REPORT")
                lines.append("-" * 70)
                for ev in self._result.gating_report:
                    gate = ev.get("gate_name", ev.get("gate", "?"))
                    nb = ev.get("n_before", ev.get("n_total", "?"))
                    na = ev.get("n_after", ev.get("n_kept", "?"))
                    pct_k = ev.get("pct_kept", "?")
                    mode = ev.get("method", ev.get("mode", "?"))
                    lines.append(
                        f"  [{gate}]  {nb} → {na}  ({pct_k:.1f}%)  mode={mode}"
                    )

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            self._log(f"Clusters exportés : {path}")
            QMessageBox.information(self, "Export réussi", f"Fichier exporté :\n{path}")

        except Exception as e:
            QMessageBox.critical(self, "Erreur export", str(e))

    def _open_combined_html(self) -> None:
        """
        Ouvre le fichier som_node_combined dans le navigateur système."""
        if self._combined_html_path and Path(self._combined_html_path).exists():
            webbrowser.open(str(Path(self._combined_html_path).resolve()))
        else:
            QMessageBox.information(self, "Info", "Vue combinée non disponible.")

    def _populate_cluster_list(self, result: Any) -> None:
        """
        Peuple la liste des nœuds SOM (clusters) dans l'onglet Clusters."""
        self.cluster_list.clear()
        self.marker_list.clear()
        self._cluster_mfi = None
        self._all_markers = []

        if result is None or result.data is None:
            return

        df = result.data
        try:
            import pandas as pd
            import numpy as np
            import traceback as tb

            if "FlowSOM_cluster" not in df.columns:
                self._log("Avertissement : colonne FlowSOM_cluster absente du résultat")
                return

            # Colonnes non-marqueurs à exclure (inclut les colonnes string du pipeline)
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
                # Colonnes ajoutées par build_cells_dataframe (string ou int non-marqueur)
                "Condition",
                "Timepoint",
                "Timepoint_Num",
            }
            # Uniquement les colonnes numériques, hors métadonnées — évite float(string.mean())
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            marker_cols = [c for c in numeric_cols if c not in _meta_cols]
            self._all_markers = marker_cols

            # Remplit la liste des marqueurs
            for m in marker_cols:
                item = QListWidgetItem(m)
                self.marker_list.addItem(item)
            self.marker_list.selectAll()  # sélectionne tout par défaut

            mc_col = (
                "FlowSOM_metacluster" if "FlowSOM_metacluster" in df.columns else None
            )

            # Calcul vectorisé MFI par nœud SOM (évite les boucles Python sur 200k+ cellules)
            mfi_df = df.groupby("FlowSOM_cluster")[marker_cols].mean()
            mfi_df["n_cells"] = df.groupby("FlowSOM_cluster").size()
            if mc_col:
                mfi_df["metacluster"] = (
                    df.groupby("FlowSOM_cluster")[mc_col]
                    .agg(lambda x: int(x.mode().iloc[0]) if len(x) > 0 else -1)
                    .astype(int)
                )

            self._cluster_mfi = mfi_df

            # Peuple la liste UI
            total_cells = len(df)
            for cl_id, row_data in mfi_df.iterrows():
                mc = (
                    int(row_data["metacluster"])
                    if "metacluster" in row_data.index
                    else -1
                )
                n = int(row_data["n_cells"])
                pct = n / total_cells * 100 if total_cells > 0 else 0
                label_text = f"Cluster {cl_id}  (MC{mc})  — {n:,} cell. ({pct:.1f}%)"
                item = QListWidgetItem(label_text)
                self.cluster_list.addItem(item)

            # Force le repaint pour s'assurer que les items sont visibles
            self.cluster_list.update()

            # Sélection automatique du premier nœud → affiche le spider plot immédiatement
            if self.cluster_list.count() > 0:
                self.cluster_list.setCurrentRow(0)

        except Exception as e:
            import traceback

            self._log(f"Erreur peuplement clusters : {e}\n{traceback.format_exc()}")

    def _on_cluster_selected(self, row: int) -> None:
        """Génère automatiquement le spider plot du nœud sélectionné."""
        if row >= 0 and self._cluster_mfi is not None:
            self._generate_spider_plot()

    def _generate_spider_plot(self) -> None:
        """
        Génère un Spider Plot pour le cluster sélectionné avec les marqueurs cochés."""
        # Arrête le worker précédent s'il tourne encore (clic rapide sur un autre nœud)
        if self._spider_worker is not None and self._spider_worker.isRunning():
            self._spider_worker.terminate()
            self._spider_worker.wait(200)

        row = self.cluster_list.currentRow()
        if row < 0 or self._cluster_mfi is None:
            QMessageBox.information(
                self, "Info", "Sélectionnez un cluster dans la liste."
            )
            return

        # Marqueurs sélectionnés
        selected_markers = [
            self.marker_list.item(i).text()
            for i in range(self.marker_list.count())
            if self.marker_list.item(i).isSelected()
        ]
        if not selected_markers:
            QMessageBox.warning(
                self, "Avertissement", "Sélectionnez au moins un marqueur."
            )
            return

        if len(selected_markers) < 3:
            QMessageBox.warning(
                self,
                "Avertissement",
                "Sélectionnez au moins 3 marqueurs pour un Spider Plot.",
            )
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
        self._spider_worker.error.connect(
            lambda msg: self._log(f"Spider erreur : {msg}")
        )
        self._spider_worker.start()

    def _on_spider_ready(self, fig: Any) -> None:
        """
        Affiche le Spider Plot dans le canvas Clusters."""
        self.star_canvas.display_figure(fig)

    # ==================================================================
    # LOGIQUE : Chargement / affichage des plots générés
    # ==================================================================

    # Correspondance fragment de nom de fichier → label du combo
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
        """
        Remonte au dossier output depuis les chemins de result.output_files."""
        if result is None or not result.output_files:
            return None
        for v in result.output_files.values():
            if v and Path(v).exists():
                p = Path(v)
                # Les fichiers sont dans output_dir/{csv,fcs,other,plots}
                candidate = p.parent.parent
                if (candidate / "plots").is_dir():
                    return candidate
                if p.parent.is_dir():
                    return p.parent
        return None

    def _load_output_plots(self, result: Any) -> None:
        """
        Scanne le dossier output/plots pour tous les fichiers PNG et HTML."""
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

        # Collecte tous les PNG et HTML récursivement
        all_files = list(plots_dir.rglob("*.png")) + list(plots_dir.rglob("*.html"))

        for file_path in all_files:
            fname = file_path.name.lower()
            # Ignore les fichiers par-fichier dans le sous-dossier sankey/per_file
            if "per_file" in str(file_path).lower():
                continue

            # Gating plots → prégating tab
            if "gating" in str(file_path.parent).lower():
                for label, keys in self._gate_plot_keys.items():
                    if any(
                        k.replace("fig_", "").replace("_", "") in fname.replace("_", "")
                        for k in keys
                    ):
                        if label not in self._gate_plot_paths:
                            self._gate_plot_paths[label] = str(file_path)
                        break
                continue

            # Mappe par fragment de nom
            for fragment, label in self._PLOT_FILENAME_MAP:
                if fragment in fname:
                    if label not in self._output_plot_paths:
                        self._output_plot_paths[label] = str(file_path)
                    # Détecte le fichier som_node_combined
                    if "som_node_combined" in fname:
                        self._combined_html_path = str(file_path)
                    break

        # Active le bouton "Vue combinée" si disponible
        if hasattr(self, "btn_open_combined"):
            self.btn_open_combined.setEnabled(bool(self._combined_html_path))

        # Charge la vue combinée dans QWebEngineView Résultats si disponible
        if self._combined_html_path and _WEBENGINE and self._results_web is not None:
            from PyQt5.QtCore import QUrl

            self._results_web.setUrl(QUrl.fromLocalFile(self._combined_html_path))

        # Charge les plots de gating dans le canvas prégating
        self._on_gate_plot_changed(self.combo_gate_plot.currentIndex())

        # Rafraîchit la visualisation principale
        self._refresh_current_plot()

    def _on_plot_selection_changed(self, index: int) -> None:
        self._refresh_current_plot()

    def _refresh_current_plot(self) -> None:
        """
        Affiche le plot sélectionné dans le combo (PNG → canvas, HTML → WebView)."""
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
        """
        Affiche un PNG dans le canvas matplotlib (page 0 du stack)."""
        self._viz_stack.setCurrentIndex(0)
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot(111)
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.axis("off")
            self.canvas.fig.patch.set_facecolor(COLORS["base"])
            self.canvas.fig.tight_layout(pad=0.5)
            self.canvas.draw()
        except Exception as e:
            self._show_placeholder(f"Erreur affichage : {e}")

    def _show_html_plot(self, path: str) -> None:
        """
        Affiche un HTML interactif dans QWebEngineView (page 1) ou ouvre dans le navigateur."""
        if _WEBENGINE and self._web_view is not None:
            from PyQt5.QtCore import QUrl

            self._viz_stack.setCurrentIndex(1)
            self._web_view.setUrl(QUrl.fromLocalFile(str(Path(path).resolve())))
        else:
            # Pas de WebEngine : ouvre dans le navigateur système automatiquement
            webbrowser.open(str(Path(path).resolve()))
            self._viz_stack.setCurrentIndex(0)
            self._show_placeholder(
                "Figure interactive (.html)\n"
                "Ouverture dans le navigateur système...\n"
                "(Installez PyQtWebEngine pour l'affichage intégré)"
            )

    def _show_placeholder(self, text: str) -> None:
        """
        Affiche un message centré dans le canvas."""
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
            color=COLORS["subtext"],
            style="italic",
            wrap=True,
        )
        self.canvas.axes.set_xlim(0, 1)
        self.canvas.axes.set_ylim(0, 1)
        self.canvas.axes.axis("off")
        self.canvas.draw()

    def _open_current_plot_browser(self) -> None:
        """
        Ouvre la figure courante dans le navigateur système."""
        label = self.combo_plot.currentText()
        path = self._output_plot_paths.get(label)
        if path and Path(path).exists():
            webbrowser.open(str(Path(path).resolve()))
        else:
            QMessageBox.information(self, "Info", f"Figure '{label}' non disponible.")

    # ── Prégating ──────────────────────────────────────────────────────

    def _populate_pregate_tab(self, result: Any) -> None:
        """
        Peuple le tableau de gating et les plots prégating."""
        if result is None:
            return

        # Tableau des événements
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

        # Le chargement des plots de gating est fait dans _load_output_plots
        self.tabs.setTabText(1, f"Prégating ({len(events)} gates)")

    def _on_gate_plot_changed(self, index: int) -> None:
        """
        Affiche le plot de gating sélectionné dans gate_canvas."""
        label = self.combo_gate_plot.currentText()
        path = (
            self._gate_plot_paths.get(label)
            if hasattr(self, "_gate_plot_paths")
            else None
        )

        if path and Path(path).exists():
            try:
                import matplotlib.image as mpimg

                self.gate_canvas.fig.clear()
                ax = self.gate_canvas.fig.add_subplot(111)
                img = mpimg.imread(path)
                ax.imshow(img)
                ax.axis("off")
                self.gate_canvas.fig.patch.set_facecolor(COLORS["base"])
                self.gate_canvas.fig.tight_layout(pad=0.3)
                self.gate_canvas.draw()
            except Exception as e:
                self._log(f"Erreur plot gating : {e}")
        else:
            self.gate_canvas.clear_and_reset()
            self.gate_canvas.axes.text(
                0.5,
                0.5,
                "Figure de gating non disponible",
                transform=self.gate_canvas.axes.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color=COLORS["subtext"],
                style="italic",
            )
            self.gate_canvas.axes.axis("off")
            self.gate_canvas.draw()

    # ==================================================================
    # LOGIQUE : Exports
    # ==================================================================

    def _export_fcs(self) -> None:
        if self._result is None or not self._result.success:
            QMessageBox.information(
                self, "Info", "Aucun résultat à exporter. Lancez d'abord le pipeline."
            )
            return

        output_files = self._result.output_files or {}
        fcs_path = output_files.get("fcs_kaluza") or output_files.get("fcs")
        if fcs_path and Path(fcs_path).exists():
            QMessageBox.information(
                self,
                "Export FCS",
                f"Fichier FCS déjà exporté par le pipeline :\n{fcs_path}",
            )
        else:
            try:
                path, _ = QFileDialog.getSaveFileName(
                    self, "Exporter FCS", "", "FCS Files (*.fcs)"
                )
                if path:
                    self._result.data.to_csv(path.replace(".fcs", ".csv"), index=False)
                    self._log(
                        f" Données exportées en CSV : {path.replace('.fcs', '.csv')}"
                    )
            except Exception as e:
                QMessageBox.critical(self, "Erreur", str(e))

    def _export_csv(self) -> None:
        if self._result is None or not self._result.success:
            QMessageBox.information(self, "Info", "Aucun résultat à exporter.")
            return

        output_files = self._result.output_files or {}
        csv_path = output_files.get("cells_csv") or output_files.get("csv")
        if csv_path and Path(csv_path).exists():
            QMessageBox.information(
                self,
                "Export CSV",
                f"Fichier CSV déjà exporté par le pipeline :\n{csv_path}",
            )
        else:
            try:
                path, _ = QFileDialog.getSaveFileName(
                    self, "Exporter CSV", "", "CSV Files (*.csv)"
                )
                if path and self._result.data is not None:
                    self._result.data.to_csv(path, index=False)
                    self._log(f" CSV exporté : {path}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", str(e))

    def _open_html_report(self) -> None:
        if self._result is None:
            QMessageBox.information(self, "Info", "Aucun résultat disponible.")
            return

        output_files = self._result.output_files or {}
        html_path = output_files.get("html_report")
        if html_path and Path(html_path).exists():
            webbrowser.open(str(Path(html_path).resolve()))
        else:
            QMessageBox.information(
                self, "Info", "Rapport HTML non trouvé dans les résultats."
            )

    def _open_output_folder(self) -> None:
        output = self.lbl_output.text()
        if output and output != "Non sélectionné" and Path(output).is_dir():
            os.startfile(output)
        elif self._result and self._result.output_files:
            # Tente de trouver le dossier de sortie depuis les fichiers exportés
            for v in self._result.output_files.values():
                if v and Path(v).exists():
                    os.startfile(str(Path(v).parent))
                    return
            QMessageBox.information(self, "Info", "Dossier de sortie non trouvé.")
        else:
            QMessageBox.information(self, "Info", "Aucun dossier de sortie configuré.")

    # ==================================================================
    # Visualisation FCS (style Kaluza)
    # ==================================================================

    def _toggle_fcs_all_cells(self, state: int) -> None:
        """Active/désactive le spinbox du nombre de cellules FCS."""
        self.spin_fcs_cells.setEnabled(state != Qt.Checked)
        self._update_fcs_plot()

    def _extract_fcs_names(self, file_path: str, n_channels: int) -> List[str]:
        """Extrait les noms de canaux depuis le segement TEXT du fichier FCS.

        Priorité : $PnS (nom marqueur/stain) > $PnN (nom paramètre court).
        Retourne une liste de longueur n_channels.
        """
        # Essayer flowio (plus simple)
        try:
            import flowio

            text = flowio.FlowData(file_path).text
            # flowio peut stocker les clés avec ou sans $, en majuscules ou minuscules
            # On normalise en créant un dict unifié en majuscules avec $
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

        # Fallback : lecture binaire du segment TEXT uniquement
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
                td[parts[j].strip().upper()] = (
                    parts[j + 1].strip() if j + 1 < len(parts) else ""
                )

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
        """Lecture binaire directe d'un fichier FCS — fallback robuste."""
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
            datatype = text_dict.get(
                "$DATATYPE", text_dict.get("DATATYPE", "F")
            ).upper()
            byteord = text_dict.get("$BYTEORD", text_dict.get("BYTEORD", "1,2,3,4"))

            if n_params == 0 or n_events == 0:
                raise ValueError(
                    f"Paramètres invalides : {n_params} params, {n_events} events"
                )

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
            else:  # 'I'
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

    def _load_fcs_for_visualization(self) -> None:
        """Ouvre un fichier FCS et peuple les combos axes/couleur."""
        import numpy as np

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Charger un fichier FCS", "", "FCS Files (*.fcs *.FCS)"
        )
        if not file_path:
            return

        try:
            self._log(f"Chargement FCS : {Path(file_path).name}")
            adata = None
            last_error: Optional[Exception] = None

            # Méthode 1 : flowsom
            try:
                import flowsom as fs

                adata = fs.io.read_FCS(file_path)
                self._log("Chargé avec flowsom")
            except Exception as e1:
                self._log(f"flowsom échoué : {str(e1)[:60]}")
                last_error = e1

            # Méthode 2 : flowio
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

            # Méthode 3 : fcsparser
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

            # Méthode 4 : lecture binaire directe
            if adata is None:
                try:
                    adata = self._read_fcs_binary(file_path)
                    self._log("Chargé avec lecture binaire directe")
                except Exception as e4:
                    self._log(f"Binaire échoué : {str(e4)[:60]}")
                    last_error = e4

            if adata is None:
                raise RuntimeError(
                    f"Impossible de charger le FCS. Dernière erreur : {last_error}"
                )

            # Toujours enrichir les noms depuis le TEXT du FCS ($PnS > $PnN)
            # pour corriger les noms génériques renvoyés par certaines librairies
            real_names = self._extract_fcs_names(file_path, adata.shape[1])
            try:
                adata.var_names = real_names
            except Exception:
                pass  # si l'AnnData refuse (noms dupliqués etc.), on garde l'existant

            self.current_fcs_adata = adata
            markers = list(adata.var_names)

            # Peupler combos axes
            for combo in (self.combo_fcs_x, self.combo_fcs_y):
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(markers)
                combo.blockSignals(False)

            # Peupler combo couleur
            self.combo_fcs_color.blockSignals(True)
            self.combo_fcs_color.clear()
            self.combo_fcs_color.addItem("Aucune")
            color_patterns = (
                "flowsom_cluster",
                "flowsom_metacluster",
                "condition",
                "cluster",
                "metacluster",
                "flowsom",
            )
            for m in markers:
                if any(p in m.lower() for p in color_patterns):
                    self.combo_fcs_color.addItem(m)
            self.combo_fcs_color.blockSignals(False)

            # Sélectionner FSC-A / SSC-A par défaut
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
            self._log(
                f"FCS chargé : {adata.shape[0]:,} cellules, {adata.shape[1]} paramètres"
            )

        except Exception as e:
            QMessageBox.critical(self, "Erreur chargement FCS", str(e))
            self._log(f"Erreur chargement FCS : {e}")

    def _update_fcs_plot(self) -> None:
        """Redessine le scatter/densité/contour FCS selon les paramètres courants."""
        if self.current_fcs_adata is None:
            return

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        try:
            x_marker = self.combo_fcs_x.currentText()
            y_marker = self.combo_fcs_y.currentText()
            plot_type = self.combo_fcs_plot_type.currentText()
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

            color_data = None
            if color_by != "Aucune" and color_by in var_names:
                color_data = X[:, var_names.index(color_by)].copy()

            # Détection coordonnées SOM pour jitter circulaire
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

            # Filtrage NaN/Inf et valeurs sentinelles (-999)
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

            # Sous-échantillonnage
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

            # Dessin
            self.fcs_viz_canvas.clear_and_reset()
            ax = self.fcs_viz_canvas.axes

            scatter_colors = COLORS["blue"]
            legend_handles = None

            if color_data is not None and plot_type == "Scatter":
                from matplotlib.patches import Patch

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
                        labelcolor="#cdd6f4",
                        edgecolor="#45475a",
                        framealpha=0.9,
                        ncol=2 if len(legend_handles) > 10 else 1,
                    )

            elif plot_type == "Densite":
                h = ax.hist2d(
                    x_data, y_data, bins=100, cmap="viridis", norm=mcolors.LogNorm()
                )
                cb = self.fcs_viz_canvas.fig.colorbar(h[3], ax=ax, label="Densité")
                cb.ax.tick_params(colors=COLORS["subtext"])
                cb.ax.yaxis.label.set_color(COLORS["subtext"])

            elif plot_type == "Contour":
                from scipy import stats as sp_stats

                try:
                    n_kde = min(5000, len(x_data))
                    kde_idx = np.random.choice(len(x_data), n_kde, replace=False)
                    xmin, xmax = x_data.min(), x_data.max()
                    ymin, ymax = y_data.min(), y_data.max()
                    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    kernel = sp_stats.gaussian_kde(
                        np.vstack([x_data[kde_idx], y_data[kde_idx]])
                    )
                    f = np.reshape(
                        kernel(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape
                    )
                    ax.contourf(xx, yy, f, levels=20, cmap="viridis")
                    ax.contour(
                        xx, yy, f, levels=10, colors="white", linewidths=0.3, alpha=0.5
                    )
                except Exception:
                    ax.scatter(
                        x_data,
                        y_data,
                        s=2,
                        alpha=0.5,
                        c=COLORS["blue"],
                        edgecolors="none",
                        rasterized=True,
                    )

            # Labels / titres
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
                color=COLORS["text"],
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel(
                x_marker, color=COLORS["text"], fontsize=11, fontweight="bold"
            )
            ax.set_ylabel(
                y_marker, color=COLORS["text"], fontsize=11, fontweight="bold"
            )

            self.fcs_viz_canvas.fig.tight_layout(pad=1.5)
            self.fcs_viz_canvas.draw()

        except Exception as e:
            self._log(f"Erreur plot FCS : {e}")

    # ==================================================================
    # Utilitaires
    # ==================================================================

    def _log(self, msg: str) -> None:
        self.log_output.append(msg)
        sb = self.log_output.verticalScrollBar()
        sb.setValue(sb.maximum())


# ══════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    import platform

    # ── Correctifs QtWebEngine sur Windows 10 ─────────────────────────
    # Sur Win10, Chromium embarqué entre en conflit avec le DWM (Desktop
    # Window Manager) et cause des crashes ou des zones noires/blanches.
    # Sur Win11, le DWM remanié gère correctement ces conflits OpenGL/DirectX.
    if platform.system() == "Windows":
        win_ver = platform.version()  # ex. "10.0.19045" ou "10.0.22621"
        build = int(win_ver.split(".")[-1]) if win_ver.count(".") >= 2 else 0
        is_win10 = build < 22000  # build ≥ 22000 → Windows 11

        if is_win10:
            # Désactive l'accélération GPU de Chromium (solution la plus fiable)
            os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
            # Force ANGLE (DirectX → OpenGL émulé), très stable sur Win10
            from PyQt5.QtCore import Qt, QCoreApplication

            QCoreApplication.setAttribute(Qt.AA_UseOpenGLES)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FlowSomAnalyzerPro()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
