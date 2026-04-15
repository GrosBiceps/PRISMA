# -*- coding: utf-8 -*-
"""
home_tab.py — Onglet Accueil MRD (résultats principaux).

Affiche après analyse :
  - Informations patient / fichier
  - Gauges MRD par méthode (selon le choix utilisateur)
  - Tableau des nœuds SOM MRD positifs avec filtre par méthode
  - Résumé clinique textuel

Avant analyse : écran d'attente avec état des données chargées.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QGridLayout,
    QStackedWidget,
    QPushButton,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

matplotlib.use("Qt5Agg")

from flowsom_pipeline_pro.gui.widgets.mrd_gauge import MRDGauge
from flowsom_pipeline_pro.gui.widgets.mrd_node_table import MRDNodeTable
from flowsom_pipeline_pro.gui.adapters.mrd_adapter import adapt_mrd_result, adapt_all_nodes


_SURFACE0 = "#313244"
_SURFACE1 = "#45475a"
_BASE = "#1e1e2e"
_MANTLE = "#181825"
_TEXT = "#cdd6f4"
_SUBTEXT = "#a6adc8"
_BLUE = "#89b4fa"
_GREEN = "#a6e3a1"
_RED = "#f38ba8"
_YELLOW = "#f9e2af"
_LAVENDER = "#b4befe"


class HomeTab(QWidget):
    """
    Onglet Accueil — résumé MRD post-analyse.

    Interfaces publiques :
      load_result(result, method_used)  → affiche les résultats
      show_waiting()                    → revient à l'écran d'attente
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._gauges: List[MRDGauge] = []
        # État du toggle dénominateur MRD
        # False = toutes cellules patho  |  True = cellules patho CD45+ seulement
        self._denom_cd45_only: bool = False
        self._raw_mrd_result: Any = None  # MRDResult brut stocké pour recalcul
        self._current_method: str = "all"
        self._build_ui()

    # ------------------------------------------------------------------
    # Construction UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Stack : page 0 = attente, page 1 = résultats
        from PyQt5.QtWidgets import QStackedWidget

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_waiting_page())
        self._stack.addWidget(self._build_results_page())
        root.addWidget(self._stack)

        self._stack.setCurrentIndex(0)

    def _build_waiting_page(self) -> QWidget:
        """Écran placeholder avant analyse."""
        page = QWidget()
        page.setObjectName("waitingPage")
        page.setStyleSheet("""
            QWidget#waitingPage {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0d0d17, stop:1 #0a0a14);
            }
        """)
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(24, 24, 24, 32)
        layout.setSpacing(20)

        # Conteneur principal
        container = QWidget()
        container.setObjectName("waitingContainer")
        container.setMinimumWidth(820)
        container.setMaximumWidth(980)
        container.setStyleSheet("""
            QWidget#waitingContainer {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(36, 38, 60, 0.82), stop:1 rgba(20, 22, 36, 0.78));
                border-radius: 18px;
                border: 1px solid rgba(137, 180, 250, 0.13);
                border-top: 1px solid rgba(137, 180, 250, 0.20);
            }
        """)
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(36, 32, 36, 28)
        c_layout.setSpacing(20)

        badge = QLabel("PRÊT POUR L'ANALYSE")
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            "background: rgba(137, 180, 250, 0.16); color: #cfe0ff; "
            "border: 1px solid rgba(137, 180, 250, 0.25); "
            "border-radius: 11px; padding: 6px 14px; font-size: 10px; "
            "font-weight: 700; letter-spacing: 0.08em;"
        )
        c_layout.addWidget(badge, alignment=Qt.AlignHCenter)

        title = QLabel("FlowSOM MRD Analyzer")
        title.setFont(QFont("Segoe UI", 27, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #dbe7ff; background: transparent;")
        c_layout.addWidget(title)

        sub = QLabel(
            "Configurez vos dossiers FCS et paramètres, puis lancez le pipeline.\n"
            "L'accueil affichera automatiquement les résultats MRD dès la fin du calcul."
        )
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #b5c0e3; background: transparent; font-size: 14px;")
        sub.setWordWrap(True)
        c_layout.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: rgba(137,180,250,0.14); max-height: 1px;")
        c_layout.addWidget(sep)

        body = QWidget()
        body.setObjectName("waitingBody")
        body.setStyleSheet("QWidget#waitingBody { background: transparent; }")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(18)

        left = QWidget()
        left.setObjectName("waitingLeftCard")
        left.setStyleSheet(
            "QWidget#waitingLeftCard {"
            "background: rgba(24, 26, 42, 0.72);"
            "border-radius: 12px;"
            "border: 1px solid rgba(137, 180, 250, 0.11);"
            "}"
        )
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(20, 18, 20, 18)
        left_v.setSpacing(12)

        left_title = QLabel("Étapes avant lancement")
        left_title.setStyleSheet(
            "color: #d5e2ff; font-size: 13px; font-weight: 700; background: transparent;"
        )
        left_v.addWidget(left_title)

        def _step_line(num: str, text: str) -> QLabel:
            lbl = QLabel(f"{num}. {text}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #c3ceef; font-size: 13px; background: transparent;")
            return lbl

        left_v.addWidget(_step_line("1", "Importer les dossiers FCS (sain et pathologique)."))
        left_v.addWidget(_step_line("2", "Vérifier les paramètres SOM, MRD et pré-gating."))
        left_v.addWidget(_step_line("3", "Cliquer sur Lancer le Pipeline dans l'étape Exécution."))

        tip = QLabel(
            "Conseil: si vous relancez une analyse, les checkpoints accélèrent les recalculs."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(
            "color: #9eaadb; font-size: 12px; background: rgba(137, 180, 250, 0.06);"
            "border: 1px solid rgba(137, 180, 250, 0.12); border-radius: 10px; padding: 10px;"
        )
        left_v.addWidget(tip)
        left_v.addStretch()

        right = QWidget()
        right.setObjectName("waitingRightCard")
        right.setStyleSheet(
            "QWidget#waitingRightCard {"
            "background: rgba(20, 22, 36, 0.72);"
            "border-radius: 12px;"
            "border: 1px solid rgba(166, 227, 161, 0.10);"
            "}"
        )
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(20, 18, 20, 18)
        right_v.setSpacing(12)

        right_title = QLabel("Ce qui apparaîtra ici après calcul")
        right_title.setStyleSheet(
            "color: #cff5cd; font-size: 13px; font-weight: 700; background: transparent;"
        )
        right_v.addWidget(right_title)

        features = [
            "Résultat MRD par méthode (gauges JF / Flo / ELN)",
            "Conclusion clinique synthétique positive/négative",
            "Tableau des nœuds SOM MRD positifs avec filtres",
            "Profils radar d'expression pour les nœuds détectés",
        ]
        for feat in features:
            lbl = QLabel(f"• {feat}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #c3ceef; font-size: 13px; background: transparent;")
            right_v.addWidget(lbl)

        right_v.addStretch()

        body_layout.addWidget(left, 1)
        body_layout.addWidget(right, 1)
        c_layout.addWidget(body)

        footer = QLabel("Statut actuel: en attente d'une exécution du pipeline")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(
            "color: #a8b8e6; background: transparent; font-size: 12px; font-weight: 600;"
        )
        c_layout.addWidget(footer)

        layout.addWidget(container, alignment=Qt.AlignCenter)
        return page

    def _build_results_page(self) -> QWidget:
        """Page principale des résultats MRD."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ background: {_BASE}; border: none; }}")

        content = QWidget()
        content.setStyleSheet(f"background: {_BASE};")
        self._results_layout = QVBoxLayout(content)
        self._results_layout.setContentsMargins(20, 20, 20, 20)
        self._results_layout.setSpacing(16)

        # ── Infos patient ──
        self._patient_card = self._build_patient_card()
        self._results_layout.addWidget(self._patient_card)

        # ── Bande de décision clinique (priorité de lecture) ──
        self._summary_card = self._build_summary_card()
        self._results_layout.addWidget(self._summary_card, 0)

        # ── Barre de contrôle dénominateur MRD ──
        self._denom_bar = self._build_denom_bar()
        self._results_layout.addWidget(self._denom_bar)

        # ── Gauges MRD ──
        self._gauge_container = QWidget()
        self._gauge_container.setStyleSheet("background: transparent;")
        # Fixed verticalement : les gauges ont une hauteur naturelle et ne
        # doivent JAMAIS voler de l'espace à la grille de validation en dessous.
        self._gauge_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._gauge_row = QHBoxLayout(self._gauge_container)
        self._gauge_row.setContentsMargins(0, 0, 0, 0)
        self._gauge_row.setSpacing(12)
        # stretch=0 : prend uniquement sa hauteur naturelle (sizeHint)
        self._results_layout.addWidget(self._gauge_container, 0)

        # ── Tableau nœuds MRD (grille de validation) ──
        self._node_table = MRDNodeTable()
        self._node_table.combo_filter.currentIndexChanged.connect(self._on_node_filter_changed)
        self._node_table.curated_ratio_changed.connect(self._on_curated_ratio_changed)
        self._node_table.manually_added_nodes_changed.connect(self._on_manually_added_nodes_changed)
        # Expanding/Expanding : la grille de validation réclame tout l'espace
        # vertical restant. setMinimumHeight est le filet de sécurité absolu
        # (déjà défini dans MRDNodeTable.__init__, répété ici par cohérence).
        self._node_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._node_table.setMinimumHeight(400)
        # stretch=1 : reçoit tout l'espace vertical non attribué par les
        # widgets à stretch=0 au-dessus (patient_card, denom_bar, gauges, summary).
        self._results_layout.addWidget(self._node_table, 1)

        # ── Mini Spider Plots des nœuds MRD ──
        self._spider_section = QWidget()
        self._spider_section.setStyleSheet("background: transparent;")
        spider_v = QVBoxLayout(self._spider_section)
        spider_v.setContentsMargins(0, 0, 0, 0)
        spider_v.setSpacing(6)

        self._spider_lbl = QLabel("Profils d'Expression — Clusters MRD (Radar)")
        self._spider_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self._spider_lbl.setStyleSheet(f"color: {_LAVENDER};")
        spider_v.addWidget(self._spider_lbl)

        # ScrollArea horizontal pour les spider plots (8 plots × 244px = déborde sinon)
        spider_scroll = QScrollArea()
        spider_scroll.setWidgetResizable(True)
        spider_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        spider_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        spider_scroll.setFixedHeight(290)
        spider_scroll.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:horizontal {{ background: {_SURFACE1}; height: 6px; border-radius: 3px; }}
            QScrollBar::handle:horizontal {{ background: {_SUBTEXT}; border-radius: 3px; }}
        """)

        self._spider_grid_widget = QWidget()
        self._spider_grid_widget.setStyleSheet("background: transparent;")
        self._spider_grid = QHBoxLayout(self._spider_grid_widget)
        self._spider_grid.setContentsMargins(4, 4, 4, 4)
        self._spider_grid.setSpacing(10)
        spider_scroll.setWidget(self._spider_grid_widget)
        spider_v.addWidget(spider_scroll)

        # stretch=0 : la section spider est masquée par défaut et ne prend de
        # place que quand elle est visible ; elle ne doit pas voler d'espace.
        self._results_layout.addWidget(self._spider_section, 0)
        self._spider_section.hide()

        # Pas de addStretch() ici : le stretch=1 sur _node_table absorbe déjà
        # tout l'espace résiduel. Un stretch supplémentaire comprimerait la grille.

        # Données MFI stockées pour les spider plots
        self._mfi_data: Any = None
        self._marker_cols: List[str] = []
        self._mrd_nodes_all: List[Dict] = []

        scroll.setWidget(content)
        return scroll

    def _build_denom_bar(self) -> QWidget:
        """Barre de contrôle explicite du dénominateur MRD actif."""
        bar = QWidget()
        bar.setObjectName("denomBar")
        bar.setStyleSheet(f"""
            QWidget#denomBar {{
                background: rgba(34, 36, 56, 0.78);
                border-radius: 10px;
                border: 1px solid rgba(137, 180, 250, 0.18);
            }}
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(12)

        icon_lbl = QLabel("÷")
        icon_lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
        icon_lbl.setStyleSheet("color: #8ca2df; background: transparent;")
        layout.addWidget(icon_lbl)

        lbl = QLabel("MODE DÉNOMINATEUR")
        lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
        lbl.setStyleSheet("color: #8f97c2; background: transparent; letter-spacing: 0.12em;")
        layout.addWidget(lbl)

        self.lbl_denom_active = QLabel("TOTAL PATHO")
        self.lbl_denom_active.setAlignment(Qt.AlignCenter)
        self.lbl_denom_active.setFixedHeight(24)
        self.lbl_denom_active.setStyleSheet(
            "color: #dbe7ff; background: rgba(137,180,250,0.20); "
            "border: 1px solid rgba(137,180,250,0.42); border-radius: 6px; "
            "padding: 0 10px; font-size: 10px; font-weight: 800; letter-spacing: 0.06em;"
        )
        layout.addWidget(self.lbl_denom_active)

        layout.addSpacing(6)

        self.lbl_denom_status = QLabel("Toutes cellules pathologiques")
        self.lbl_denom_status.setFont(QFont("Segoe UI", 10, QFont.DemiBold))
        self.lbl_denom_status.setStyleSheet("color: #d3def9; background: transparent;")
        layout.addWidget(self.lbl_denom_status)

        self.lbl_denom_count = QLabel("")
        self.lbl_denom_count.setStyleSheet(
            "color: #b6c3ea; background: transparent; font-size: 10px; font-weight: 600;"
        )
        layout.addWidget(self.lbl_denom_count)

        layout.addStretch()

        self.btn_toggle_denom = QPushButton("  Activer mode CD45+")
        self.btn_toggle_denom.setEnabled(False)
        self.btn_toggle_denom.setFixedHeight(30)
        self.btn_toggle_denom.setStyleSheet(f"""
            QPushButton {{
                background: rgba(137, 180, 250, 0.18);
                color: #d4e2ff;
                border: 1px solid rgba(137, 180, 250, 0.45);
                border-radius: 7px;
                padding: 0 14px;
                font-size: 10px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: rgba(137, 180, 250, 0.30);
                border-color: rgba(137, 180, 250, 0.65);
            }}
            QPushButton:disabled {{
                background: rgba(69, 71, 90, 0.3);
                color: #646b93;
                border-color: rgba(69, 71, 90, 0.4);
            }}
        """)
        self.btn_toggle_denom.clicked.connect(self._toggle_mrd_denominator)
        layout.addWidget(self.btn_toggle_denom)

        return bar

    def _build_patient_card(self) -> QWidget:
        """Carte infos patient/fichier."""
        card = QWidget()
        card.setObjectName("patientCard")
        card.setStyleSheet(f"""
            QWidget#patientCard {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(44, 46, 66, 0.75), stop:1 rgba(28, 30, 48, 0.7));
                border-radius: 12px;
                border: 1px solid rgba(137, 180, 250, 0.16);
                border-top: 2px solid rgba(137, 180, 250, 0.28);
            }}
        """)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(32)

        def _info_block(label: str, attr: str) -> QWidget:
            block = QWidget()
            block.setStyleSheet("background: transparent;")
            v = QVBoxLayout(block)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(3)
            lbl = QLabel(label.upper())
            lbl.setStyleSheet(
                f"color: #7f88b7; font-size: 9px; background: transparent; "
                f"font-weight: 700; letter-spacing: 0.1em;"
            )
            val = QLabel("—")
            val.setFont(QFont("Segoe UI", 11, QFont.Bold))
            val.setStyleSheet(f"color: {_TEXT}; background: transparent;")
            val.setWordWrap(True)
            setattr(self, attr, val)
            v.addWidget(lbl)
            v.addWidget(val)
            return block

        layout.addWidget(_info_block("Échantillon", "lbl_patient_stem"))

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: rgba(137,180,250,0.1);")
        layout.addWidget(sep1)

        layout.addWidget(_info_block("Date", "lbl_patient_date"))

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: rgba(137,180,250,0.1);")
        layout.addWidget(sep2)

        layout.addWidget(_info_block("Cellules totales", "lbl_patient_cells"))

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet("color: rgba(137,180,250,0.1);")
        layout.addWidget(sep3)

        layout.addWidget(_info_block("Cellules pathologiques", "lbl_patient_patho"))
        layout.addStretch()

        self.lbl_run_time = QLabel("")
        self.lbl_run_time.setStyleSheet(
            "color: #8792bf; font-size: 9px; background: transparent; font-weight: 600;"
        )
        self.lbl_run_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.lbl_run_time)

        return card

    def _build_summary_card(self) -> QWidget:
        """Bande de décision clinique prioritaire."""
        card = QWidget()
        card.setObjectName("summaryCard")
        card.setStyleSheet(f"""
            QWidget#summaryCard {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(42, 34, 66, 0.82), stop:1 rgba(25, 26, 46, 0.82));
                border-radius: 12px;
                border: 1px solid rgba(180, 190, 255, 0.20);
                border-left: 4px solid rgba(180, 190, 255, 0.65);
            }}
        """)
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 14, 20, 14)
        v.setSpacing(5)

        lbl = QLabel("DÉCISION CLINIQUE")
        lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
        lbl.setStyleSheet("color: #9ca7d8; background: transparent; letter-spacing: 0.12em;")
        v.addWidget(lbl)

        self.lbl_clinical = QLabel("—")
        self.lbl_clinical.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.lbl_clinical.setWordWrap(True)
        self.lbl_clinical.setStyleSheet(f"color: {_TEXT}; background: transparent;")
        v.addWidget(self.lbl_clinical)

        self.lbl_decision_ref = QLabel("")
        self.lbl_decision_ref.setWordWrap(True)
        self.lbl_decision_ref.setStyleSheet(
            "color: #c9d5ff; background: transparent; font-size: 11px; font-weight: 600;"
        )
        v.addWidget(self.lbl_decision_ref)

        self.lbl_decision_denom = QLabel("")
        self.lbl_decision_denom.setWordWrap(True)
        self.lbl_decision_denom.setStyleSheet(
            "color: #b0bee8; background: transparent; font-size: 10.5px;"
        )
        v.addWidget(self.lbl_decision_denom)

        self.lbl_clinical_detail = QLabel("")
        self.lbl_clinical_detail.setWordWrap(True)
        self.lbl_clinical_detail.setStyleSheet(
            "color: #9aa7d3; background: transparent; font-size: 11px;"
        )
        v.addWidget(self.lbl_clinical_detail)

        return card

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def show_waiting(self) -> None:
        """Revient à l'écran d'attente (ex: avant un nouveau run)."""
        self._stack.setCurrentIndex(0)

    def load_result(self, result: Any, method_used: str = "all") -> None:
        """
        Met à jour l'affichage avec un nouveau PipelineResult.

        Args:
            result: PipelineResult post-pipeline.
            method_used: méthode MRD choisie ("all", "jf", "flo", "eln").
        """
        # Stocker le résultat brut pour le toggle dénominateur
        self._raw_mrd_result = getattr(result, "mrd_result", None)
        self._current_method = method_used
        self._denom_cd45_only = False  # Réinitialiser à chaque nouveau résultat
        self._last_patient_stem = getattr(result, "patho_stem", "") or ""

        # Configurer le bouton toggle selon la disponibilité du dénominateur CD45+
        self._refresh_denom_bar()

        data = adapt_mrd_result(result, method_used)

        if not data["has_data"]:
            self.show_waiting()
            return

        # Stocker les données MFI pour les spider plots
        # Priorité : node_mfi_matrix (médiane par nœud SOM, même source que le radar HTML)
        # Fallback : recalcul depuis result.data (moyenne, moins précis)
        self._mfi_data = None
        self._marker_cols = []
        try:
            import numpy as np

            node_mfi = getattr(result, "node_mfi_matrix", None)
            if node_mfi is not None and not node_mfi.empty:
                self._mfi_data = node_mfi
                self._marker_cols = list(node_mfi.columns)
            else:
                df = result.data
                if df is not None and "FlowSOM_cluster" in df.columns:
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
                    self._marker_cols = [c for c in numeric_cols if c not in _meta_cols]
                    if self._marker_cols:
                        self._mfi_data = df.groupby("FlowSOM_cluster")[self._marker_cols].mean()
        except Exception:
            pass

        self._update_patient_info(data["patient_info"])
        self._update_gauges(data["gauges"])
        self._last_gauges_data: List[Dict] = data["gauges"]  # exposé pour export dashboard
        self._update_summary(data["gauges"], data["patient_info"])
        self._update_node_table(data["nodes"], data["gauges"])

        self._stack.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Mise à jour des sous-widgets
    # ------------------------------------------------------------------

    def _update_patient_info(self, info: Dict) -> None:
        self.lbl_patient_stem.setText(info.get("stem", "—") or "—")
        self.lbl_patient_date.setText(info.get("date", "—") or "—")
        n_cells = info.get("n_cells", 0)
        n_patho = info.get("n_cells_patho", 0)
        self.lbl_patient_cells.setText(f"{n_cells:,}" if n_cells else "—")
        self.lbl_patient_patho.setText(f"{n_patho:,}" if n_patho else "—")
        ts = info.get("timestamp", "")
        self.lbl_run_time.setText(f"Analyse : {ts}" if ts else "")

    def _update_gauges(self, gauges: List[Dict]) -> None:
        """Recrée les gauges selon les méthodes à afficher."""
        # Vider les gauges existantes
        while self._gauge_row.count():
            item = self._gauge_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._gauges.clear()

        for g_data in gauges:
            gauge = MRDGauge(method_name=g_data["method"])
            gauge.update_data(g_data)
            gauge.setMinimumWidth(220)
            gauge.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._gauge_row.addWidget(gauge)
            self._gauges.append(gauge)

    def _get_active_denom_text(self) -> str:
        """Retourne une description lisible du dénominateur actif."""
        mrd = self._raw_mrd_result
        if mrd is None:
            return "Total pathologique"

        n_pre = getattr(mrd, "n_patho_pre_cd45", 0)
        total_patho = n_pre if n_pre > 0 else getattr(mrd, "total_cells_patho", 0)
        n_cd45pos = getattr(mrd, "n_patho_cd45pos", 0)

        if self._denom_cd45_only:
            return f"CD45+ pathologique ({n_cd45pos:,} cellules)"
        return f"Total pathologique ({total_patho:,} cellules)"

    def _update_summary(self, gauges: List[Dict], info: Dict) -> None:
        """Génère le texte de conclusion clinique.

        La conclusion est basée sur JF et Flo uniquement (pas ELN).
        ELN est affiché en gauge mais ne dicte pas la conclusion clinique finale.
        """
        if not gauges:
            self.lbl_clinical.setText("Aucune donnée MRD disponible")
            self.lbl_decision_ref.setText("")
            self.lbl_decision_denom.setText("")
            self.lbl_clinical_detail.setText("")
            return

        # Exclure ELN de la conclusion principale
        non_eln_gauges = [g for g in gauges if g["method"] not in ("ELN 2025", "ELN")]
        ref_gauges = non_eln_gauges if non_eln_gauges else gauges

        positives = [g for g in ref_gauges if g.get("positive") or g.get("low_level")]

        if not positives:
            self.lbl_clinical.setText("MRD Négative")
            self.lbl_clinical.setStyleSheet(
                f"color: {_GREEN}; background: transparent; font-size: 16px;"
            )
            self.lbl_decision_ref.setText(
                "Méthodes de référence concordantes: JF/FLO non détectées"
            )
            self.lbl_decision_denom.setText(f"Dénominateur actif: {self._get_active_denom_text()}")
            details = [f"{g['method']} : {g['pct']:.4f} %" for g in gauges]
            self.lbl_clinical_detail.setText("  ·  ".join(details))
        else:
            # La méthode de référence prioritaire est JF, puis Flo
            priority = ["JF", "Flo"]
            ref = None
            for p in priority:
                ref = next((g for g in positives if g["method"] == p), None)
                if ref:
                    break
            if ref is None:
                ref = max(positives, key=lambda g: g.get("pct", 0))

            self.lbl_clinical.setText(f"MRD Positive — {ref['pct']:.4f} % ({ref['method']})")
            self.lbl_clinical.setStyleSheet(
                f"color: {_RED}; background: transparent; font-size: 16px;"
            )
            self.lbl_decision_ref.setText(
                f"Méthode de référence retenue: {ref['method']} ({ref['pct']:.4f} %)"
            )
            self.lbl_decision_denom.setText(f"Dénominateur actif: {self._get_active_denom_text()}")
            details = []
            for g in gauges:
                status = "POSITIF" if (g.get("positive") or g.get("low_level")) else "négatif"
                details.append(f"{g['method']} : {g['pct']:.4f} % ({status})")
            self.lbl_clinical_detail.setText("  ·  ".join(details))

    def _update_node_table(self, nodes: List[Dict], gauges: List[Dict]) -> None:
        """Charge la grille de validation des nœuds MRD."""
        available_methods = [g["method"] for g in gauges]
        self._mrd_nodes_all = nodes

        # Dénominateur : total cellules viables (patho) pour le ratio validé
        mrd = self._raw_mrd_result
        n_pre = getattr(mrd, "n_patho_pre_cd45", 0) if mrd else 0
        total_viable = n_pre if n_pre > 0 else (getattr(mrd, "total_cells_patho", 0) if mrd else 0)

        # Fournir TOUS les nœuds SOM (y compris non-MRD) à ExpertFocusDialog
        all_patient_nodes = adapt_all_nodes(self._raw_mrd_result)
        self._node_table.set_all_patient_nodes(all_patient_nodes if all_patient_nodes else nodes)

        self._node_table.load_nodes(
            nodes,
            available_methods,
            mfi_data=self._mfi_data,
            marker_cols=self._marker_cols,
            total_viable_cells=max(total_viable, 1),
        )
        # Afficher les spider plots pour la sélection initiale
        self._refresh_spider_plots(nodes, method_label="")

    def _on_curated_ratio_changed(self, method: str, ratio: float, n_mrd_cells: int) -> None:
        """
        Slot connecté à MRDNodeTable.curated_ratio_changed.

        Met à jour (ou crée) une jauge "Validé" reflétant le ratio
        recalculé après validation experte des nœuds.
        Toutes les gauges existantes JF/Flo/ELN sont conservées.
        """
        # Chercher une jauge "Validé" déjà présente
        curated_gauge: Optional[MRDGauge] = None
        for g in self._gauges:
            if g.method_name == "Curated":
                curated_gauge = g
                break

        if curated_gauge is None:
            curated_gauge = MRDGauge(method_name="Validé")
            curated_gauge.setMinimumWidth(220)
            curated_gauge.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._gauge_row.addWidget(curated_gauge)
            self._gauges.append(curated_gauge)

        curated_gauge.update_data(
            {
                "pct": ratio,
                "n_cells": n_mrd_cells,
                "n_nodes": sum(1 for c in self._node_table._cards if c.is_included),
                "positive": ratio > 0.01,
                "low_level": 0 < ratio <= 0.01,
                "positivity_threshold": 0.01,
            }
        )

    def _on_manually_added_nodes_changed(self, manual_nodes: list) -> None:
        """
        Slot connecté à MRDNodeTable.manually_added_nodes_changed.

        Appelé chaque fois que l'utilisateur valide des ajouts manuels
        via ExpertFocusDialog. Peut être étendu pour :
          - persister les ajouts dans un fichier de session,
          - mettre à jour un indicateur visuel dans HomeTab,
          - recalculer les spider plots.
        """
        # Rafraîchir les spider plots pour inclure les nœuds manuels
        if manual_nodes:
            all_curated = self._node_table.get_human_curated_results()
            self._refresh_spider_plots(all_curated, method_label="Manuel")

    # ------------------------------------------------------------------
    # Toggle dénominateur MRD
    # ------------------------------------------------------------------

    def _refresh_denom_bar(self) -> None:
        """Met à jour l'état de la barre de contrôle dénominateur."""
        mrd = self._raw_mrd_result
        if mrd is None:
            self.btn_toggle_denom.setEnabled(False)
            self.lbl_denom_active.setText("TOTAL PATHO")
            self.lbl_denom_status.setText("Toutes cellules pathologiques")
            self.lbl_denom_count.setText("")
            return

        # n_patho_pre_cd45 = total patho avant gate CD45 (CD45+ + CD45-)
        # Si non disponible (gate inactive), fallback sur total_cells_patho.
        n_pre = getattr(mrd, "n_patho_pre_cd45", 0)
        total_patho = n_pre if n_pre > 0 else getattr(mrd, "total_cells_patho", 0)
        n_cd45pos = getattr(mrd, "n_patho_cd45pos", 0)

        # Le toggle est disponible uniquement si les deux valeurs sont distinctes
        cd45_available = n_cd45pos > 0 and n_cd45pos != total_patho

        if self._denom_cd45_only:
            self.lbl_denom_active.setText("CD45+ ACTIF")
            self.lbl_denom_active.setStyleSheet(
                "color: #d6fff0; background: rgba(137,220,235,0.20); "
                "border: 1px solid rgba(137,220,235,0.45); border-radius: 6px; "
                "padding: 0 10px; font-size: 10px; font-weight: 800; letter-spacing: 0.06em;"
            )
            self.lbl_denom_status.setText("Cellules pathologiques CD45+")
            self.lbl_denom_status.setStyleSheet(
                "color: #c7f6ff; background: transparent; font-size: 10px; font-weight: bold;"
            )
            self.lbl_denom_count.setText(f"({n_cd45pos:,} cellules)" if n_cd45pos > 0 else "")
            btn_label = "  Revenir en mode total patho"
        else:
            self.lbl_denom_active.setText("TOTAL PATHO")
            self.lbl_denom_active.setStyleSheet(
                "color: #dbe7ff; background: rgba(137,180,250,0.20); "
                "border: 1px solid rgba(137,180,250,0.42); border-radius: 6px; "
                "padding: 0 10px; font-size: 10px; font-weight: 800; letter-spacing: 0.06em;"
            )
            self.lbl_denom_status.setText("Toutes cellules pathologiques")
            self.lbl_denom_status.setStyleSheet(
                "color: #d3def9; background: transparent; font-size: 10px; font-weight: bold;"
            )
            self.lbl_denom_count.setText(f"({total_patho:,} cellules)" if total_patho > 0 else "")
            btn_label = "  Activer mode CD45+"

        self.btn_toggle_denom.setText(btn_label)
        self.btn_toggle_denom.setEnabled(cd45_available)

    def _toggle_mrd_denominator(self) -> None:
        """Bascule entre le dénominateur total patho et CD45+ patho."""
        self._denom_cd45_only = not self._denom_cd45_only
        self._refresh_denom_bar()

        mrd = self._raw_mrd_result
        if mrd is None:
            return

        # Recalculer les pourcentages MRD avec le nouveau dénominateur
        new_gauges = self._recompute_gauges_with_denom(mrd, self._current_method)
        self._update_gauges(new_gauges)
        self._update_summary(new_gauges, {"stem": self._last_patient_stem})

        # Mettre à jour le compteur "Cellules pathologiques" dans la carte patient
        n_cd45pos = getattr(mrd, "n_patho_cd45pos", 0)
        n_pre = getattr(mrd, "n_patho_pre_cd45", 0)
        total_patho = n_pre if n_pre > 0 else getattr(mrd, "total_cells_patho", 0)
        displayed = n_cd45pos if self._denom_cd45_only else total_patho
        if hasattr(self, "lbl_patient_patho") and displayed > 0:
            suffix = " (CD45+)" if self._denom_cd45_only else ""
            self.lbl_patient_patho.setText(f"{displayed:,}{suffix}")

    def _recompute_gauges_with_denom(self, mrd: Any, method_used: str) -> List[Dict]:
        """
        Recalcule les gauges MRD avec le dénominateur choisi.

        Si _denom_cd45_only=True  → dénominateur = mrd.n_patho_cd45pos (CD45+ patho)
        Sinon                     → dénominateur = mrd.n_patho_pre_cd45 (total patho avant gate)
        """
        n_pre = getattr(mrd, "n_patho_pre_cd45", 0)
        total_patho = n_pre if n_pre > 0 else getattr(mrd, "total_cells_patho", 0)
        n_cd45pos = getattr(mrd, "n_patho_cd45pos", 0)

        denom = n_cd45pos if (self._denom_cd45_only and n_cd45pos > 0) else total_patho
        if denom <= 0:
            denom = max(total_patho, 1)

        def _pct(n_cells: int) -> float:
            return round(n_cells / denom * 100.0, 4) if denom > 0 else 0.0

        gauges: List[Dict] = []
        _show_jf = method_used in ("all", "jf")
        _show_flo = method_used in ("all", "flo")
        _show_eln = method_used == "eln"

        if _show_jf:
            n = getattr(mrd, "mrd_cells_jf", 0)
            p = _pct(n)
            gauges.append(
                {
                    "method": "JF",
                    "pct": p,
                    "n_cells": n,
                    "n_nodes": getattr(mrd, "n_nodes_mrd_jf", 0),
                    "positive": p > 0,
                    "positivity_threshold": None,
                }
            )

        if _show_flo:
            n = getattr(mrd, "mrd_cells_flo", 0)
            p = _pct(n)
            gauges.append(
                {
                    "method": "Flo",
                    "pct": p,
                    "n_cells": n,
                    "n_nodes": getattr(mrd, "n_nodes_mrd_flo", 0),
                    "positive": p > 0,
                    "positivity_threshold": None,
                }
            )

        if _show_eln:
            n = getattr(mrd, "mrd_cells_eln", 0)
            p = _pct(n)
            cfg = getattr(mrd, "config_snapshot", {})
            eln_cfg = cfg.get("eln_standards", {}) if isinstance(cfg, dict) else {}
            threshold = eln_cfg.get("clinical_positivity_pct", 0.1)
            gauges.append(
                {
                    "method": "ELN 2025",
                    "pct": p,
                    "n_cells": n,
                    "n_nodes": getattr(mrd, "n_nodes_mrd_eln", 0),
                    "positive": p >= threshold,
                    "low_level": (n > 0) and (p < threshold),
                    "positivity_threshold": threshold,
                }
            )

        return gauges

    def _on_node_filter_changed(self) -> None:
        """Rafraîchit les spider plots quand le filtre de méthode change."""
        selected = self._node_table.combo_filter.currentText()
        if selected == "Toutes les méthodes":
            filtered = self._mrd_nodes_all
            method_label = ""
        elif selected == "JF":
            filtered = [n for n in self._mrd_nodes_all if n["is_mrd_jf"]]
            method_label = "JF"
        elif selected == "Flo":
            filtered = [n for n in self._mrd_nodes_all if n["is_mrd_flo"]]
            method_label = "Flo"
        elif selected in ("ELN 2025", "ELN"):
            filtered = [n for n in self._mrd_nodes_all if n["is_mrd_eln"]]
            method_label = "ELN"
        else:
            filtered = self._mrd_nodes_all
            method_label = ""
        self._refresh_spider_plots(filtered, method_label=method_label)

    # Marqueurs techniques à exclure du spider plot (pas cliniquement pertinents)
    _TECHNICAL_MARKERS = {
        "fsc-a",
        "fsc-h",
        "fsc-w",
        "ssc-a",
        "ssc-h",
        "ssc-w",
        "time",
        "event_",
        "event",
        "width",
        "height",
        "area",
        "fsc",
        "ssc",
    }

    def _filter_clinical_markers(self, markers: List[str]) -> List[str]:
        """Retourne uniquement les marqueurs cliniques (exclut FSC, SSC, Time, etc.)."""
        result = []
        for m in markers:
            m_low = m.lower().strip()
            if any(m_low == t or m_low.startswith(t) for t in self._TECHNICAL_MARKERS):
                continue
            result.append(m)
        return result

    # Palette de couleurs distinctes par nœud (calquée sur Set3 de Plotly)
    _NODE_COLORS = [
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#d9d9d9",
        "#bc80bd",
        "#ccebc5",
        "#ffed6f",
    ]

    def _refresh_spider_plots(self, nodes: List[Dict], *, method_label: str = "") -> None:
        """
        Génère les mini spider plots pour les nœuds MRD filtrés (max 8).

        Normalisation par nœud (identique à plot_cluster_radar_mrd) :
        chaque profil est mis à l'échelle sur son propre min/max pour que
        la forme du radar reflète fidèlement le profil d'expression relatif,
        exactement comme dans le rapport HTML Méthode JF/Flo.
        """
        # Vider l'ancienne grille
        while self._spider_grid.count():
            item = self._spider_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not nodes or self._mfi_data is None or not self._marker_cols:
            self._spider_section.hide()
            return

        import numpy as np

        # Exclure les marqueurs techniques
        markers = self._filter_clinical_markers(self._marker_cols)
        n_markers = len(markers)
        if n_markers < 3:
            self._spider_section.hide()
            return

        # Mettre à jour le titre de section avec méthode + nombre de nœuds
        n_nodes = len(nodes)
        if method_label:
            title_text = (
                f"Profils d'Expression — Clusters MRD Méthode {method_label} (Radar)"
                f"  ·  {n_nodes} nœud(s)"
            )
        else:
            title_text = f"Profils d'Expression — Clusters MRD (Radar)  ·  {n_nodes} nœud(s)"
        self._spider_lbl.setText(title_text)

        angles = np.linspace(0, 2 * np.pi, n_markers, endpoint=False).tolist()
        angles_closed = angles + angles[:1]

        max_plots = min(n_nodes, 8)
        display_nodes = nodes[:max_plots]

        # Labels marqueurs courts
        short_labels = []
        for m in markers:
            parts = m.strip().split()
            candidate = parts[-1] if len(parts) > 1 else parts[0]
            short_labels.append(candidate[:10])

        for idx, node in enumerate(display_nodes):
            node_id = node["node_id"]
            if node_id not in self._mfi_data.index:
                continue

            mfi_row = self._mfi_data.loc[node_id, markers].values.astype(float)

            # ── Normalisation par nœud (identique à plot_cluster_radar_mrd) ──
            v_min, v_max = float(mfi_row.min()), float(mfi_row.max())
            norm_vals = (mfi_row - v_min) / (v_max - v_min + 1e-10)
            vals = list(norm_vals) + [norm_vals[0]]

            pct_p = node["pct_patho"]
            node_color = self._NODE_COLORS[idx % len(self._NODE_COLORS)]

            # Widget conteneur (radar + label sous)
            container = QWidget()
            container.setStyleSheet(f"background: {_SURFACE0}; border-radius: 8px;")
            c_layout = QVBoxLayout(container)
            c_layout.setContentsMargins(6, 6, 6, 6)
            c_layout.setSpacing(3)

            # Figure matplotlib radar
            fig = Figure(figsize=(2.8, 2.8), dpi=88)
            fig.patch.set_facecolor(_SURFACE0)
            ax = fig.add_subplot(111, polar=True)
            ax.set_facecolor("#1e1e2e")

            # Tracé avec couleur distincte par nœud
            ax.plot(angles_closed, vals, color=node_color, linewidth=2.0, zorder=3)
            ax.fill(angles_closed, vals, color=node_color, alpha=0.20, zorder=2)

            # Grille radiale discrète
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0.33, 0.66, 1.0])
            ax.set_yticklabels([])
            ax.yaxis.grid(True, color=(1, 1, 1, 0.15), linewidth=0.5, linestyle=":")
            ax.spines["polar"].set_color((1, 1, 1, 0.3))
            ax.spines["polar"].set_linewidth(0.8)

            ax.set_xticks(angles)
            ax.set_xticklabels(
                short_labels,
                fontsize=7,
                color=_TEXT,
                fontweight="bold",
            )
            ax.tick_params(axis="x", pad=9)

            # Marges pour laisser de la place aux labels
            fig.subplots_adjust(top=0.88, bottom=0.10, left=0.10, right=0.90)

            canvas = FigureCanvas(fig)
            canvas.setMinimumSize(246, 246)
            canvas.setMaximumSize(280, 280)
            c_layout.addWidget(canvas)

            # Label sous le radar : nœud + % patho + méthode
            label_parts = [f"Nœud {node_id}"]
            if method_label:
                label_parts.append(f"[{method_label}]")
            label_parts.append(f"{pct_p:.0f}% patho")
            lbl_node = QLabel("  —  ".join(label_parts))
            lbl_node.setAlignment(Qt.AlignCenter)
            lbl_node.setStyleSheet(
                f"color: {node_color}; font-size: 10px; font-weight: bold; background: transparent;"
            )
            c_layout.addWidget(lbl_node)

            container.setFixedWidth(290)
            self._spider_grid.addWidget(container)

        self._spider_grid.addStretch()
        self._spider_section.show()
