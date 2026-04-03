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
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QSizePolicy, QGridLayout, QStackedWidget,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("Qt5Agg")

from flowsom_pipeline_pro.gui.widgets.mrd_gauge import MRDGauge
from flowsom_pipeline_pro.gui.widgets.mrd_node_table import MRDNodeTable
from flowsom_pipeline_pro.gui.adapters.mrd_adapter import adapt_mrd_result


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
        page.setStyleSheet(f"background: {_BASE};")
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)

        title = QLabel("FlowSOM MRD Analyzer")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {_BLUE}; background: transparent;")
        layout.addWidget(title)

        sub = QLabel(
            "Configurez les dossiers FCS et les paramètres dans le panneau gauche,\n"
            "puis cliquez sur  Lancer le Pipeline  pour démarrer l'analyse."
        )
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {_SUBTEXT}; background: transparent; font-size: 13px;")
        sub.setWordWrap(True)
        layout.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFixedWidth(300)
        sep.setStyleSheet(f"color: {_SURFACE1};")
        layout.addWidget(sep, alignment=Qt.AlignCenter)

        hint = QLabel(
            "Les résultats MRD (% résiduel, nœuds SOM positifs) s'afficheront ici\n"
            "dès que le pipeline sera terminé."
        )
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color: {_SURFACE1}; background: transparent; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

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

        # ── Gauges MRD ──
        self._gauge_container = QWidget()
        self._gauge_container.setStyleSheet("background: transparent;")
        self._gauge_row = QHBoxLayout(self._gauge_container)
        self._gauge_row.setContentsMargins(0, 0, 0, 0)
        self._gauge_row.setSpacing(12)
        self._results_layout.addWidget(self._gauge_container)

        # ── Résumé clinique ──
        self._summary_card = self._build_summary_card()
        self._results_layout.addWidget(self._summary_card)

        # ── Tableau nœuds MRD ──
        self._node_table = MRDNodeTable()
        self._node_table.combo_filter.currentIndexChanged.connect(self._on_node_filter_changed)
        self._results_layout.addWidget(self._node_table)

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

        self._results_layout.addWidget(self._spider_section)
        self._spider_section.hide()

        self._results_layout.addStretch()

        # Données MFI stockées pour les spider plots
        self._mfi_data: Any = None
        self._marker_cols: List[str] = []
        self._mrd_nodes_all: List[Dict] = []

        scroll.setWidget(content)
        return scroll

    def _build_patient_card(self) -> QWidget:
        """Carte infos patient/fichier."""
        card = QWidget()
        card.setObjectName("patientCard")
        card.setStyleSheet(f"""
            QWidget#patientCard {{
                background: {_SURFACE0};
                border-radius: 10px;
                border: 1px solid {_SURFACE1};
            }}
        """)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(24)

        def _info_block(label: str, attr: str) -> QWidget:
            block = QWidget()
            block.setStyleSheet("background: transparent;")
            v = QVBoxLayout(block)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(2)
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {_SUBTEXT}; font-size: 10px; background: transparent;")
            val = QLabel("—")
            val.setFont(QFont("Segoe UI", 11))
            val.setStyleSheet(f"color: {_TEXT}; background: transparent;")
            val.setWordWrap(True)
            setattr(self, attr, val)
            v.addWidget(lbl)
            v.addWidget(val)
            return block

        layout.addWidget(_info_block("Échantillon", "lbl_patient_stem"))
        layout.addWidget(_info_block("Date", "lbl_patient_date"))
        layout.addWidget(_info_block("Cellules totales", "lbl_patient_cells"))
        layout.addWidget(_info_block("Cellules pathologiques", "lbl_patient_patho"))
        layout.addStretch()

        self.lbl_run_time = QLabel("")
        self.lbl_run_time.setStyleSheet(f"color: {_SUBTEXT}; font-size: 10px; background: transparent;")
        self.lbl_run_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.lbl_run_time)

        return card

    def _build_summary_card(self) -> QWidget:
        """Carte résumé clinique."""
        card = QWidget()
        card.setObjectName("summaryCard")
        card.setStyleSheet(f"""
            QWidget#summaryCard {{
                background: {_SURFACE0};
                border-radius: 10px;
                border: 1px solid {_SURFACE1};
            }}
        """)
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(6)

        lbl = QLabel("Conclusion clinique")
        lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl.setStyleSheet(f"color: {_LAVENDER}; background: transparent;")
        v.addWidget(lbl)

        self.lbl_clinical = QLabel("—")
        self.lbl_clinical.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.lbl_clinical.setWordWrap(True)
        self.lbl_clinical.setStyleSheet(f"color: {_TEXT}; background: transparent;")
        v.addWidget(self.lbl_clinical)

        self.lbl_clinical_detail = QLabel("")
        self.lbl_clinical_detail.setWordWrap(True)
        self.lbl_clinical_detail.setStyleSheet(f"color: {_SUBTEXT}; background: transparent; font-size: 11px;")
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
                        "FlowSOM_cluster", "FlowSOM_metacluster", "condition",
                        "file_origin", "xGrid", "yGrid", "xNodes", "yNodes",
                        "size", "Condition_Num", "Condition", "Timepoint", "Timepoint_Num",
                    }
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    self._marker_cols = [c for c in numeric_cols if c not in _meta_cols]
                    if self._marker_cols:
                        self._mfi_data = df.groupby("FlowSOM_cluster")[self._marker_cols].mean()
        except Exception:
            pass

        self._update_patient_info(data["patient_info"])
        self._update_gauges(data["gauges"])
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

    def _update_summary(self, gauges: List[Dict], info: Dict) -> None:
        """Génère le texte de conclusion clinique.

        La conclusion est basée sur JF et Flo uniquement (pas ELN).
        ELN est affiché en gauge mais ne dicte pas la conclusion clinique finale.
        """
        if not gauges:
            self.lbl_clinical.setText("Aucune donnée MRD disponible")
            self.lbl_clinical_detail.setText("")
            return

        # Exclure ELN de la conclusion principale
        non_eln_gauges = [g for g in gauges if g["method"] not in ("ELN 2025", "ELN")]
        ref_gauges = non_eln_gauges if non_eln_gauges else gauges

        positives = [g for g in ref_gauges if g.get("positive") or g.get("low_level")]

        if not positives:
            self.lbl_clinical.setText("MRD Négative")
            self.lbl_clinical.setStyleSheet(f"color: {_GREEN}; background: transparent; font-size: 16px;")
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
            self.lbl_clinical.setStyleSheet(f"color: {_RED}; background: transparent; font-size: 16px;")
            details = []
            for g in gauges:
                status = "POSITIF" if (g.get("positive") or g.get("low_level")) else "négatif"
                details.append(f"{g['method']} : {g['pct']:.4f} % ({status})")
            self.lbl_clinical_detail.setText("  ·  ".join(details))

    def _update_node_table(self, nodes: List[Dict], gauges: List[Dict]) -> None:
        """Charge le tableau des nœuds MRD."""
        available_methods = [g["method"] for g in gauges]
        self._mrd_nodes_all = nodes
        self._node_table.load_nodes(nodes, available_methods)
        # Afficher les spider plots pour la sélection initiale
        self._refresh_spider_plots(nodes, method_label="")

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
        "fsc-a", "fsc-h", "fsc-w", "ssc-a", "ssc-h", "ssc-w",
        "time", "event_", "event", "width", "height", "area",
        "fsc", "ssc",
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
        "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
        "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
        "#ccebc5", "#ffed6f",
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
