# -*- coding: utf-8 -*-
"""
mrd_node_table.py — Tableau des nœuds SOM MRD positifs.

Affiche les nœuds MRD positifs avec filtre par méthode.
Colonnes : Nœud | Méthodes | % Sain (nœud) | % Patho (nœud) | Cellules patho
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont


_SURFACE0 = "#313244"
_SURFACE1 = "#45475a"
_RED = "#f38ba8"
_GREEN = "#a6e3a1"
_YELLOW = "#f9e2af"
_BLUE = "#89b4fa"
_SUBTEXT = "#a6adc8"
_TEXT = "#cdd6f4"
_BASE = "#1e1e2e"


class MRDNodeTable(QWidget):
    """
    Tableau compact des nœuds MRD positifs avec filtre par méthode.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._all_nodes: List[Dict[str, Any]] = []
        self._available_methods: List[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # ── En-tête + sélecteur méthode ──
        header = QHBoxLayout()
        lbl = QLabel("Nœuds SOM MRD positifs")
        lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl.setStyleSheet(f"color: {_TEXT};")
        header.addWidget(lbl)
        header.addStretch()

        header.addWidget(QLabel("Filtrer par méthode :"))
        self.combo_filter = QComboBox()
        self.combo_filter.setMinimumWidth(140)
        self.combo_filter.currentIndexChanged.connect(self._apply_filter)
        header.addWidget(self.combo_filter)
        root.addLayout(header)

        # ── Tableau ──
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Nœud SOM",
            "Méthodes",
            "% Sain (nœud)",
            "% Patho (nœud)",
            "Cellules patho",
            "Total nœud",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background: {_SURFACE0};
                border: 1px solid {_SURFACE1};
                border-radius: 6px;
                gridline-color: {_SURFACE1};
                color: {_TEXT};
            }}
            QHeaderView::section {{
                background: {_SURFACE1};
                color: {_SUBTEXT};
                padding: 4px;
                border: none;
                font-weight: bold;
            }}
            QTableWidget::item:selected {{
                background: {_BLUE}44;
            }}
        """)
        root.addWidget(self.table)

        self.lbl_empty = QLabel("Aucun nœud MRD positif détecté")
        self.lbl_empty.setAlignment(Qt.AlignCenter)
        self.lbl_empty.setStyleSheet(f"color: {_SUBTEXT}; font-style: italic; padding: 20px;")
        self.lbl_empty.hide()
        root.addWidget(self.lbl_empty)

    # ------------------------------------------------------------------

    def load_nodes(self, nodes: List[Dict[str, Any]], available_methods: List[str]) -> None:
        """Charge les nœuds MRD et peuple le filtre + tableau."""
        self._all_nodes = nodes
        self._available_methods = available_methods

        self.combo_filter.blockSignals(True)
        self.combo_filter.clear()
        self.combo_filter.addItem("Toutes les méthodes")
        for m in available_methods:
            self.combo_filter.addItem(m)
        self.combo_filter.blockSignals(False)

        self._apply_filter()

    def _apply_filter(self) -> None:
        """Filtre les nœuds selon la méthode sélectionnée."""
        selected = self.combo_filter.currentText()

        if selected == "Toutes les méthodes":
            filtered = self._all_nodes
        elif selected == "JF":
            filtered = [n for n in self._all_nodes if n["is_mrd_jf"]]
        elif selected == "Flo":
            filtered = [n for n in self._all_nodes if n["is_mrd_flo"]]
        elif selected in ("ELN 2025", "ELN"):
            filtered = [n for n in self._all_nodes if n["is_mrd_eln"]]
        else:
            filtered = self._all_nodes

        self._populate(filtered)

    def _populate(self, nodes: List[Dict[str, Any]]) -> None:
        """Remplit le tableau avec la liste filtrée."""
        self.table.setRowCount(0)

        if not nodes:
            self.table.hide()
            self.lbl_empty.show()
            return

        self.lbl_empty.hide()
        self.table.show()
        self.table.setRowCount(len(nodes))

        for i, node in enumerate(nodes):
            # Nœud ID
            item_id = QTableWidgetItem(str(node["node_id"]))
            item_id.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 0, item_id)

            # Méthodes positives
            methods = []
            if node["is_mrd_jf"]:
                methods.append("JF")
            if node["is_mrd_flo"]:
                methods.append("Flo")
            if node["is_mrd_eln"]:
                methods.append("ELN")
            item_methods = QTableWidgetItem(" · ".join(methods))
            item_methods.setTextAlignment(Qt.AlignCenter)
            item_methods.setForeground(QColor(_RED))
            self.table.setItem(i, 1, item_methods)

            # % Sain dans le nœud
            item_sain = QTableWidgetItem(f"{node['pct_sain']:.1f} %")
            item_sain.setTextAlignment(Qt.AlignCenter)
            item_sain.setForeground(QColor(_GREEN))
            self.table.setItem(i, 2, item_sain)

            # % Patho dans le nœud
            item_patho = QTableWidgetItem(f"{node['pct_patho']:.1f} %")
            item_patho.setTextAlignment(Qt.AlignCenter)
            item_patho.setForeground(QColor(_RED))
            self.table.setItem(i, 3, item_patho)

            # Cellules patho
            item_n_patho = QTableWidgetItem(f"{node['n_patho']:,}")
            item_n_patho.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 4, item_n_patho)

            # Total nœud
            item_total = QTableWidgetItem(f"{node['n_cells']:,}")
            item_total.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 5, item_total)

    def clear(self) -> None:
        self._all_nodes = []
        self.table.setRowCount(0)
        self.combo_filter.clear()
        self.lbl_empty.show()
        self.table.hide()
