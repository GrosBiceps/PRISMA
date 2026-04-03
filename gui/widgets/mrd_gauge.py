# -*- coding: utf-8 -*-
"""
mrd_gauge.py — Widget gauge MRD par méthode.

Affiche pour une méthode donnée :
  - Nom de la méthode
  - Pourcentage MRD (barre de progression + valeur numérique)
  - Badge POSITIF / NÉGATIF / INDÉTECTABLE
  - Nombre de nœuds MRD et cellules impliquées
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QFrame,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor


# Couleurs Catppuccin Mocha (cohérence avec styles.py)
_RED = "#f38ba8"
_YELLOW = "#f9e2af"
_GREEN = "#a6e3a1"
_BLUE = "#89b4fa"
_SUBTEXT = "#a6adc8"
_SURFACE0 = "#313244"
_SURFACE1 = "#45475a"
_BASE = "#1e1e2e"
_MANTLE = "#181825"
_TEXT = "#cdd6f4"


class MRDGauge(QWidget):
    """
    Widget compact affichant le résultat MRD pour une méthode.

    Usage :
        gauge = MRDGauge(method_name="ELN 2025")
        gauge.update_data({
            "pct": 0.42,
            "n_cells": 1234,
            "n_nodes": 3,
            "positive": True,
            "positivity_threshold": 0.1,  # optionnel
        })
    """

    def __init__(self, method_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.method_name = method_name
        self._build_ui()
        self._set_empty()

    def _build_ui(self) -> None:
        self.setObjectName("mrdGaugeCard")
        self.setStyleSheet(f"""
            QWidget#mrdGaugeCard {{
                background: {_SURFACE0};
                border-radius: 10px;
                border: 1px solid {_SURFACE1};
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(8)

        # ── Titre méthode ──
        self.lbl_method = QLabel(self.method_name)
        self.lbl_method.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.lbl_method.setStyleSheet(f"color: {_BLUE}; background: transparent;")
        self.lbl_method.setAlignment(Qt.AlignCenter)
        root.addWidget(self.lbl_method)

        # ── Badge statut ──
        self.lbl_badge = QLabel("EN ATTENTE")
        self.lbl_badge.setAlignment(Qt.AlignCenter)
        self.lbl_badge.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.lbl_badge.setFixedHeight(24)
        self.lbl_badge.setStyleSheet(
            f"color: {_SUBTEXT}; background: {_SURFACE1}; "
            f"border-radius: 6px; padding: 0 8px;"
        )
        root.addWidget(self.lbl_badge)

        # ── Valeur MRD ──
        self.lbl_pct = QLabel("—")
        self.lbl_pct.setAlignment(Qt.AlignCenter)
        self.lbl_pct.setFont(QFont("Segoe UI", 28, QFont.Bold))
        self.lbl_pct.setStyleSheet(f"color: {_TEXT}; background: transparent;")
        root.addWidget(self.lbl_pct)

        # ── Barre de progression ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)  # x10 pour afficher 0.01% → 10
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background: {_SURFACE1};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                border-radius: 4px;
                background: {_BLUE};
            }}
        """)
        root.addWidget(self.progress)

        # ── Infos secondaires ──
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {_SURFACE1};")
        root.addWidget(sep)

        info_row = QHBoxLayout()
        info_row.setSpacing(12)

        self.lbl_nodes = QLabel("— nœuds")
        self.lbl_nodes.setStyleSheet(f"color: {_SUBTEXT}; background: transparent; font-size: 11px;")
        self.lbl_nodes.setAlignment(Qt.AlignCenter)

        self.lbl_cells = QLabel("— cellules")
        self.lbl_cells.setStyleSheet(f"color: {_SUBTEXT}; background: transparent; font-size: 11px;")
        self.lbl_cells.setAlignment(Qt.AlignCenter)

        info_row.addWidget(self.lbl_nodes)
        info_row.addWidget(self.lbl_cells)
        root.addLayout(info_row)

    # ------------------------------------------------------------------

    def update_data(self, data: Dict[str, Any]) -> None:
        """Met à jour l'affichage avec les données MRD de la méthode."""
        pct = data.get("pct", 0.0)
        n_nodes = data.get("n_nodes", 0)
        n_cells = data.get("n_cells", 0)
        positive = data.get("positive", False)
        low_level = data.get("low_level", False)
        threshold = data.get("positivity_threshold", None)

        # Valeur numérique
        self.lbl_pct.setText(f"{pct:.4f} %")

        # Barre (max représente 5%, clampé)
        bar_val = min(int(pct * 200), 1000)  # 5% → 1000
        self.progress.setValue(bar_val)

        # Badge et couleurs
        if positive:
            badge_text = "POSITIF"
            badge_color = _RED
            bar_color = _RED
        elif low_level:
            badge_text = "INDÉTECTABLE (bas niveau)"
            badge_color = _YELLOW
            bar_color = _YELLOW
        elif n_nodes > 0:
            badge_text = "POSITIF"
            badge_color = _RED
            bar_color = _RED
        else:
            badge_text = "NÉGATIF"
            badge_color = _GREEN
            bar_color = _GREEN

        self.lbl_badge.setText(badge_text)
        self.lbl_badge.setStyleSheet(
            f"color: {_BASE}; background: {badge_color}; "
            f"border-radius: 6px; padding: 0 8px; font-weight: bold;"
        )
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background: {_SURFACE1};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                border-radius: 4px;
                background: {bar_color};
            }}
        """)

        # Infos
        seuil_txt = f" (seuil : {threshold}%)" if threshold else ""
        self.lbl_nodes.setText(f"{n_nodes} nœud{'s' if n_nodes != 1 else ''} MRD{seuil_txt}")
        self.lbl_cells.setText(f"{n_cells:,} cellule{'s' if n_cells != 1 else ''}")

    def _set_empty(self) -> None:
        """Affichage vide avant le premier résultat."""
        self.lbl_pct.setText("—")
        self.lbl_badge.setText("EN ATTENTE")
        self.lbl_badge.setStyleSheet(
            f"color: {_SUBTEXT}; background: {_SURFACE1}; "
            f"border-radius: 6px; padding: 0 8px;"
        )
        self.progress.setValue(0)
        self.lbl_nodes.setText("— nœuds")
        self.lbl_cells.setText("— cellules")
