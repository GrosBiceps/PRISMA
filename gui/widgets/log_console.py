# -*- coding: utf-8 -*-
"""
log_console.py — Terminal de logs avec coloration syntaxique.

LogConsole est un QPlainTextEdit enrichi avec :
  - Police monospace (Cascadia Code / Fira Code / Consolas)
  - Fond #0a0a14 (terminal sombre)
  - Coloration syntaxique : [INFO] vert, [WARNING] orange, [ERROR] rouge,
    [SUCCESS] cyan, timestamps en gris
  - Méthode append_log(msg) qui parse et colore automatiquement

Design System "Deep Medical Clarity" :
  - [INFO]    → #2ECC71 (Vert Santé)
  - [WARNING] → #F39C12 (Orange Alerte)
  - [ERROR]   → #E74C3C (Rouge Alerte)
  - [SUCCESS] → #00A3FF (Bleu Technologie)
  - Timestamp → #45475a (Gris surface)
"""

from __future__ import annotations

import re
from typing import Optional

from PyQt5.QtWidgets import QPlainTextEdit, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QFont,
    QTextCharFormat,
    QColor,
    QTextCursor,
    QBrush,
)

# ── Patterns de coloration ────────────────────────────────────────────

_LOG_RULES: list[tuple[re.Pattern, str]] = [
    # Timestamps ISO : 2024-01-15 12:34:56
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"), "#45475a"),
    # Horodatage court : [12:34:56]
    (re.compile(r"\[\d{2}:\d{2}:\d{2}\]"), "#45475a"),
    # Niveaux de log
    (re.compile(r"\[INFO\]|\[info\]"), "#2ECC71"),  # Vert santé
    (re.compile(r"\bINFO\b"), "#2ECC71"),  # Format logger sans []
    (re.compile(r"\[SUCCESS\]|\[success\]"), "#00A3FF"),  # Bleu technologie
    (re.compile(r"\bSUCCESS\b"), "#00A3FF"),
    (re.compile(r"\[WARNING\]|\[warning\]|\[WARN\]|\[warn\]"), "#F39C12"),  # Orange
    (re.compile(r"\bWARNING\b|\bWARN\b"), "#F39C12"),
    (re.compile(r"\[ERROR\]|\[error\]|\[ERR\]|\[err\]"), "#E74C3C"),  # Rouge
    (re.compile(r"\bERROR\b|\bERR\b|\bCRITICAL\b"), "#E74C3C"),
    (re.compile(r"\[DEBUG\]|\[debug\]"), "#585b70"),  # Gris discret
    (re.compile(r"\bDEBUG\b"), "#585b70"),
    # Marqueurs pipeline / gating importants
    (re.compile(r"\bG[1-4]_[a-zA-Z0-9_]+\b|\bGate\s+[1-4]\b"), "#89b4fa"),
    (re.compile(r"\b(MRD|KDE|RANSAC|GMM|UMAP|FlowSOM)\b", re.IGNORECASE), "#74c7ec"),
    (
        re.compile(
            r"\b(Marqueurs\s+retenus\s+dans\s+le\s+panel\s+commun|Marqueurs\s+pour\s+FlowSOM)\b",
            re.IGNORECASE,
        ),
        "#f9e2af",
    ),
    (re.compile(r"\bD[ée]s[ée]quilibre\s+Ma[îi]tris[ée]\s+activ[ée]\b", re.IGNORECASE), "#fab387"),
    (re.compile(r"\bfallback\b|\bfallbacks\b|\béchou[ée]\b", re.IGNORECASE), "#F39C12"),
    (re.compile(r"\bPipeline\b|\bÉtape\b", re.IGNORECASE), "#74c7ec"),
    # Valeurs numériques importantes (pourcentages, MRD)
    (re.compile(r"\b\d+\.?\d*\s*%"), "#cba6f7"),  # Mauve — pourcentages
    (re.compile(r"\bratio\s*=\s*\d+(?:\.\d+)?×\b", re.IGNORECASE), "#f38ba8"),
    (re.compile(r"\bseed\s*=\s*\d+\b", re.IGNORECASE), "#89dceb"),
    (re.compile(r"\([^)]*\)"), "#b4befe"),
    (re.compile(r"\b\d[\d,]*\s*/\s*\d[\d,]*\b"), "#f9e2af"),  # Ratios n_kept/n_total
    # Chemins de fichiers
    (re.compile(r"[A-Za-z]:\\[^\s]+|/[^\s]+\.[a-z]+"), "#f9e2af"),  # Jaune doux
    # Flèches / transitions de comptage
    (re.compile(r"→|->|=>|\|"), "#6c7086"),
    # Mots-clés état
    (re.compile(r"\b(terminé|sauvegardé|chargé|exporté|conservées?)\b", re.IGNORECASE), "#a6e3a1"),
    (re.compile(r"\b(absent|ignoré|déséquilibre|warning|erreur)\b", re.IGNORECASE), "#fab387"),
    # Étapes pipelines (ex: "Étape 1/5" ou "Step 1 of 5")
    (re.compile(r"[ÉEé]tape\s+\d+\s*/\s*\d+|Step\s+\d+\s+of\s+\d+"), "#89b4fa"),
]

_MAJOR_LINE_PATTERNS: list[tuple[re.Pattern, str, int]] = [
    # Blocs de séparation visuels
    (re.compile(r"^\s*[=]{20,}\s*$"), "#94e2d5", 63),
    (re.compile(r"^\s*[═]{10,}\s*$"), "#94e2d5", 63),
    # Démarrage / fin du pipeline
    (re.compile(r"\bPIPELINE\b.*\bD[ÉE]MARRAGE\b", re.IGNORECASE), "#74c7ec", 75),
    (re.compile(r"\bD[ÉE]MARRAGE\s+DU\s+PIPELINE\b", re.IGNORECASE), "#74c7ec", 75),
    (re.compile(r"\bPIPELINE\b.*\bTERMIN[ÉE]\b", re.IGNORECASE), "#a6e3a1", 75),
    # Bloc d'analyse important
    (re.compile(r"\bANALYSE\s+DES\s+CLUSTERS\s+EXCLUSIFS\b", re.IGNORECASE), "#f9e2af", 75),
    # Étapes du pipeline: colorer toute la ligne, pas seulement le token "Étape"
    (re.compile(r"\b[ÉEé]tape\s+\d+\b.*", re.IGNORECASE), "#74c7ec", 75),
    (re.compile(r"\bStep\s+\d+\b.*", re.IGNORECASE), "#74c7ec", 75),
    # KPI finaux de synthèse
    (re.compile(r"\bCellules\s*:", re.IGNORECASE), "#89dceb", 75),
    (re.compile(r"\bMarqueurs\s*:", re.IGNORECASE), "#cba6f7", 75),
    (re.compile(r"\bM[ée]taclusters\s*:", re.IGNORECASE), "#f38ba8", 75),
    # Lignes stratégiques de sélection des marqueurs / balance
    (
        re.compile(r"\bMarqueurs\s+retenus\s+dans\s+le\s+panel\s+commun\b", re.IGNORECASE),
        "#f9e2af",
        75,
    ),
    (re.compile(r"\bMarqueurs\s+pour\s+FlowSOM\b", re.IGNORECASE), "#f5c2e7", 75),
    (
        re.compile(r"\bD[ée]s[ée]quilibre\s+Ma[îi]tris[ée]\s+activ[ée]\b", re.IGNORECASE),
        "#fab387",
        75,
    ),
]


class LogConsole(QPlainTextEdit):
    """
    Terminal de logs avec coloration syntaxique intégrée.

    Usage :
        console = LogConsole()
        console.append_log("[INFO] Pipeline démarré.")
        console.append_log("[ERROR] Fichier introuvable : data.fcs")
    """

    # Nombre max de blocs (lignes) pour éviter une croissance infinie
    MAX_BLOCKS = 5000

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.setObjectName("logConsole")
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_BLOCKS)
        self.setPlaceholderText("Les logs du pipeline apparaîtront ici…")

        # Police monospace
        font = QFont()
        font.setFamilies(["Cascadia Code", "Fira Code", "Consolas", "Courier New"])
        font.setPointSize(9)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)

        # Style de base (complété par QSS global)
        self.setStyleSheet("""
            QPlainTextEdit#logConsole {
                background: #0a0a14;
                border: 1px solid rgba(137, 180, 250, 0.10);
                border-radius: 10px;
                color: #bac2de;
                padding: 12px;
                selection-background-color: rgba(137, 180, 250, 0.28);
                line-height: 1.5;
            }
            QPlainTextEdit#logConsole QScrollBar:vertical {
                background: #0d0d18;
                width: 8px;
                border: none;
            }
            QPlainTextEdit#logConsole QScrollBar::handle:vertical {
                background: rgba(137, 180, 250, 0.2);
                border-radius: 4px;
                min-height: 20px;
            }
            QPlainTextEdit#logConsole QScrollBar::handle:vertical:hover {
                background: rgba(137, 180, 250, 0.4);
            }
        """)

        # Format de texte par défaut (texte brut)
        self._default_format = QTextCharFormat()
        self._default_format.setForeground(QBrush(QColor("#bac2de")))

        # Nettoyage visuel des préfixes doublés: "[hh:mm:ss] INFO [hh:mm:ss] INFO ..."
        self._dup_prefix_re = re.compile(
            r"^(\[\d{2}:\d{2}:\d{2}\]\s+(?:INFO|WARNING|WARN|ERROR|ERR|DEBUG|SUCCESS)\s+)"
            r"\1+"
        )
        self._ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

    # ── API publique ──────────────────────────────────────────────────

    def append_log(self, message: str) -> None:
        """
        Ajoute une ligne de log avec coloration syntaxique automatique.
        Scinde les spans par regex et insère chaque segment coloré.
        """
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Saut de ligne si le document n'est pas vide
        if not self.document().isEmpty():
            cursor.insertText("\n", self._default_format)

        line = self._normalize_line(message)
        self._insert_colored(cursor, line)

        # Auto-scroll vers le bas
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def append(self, text: str) -> None:
        """
        Surcharge de QPlainTextEdit.appendPlainText pour compatibilité avec
        l'API QTextEdit.append() utilisée dans main_window.py.
        Redirige vers append_log() pour appliquer la coloration syntaxique.
        """
        # Retirer les balises HTML simples si le texte vient d'un insertHtml
        clean = re.sub(r"<[^>]+>", "", text)
        self.append_log(clean)

    def clear_logs(self) -> None:
        """Efface tous les logs."""
        self.clear()

    # ── Coloration syntaxique ─────────────────────────────────────────

    def _insert_colored(self, cursor: QTextCursor, line: str) -> None:
        """
        Insère une ligne de log en colorant les tokens correspondant aux règles.
        Algorithme : on calcule tous les spans qui matchent, on les trie,
        on insère segment par segment.
        """
        if not line:
            cursor.insertText("", self._default_format)
            return

        # Collecter tous les spans (start, end, color)
        spans: list[tuple[int, int, str]] = []
        for pattern, color in _LOG_RULES:
            for m in pattern.finditer(line):
                spans.append((m.start(), m.end(), color))

        if not spans:
            # Ligne sans token spécial : couleur de base selon niveau
            fmt = self._format_for_line(line)
            cursor.insertText(line, fmt)
            return

        # Trier et dédupliquer (priorité au premier match en cas de chevauchement)
        # A position égale, donner priorité au motif le plus long
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        merged: list[tuple[int, int, str]] = []
        last_end = 0
        for start, end, color in spans:
            if start >= last_end:
                merged.append((start, end, color))
                last_end = end

        # Couleur de base de ligne (niveau log + sections majeures)
        base_fmt = self._format_for_line(line)
        major_fmt = self._major_line_format(line)
        if major_fmt is not None:
            base_fmt = major_fmt

        pos = 0
        for start, end, color in merged:
            # Segment avant le match
            if pos < start:
                cursor.insertText(line[pos:start], base_fmt)
            # Segment coloré
            fmt = QTextCharFormat()
            fmt.setForeground(QBrush(QColor(color)))
            fmt.setFont(self.font())
            cursor.insertText(line[start:end], fmt)
            pos = end

        # Reste après le dernier match
        if pos < len(line):
            cursor.insertText(line[pos:], base_fmt)

    def _format_for_line(self, line: str) -> QTextCharFormat:
        """Détermine la couleur de base d'une ligne selon son niveau de log."""
        fmt = QTextCharFormat()
        fmt.setFont(self.font())
        line_up = line.upper()
        if (
            "[ERROR]" in line_up
            or "[ERR]" in line_up
            or " ERROR " in f" {line_up} "
            or " ERR " in f" {line_up} "
            or " CRITICAL " in f" {line_up} "
        ):
            fmt.setForeground(QBrush(QColor("#E74C3C")))
        elif (
            "[WARNING]" in line_up
            or "[WARN]" in line_up
            or " WARNING " in f" {line_up} "
            or " WARN " in f" {line_up} "
            or " FALLBACK" in line_up
        ):
            fmt.setForeground(QBrush(QColor("#F39C12")))
        elif "[SUCCESS]" in line_up or " SUCCESS " in f" {line_up} ":
            fmt.setForeground(QBrush(QColor("#00A3FF")))
        elif "[DEBUG]" in line_up or " DEBUG " in f" {line_up} ":
            fmt.setForeground(QBrush(QColor("#585b70")))
        elif " INFO " in f" {line_up} " or "[INFO]" in line_up:
            fmt.setForeground(QBrush(QColor("#cdd6f4")))
        else:
            fmt.setForeground(QBrush(QColor("#bac2de")))
        return fmt

    def _major_line_format(self, line: str) -> Optional[QTextCharFormat]:
        """Retourne un format renforcé pour les lignes structurantes du pipeline."""
        for pattern, color, weight in _MAJOR_LINE_PATTERNS:
            if pattern.search(line):
                fmt = QTextCharFormat()
                fmt.setFont(self.font())
                fmt.setForeground(QBrush(QColor(color)))
                fmt.setFontWeight(weight)
                return fmt
        return None

    def _normalize_line(self, line: str) -> str:
        """Nettoie une ligne brute pour éviter le bruit visuel dans la console."""
        if not line:
            return ""
        cleaned = self._ansi_re.sub("", line)
        # Collapse des préfixes log dupliqués en tête de ligne.
        while True:
            collapsed = self._dup_prefix_re.sub(r"\1", cleaned)
            if collapsed == cleaned:
                break
            cleaned = collapsed
        return cleaned
