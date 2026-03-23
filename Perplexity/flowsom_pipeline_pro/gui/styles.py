# -*- coding: utf-8 -*-
"""
styles.py — Thème Catppuccin Mocha / Glassmorphism pour FlowSomAnalyzerPro.

Palette Catppuccin Mocha :
    Base       #1e1e2e    Mantle   #181825    Crust    #11111b
    Surface0   #313244    Surface1 #45475a    Surface2 #585b70
    Overlay0   #6c7086    Text     #cdd6f4    Subtext  #a6adc8
    Blue       #89b4fa    Lavender #b4befe    Mauve    #cba6f7
    Pink       #f5c2e7    Red      #f38ba8    Peach    #fab387
    Green      #a6e3a1    Teal     #94e2d5    Sky      #89dceb
    Sapphire   #74c7ec
"""

# ---------------------------------------------------------------------------
# Couleurs réutilisables
# ---------------------------------------------------------------------------
COLORS = {
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "overlay0": "#6c7086",
    "text": "#cdd6f4",
    "subtext": "#a6adc8",
    "blue": "#89b4fa",
    "lavender": "#b4befe",
    "mauve": "#cba6f7",
    "pink": "#f5c2e7",
    "red": "#f38ba8",
    "peach": "#fab387",
    "green": "#a6e3a1",
    "teal": "#94e2d5",
    "sky": "#89dceb",
    "sapphire": "#74c7ec",
}

# ---------------------------------------------------------------------------
# QSS global
# ---------------------------------------------------------------------------
STYLESHEET = """
/* ============================================
   THÈME PRINCIPAL — CATPPUCCIN MOCHA
   ============================================ */

QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #11111b, stop:0.5 #1e1e2e, stop:1 #181825);
}

QWidget {
    background-color: transparent;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'SF Pro Display', Arial, sans-serif;
    font-size: 10pt;
}

QWidget#centralWidget {
    background: transparent;
}

QWidget#leftPanel {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(49, 50, 68, 0.95), stop:1 rgba(30, 30, 46, 0.95));
    border-right: 1px solid rgba(137, 180, 250, 0.3);
}

QWidget#rightPanel {
    background: rgba(30, 30, 46, 0.7);
}

/* ============================================
   GROUPBOX — GLASSMORPHISM
   ============================================ */

QGroupBox {
    font-weight: 600;
    font-size: 11pt;
    border: 1px solid rgba(137, 180, 250, 0.2);
    border-radius: 12px;
    margin-top: 20px;
    padding: 20px 15px 15px 15px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(69, 71, 90, 0.6), stop:1 rgba(49, 50, 68, 0.8));
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    top: 5px;
    padding: 5px 15px;
    color: #89b4fa;
    background: linear-gradient(90deg, rgba(137, 180, 250, 0.2), transparent);
    border-radius: 8px;
    font-size: 11pt;
}

/* ============================================
   BOUTONS
   ============================================ */

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #585b70, stop:1 #45475a);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 10px 20px;
    color: #cdd6f4;
    font-weight: 600;
    font-size: 10pt;
    min-height: 18px;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #6c7086, stop:1 #585b70);
    border: 1px solid rgba(137, 180, 250, 0.4);
}

QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #45475a, stop:1 #313244);
}

QPushButton:disabled {
    background: rgba(49, 50, 68, 0.5);
    color: #6c7086;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

QPushButton#primaryBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #89b4fa, stop:1 #74c7ec);
    color: #11111b;
    border: none;
    font-weight: 700;
}

QPushButton#primaryBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #b4befe, stop:1 #89dceb);
}

QPushButton#primaryBtn:disabled {
    background: rgba(137, 180, 250, 0.3);
    color: rgba(17, 17, 27, 0.5);
}

QPushButton#successBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #a6e3a1, stop:1 #94e2d5);
    color: #11111b;
    border: none;
    font-weight: 600;
}

QPushButton#successBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #b4f5b0, stop:1 #a6f3e8);
}

QPushButton#dangerBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #f38ba8, stop:1 #fab387);
    color: #11111b;
    border: none;
    font-weight: 600;
}

QPushButton#dangerBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #f5a0ba, stop:1 #fcc8a0);
}

QPushButton#exportBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #cba6f7, stop:1 #f5c2e7);
    color: #11111b;
    border: none;
    font-weight: 600;
}

QPushButton#exportBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #dbb8ff, stop:1 #f8d5ed);
}

/* ============================================
   LABELS
   ============================================ */

QLabel {
    color: #cdd6f4;
    background: transparent;
}

QLabel#titleLabel {
    font-size: 20pt;
    font-weight: 700;
    color: #89b4fa;
}

QLabel#subtitleLabel {
    font-size: 10pt;
    color: #a6adc8;
    font-style: italic;
}

QLabel#sectionLabel {
    font-size: 11pt;
    font-weight: 600;
    color: #89b4fa;
}

QLabel#fileLabel {
    font-size: 9pt;
    color: #a6adc8;
    padding: 5px 10px;
    background: rgba(69, 71, 90, 0.4);
    border-radius: 6px;
}

QLabel#statusSuccess {
    color: #a6e3a1;
    font-weight: 600;
}

QLabel#statusError {
    color: #f38ba8;
    font-weight: 600;
}

/* ============================================
   INPUTS
   ============================================ */

QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background: rgba(69, 71, 90, 0.6);
    border: 1px solid rgba(137, 180, 250, 0.2);
    border-radius: 8px;
    padding: 8px 10px;
    color: #cdd6f4;
    min-height: 18px;
    font-size: 10pt;
    selection-background-color: #89b4fa;
}

QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QLineEdit:hover {
    border: 1px solid rgba(137, 180, 250, 0.4);
    background: rgba(69, 71, 90, 0.8);
}

QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
    border: 2px solid #89b4fa;
    background: rgba(69, 71, 90, 0.9);
}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: rgba(137, 180, 250, 0.2);
    border: none;
    border-radius: 4px;
    width: 18px;
    margin: 2px;
}

QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background: rgba(137, 180, 250, 0.4);
}

QComboBox::drop-down {
    border: none;
    width: 28px;
    background: rgba(137, 180, 250, 0.2);
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}

/* Conteneur natif du popup (QComboBoxPrivateContainer) */
QComboBoxPrivateContainer {
    background: #313244;
    border: 1px solid rgba(137, 180, 250, 0.3);
    margin: 0px;
    padding: 0px;
}

/* Petites barres de défilement haut/bas — source des barres blanches */
QComboBoxPrivateScroller {
    background: #313244;
    border: none;
    margin: 0px;
    padding: 0px;
    min-height: 0px;
}

/* Viewport et tous les widgets enfants du popup */
QComboBoxPrivateContainer QWidget {
    background: #313244;
    border: none;
}

QComboBox QAbstractItemView {
    background: #313244;
    /* Ligne magique : même couleur que le fond pour masquer le cadre blanc Windows */
    border: 1px solid #313244;
    selection-background-color: #89b4fa;
    selection-color: #11111b;
    outline: none;
    padding: 0px;
    margin: 0px;
}

QComboBox QAbstractItemView::item {
    background: #313244;
    color: #cdd6f4;
    padding: 6px 12px;
    border: none;
    min-height: 22px;
}

QComboBox QAbstractItemView::item:hover,
QComboBox QAbstractItemView::item:focus {
    background: rgba(137, 180, 250, 0.2);
    color: #cdd6f4;
}

QComboBox QAbstractItemView::item:selected {
    background: #89b4fa;
    color: #11111b;
}

QComboBox QScrollBar:vertical {
    background: #313244;
    width: 8px;
    border: none;
}

QComboBox QScrollBar::handle:vertical {
    background: rgba(137, 180, 250, 0.4);
    border-radius: 4px;
    min-height: 20px;
}

QComboBox QFrame {
    background: #313244;
    border: none;
}

/* ============================================
   CHECKBOX
   ============================================ */

QCheckBox {
    spacing: 10px;
    color: #cdd6f4;
    font-size: 10pt;
}

QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 6px;
    border: 2px solid rgba(137, 180, 250, 0.3);
    background: rgba(69, 71, 90, 0.4);
}

QCheckBox::indicator:hover {
    border-color: rgba(137, 180, 250, 0.6);
    background: rgba(69, 71, 90, 0.6);
}

QCheckBox::indicator:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #89b4fa, stop:1 #74c7ec);
    border-color: #89b4fa;
}

/* ============================================
   PROGRESSBAR ANIMÉE
   ============================================ */

QProgressBar {
    border: none;
    border-radius: 10px;
    background: rgba(69, 71, 90, 0.4);
    text-align: center;
    color: #cdd6f4;
    font-weight: 600;
    font-size: 9pt;
    min-height: 20px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #89b4fa, stop:0.5 #cba6f7, stop:1 #f5c2e7);
    border-radius: 10px;
}

/* ============================================
   STATUSBAR
   ============================================ */

QStatusBar {
    background: rgba(49, 50, 68, 0.9);
    color: #a6adc8;
    border-top: 1px solid rgba(137, 180, 250, 0.2);
    font-size: 9pt;
    padding: 5px;
}

/* ============================================
   TABWIDGET
   ============================================ */

QTabWidget::pane {
    border: 1px solid rgba(137, 180, 250, 0.2);
    border-radius: 12px;
    background: rgba(49, 50, 68, 0.6);
    padding-top: 5px;
}

QTabWidget::tab-bar {
    left: 10px;
}

QTabBar {
    qproperty-drawBase: 0;
}

QTabBar::tab {
    background: rgba(69, 71, 90, 0.6);
    border: 1px solid rgba(137, 180, 250, 0.15);
    border-bottom: none;
    padding: 10px 18px;
    margin-right: 4px;
    margin-top: 3px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    color: #a6adc8;
    font-weight: 500;
    font-size: 9pt;
    min-width: 80px;
}

QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #89b4fa, stop:1 #cba6f7);
    color: #11111b;
    font-weight: 600;
    margin-top: 0px;
    padding-bottom: 12px;
}

QTabBar::tab:hover:!selected {
    background: rgba(137, 180, 250, 0.3);
    color: #cdd6f4;
}

/* ============================================
   SCROLLAREA / SCROLLBAR
   ============================================ */

QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background: rgba(69, 71, 90, 0.3);
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: rgba(137, 180, 250, 0.4);
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(137, 180, 250, 0.6);
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: rgba(69, 71, 90, 0.3);
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: rgba(137, 180, 250, 0.4);
    border-radius: 5px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: rgba(137, 180, 250, 0.6);
}

/* ============================================
   TABLEWIDGET
   ============================================ */

QTableWidget {
    background: rgba(49, 50, 68, 0.6);
    border: 1px solid rgba(137, 180, 250, 0.2);
    border-radius: 10px;
    gridline-color: rgba(137, 180, 250, 0.1);
    font-size: 10pt;
}

QTableWidget::item {
    padding: 8px;
    color: #cdd6f4;
    border-bottom: 1px solid rgba(137, 180, 250, 0.1);
}

QTableWidget::item:selected {
    background: rgba(137, 180, 250, 0.3);
    color: #cdd6f4;
}

QTableWidget::item:hover {
    background: rgba(137, 180, 250, 0.15);
}

QHeaderView::section {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(137, 180, 250, 0.3), stop:1 rgba(137, 180, 250, 0.1));
    color: #cdd6f4;
    padding: 10px;
    border: none;
    font-weight: 600;
    font-size: 10pt;
}

/* ============================================
   TEXTEDIT (LOGS)
   ============================================ */

QTextEdit {
    background: rgba(17, 17, 27, 0.8);
    border: 1px solid rgba(137, 180, 250, 0.2);
    border-radius: 10px;
    color: #a6e3a1;
    padding: 12px;
    font-family: 'Consolas', 'Fira Code', monospace;
    font-size: 9pt;
    selection-background-color: #89b4fa;
}

/* ============================================
   LISTWIDGET (CLUSTERS)
   ============================================ */

QListWidget {
    background: rgba(49, 50, 68, 0.6);
    border: 1px solid rgba(137, 180, 250, 0.2);
    border-radius: 10px;
    padding: 5px;
    font-size: 10pt;
    outline: none;
}

QListWidget::item {
    padding: 8px 12px;
    border-radius: 6px;
    margin: 2px 4px;
    color: #cdd6f4;
}

QListWidget::item:selected {
    background: rgba(137, 180, 250, 0.3);
    color: #cdd6f4;
    border: 1px solid rgba(137, 180, 250, 0.4);
}

QListWidget::item:hover:!selected {
    background: rgba(137, 180, 250, 0.12);
}

/* ============================================
   SPLITTER
   ============================================ */

QSplitter::handle {
    background: rgba(137, 180, 250, 0.2);
    width: 3px;
}

QSplitter::handle:hover {
    background: rgba(137, 180, 250, 0.5);
}

/* ============================================
   TOOLTIPS
   ============================================ */

QToolTip {
    background: #313244;
    color: #cdd6f4;
    border: 1px solid rgba(137, 180, 250, 0.3);
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 9pt;
}
"""
