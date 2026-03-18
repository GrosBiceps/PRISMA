"""
html_report.py — Génération d'un rapport HTML self-contained avec Plotly + Matplotlib.

Assemble toutes les figures interactives (Plotly) et statiques (Matplotlib)
en un seul fichier HTML autonome, sans dépendance CDN ou fichier externe.

Le rapport inclut :
- En-tête avec résumé de l'analyse
- Table des matières navigable
- Paramètres de l'analyse
- Statistiques par métacluster
- Figures Plotly interactives (Sankey, heatmaps, spider plots, etc.)
- Figures Matplotlib en base64 inline (gating, RANSAC, etc.)
"""

from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from flowsom_pipeline_pro.src.utils.logger import get_logger

_logger = get_logger("visualization.html_report")

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    import plotly.offline

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

try:
    import matplotlib.figure

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def _fig_to_base64(fig_mpl: Any) -> str:
    """Convertit une figure matplotlib en string base64 PNG."""
    buf = BytesIO()
    fig_mpl.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _plotly_to_html_div(fig_plotly: Any, fig_id: str = "") -> str:
    """Convertit une figure Plotly en div HTML (sans plotly.js embarqué)."""
    fig_height = fig_plotly.layout.height or 500
    fig_width_val = fig_plotly.layout.width
    default_w = f"{fig_width_val}px" if fig_width_val else "100%"
    return pio.to_html(
        fig_plotly,
        full_html=False,
        include_plotlyjs=False,
        div_id=fig_id if fig_id else None,
        default_height=f"{fig_height}px",
        default_width=default_w,
        config={"responsive": True},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CSS Template (réplique exacte du monolithique)
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
:root {
    --primary: #667eea;
    --primary-dark: #764ba2;
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --text: #2d3748;
    --text-light: #718096;
    --border: #e2e8f0;
    --success: #48bb78;
    --warning: #ed8936;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
}
.header {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white; padding: 40px 0; text-align: center; margin-bottom: 30px;
}
.header h1 { font-size: 2.2em; margin-bottom: 8px; font-weight: 700; }
.header .subtitle { font-size: 1.1em; opacity: 0.9; }
.container { max-width: 1400px; margin: 0 auto; padding: 0 20px; }
.section {
    background: var(--card-bg); border-radius: 12px; padding: 30px;
    margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border: 1px solid var(--border);
}
.section h2 {
    font-size: 1.4em; color: var(--primary); margin-bottom: 20px;
    padding-bottom: 10px; border-bottom: 2px solid var(--border);
}
.grid-3 {
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;
}
.stat-card {
    background: linear-gradient(135deg, #f6f8ff, #f0f4ff);
    border-radius: 10px; padding: 20px; text-align: center;
    border: 1px solid #dde4f0;
}
.stat-card .value { font-size: 2em; font-weight: 700; color: var(--primary); }
.stat-card .label { font-size: 0.9em; color: var(--text-light); margin-top: 4px; }
table { width: 100%; border-collapse: collapse; margin-top: 15px; }
th {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white; padding: 12px 16px; text-align: left; font-weight: 600;
}
td { padding: 10px 16px; border-bottom: 1px solid var(--border); }
tr:nth-child(even) { background: #f7fafc; }
tr:hover { background: #edf2f7; }
.marker-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea22, #764ba222);
    color: var(--primary-dark); padding: 4px 12px; border-radius: 20px;
    font-size: 0.85em; margin: 3px; border: 1px solid #667eea44;
    font-weight: 500;
}
.plotly-container {
    width: 100%; overflow-x: auto; display: flex; justify-content: center;
}
.plotly-container > div { min-width: 0; }
.param-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 12px;
}
.param-item {
    background: #f7fafc; padding: 10px 15px; border-radius: 8px;
    border-left: 3px solid var(--primary);
}
.param-item .param-label {
    font-size: 0.8em; color: var(--text-light); text-transform: uppercase;
    letter-spacing: 0.5px;
}
.param-item .param-value { font-size: 1.1em; font-weight: 600; color: var(--text); }
.toc {
    background: #f0f4ff; border-radius: 10px; padding: 20px 30px;
    margin-bottom: 24px;
}
.toc h3 { margin-bottom: 10px; color: var(--primary-dark); }
.toc ul { list-style: none; columns: 2; }
.toc li { padding: 4px 0; }
.toc a { color: var(--primary); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.footer {
    text-align: center; padding: 30px; color: var(--text-light);
    font-size: 0.9em;
}
@media (max-width: 768px) {
    .grid-3 { grid-template-columns: 1fr; }
    .toc ul { columns: 1; }
}
"""


def generate_html_report(
    output_path: Path | str,
    *,
    plotly_figures: Optional[Dict[str, Any]] = None,
    matplotlib_figures: Optional[Dict[str, Any]] = None,
    figure_labels: Optional[Dict[str, str]] = None,
    analysis_params: Optional[Dict[str, Any]] = None,
    summary_stats: Optional[Dict[str, Any]] = None,
    metacluster_table: Optional[List[Dict[str, Any]]] = None,
    markers: Optional[List[str]] = None,
    condition_data: Optional[List[Dict[str, Any]]] = None,
    files_data: Optional[List[Dict[str, Any]]] = None,
    export_paths: Optional[Dict[str, str]] = None,
    self_contained: bool = True,
) -> bool:
    """
    Génère un rapport HTML complet avec toutes les visualisations.

    Le rapport est self-contained : plotly.js est embarqué directement dans le
    fichier HTML pour un fonctionnement hors-ligne.

    Args:
        output_path: Chemin du fichier HTML de sortie.
        plotly_figures: Dict {nom: go.Figure} — figures Plotly interactives.
        matplotlib_figures: Dict {nom: mpl.Figure} — figures matplotlib.
        figure_labels: Dict {nom: "Titre lisible"} pour les légendes.
        analysis_params: Dict des paramètres d'analyse (transformation, grille, etc.).
        summary_stats: Dict avec n_cells, n_markers, n_files, n_clusters.
        metacluster_table: Liste de dicts [{metacluster, n_cells, pct, top_markers}].
        markers: Liste des marqueurs utilisés pour le clustering.
        self_contained: Si True, embarque plotly.js (~3.5 MB). Sinon, CDN.

    Returns:
        True si succès.
    """
    if not _PLOTLY_AVAILABLE:
        _logger.error("plotly requis pour le rapport HTML")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotly_figures = plotly_figures or {}
    matplotlib_figures = matplotlib_figures or {}
    figure_labels = figure_labels or {}
    analysis_params = analysis_params or {}
    summary_stats = summary_stats or {}
    metacluster_table = metacluster_table or []
    markers = markers or []
    condition_data = condition_data or []
    files_data = files_data or []
    export_paths = export_paths or {}

    now_str = datetime.now().strftime("%d/%m/%Y à %H:%M")
    n_cells = summary_stats.get("n_cells", 0)
    n_markers = summary_stats.get("n_markers", len(markers))
    n_files = summary_stats.get("n_files", 0)
    n_clusters = summary_stats.get("n_clusters", 0)

    # ── Script Plotly.js ──────────────────────────────────────────────────
    if self_contained:
        plotly_js = plotly.offline.get_plotlyjs()
        plotly_script = f'<script type="text/javascript">{plotly_js}</script>'
    else:
        plotly_script = (
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
        )

    # ── Sections de paramètres ────────────────────────────────────────────
    param_items = ""
    for key, val in analysis_params.items():
        param_items += (
            f'<div class="param-item">'
            f'<div class="param-label">{key}</div>'
            f'<div class="param-value">{val}</div>'
            f"</div>\n"
        )

    # ── Statistiques KPI ──────────────────────────────────────────────────
    stats_cards = f"""
    <div class="grid-3">
        <div class="stat-card">
            <div class="value">{n_cells:,}</div>
            <div class="label">Cellules totales</div>
        </div>
        <div class="stat-card">
            <div class="value">{n_markers}</div>
            <div class="label">Marqueurs (clustering)</div>
        </div>
        <div class="stat-card">
            <div class="value">{n_files}</div>
            <div class="label">Fichiers analysés</div>
        </div>
    </div>
    """

    # ── Tableau métaclusters ──────────────────────────────────────────────
    mc_rows = ""
    for row in metacluster_table:
        mc_rows += (
            f"<tr>"
            f'<td style="font-weight:bold; text-align:center;">{row.get("metacluster", "")}</td>'
            f'<td style="text-align:right;">{row.get("n_cells", 0):,}</td>'
            f'<td style="text-align:right;">{row.get("pct", 0):.1f}%</td>'
            f"<td>{row.get('top_markers', 'N/A')}</td>"
            f"</tr>\n"
        )

    # ── Badges marqueurs ─────────────────────────────────────────────────
    markers_html = "\n".join(f'<span class="marker-badge">{m}</span>' for m in markers)

    # ── Données par condition ────────────────────────────────────────────
    cond_rows = ""
    for row in condition_data:
        cond_rows += (
            f"<tr>"
            f'<td style="font-weight:bold;">{row.get("condition", "")}</td>'
            f'<td style="text-align:right;">{row.get("n_cells", 0):,}</td>'
            f'<td style="text-align:right;">{row.get("pct", 0):.1f}%</td>'
            f"</tr>\n"
        )

    # ── Données par fichier source ────────────────────────────────────────
    files_rows = ""
    for row in files_data:
        files_rows += (
            f"<tr>"
            f"<td>{row.get('file', '')}</td>"
            f'<td style="text-align:right;">{row.get("n_cells", 0):,}</td>'
            f"</tr>\n"
        )

    # ── Table exports ─────────────────────────────────────────────────────
    _EXPORT_LABELS = {
        "csv_complete": "CSV complet",
        "fcs_complete": "FCS (Kaluza compatible)",
        "csv_statistics": "Statistiques par cluster",
        "csv_mfi": "Matrice MFI",
        "json_metadata": "Métadonnées JSON",
        "gating_log": "Log de gating JSON",
        "html_report": "Rapport HTML",
        "sankey_global": "Sankey global (HTML interactif)",
        "mrd_results": "Résultats MRD (JSON)",
    }
    export_rows = ""
    for key, path_val in export_paths.items():
        if not isinstance(path_val, str) or not path_val.endswith(
            (".csv", ".fcs", ".json", ".html", ".png")
        ):
            continue
        label = _EXPORT_LABELS.get(key, key)
        export_rows += (
            f"<tr><td>{label}</td>"
            f"<td style='font-family:monospace; font-size:0.85em;'>{path_val}</td></tr>\n"
        )

    # ── Sections Plotly ──────────────────────────────────────────────────
    plotly_sections = ""
    for fig_name, fig_obj in plotly_figures.items():
        label = figure_labels.get(fig_name, fig_name)
        try:
            div_html = _plotly_to_html_div(fig_obj, fig_id=fig_name)
            plotly_sections += (
                f'<div class="section">\n'
                f"  <h2>{label}</h2>\n"
                f'  <div class="plotly-container">{div_html}</div>\n'
                f"</div>\n"
            )
        except Exception as exc:
            _logger.warning("Erreur conversion Plotly %s: %s", fig_name, exc)

    # ── Sections Matplotlib ──────────────────────────────────────────────
    mpl_sections = ""
    if _MPL_AVAILABLE:
        for fig_name, fig_obj in matplotlib_figures.items():
            label = figure_labels.get(fig_name, fig_name)
            try:
                b64 = _fig_to_base64(fig_obj)
                mpl_sections += (
                    f'<div class="section">\n'
                    f"  <h2>{label}</h2>\n"
                    f'  <div style="text-align:center;">\n'
                    f'    <img src="data:image/png;base64,{b64}" '
                    f'style="max-width:100%; border-radius:8px; '
                    f'box-shadow:0 2px 8px rgba(0,0,0,0.1);" />\n'
                    f"  </div>\n"
                    f"</div>\n"
                )
            except Exception as exc:
                _logger.warning("Erreur conversion Matplotlib %s: %s", fig_name, exc)

    # ── Assemblage HTML ──────────────────────────────────────────────────
    _cond_section = ""
    if cond_rows:
        _cond_section = f"""
    <h3 style="margin-top:25px; margin-bottom:10px;">Par condition</h3>
    <table>
        <tr><th>Condition</th><th>Cellules</th><th>Pourcentage</th></tr>
        {cond_rows}
    </table>"""

    _files_section = ""
    if files_rows:
        _files_section = f"""
    <h3 style="margin-top:25px; margin-bottom:10px;">Par fichier source</h3>
    <table>
        <tr><th>Fichier</th><th>Cellules</th></tr>
        {files_rows}
    </table>"""

    _exports_section = ""
    if export_rows:
        _exports_section = f"""
<div class="section" id="exports">
    <h2>7. Fichiers Exportés</h2>
    <table>
        <tr><th>Type</th><th>Fichier</th></tr>
        {export_rows}
    </table>
</div>"""

    _toc_exports = (
        '\n        <li><a href="#exports">7. Fichiers exportés</a></li>'
        if export_rows
        else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlowSOM Analysis Report — {now_str}</title>
    {plotly_script}
    <style>{_CSS}</style>
</head>
<body>

<div class="header">
    <div class="container">
        <h1>FlowSOM Analysis Report</h1>
        <div class="subtitle">
            Analyse générée le {now_str} —
            {n_cells:,} cellules · {n_markers} marqueurs · {n_clusters} métaclusters
        </div>
    </div>
</div>

<div class="container">

<div class="toc">
    <h3>Table des matières</h3>
    <ul>
        <li><a href="#params">1. Paramètres de l'analyse</a></li>
        <li><a href="#data">2. Résumé des données</a></li>
        <li><a href="#markers">3. Marqueurs utilisés</a></li>
        <li><a href="#metaclusters">4. Métaclusters</a></li>
        <li><a href="#plotly-viz">5. Visualisations interactives</a></li>
        <li><a href="#static-viz">6. Visualisations statiques</a></li>{_toc_exports}
    </ul>
</div>

<div class="section" id="params">
    <h2>1. Paramètres de l'Analyse</h2>
    <div class="param-grid">
        {param_items}
    </div>
</div>

<div class="section" id="data">
    <h2>2. Résumé des Données</h2>
    {stats_cards}
    {_cond_section}
    {_files_section}
</div>

<div class="section" id="markers">
    <h2>3. Marqueurs Utilisés pour le Clustering</h2>
    <p style="margin-bottom:15px; color:var(--text-light);">
        {n_markers} marqueurs sélectionnés (scatter et Time exclus)
    </p>
    {markers_html}
</div>

<div class="section" id="metaclusters">
    <h2>4. Résumé des Métaclusters</h2>
    <table>
        <tr>
            <th>Métacluster</th>
            <th>Cellules</th>
            <th>% Total</th>
            <th>Top 3 Marqueurs</th>
        </tr>
        {mc_rows}
    </table>
</div>

<div id="plotly-viz">
    <div class="section">
        <h2>5. Visualisations Interactives (Plotly)</h2>
        <p style="color:var(--text-light); margin-bottom:10px;">
            {len(plotly_figures)} figures interactives — zoom, pan, hover
        </p>
    </div>
    {plotly_sections}
</div>

<div id="static-viz">
    <div class="section">
        <h2>6. Visualisations Statiques (Matplotlib)</h2>
        <p style="color:var(--text-light); margin-bottom:10px;">
            {len(matplotlib_figures)} figures haute résolution
        </p>
    </div>
    {mpl_sections}
</div>

{_exports_section}

</div>

<div class="footer">
    <p>FlowSOM Analysis Pipeline Pro — Rapport généré le {now_str}</p>
    <p>{n_cells:,} cellules · {n_markers} marqueurs · {n_clusters} métaclusters</p>
</div>

</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    html_size_mb = output_path.stat().st_size / (1024 * 1024)
    _logger.info(
        "Rapport HTML exporté: %s (%.1f MB, %d Plotly + %d Matplotlib)",
        output_path.name,
        html_size_mb,
        len(plotly_figures),
        len(matplotlib_figures),
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Alias publics — compatibilité avec flowsom_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────


def fig_to_base64(fig_mpl: Any) -> str:
    """
    Alias public de ``_fig_to_base64``.

    Convertit une figure matplotlib en chaîne base64 PNG embarquable
    directement dans un bloc HTML ``<img src="data:image/png;base64,...">``.

    Args:
        fig_mpl: Figure matplotlib.

    Returns:
        Chaîne base64 encodée (str).
    """
    return _fig_to_base64(fig_mpl)


def plotly_to_html_div(fig_plotly: Any, fig_id: str = "") -> str:
    """
    Alias public de ``_plotly_to_html_div``.

    Convertit une figure Plotly en div HTML auto-contenu (sans CDN externe),
    prêt à être inséré dans un rapport HTML.

    Args:
        fig_plotly: Figure Plotly.
        fig_id: Identifiant CSS optionnel du div englobant.

    Returns:
        Chaîne HTML contenant le div Plotly.
    """
    return _plotly_to_html_div(fig_plotly, fig_id)
