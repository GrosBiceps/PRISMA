"""
population_viz.py — Visualisations interactives des populations cellulaires.

Fonctions de visualisation Plotly pour :
  §10.4c — Blast scoring (heatmap, radar, bar chart)
  §10.4d — Traçabilité FCS (stacked bar par source pathologique)
  §10.5  — MST interactif coloré par population
  §10.5b — Grille SOM ScatterGL (MC / Condition / Population)
  §10.6  — Heatmap comparative CSV-ref vs MetaClusters (z-score)

Toutes les fonctions retournent un `plotly.graph_objects.Figure` et sauvegardent
le HTML dans output_dir si spécifié.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from flowsom_pipeline_pro.src.utils.logger import get_logger

_logger = get_logger("visualization.population_viz")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    _logger.warning(
        "plotly absent — visualisations population non disponibles (pip install plotly)"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  §10.4c — Blast scoring visualizations
# ─────────────────────────────────────────────────────────────────────────────


def plot_blast_heatmap(
    blast_df: pd.DataFrame,
    marker_cols: Optional[List[str]] = None,
    title: str = "Blast Score — Profil Marqueurs (ELN 2022)",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
) -> Optional["go.Figure"]:
    """
    Heatmap divergente des scores normalisés par marqueur pour les nœuds blast.

    Axe X = marqueurs, axe Y = nœuds (triés par blast_score décroissant).
    Colorscale : bleu = en-dessous référence, rouge = au-dessus.

    Args:
        blast_df: DataFrame de build_blast_score_dataframe.
        marker_cols: Colonnes '_M8' à visualiser (auto-detected si None).
        title: Titre du graphe.
        output_dir: Répertoire de sortie HTML.
        timestamp: Suffixe de nom de fichier.

    Returns:
        Figure Plotly ou None si plotly absent.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    # Détecter les colonnes de valeurs normalisées
    if marker_cols is None:
        marker_cols = [c for c in blast_df.columns if c.endswith("_M8")]

    if not marker_cols:
        _logger.warning("plot_blast_heatmap: aucune colonne marqueur trouvée.")
        return None

    df_sorted = blast_df.sort_values("blast_score", ascending=False)
    Z = df_sorted[marker_cols].values.astype(float)
    y_labels = [
        f"Node {int(r['node_id'])} | {r['blast_category']} | {r['blast_score']:.1f}"
        for _, r in df_sorted.iterrows()
    ]
    x_labels = [c.replace("_M8", "") for c in marker_cols]

    fig = go.Figure(
        data=go.Heatmap(
            z=Z,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu_r",
            zmid=0.0,
            colorbar=dict(title="Valeur norm."),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(tickangle=-45),
        height=max(400, len(df_sorted) * 30 + 150),
        template="plotly_dark",
        margin=dict(l=200, r=80, t=80, b=120),
    )

    if output_dir is not None:
        _save_html(fig, output_dir, f"blast_heatmap_{timestamp}.html")

    return fig


def plot_blast_radar(
    blast_df: pd.DataFrame,
    marker_cols: Optional[List[str]] = None,
    max_nodes: int = 12,
    title: str = "Profils Blast — Radar Charts",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
) -> Optional["go.Figure"]:
    """
    Radar (spider) charts : un subplot par nœud candidat blast.

    Chaque trace montre le profil multivarié du nœud dans l'espace
    de référence normalisé. Les projections > 1 indiquent une
    sur-expression par rapport à la population de référence.

    Args:
        blast_df: DataFrame de build_blast_score_dataframe.
        marker_cols: Colonnes '_M8' (auto-détectées si None).
        max_nodes: Nombre maximum de nœuds à afficher.
        title: Titre global.
        output_dir: Répertoire de sortie.
        timestamp: Suffixe du fichier.

    Returns:
        Figure Plotly.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    if marker_cols is None:
        marker_cols = [c for c in blast_df.columns if c.endswith("_M8")]

    if not marker_cols:
        return None

    category_order = ["BLAST_HIGH", "BLAST_MODERATE", "BLAST_WEAK", "NON_BLAST_UNK"]
    df_sorted = (
        blast_df.sort_values(
            ["blast_category", "blast_score"],
            key=lambda col: (
                col.map({c: i for i, c in enumerate(category_order)})
                if col.name == "blast_category"
                else col
            ),
            ascending=[True, False],
        )
        .head(max_nodes)
        .reset_index(drop=True)
    )

    n_plots = min(len(df_sorted), max_nodes)
    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    specs = [[{"type": "polar"} for _ in range(n_cols)] for _ in range(n_rows)]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        subplot_titles=[
            f"Node {int(df_sorted.loc[i, 'node_id'])} | {df_sorted.loc[i, 'blast_category']}"
            for i in range(n_plots)
        ],
    )

    theta = [c.replace("_M8", "") for c in marker_cols] + [
        marker_cols[0].replace("_M8", "")
    ]
    color_map = {
        "BLAST_HIGH": "red",
        "BLAST_MODERATE": "orange",
        "BLAST_WEAK": "yellow",
        "NON_BLAST_UNK": "lightblue",
    }

    for i, row_data in df_sorted.iterrows():
        values = list(row_data[marker_cols].values.astype(float))
        values_closed = values + [values[0]]  # fermer le polygone
        r_idx = (i // n_cols) + 1
        c_idx = (i % n_cols) + 1
        cat = str(row_data["blast_category"])
        color = color_map.get(cat, "gray")

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=theta,
                fill="toself",
                name=f"Node {int(row_data['node_id'])}",
                line=dict(color=color),
                fillcolor=color,
                opacity=0.4,
                showlegend=False,
            ),
            row=r_idx,
            col=c_idx,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=300 * n_rows,
        template="plotly_dark",
    )

    if output_dir is not None:
        _save_html(fig, output_dir, f"blast_radar_{timestamp}.html")

    return fig


def plot_blast_scores_bar(
    blast_df: pd.DataFrame,
    title: str = "Scores Blast par Nœud (ELN 2022)",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
) -> Optional["go.Figure"]:
    """
    Bar chart horizontal des scores blast, coloré par catégorie clinique.

    Lignes de référence : HIGH=6, MODERATE=3.

    Args:
        blast_df: DataFrame de build_blast_score_dataframe.
        title: Titre du graphe.
        output_dir: Répertoire de sortie.
        timestamp: Suffixe du fichier.

    Returns:
        Figure Plotly.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    color_map = {
        "BLAST_HIGH": "#d62728",
        "BLAST_MODERATE": "#ff7f0e",
        "BLAST_WEAK": "#ffdd55",
        "NON_BLAST_UNK": "#7f7f7f",
    }

    df_s = blast_df.sort_values("blast_score", ascending=True)
    y_labels = [f"Node {int(r['node_id'])}" for _, r in df_s.iterrows()]
    colors = [
        color_map.get(str(r["blast_category"]), "#999") for _, r in df_s.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_s["blast_score"].values,
            y=y_labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}" for v in df_s["blast_score"].values],
            textposition="outside",
            name="Score Blast",
        )
    )

    # Lignes de seuil ELN
    fig.add_vline(
        x=6.0,
        line_dash="dash",
        line_color="red",
        annotation_text="BLAST_HIGH (6)",
        annotation_position="top right",
    )
    fig.add_vline(
        x=3.0,
        line_dash="dot",
        line_color="orange",
        annotation_text="BLAST_MODERATE (3)",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Score /10", range=[0, 11]),
        height=max(400, len(df_s) * 25 + 150),
        template="plotly_dark",
        showlegend=False,
    )

    if output_dir is not None:
        _save_html(fig, output_dir, f"blast_scores_bar_{timestamp}.html")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  §10.4d — Traçabilité FCS source
# ─────────────────────────────────────────────────────────────────────────────


def plot_blast_fcs_source(
    source_df: pd.DataFrame,
    title: str = "Cellules Blast par Fichier Source FCS",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
) -> Optional["go.Figure"]:
    """
    Stacked bar : pour chaque nœud blast, proportion Patho/Sain par FCS source.

    Barre rouge = Patho (diagnositc/FU positif), vert = Sain/NBM.
    ALERTE CLINIQUE en annotaion si >50% Patho.

    Args:
        source_df: Résultat de trace_blast_cells_to_fcs_source.
        title: Titre du graphe.
        output_dir: Répertoire de sortie.
        timestamp: Suffixe du fichier.

    Returns:
        Figure Plotly.
    """
    if not _PLOTLY_AVAILABLE or source_df.empty:
        return None

    df_s = source_df.sort_values("blast_score", ascending=False)
    x_labels = [
        f"Node {int(r['node_id'])} ({r['blast_category']})" for _, r in df_s.iterrows()
    ]
    n_patho = df_s["n_cells_patho"].fillna(0).values
    n_sain = df_s["n_cells_sain"].fillna(0).values
    n_other = np.maximum(0, df_s["n_cells_total"].values - n_patho - n_sain)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=n_patho,
            name="Pathologique",
            marker_color="#d62728",
            text=n_patho,
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=n_sain,
            name="Sain/NBM",
            marker_color="#2ca02c",
            text=n_sain,
            textposition="inside",
        )
    )
    if n_other.sum() > 0:
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=n_other,
                name="Autre",
                marker_color="#7f7f7f",
            )
        )

    # Annotations ALERTE CLINIQUE
    for i, row in enumerate(df_s.itertuples()):
        if getattr(row, "clinical_alert", False):
            fig.add_annotation(
                x=x_labels[i],
                y=row.n_cells_total,
                text="⚠ ALERTE",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                font=dict(color="red", size=11),
                yshift=10,
            )

    fig.update_layout(
        barmode="stack",
        title=dict(text=title, x=0.5),
        xaxis=dict(tickangle=-30),
        yaxis=dict(title="N cellules"),
        height=500,
        template="plotly_dark",
    )

    if output_dir is not None:
        _save_html(fig, output_dir, f"blast_source_fcs_{timestamp}.html")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  §10.5 — MST interactif coloré par population
# ─────────────────────────────────────────────────────────────────────────────


def plot_mst_interactive(
    mapping_df: pd.DataFrame,
    node_coords_df: pd.DataFrame,
    node_counts: np.ndarray,
    mst_edges: Optional[List[Tuple[int, int]]] = None,
    color_map: Optional[Dict[str, str]] = None,
    mc_per_node: Optional[np.ndarray] = None,
    title: str = "MST FlowSOM — Populations cytométriques",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
    dashboard_dir: Optional[Path] = None,
) -> Optional["go.Figure"]:
    """
    MST interactif Plotly : bulles colorées par population, taille ∝ n_cells.

    Chaque nœud est une bulle :
      - Couleur → population assignée
      - Taille  → √(n_cells) (rendu visuel équilibré)
      - Hover   → node_id | pop | mc | n_cells | dist | seuil

    Les arêtes MST sont tracées en gris transparent.

    Args:
        mapping_df: Résultat de map_populations_to_nodes_v5 (colonnes:
                    node_id, assigned_pop, best_dist, threshold, method).
        node_coords_df: DataFrame (n_nodes, [xNodes, yNodes, xGrid, yGrid]).
        node_counts: np.ndarray (n_nodes,) — cellules par nœud.
        mst_edges: Liste de (i, j) 0-indexés (None = pas d'arêtes MST).
        color_map: dict {pop_name: hex_color}.
        mc_per_node: Métacluster par nœud (optionnel pour le hover).
        title: Titre du graphe.
        output_dir: Répertoire global de sortie.
        timestamp: Suffixe du fichier.
        dashboard_dir: Répertoire dashboard additionnel (copie du HTML).

    Returns:
        Figure Plotly.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    from flowsom_pipeline_pro.src.analysis.population_mapping import (
        POPULATION_COLORS,
        build_population_color_map,
    )

    pops = list(mapping_df["assigned_pop"].unique())
    if color_map is None:
        color_map = build_population_color_map(pops)

    # Choisir les coordonnées de layout
    x_col = "xNodes" if "xNodes" in node_coords_df.columns else "xGrid"
    y_col = "yNodes" if "yNodes" in node_coords_df.columns else "yGrid"

    n_nodes = len(mapping_df)
    x_pos = np.zeros(n_nodes)
    y_pos = np.zeros(n_nodes)
    for i, node_id in enumerate(mapping_df["node_id"].values):
        if node_id < len(node_coords_df):
            x_pos[i] = float(node_coords_df.loc[node_id, x_col])
            y_pos[i] = float(node_coords_df.loc[node_id, y_col])

    fig = go.Figure()

    # ── Arêtes MST ────────────────────────────────────────────────────────────
    if mst_edges:
        node_id_map = {
            int(row["node_id"]): idx
            for idx, (_, row) in enumerate(mapping_df.iterrows())
        }
        for e_from, e_to in mst_edges:
            i1 = node_id_map.get(e_from)
            i2 = node_id_map.get(e_to)
            if i1 is not None and i2 is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[x_pos[i1], x_pos[i2], None],
                        y=[y_pos[i1], y_pos[i2], None],
                        mode="lines",
                        line=dict(color="rgba(150,150,150,0.3)", width=0.8),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # ── Nœuds par population ─────────────────────────────────────────────────
    for pop in sorted(pops):
        mask = mapping_df["assigned_pop"] == pop
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue

        node_ids = mapping_df.loc[mask, "node_id"].values
        counts = np.array(
            [node_counts[nid] if nid < len(node_counts) else 1 for nid in node_ids]
        )
        sizes = np.sqrt(np.maximum(counts, 1)) * 3.0
        sizes = np.clip(sizes, 5, 40)

        mc_labels = [
            f"MC{int(mc_per_node[nid])}"
            if mc_per_node is not None and nid < len(mc_per_node)
            else "N/A"
            for nid in node_ids
        ]
        dists = mapping_df.loc[mask, "best_dist"].values
        thresholds = mapping_df.loc[mask, "threshold"].values

        hover_texts = [
            f"Nœud {nid}<br>Population: {pop}<br>MC: {mc}<br>"
            f"N cellules: {cnt:.0f}<br>Dist: {d:.4f}<br>Seuil: {th:.4f}"
            for nid, mc, cnt, d, th in zip(
                node_ids, mc_labels, counts, dists, thresholds
            )
        ]

        fig.add_trace(
            go.Scatter(
                x=x_pos[idxs],
                y=y_pos[idxs],
                mode="markers",
                name=pop,
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=sizes,
                    color=color_map.get(pop, "#7f7f7f"),
                    line=dict(width=0.5, color="white"),
                    opacity=0.85,
                ),
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(showgrid=False, zeroline=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, title=""),
        template="plotly_dark",
        height=700,
        legend=dict(title="Population"),
        hovermode="closest",
    )

    fname = f"mst_population_v3_{timestamp}.html"
    if output_dir is not None:
        _save_html(fig, output_dir, fname)
    if dashboard_dir is not None:
        _save_html(fig, dashboard_dir, fname)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  §10.5b — Grille SOM ScatterGL interactive
# ─────────────────────────────────────────────────────────────────────────────


def plot_som_grid_interactive(
    cell_data: Any,
    mapping_df: pd.DataFrame,
    color_map: Optional[Dict[str, str]] = None,
    condition_colors: Optional[Dict[str, str]] = None,
    viz_max_points: int = 50_000,
    title: str = "Grille SOM FlowSOM — Vue Populations",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
    dashboard_dir: Optional[Path] = None,
) -> Optional["go.Figure"]:
    """
    Trois ScatterGL côte à côte sur la grille SOM : Métacluster | Condition | Population.

    Utilise un jitter circulaire pour disperser les cellules à l'intérieur
    de chaque nœud (style FlowSOM-R).

    Args:
        cell_data: AnnData avec .obs (colonnes: FlowSOM_cluster, xGrid, yGrid,
                   FlowSOM_metacluster, Condition).
        mapping_df: Résultat du mapping population → nœud.
        color_map: dict {pop_name: hex_color}.
        condition_colors: dict {condition_name: hex_color}.
        viz_max_points: Maximum de cellules à afficher (sous-échantillonnage).
        title: Titre global.
        output_dir: Répertoire de sortie.
        timestamp: Suffixe du fichier.
        dashboard_dir: Répertoire dashboard additionnel.

    Returns:
        Figure Plotly avec 3 subplots.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    # ── Préparer les données cellulaires ─────────────────────────────────────
    try:
        obs = cell_data.obs.copy()
    except AttributeError:
        if isinstance(cell_data, pd.DataFrame):
            obs = cell_data.copy()
        else:
            _logger.warning("cell_data inaccessible pour SOM grid.")
            return None

    # Sous-échantillonnage si nécessaire
    if len(obs) > viz_max_points:
        rng = np.random.default_rng(42)
        idx_sample = rng.choice(len(obs), viz_max_points, replace=False)
        obs = obs.iloc[idx_sample].reset_index(drop=True)

    # Colonnes nécessaires
    required = ["FlowSOM_cluster", "xGrid", "yGrid"]
    for col in required:
        if col not in obs.columns:
            _logger.warning(
                "Colonne '%s' absente de cell_data.obs — SOM grid incomplet.", col
            )

    # Jitter circulaire style FlowSOM-R
    n_cells = len(obs)
    rng = np.random.default_rng(42)
    angles = rng.uniform(0, 2 * np.pi, n_cells)
    radii = rng.uniform(0, 0.4, n_cells)
    jitter_x = radii * np.cos(angles)
    jitter_y = radii * np.sin(angles)

    if "xGrid" in obs.columns and "yGrid" in obs.columns:
        x_cells = obs["xGrid"].to_numpy(dtype=float) + jitter_x
        y_cells = obs["yGrid"].to_numpy(dtype=float) + jitter_y
    else:
        x_cells = jitter_x
        y_cells = jitter_y

    # Construire le map nœud → population assignée
    node_to_pop = {}
    if "node_id" in mapping_df.columns and "assigned_pop" in mapping_df.columns:
        node_to_pop = dict(zip(mapping_df["node_id"], mapping_df["assigned_pop"]))

    # Couleurs par population
    from flowsom_pipeline_pro.src.analysis.population_mapping import (
        build_population_color_map,
    )

    all_pops = (
        list(mapping_df["assigned_pop"].unique())
        if "assigned_pop" in mapping_df.columns
        else []
    )
    if color_map is None:
        color_map = build_population_color_map(all_pops)

    # Populations par cellule
    if "FlowSOM_cluster" in obs.columns:
        cluster_arr = obs["FlowSOM_cluster"].to_numpy(dtype=np.int32) - 1  # 0-indexed
        pop_labels = np.array(
            [node_to_pop.get(int(nid), "Unknown") for nid in cluster_arr]
        )
    else:
        pop_labels = np.full(n_cells, "Unknown", dtype=object)

    pop_colors = np.array([color_map.get(p, "#7f7f7f") for p in pop_labels])

    # Métacluster par cellule
    if "FlowSOM_metacluster" in obs.columns:
        mc_arr = obs["FlowSOM_metacluster"].to_numpy(dtype=np.int32)
        n_mc = max(int(mc_arr.max()), 1) + 1
        mc_palette = (
            px.colors.qualitative.Plotly if _PLOTLY_AVAILABLE else ["#999"] * 50
        )
        mc_colors = np.array([mc_palette[int(m) % len(mc_palette)] for m in mc_arr])
        mc_labels = [f"MC{int(m)}" for m in mc_arr]
    else:
        mc_colors = np.full(n_cells, "#7f7f7f", dtype=object)
        mc_labels = ["N/A"] * n_cells

    # Condition par cellule
    if "Condition" in obs.columns:
        cond_arr = obs["Condition"].astype(str).values
    elif condition_colors and len(obs.columns) > 0:
        cond_arr = np.full(n_cells, "Unknown", dtype=object)
    else:
        cond_arr = np.full(n_cells, "Unknown", dtype=object)

    default_cond_colors = {
        "Sain": "#2ca02c",
        "Normal": "#2ca02c",
        "NBM": "#2ca02c",
        "Patho": "#d62728",
        "Diag": "#d62728",
        "Dx": "#d62728",
        "FU": "#ff7f0e",
        "Suivi": "#ff7f0e",
        "Unknown": "#7f7f7f",
    }
    if condition_colors:
        default_cond_colors.update(condition_colors)
    cond_colors = np.array(
        [
            next(
                (v for k, v in default_cond_colors.items() if k.upper() in c.upper()),
                "#7f7f7f",
            )
            for c in cond_arr
        ]
    )

    # ── Construction des 3 subplots ──────────────────────────────────────────
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Métacluster", "Condition", "Population cellulaire"],
        shared_yaxes=True,
    )

    common_kwargs = dict(
        x=x_cells,
        y=y_cells,
        mode="markers",
        marker=dict(size=2, opacity=0.6),
        showlegend=False,
        hoverinfo="skip",
    )

    # Subplot 1 : Métacluster
    fig.add_trace(
        go.Scattergl(
            **common_kwargs,
            marker=dict(size=2, color=mc_colors, opacity=0.6),
            text=mc_labels,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Subplot 2 : Condition
    fig.add_trace(
        go.Scattergl(
            **common_kwargs,
            marker=dict(size=2, color=cond_colors, opacity=0.6),
            text=cond_arr,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Subplot 3 : Population (légende)
    for pop in sorted(set(pop_labels)):
        mask_pop = pop_labels == pop
        fig.add_trace(
            go.Scattergl(
                x=x_cells[mask_pop],
                y=y_cells[mask_pop],
                mode="markers",
                name=pop,
                marker=dict(size=2, color=color_map.get(pop, "#7f7f7f"), opacity=0.6),
                showlegend=True,
                hovertemplate=f"{pop}<extra></extra>",
            ),
            row=1,
            col=3,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=600,
        template="plotly_dark",
        legend=dict(title="Population", itemsizing="constant"),
    )
    for ax in ["xaxis", "xaxis2", "xaxis3", "yaxis"]:
        fig.update_layout(**{ax: dict(showgrid=False, zeroline=False)})

    fname = f"som_grid_populations_{timestamp}.html"
    if output_dir is not None:
        _save_html(fig, output_dir, fname)
    if dashboard_dir is not None:
        _save_html(fig, dashboard_dir, fname)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  §10.6 — Heatmap comparative CSV-ref vs MetaClusters (z-score)
# ─────────────────────────────────────────────────────────────────────────────


def plot_heatmap_comparative(
    df_ref_mfi: pd.DataFrame,
    node_mfi_df: pd.DataFrame,
    mc_per_node: Optional[np.ndarray] = None,
    method_name: str = "M12",
    title: str = "Heatmap Comparative — CSV Référence vs MetaClusters SOM",
    output_dir: Optional[Path] = None,
    timestamp: str = "",
    dashboard_dir: Optional[Path] = None,
) -> Optional["go.Figure"]:
    """
    Double heatmap z-score : populations de référence CSV (gauche) vs
    centroïdes des métaclusters SOM (droite).

    Permet de vérifier visuellement la cohérence entre le mapping calculé
    et les profils biologiques de référence.

    Args:
        df_ref_mfi: DataFrame [n_pops × n_markers] — MFI CSV de référence.
        node_mfi_df: DataFrame [n_nodes × n_markers] — centroïdes SOM.
        mc_per_node: Vecteur métacluster par nœud (groupement par MC).
        method_name: Méthode de mapping pour le titre.
        title: Titre global.
        output_dir: Répertoire de sortie.
        timestamp: Suffixe du fichier.
        dashboard_dir: Répertoire dashboard.

    Returns:
        Figure Plotly avec 2 subplots côte à côte.
    """
    if not _PLOTLY_AVAILABLE:
        return None

    # Aligner les colonnes communes
    cols_common = sorted(set(df_ref_mfi.columns) & set(node_mfi_df.columns))
    if not cols_common:
        _logger.warning("plot_heatmap_comparative: aucune colonne commune.")
        return None

    X_ref = df_ref_mfi[cols_common].values.astype(float)
    pop_labels = list(df_ref_mfi.index)

    # Calculer les centroïdes par métacluster si mc_per_node fourni
    if mc_per_node is not None and len(mc_per_node) == len(node_mfi_df):
        unique_mcs = sorted(set(mc_per_node.tolist()))
        mc_means: List[np.ndarray] = []
        mc_row_labels: List[str] = []
        for mc in unique_mcs:
            mask_mc = mc_per_node == mc
            if mask_mc.any():
                mc_means.append(
                    node_mfi_df.iloc[np.where(mask_mc)[0]][cols_common].mean().values
                )
                mc_row_labels.append(f"MC{int(mc)}")
        X_mc = (
            np.array(mc_means)
            if mc_means
            else node_mfi_df[cols_common].values.astype(float)
        )
        mc_labels = (
            mc_row_labels if mc_row_labels else [f"Node {i}" for i in range(len(X_mc))]
        )
    else:
        X_mc = node_mfi_df[cols_common].values.astype(float)
        mc_labels = [f"Node {i}" for i in range(len(X_mc))]

    # Z-score par colonne (marqueur)
    def _zscore_cols(X: np.ndarray) -> np.ndarray:
        std = X.std(axis=0)
        std[std == 0] = 1.0
        return (X - X.mean(axis=0)) / std

    Z_ref = _zscore_cols(X_ref)
    Z_mc = _zscore_cols(X_mc)

    zmin = min(Z_ref.min(), Z_mc.min(), -2.0)
    zmax = max(Z_ref.max(), Z_mc.max(), 2.0)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Populations de référence CSV (z-score)",
            f"Métaclusters SOM — méthode {method_name} (z-score)",
        ],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Heatmap(
            z=Z_ref,
            x=cols_common,
            y=pop_labels,
            colorscale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
            zmid=0.0,
            colorbar=dict(title="z-score", x=0.44, len=0.8),
            showscale=True,
            hoverongaps=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=Z_mc,
            x=cols_common,
            y=mc_labels,
            colorscale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
            zmid=0.0,
            showscale=False,
            hoverongaps=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=max(500, max(len(pop_labels), len(mc_labels)) * 22 + 200),
        template="plotly_dark",
        margin=dict(l=180, r=60, t=80, b=150),
    )
    for ax in ["xaxis", "xaxis2"]:
        fig.update_layout(**{ax: dict(tickangle=-45)})

    fname = f"heatmap_{method_name}_{timestamp}.html"
    if output_dir is not None:
        _save_html(fig, output_dir, fname)
    if dashboard_dir is not None:
        _save_html(fig, dashboard_dir, fname)

    # Log de la table de correspondance population → MC dominant
    _log_population_mc_correspondence(
        df_ref_mfi, node_mfi_df, mc_per_node, cols_common, pop_labels
    )

    return fig


def _log_population_mc_correspondence(
    df_ref_mfi: pd.DataFrame,
    node_mfi_df: pd.DataFrame,
    mc_per_node: Optional[np.ndarray],
    cols_common: List[str],
    pop_labels: List[str],
) -> None:
    """Log la table : Population → MC dominant + Top 3 MC par cosine."""
    if mc_per_node is None:
        return

    from scipy.spatial.distance import cdist

    X_ref = df_ref_mfi[cols_common].values.astype(float)
    Z_ref = X_ref - X_ref.mean(axis=0)

    unique_mcs = sorted(set(mc_per_node.tolist()))
    mc_means = {
        mc: node_mfi_df.iloc[np.where(mc_per_node == mc)[0]][cols_common].mean().values
        for mc in unique_mcs
        if (mc_per_node == mc).any()
    }

    if not mc_means:
        return

    mc_keys = list(mc_means.keys())
    X_mc = np.array([mc_means[k] for k in mc_keys])
    Z_mc = X_mc - X_mc.mean(axis=0)

    D = cdist(Z_ref, Z_mc, metric="cosine")

    _logger.info("Table correspondance Population → Métacluster:")
    _logger.info("  %-30s | MC dominant | Top 3 MC", "Population")
    _logger.info("  " + "-" * 65)
    for i, pop in enumerate(pop_labels):
        sorted_mc_idx = np.argsort(D[i])
        dom_mc = mc_keys[sorted_mc_idx[0]]
        top3 = [f"MC{mc_keys[j]}" for j in sorted_mc_idx[:3]]
        _logger.info("  %-30s | MC%-8d | %s", pop, dom_mc, ", ".join(top3))


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitaire de sauvegarde HTML
# ─────────────────────────────────────────────────────────────────────────────


def _save_html(
    fig: "go.Figure",
    output_dir: Path,
    filename: str,
    include_plotlyjs: str = "cdn",
) -> Path:
    """Sauvegarde un figure Plotly en HTML dans output_dir/other/."""
    dest_dir = Path(output_dir) / "other"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    fig.write_html(str(dest), include_plotlyjs=include_plotlyjs)
    _logger.info("Sauvegardé: %s", dest)
    return dest
