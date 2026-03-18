"""
flowsom_plots.py — Visualisations FlowSOM (MST, heatmap MFI, UMAP).

Génère les graphiques spécifiques à l'analyse FlowSOM:
  - Heatmap des profils d'expression par métacluster
  - UMAP coloré par métacluster (attention au réglage de max_pts pour les gros datasets)
  - Graphique en barres de la taille des métaclusters
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.figure as _mpl_figure

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import seaborn as sns

    _SNS_AVAILABLE = True
except ImportError:
    _SNS_AVAILABLE = False

from flowsom_pipeline_pro.src.visualization.plot_helpers import (
    BG_COLOR,
    TEXT_COLOR,
    SPINE_COLOR,
    apply_dark_style,
    save_figure,
)
from flowsom_pipeline_pro.src.utils.logger import get_logger

_logger = get_logger("visualization.flowsom_plots")


def plot_mfi_heatmap(
    mfi_matrix: pd.DataFrame,
    output_path: Path | str,
    *,
    figsize: Tuple[int, int] = (16, 8),
    cmap: str = "magma",
    title: str = "Profils d'expression par métacluster (MFI normalisée)",
) -> Optional["matplotlib.figure.Figure"]:
    """
    Heatmap des profils d'expression MFI par métacluster.

    Chaque ligne = un métacluster, chaque colonne = un marqueur.
    Les valeurs sont normalisées par marqueur (z-score) pour
    rendre toutes les intensités comparables visuellement.

    Args:
        mfi_matrix: DataFrame [n_clusters × n_markers] avec les MFI.
        output_path: Chemin PNG de sortie.
        figsize: Taille de la figure.
        cmap: Colormap matplotlib.
        title: Titre du graphique.

    Returns:
        True si succès.
    """
    if not _MPL_AVAILABLE or not _SNS_AVAILABLE:
        _logger.warning("matplotlib et seaborn requis pour plot_mfi_heatmap")
        return None

    try:
        # Normalisation par marqueur (z-score colonne)
        mfi_norm = mfi_matrix.copy().astype(float)
        stds = mfi_norm.std()
        stds[stds == 0] = 1.0
        mfi_norm = (mfi_norm - mfi_norm.mean()) / stds

        fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)

        sns.heatmap(
            mfi_norm,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor=SPINE_COLOR,
            cbar_kws={"label": "z-score MFI", "shrink": 0.8},
        )

        ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=15)
        ax.set_xlabel("Marqueurs", fontsize=12, color=TEXT_COLOR)
        ax.set_ylabel("Métacluster", fontsize=12, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=10, rotation=45)
        ax.set_facecolor(BG_COLOR)

        # Style colorbar
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(colors=TEXT_COLOR)
            cbar.ax.yaxis.label.set_color(TEXT_COLOR)

        save_figure(fig, output_path)
        _logger.info("Heatmap MFI sauvegardée: %s", output_path)
        return fig

    except Exception as exc:
        _logger.error("Échec plot_mfi_heatmap: %s", exc)
        return None


def plot_metacluster_sizes(
    metaclustering: np.ndarray,
    n_metaclusters: int,
    output_path: Path | str,
    condition_labels: Optional[np.ndarray] = None,
    title: str = "Distribution des cellules par métacluster",
) -> Optional["matplotlib.figure.Figure"]:
    """
    Figure 1×2 : pie chart (gauche) + bar chart horizontal (droite).

    Fidèle à la section §15 du pipeline monolithique flowsom_pipeline.py :
    le panneau gauche est un camembert des proportions, le panneau droit
    un bar chart horizontal des tailles absolues.
    Si condition_labels est fourni, le panneau droit utilise des barres
    empilées par condition (Sain/Patho).

    Args:
        metaclustering: Assignation par cellule (n_cells,).
        n_metaclusters: Nombre total de métaclusters.
        output_path: Chemin PNG de sortie.
        condition_labels: Labels de condition par cellule (optionnel).
        title: Titre du graphique.

    Returns:
        Figure matplotlib ou None si matplotlib absent.
    """
    if not _MPL_AVAILABLE:
        return None

    try:
        cluster_ids = np.arange(n_metaclusters)
        counts_total = np.bincount(metaclustering.astype(int), minlength=n_metaclusters)
        labels = [f"MC{i}" for i in cluster_ids]
        colors = plt.cm.tab20(np.linspace(0, 1, n_metaclusters))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_COLOR)

        # ── Panneau gauche : Pie chart ───────────────────────────────────────
        ax_pie = axes[0]
        ax_pie.set_facecolor(BG_COLOR)
        wedges, texts, autotexts = ax_pie.pie(
            counts_total,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            pctdistance=0.8,
            textprops={"color": TEXT_COLOR, "fontsize": 9},
        )
        for at in autotexts:
            at.set_color(TEXT_COLOR)
            at.set_fontsize(8)
        ax_pie.set_title(
            "Distribution des Cellules par Métacluster",
            fontsize=12,
            fontweight="bold",
            color=TEXT_COLOR,
        )

        # ── Panneau droite : Bar chart ───────────────────────────────────────
        ax_bar = axes[1]
        ax_bar.set_facecolor(BG_COLOR)

        if condition_labels is None:
            ax_bar.barh(
                range(n_metaclusters), counts_total, color=colors, edgecolor="white"
            )
            ax_bar.set_xlabel("Nombre de cellules", fontsize=11, color=TEXT_COLOR)
        else:
            unique_conds = sorted(set(condition_labels))
            cond_colors = ["#89b4fa", "#f38ba8", "#a6e3a1", "#f9e2af"]
            left = np.zeros(n_metaclusters)
            for ci, cond in enumerate(unique_conds):
                cond_mask = condition_labels == cond
                cnts = np.array(
                    [
                        int(((metaclustering == k) & cond_mask).sum())
                        for k in cluster_ids
                    ]
                )
                ax_bar.barh(
                    range(n_metaclusters),
                    cnts,
                    left=left,
                    color=cond_colors[ci % len(cond_colors)],
                    edgecolor="white",
                    linewidth=0.5,
                    label=str(cond),
                )
                left += cnts
            ax_bar.legend(
                facecolor="#313244",
                labelcolor=TEXT_COLOR,
                edgecolor=SPINE_COLOR,
                fontsize=9,
            )
            ax_bar.set_xlabel("Nombre de cellules", fontsize=11, color=TEXT_COLOR)

        ax_bar.set_yticks(range(n_metaclusters))
        ax_bar.set_yticklabels(labels, fontsize=9, color=TEXT_COLOR)
        ax_bar.set_title(
            "Taille des Métaclusters", fontsize=12, fontweight="bold", color=TEXT_COLOR
        )
        ax_bar.grid(axis="x", alpha=0.3, linestyle="--", color=SPINE_COLOR)
        ax_bar.tick_params(colors=TEXT_COLOR)
        for sp in ax_bar.spines.values():
            sp.set_color(SPINE_COLOR)

        plt.tight_layout()
        save_figure(fig, output_path)
        _logger.info("Distribution métaclusters sauvegardée: %s", output_path)
        return fig

    except Exception as exc:
        _logger.error("Échec plot_metacluster_sizes: %s", exc)
        return None


def plot_umap(
    umap_coords: np.ndarray,
    metaclustering: np.ndarray,
    output_path: Path | str,
    n_metaclusters: int = 10,
    title: str = "UMAP — Coloré par métacluster FlowSOM",
    max_pts: int = 10000,
    seed: int = 42,
) -> Optional["matplotlib.figure.Figure"]:
    """
    Scatter UMAP coloré par métacluster.

    Args:
        umap_coords: Coordonnées UMAP (n_cells, 2).
        metaclustering: Assignation de métacluster (n_cells,).
        output_path: Chemin PNG de sortie.
        n_metaclusters: Nombre de métaclusters (pour colormap discrète).
        title: Titre du graphique.
        max_pts: Sous-échantillonnage si > max_pts cellules.
        seed: Graine pour sous-échantillonnage.

    Returns:
        True si succès.
    """
    if not _MPL_AVAILABLE:
        return None

    try:
        if len(umap_coords) > max_pts:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(umap_coords), max_pts, replace=False)
            umap_coords = umap_coords[idx]
            metaclustering = metaclustering[idx]

        # Colormap discrète Tab20 (jusqu'à 20 couleurs distinctes)
        cmap = plt.cm.get_cmap("tab20", n_metaclusters)
        colors = [cmap(i % 20) for i in metaclustering]

        fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG_COLOR)

        scatter = ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=metaclustering,
            cmap="tab20",
            vmin=0,
            vmax=n_metaclusters,
            s=3,
            alpha=0.5,
            edgecolors="none",
            rasterized=True,
        )

        ax.set_xlabel("UMAP 1", fontsize=13, fontweight="bold", color=TEXT_COLOR)
        ax.set_ylabel("UMAP 2", fontsize=13, fontweight="bold", color=TEXT_COLOR)
        ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=12)
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

        for spine in ax.spines.values():
            spine.set_color(SPINE_COLOR)

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Métacluster", color=TEXT_COLOR, fontsize=11)
        cbar.ax.tick_params(colors=TEXT_COLOR)
        cbar.set_ticks(np.arange(n_metaclusters))
        cbar.set_ticklabels([f"MC{i}" for i in range(n_metaclusters)])

        save_figure(fig, output_path)
        _logger.info("UMAP sauvegardé: %s", output_path)
        return fig

    except Exception as exc:
        _logger.error("Échec plot_umap: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitaires de jitter pour les graphiques FlowSOM-style
# ─────────────────────────────────────────────────────────────────────────────


def circular_jitter(
    n_points: int,
    cluster_ids: np.ndarray,
    node_sizes: np.ndarray,
    max_radius: float = 0.45,
    min_radius: float = 0.10,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jitter circulaire vectorisé pour représentation FlowSOM-R.

    Chaque cellule est décalée d'une distance aléatoire dans un disque
    centré sur son nœud SOM. Le rayon maximum du disque est proportionnel
    à la racine carrée de la taille du nœud (conforme à FlowSOM R package).

    Formule du rayon:
        r_max = min_radius + (max_radius - min_radius) × √(node_size / max_node_size)

    Args:
        n_points: Nombre de cellules.
        cluster_ids: Assignation de nœud par cellule (n_cells,), entiers.
        node_sizes: Taille (n_cells) de chaque nœud SOM (n_nodes,).
        max_radius: Rayon maximum pour le nœud le plus grand (défaut 0.45).
        min_radius: Rayon minimum pour les petits nœuds (défaut 0.10).
        seed: Graine aléatoire pour la reproductibilité.

    Returns:
        Tuple (jitter_x, jitter_y) de dtype float32.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
    u = rng.uniform(0.0, 1.0, n_points)

    ids = cluster_ids.astype(int)
    max_size_val = float(node_sizes.max())
    if max_size_val <= 0.0:
        max_size_val = 1.0

    radii = min_radius + (max_radius - min_radius) * np.sqrt(
        node_sizes[ids] / max_size_val
    )
    # √u pour une densité uniforme dans le disque (pas une concentration au centre)
    r = np.sqrt(u) * radii

    jitter_x = (r * np.cos(theta)).astype(np.float32)
    jitter_y = (r * np.sin(theta)).astype(np.float32)
    return jitter_x, jitter_y


# ─────────────────────────────────────────────────────────────────────────────
#  MST matplotlib statique
# ─────────────────────────────────────────────────────────────────────────────


def plot_mst_static(
    clusterer: Any,
    mfi_matrix: pd.DataFrame,
    metaclustering: np.ndarray,
    output_path: Path | str,
    *,
    figsize: Tuple[int, int] = (14, 12),
    title: str = "Arbre MST FlowSOM — Topologie des nodes SOM",
) -> Optional["matplotlib.figure.Figure"]:
    """
    Graphique MST statique matplotlib.

    Nodes colorés par métacluster dominant, taille proportionnelle au nombre
    de cellules. Arêtes de l'arbre minimal (MST sur le codebook).

    Args:
        clusterer: FlowSOMClusterer après fit().
        mfi_matrix: DataFrame [n_metaclusters × n_markers] — MFI médiane.
        metaclustering: Assignation métacluster par cellule (n_cells,).
        output_path: Chemin PNG de sortie.
        figsize: Taille de la figure.
        title: Titre du graphique.

    Returns:
        Figure matplotlib ou None si échec.
    """
    if not _MPL_AVAILABLE:
        return None

    try:
        from matplotlib.patches import Patch

        layout_coords = clusterer.get_layout_coords()  # (n_nodes, 2)
        node_sizes = clusterer.get_node_sizes()  # (n_nodes,)
        n_nodes = clusterer.n_nodes

        # ── Métacluster par node — utilise metacluster_map_ en priorité ───────
        mc_per_node = getattr(clusterer, "metacluster_map_", None)
        if mc_per_node is None:
            na = getattr(clusterer, "node_assignments_", None)
            ma = getattr(clusterer, "metacluster_assignments_", None)
            if na is not None and ma is not None:
                mc_per_node = np.array(
                    [
                        int(np.bincount(ma[na == i]).argmax()) if (na == i).any() else 0
                        for i in range(n_nodes)
                    ],
                    dtype=int,
                )
            else:
                mc_per_node = np.zeros(n_nodes, dtype=int)
        mc_per_node = np.asarray(mc_per_node, dtype=int)

        # ── Style identique au notebook ───────────────────────────────────────
        max_size = float(node_sizes.max()) if node_sizes.max() > 0 else 1.0
        sizes = 100 + (node_sizes / max_size) * 800

        n_meta = len(np.unique(mc_per_node))
        cmap = plt.cm.tab20 if n_meta <= 20 else plt.cm.turbo
        # Couleur indexée comme le notebook : cmap(int(m) / max(n_meta-1, 1))
        colors = [cmap(int(m) / max(n_meta - 1, 1)) for m in mc_per_node]

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            layout_coords[:, 0],
            layout_coords[:, 1],
            s=sizes,
            c=colors,
            edgecolors="white",
            linewidths=1.5,
            alpha=0.9,
            zorder=2,
        )

        # Label par node (MC id) — identique au notebook
        for i in range(n_nodes):
            ax.annotate(
                str(int(mc_per_node[i])),
                (layout_coords[i, 0], layout_coords[i, 1]),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

        ax.set_xlabel("xNodes", fontsize=12, fontweight="bold")
        ax.set_ylabel("yNodes", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Arbre MST - {n_nodes} nodes, {n_meta} métaclusters",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.grid(True, alpha=0.15, linestyle="--")

        if n_meta <= 15:
            legend_elements = [
                Patch(facecolor=cmap(i / max(n_meta - 1, 1)), label=f"MC {i}")
                for i in range(n_meta)
            ]
            ax.legend(
                handles=legend_elements,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=9,
            )

        plt.tight_layout()
        save_figure(fig, output_path)
        _logger.info("MST statique sauvegardé: %s", output_path)
        return fig

    except Exception as exc:
        _logger.error("Échec plot_mst_static: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  MST Plotly interactif
# ─────────────────────────────────────────────────────────────────────────────


def plot_mst_plotly(
    clusterer: Any,
    mfi_matrix: pd.DataFrame,
    metaclustering: np.ndarray,
    output_path: Optional[Path | str] = None,
    *,
    title: str = "MST FlowSOM — Vue Interactive",
) -> Optional[Any]:
    """
    MST FlowSOM interactif Plotly.

    Nodes colorés par métacluster, taille proportionnelle au nb de cellules.
    Hover: ID node, métacluster, cellules, top 2 marqueurs.

    Args:
        clusterer: FlowSOMClusterer après fit().
        mfi_matrix: DataFrame [n_metaclusters × n_markers].
        metaclustering: Assignation métacluster par cellule.
        output_path: Chemin HTML de sortie (optionnel).
        title: Titre du graphique.

    Returns:
        Figure Plotly ou None.
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from scipy.spatial.distance import cdist
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.sparse import csr_matrix
    except ImportError:
        _logger.warning("plotly / scipy requis pour plot_mst_plotly")
        return None

    try:
        layout_coords = clusterer.get_layout_coords()
        node_sizes = clusterer.get_node_sizes()
        n_nodes = clusterer.n_nodes

        # Métacluster par node — utilise metacluster_map_ en priorité
        mc_per_node = getattr(clusterer, "metacluster_map_", None)
        if mc_per_node is None:
            na = getattr(clusterer, "node_assignments_", None)
            ma = getattr(clusterer, "metacluster_assignments_", None)
            if na is not None and ma is not None:
                mc_per_node = np.array(
                    [
                        int(np.bincount(ma[na == i]).argmax()) if (na == i).any() else 0
                        for i in range(n_nodes)
                    ],
                    dtype=int,
                )
            else:
                mc_per_node = np.zeros(n_nodes, dtype=int)
        mc_per_node = np.asarray(mc_per_node, dtype=int)

        # ── Arêtes MST ────────────────────────────────────────────────────────
        # Priorité 1 (CPU) : graphe MST stocké dans cluster_data.uns['mst'] (igraph)
        # Priorité 2 (GPU/fallback) : recompute depuis codebook
        edge_x: List[Optional[float]] = []
        edge_y: List[Optional[float]] = []
        fsm = getattr(clusterer, "_fsom_model", None)
        _mst_loaded = False

        # Tentative via cluster_data.uns['mst'] (fs.FlowSOM CPU)
        if fsm is not None and hasattr(fsm, "get_cluster_data"):
            try:
                import igraph as _ig

                cluster_data_mst = fsm.get_cluster_data()
                if "mst" in cluster_data_mst.uns:
                    _mst_graph = cluster_data_mst.uns["mst"]
                    if isinstance(_mst_graph, _ig.Graph):
                        for edge in _mst_graph.es:
                            s, t = edge.source, edge.target
                            if s < n_nodes and t < n_nodes:
                                edge_x += [
                                    layout_coords[s, 0],
                                    layout_coords[t, 0],
                                    None,
                                ]
                                edge_y += [
                                    layout_coords[s, 1],
                                    layout_coords[t, 1],
                                    None,
                                ]
                        _mst_loaded = True
            except Exception:
                pass

        # Fallback : recompute depuis le codebook (GPU ou cluster_data sans uns['mst'])
        if not _mst_loaded:
            codebook = None
            if fsm is not None:
                if hasattr(fsm, "codes"):
                    codebook = np.asarray(fsm.codes, dtype=float)
                elif hasattr(fsm, "model") and hasattr(fsm.model, "codes"):
                    codebook = np.asarray(fsm.model.codes, dtype=float)
                elif hasattr(fsm, "get_cluster_data"):
                    try:
                        codebook = np.asarray(fsm.get_cluster_data().X, dtype=float)
                    except Exception:
                        pass
            if codebook is not None:
                from scipy.spatial.distance import cdist
                from scipy.sparse.csgraph import minimum_spanning_tree
                from scipy.sparse import csr_matrix

                dist_mat = cdist(codebook, codebook, metric="euclidean")
                mst_sparse = minimum_spanning_tree(csr_matrix(dist_mat))
                coo = mst_sparse.tocoo()
                for i, j in zip(coo.row.tolist(), coo.col.tolist()):
                    edge_x += [layout_coords[i, 0], layout_coords[j, 0], None]
                    edge_y += [layout_coords[i, 1], layout_coords[j, 1], None]

        # ── Palette & sizing ─────────────────────────────────────────────────
        n_meta = len(np.unique(mc_per_node))
        max_sz = float(node_sizes.max()) if node_sizes.max() > 0 else 1.0

        if n_meta <= 20:
            palette = px.colors.qualitative.Alphabet[:n_meta]
        else:
            palette = [f"hsl({int(i * 360 / n_meta)},70%,55%)" for i in range(n_meta)]

        traces: List[Any] = []
        # Arêtes en premier
        if edge_x:
            traces.append(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(color="rgba(100,100,100,0.5)", width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Nodes par métacluster
        _bubble_sizes = 10 + (node_sizes / max_sz) * 40
        for mc_id in range(n_meta):
            indices = [i for i in range(n_nodes) if mc_per_node[i] == mc_id]
            if not indices:
                continue
            idx = np.array(indices)

            mc_key = f"MC{mc_id}"
            if mc_key in mfi_matrix.index:
                top2 = mfi_matrix.loc[mc_key].nlargest(2).index.tolist()
                hover = [
                    f"<b>Node {ni}</b><br>MC {mc_id}<br>Cellules: {int(node_sizes[ni]):,}<br>Top: {', '.join(top2)}"
                    for ni in idx
                ]
            else:
                hover = [
                    f"<b>Node {ni}</b><br>MC {mc_id}<br>Cellules: {int(node_sizes[ni]):,}<br>x: {layout_coords[ni, 0]:.2f}, y: {layout_coords[ni, 1]:.2f}"
                    for ni in idx
                ]

            traces.append(
                go.Scatter(
                    x=layout_coords[idx, 0].tolist(),
                    y=layout_coords[idx, 1].tolist(),
                    mode="markers+text",
                    marker=dict(
                        size=_bubble_sizes[idx].tolist(),
                        color=palette[mc_id % len(palette)],
                        line=dict(color="white", width=1.5),
                        opacity=0.9,
                    ),
                    text=[str(int(mc_id))] * len(idx),
                    textfont=dict(size=9, color="white", family="Arial Black"),
                    textposition="middle center",
                    name=f"MC{mc_id} ({int(node_sizes[idx].sum()):,} cells)",
                    hovertext=hover,
                    hoverinfo="text",
                )
            )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>Arbre MST — {n_nodes} nodes, {n_meta} métaclusters</b><br>"
                    "<sup>Taille des bulles ∝ nombre de cellules · Cliquez la légende pour filtrer</sup>"
                ),
                font=dict(size=14),
            ),
            xaxis=dict(
                title="xNodes",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.06)",
                zeroline=False,
            ),
            yaxis=dict(
                title="yNodes",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.06)",
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,
            ),
            height=750,
            width=900,
            paper_bgcolor="#fafafa",
            plot_bgcolor="#f5f5f5",
            legend=dict(
                title="Métacluster",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc",
                borderwidth=1,
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(t=80, b=50, l=60, r=200),
        )

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out), include_plotlyjs="cdn")
            _logger.info("MST Plotly sauvegardé: %s", out.name)

        return fig

    except Exception as exc:
        _logger.error("Échec plot_mst_plotly: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Grille SOM Plotly (ScatterGL)
# ─────────────────────────────────────────────────────────────────────────────


def plot_som_grid_plotly(
    clustering: np.ndarray,
    metaclustering: np.ndarray,
    clusterer: Any,
    output_path: Optional[Path | str] = None,
    *,
    max_cells: int = 50_000,
    seed: int = 42,
    title: str = "Grille SOM — Cellules par Métacluster",
) -> Optional[Any]:
    """
    Grille SOM Plotly ScatterGL — chaque cellule positionnée sur son node SOM avec jitter.

    Args:
        clustering: Assignation de node par cellule (n_cells,).
        metaclustering: Assignation métacluster par cellule (n_cells,).
        clusterer: FlowSOMClusterer après fit().
        output_path: Chemin HTML de sortie (optionnel).
        max_cells: Sous-échantillonnage si > max_cells cellules.
        seed: Graine pour reproducibilité.
        title: Titre du graphique.

    Returns:
        Figure Plotly ou None.
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        _logger.warning("plotly requis pour plot_som_grid_plotly")
        return None

    try:
        grid_coords = clusterer.get_grid_coords()  # (n_nodes, 2)
        node_sizes = clusterer.get_node_sizes()  # (n_nodes,)

        # Sous-échantillonnage
        rng = np.random.default_rng(seed)
        n = min(max_cells, len(clustering))
        idx_sample = rng.choice(len(clustering), n, replace=False)
        cl_sub = clustering[idx_sample].astype(int)
        mc_sub = metaclustering[idx_sample].astype(int)

        # Jitter circulaire proportionnel à la taille du node
        max_sz = float(node_sizes.max()) if node_sizes.max() > 0 else 1.0
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        u = rng.uniform(0.0, 1.0, n)
        radii = 0.1 + 0.35 * np.sqrt(node_sizes[cl_sub] / max_sz)
        r = np.sqrt(u) * radii
        x_pos = grid_coords[cl_sub, 0] + r * np.cos(theta)
        y_pos = grid_coords[cl_sub, 1] + r * np.sin(theta)

        palette = (
            px.colors.qualitative.Set1
            + px.colors.qualitative.Pastel
            + px.colors.qualitative.Set2
        )
        n_meta = int(metaclustering.max()) + 1 if len(metaclustering) > 0 else 1

        traces: List[Any] = []
        for mc_id in range(n_meta):
            mask = mc_sub == mc_id
            if not mask.any():
                continue
            traces.append(
                go.Scattergl(
                    x=x_pos[mask].tolist(),
                    y=y_pos[mask].tolist(),
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=palette[mc_id % len(palette)],
                        opacity=0.55,
                    ),
                    name=f"MC{mc_id}",
                    hoverinfo="skip",
                )
            )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=16)),
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            font=dict(color="#e2e8f0"),
            height=620,
            legend=dict(bgcolor="#313244", bordercolor="#585b70"),
            xaxis=dict(
                title="SOM Grid X",
                showgrid=False,
                zeroline=False,
                color="#e2e8f0",
            ),
            yaxis=dict(
                title="SOM Grid Y",
                showgrid=False,
                zeroline=False,
                color="#e2e8f0",
            ),
            margin=dict(l=40, r=20, t=60, b=40),
        )

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out), include_plotlyjs="cdn")
            _logger.info("SOM Grid Plotly sauvegardé: %s", out.name)

        return fig

    except Exception as exc:
        _logger.error("Échec plot_som_grid_plotly: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Optimisation FlowSOM — Visualisation multi-critères
# ─────────────────────────────────────────────────────────────────────────────


def plot_optimization_results(
    results_df: pd.DataFrame,
    best_k: int,
    stability_results: Optional[Dict[int, Any]] = None,
    w_stability: float = 0.65,
    w_silhouette: float = 0.35,
    min_stability_threshold: float = 0.75,
    output_path: Optional[Path | str] = None,
) -> Optional["_mpl_figure.Figure"]:
    """
    Visualisation des résultats d'optimisation multi-critères FlowSOM.

    Produit une figure matplotlib à 2 ou 3 panneaux :
      - Panneau 1 : Silhouette score vs k
      - Panneau 2 : ARI moyen/stabilité bootstrap (si stability_results fourni)
      - Panneau 3 (ou 2) : Score composite pondéré (w_stability × ARI + w_silhouette × Sil)

    Args:
        results_df: DataFrame avec colonnes 'k', 'silhouette', optionnellement
                    'composite_score'.
        best_k: Valeur optimale de k retenue.
        stability_results: Dict {k: {'mean_ari': float, 'std_ari': float}} (optionnel).
        w_stability: Poids de l'ARI dans le score composite (défaut 0.65).
        w_silhouette: Poids du silhouette dans le score composite (défaut 0.35).
        min_stability_threshold: Seuil ARI minimal pour déclarer un k stable.
        output_path: Chemin PNG de sauvegarde (optionnel).

    Returns:
        Figure matplotlib ou None si matplotlib absent.
    """
    if not _MPL_AVAILABLE:
        _logger.warning("matplotlib requis pour plot_optimization_results")
        return None

    has_stability = bool(stability_results and len(stability_results) > 0)
    n_plots = 3 if has_stability else 2

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), facecolor=BG_COLOR)
    axes = list(axes)

    ks = results_df["k"].values
    sils = results_df["silhouette"].values

    # ── Panneau 1 : Silhouette ─────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG_COLOR)
    ax.plot(
        ks, sils, "o-", color="#2196F3", linewidth=2, markersize=5, label="Silhouette"
    )
    ax.axvline(
        best_k,
        color="#F44336",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Optimal k={best_k}",
    )
    ax.set_xlabel("Nombre de métaclusters (k)", fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel("Silhouette Score", fontsize=11, color=TEXT_COLOR)
    ax.set_title(
        "Silhouette sur Codebook SOM", fontsize=12, fontweight="bold", color=TEXT_COLOR
    )
    ax.legend(fontsize=9, facecolor="#313244", labelcolor=TEXT_COLOR)
    ax.grid(True, alpha=0.3, color=SPINE_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for sp in ax.spines.values():
        sp.set_color(SPINE_COLOR)

    # ── Panneau 2 : Stabilité ARI ──────────────────────────────────────────
    if has_stability:
        ax = axes[1]
        ax.set_facecolor(BG_COLOR)
        stab_ks = sorted(stability_results.keys())
        stab_aris = [stability_results[k]["mean_ari"] for k in stab_ks]
        stab_stds = [stability_results[k]["std_ari"] for k in stab_ks]
        ax.errorbar(
            stab_ks,
            stab_aris,
            yerr=stab_stds,
            fmt="s-",
            color="#4CAF50",
            linewidth=2,
            markersize=6,
            capsize=3,
            label="ARI moyen ± σ",
        )
        ax.axhline(
            min_stability_threshold,
            color="#FF9800",
            linestyle=":",
            linewidth=1.5,
            label=f"Seuil stabilité ({min_stability_threshold})",
        )
        ax.axvline(best_k, color="#F44336", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel("Nombre de métaclusters (k)", fontsize=11, color=TEXT_COLOR)
        ax.set_ylabel("ARI moyen (stabilité)", fontsize=11, color=TEXT_COLOR)
        ax.set_title(
            "Stabilité Bootstrap (ARI pairwise)",
            fontsize=12,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        ax.legend(fontsize=9, facecolor="#313244", labelcolor=TEXT_COLOR)
        ax.grid(True, alpha=0.3, color=SPINE_COLOR)
        ax.set_ylim(0, 1.05)
        ax.tick_params(colors=TEXT_COLOR)
        for sp in ax.spines.values():
            sp.set_color(SPINE_COLOR)

    # ── Panneau 3 (ou 2) : Score composite ────────────────────────────────
    ax = axes[-1]
    ax.set_facecolor(BG_COLOR)
    if "composite_score" in results_df.columns:
        valid = results_df.dropna(subset=["composite_score"])
        ax.bar(
            valid["k"],
            valid["composite_score"],
            color="#9C27B0",
            alpha=0.7,
            label="Score composite",
        )
        ax.axvline(
            best_k,
            color="#F44336",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Optimal k={best_k}",
        )
        ax.set_xlabel("Nombre de métaclusters (k)", fontsize=11, color=TEXT_COLOR)
        ax.set_ylabel("Score composite", fontsize=11, color=TEXT_COLOR)
        ax.set_title(
            f"Score Composite (w_stab={w_stability}, w_sil={w_silhouette})",
            fontsize=12,
            fontweight="bold",
            color=TEXT_COLOR,
        )
    else:
        diffs = np.diff(sils)
        ax.plot(ks[1:], diffs, "o-", color="#FF5722", linewidth=2, markersize=4)
        ax.axvline(best_k, color="#F44336", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel("k", fontsize=11, color=TEXT_COLOR)
        ax.set_ylabel("Δ Silhouette", fontsize=11, color=TEXT_COLOR)
        ax.set_title(
            "Variation Silhouette (Elbow)",
            fontsize=12,
            fontweight="bold",
            color=TEXT_COLOR,
        )
    ax.legend(fontsize=9, facecolor="#313244", labelcolor=TEXT_COLOR)
    ax.grid(True, alpha=0.3, color=SPINE_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for sp in ax.spines.values():
        sp.set_color(SPINE_COLOR)

    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        _logger.info("Figure optimisation sauvegardée → %s", out)
        plt.close(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Jitter circulaire (version non-seeded — style FlowSOM R)
# ─────────────────────────────────────────────────────────────────────────────


def circular_jitter_viz(
    n_points: int,
    cluster_ids: np.ndarray,
    node_sizes: np.ndarray,
    max_radius: float = 0.45,
    min_radius: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jitter circulaire style FlowSOM R — vectorisé, sans graine fixe.

    Variante de ``circular_jitter`` utilisée dans les représentations SOM
    en temps réel où la reproductibilité n'est pas requise (exploration
    interactive, notebooks). Le rayon dépend proportionnellement à la
    racine carrée de la taille de chaque nœud.

    Args:
        n_points: Nombre de cellules à positionner.
        cluster_ids: Assignation de nœud par cellule (n_cells,), entiers.
        node_sizes: Taille de chaque nœud SOM (n_nodes,).
        max_radius: Rayon maximum pour le nœud le plus grand (défaut 0.45).
        min_radius: Rayon minimum pour les petits nœuds (défaut 0.10).

    Returns:
        Tuple (jitter_x, jitter_y) de dtype float32.
    """
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    u = np.random.uniform(0, 1, n_points)

    max_size_val = float(node_sizes.max())
    if max_size_val <= 0.0:
        max_size_val = 1.0

    size_ratios = np.sqrt(node_sizes[cluster_ids.astype(int)] / max_size_val)
    radii = (min_radius + (max_radius - min_radius) * size_ratios).astype(np.float32)
    r = np.sqrt(u) * radii

    return (r * np.cos(theta)).astype(np.float32), (r * np.sin(theta)).astype(
        np.float32
    )


# =============================================================================
# SECTION §13 — Star Chart (MST view via fs.pl.plot_stars)
# =============================================================================


def plot_star_chart(
    fsom: Any,
    output_path: Path | str,
    *,
    title: str = "FlowSOM Star Chart (MST View)",
    dpi: int = 150,
) -> Optional["_mpl_figure.Figure"]:
    """
    Génère le Star Chart FlowSOM en vue MST via ``fs.pl.plot_stars()``.

    Le Star Chart est la visualisation officielle du package FlowSOM (R/Python) :
    chaque nœud du MST est représenté par un étoile dont les branches traduisent
    l'intensité relative de chaque marqueur dans ce nœud. La couleur de fond
    correspond au métacluster dominant du nœud.

    Args:
        fsom: Objet FlowSOM entraîné (retourné par ``fs.FlowSOM()``).
        output_path: Chemin PNG de sauvegarde.
        title: Titre de la figure.
        dpi: Résolution de sortie (défaut 150).

    Returns:
        Figure matplotlib ou None si la génération échoue.
    """
    if not _MPL_AVAILABLE:
        _logger.warning("matplotlib requis pour plot_star_chart")
        return None

    try:
        import flowsom as fs  # noqa: F401 — vérifie la disponibilité

        fig = fs.pl.plot_stars(
            fsom,
            background_values=fsom.get_cluster_data().obs.metaclustering,
            view="MST",
        )
        plt.suptitle(title, fontsize=14, fontweight="bold")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        plt.close("all")
        _logger.info("Star Chart sauvegardé: %s", output_path)
        return fig
    except Exception as exc:
        _logger.warning(
            "Échec plot_star_chart (fs.pl.plot_stars indisponible): %s", exc
        )
        return None


# =============================================================================
# SECTION §13b — Star Chart custom (compatible CPU + GPU)
#
# Reproduit visuellement le Star Chart FlowSOM sans dépendre de fs.pl.plot_stars:
#   • Chaque nœud est dessiné en étoile dont les branches = MFI normalisée par marqueur
#   • Taille du disque central ∝ nombre de cellules dans le nœud
#   • Couleur du disque = métacluster dominant
#   • Les arêtes MST sont tracées entre les nœuds voisins
#   • Layout Kamada-Kawai (identique à fs.FlowSOM.build_MST)
#
# Compatible GPU (utilise clusterer.get_layout_coords() + codebook.codes)
# et CPU (mêmes appels — get_layout_coords() retourne obsm["layout"] si disponible).
# =============================================================================


def plot_star_chart_custom(
    clusterer: Any,
    output_path: "Path | str",
    *,
    marker_names: "Optional[List[str]]" = None,
    title: str = "FlowSOM Star Chart (MST View — Custom)",
    dpi: int = 150,
    figsize: "Tuple[int, int]" = (14, 12),
    star_scale: float = 0.35,
    max_bubble: float = 500.0,
    min_bubble: float = 30.0,
) -> "Optional[_mpl_figure.Figure]":
    """
    Génère un Star Chart FlowSOM compatible CPU **et** GPU.

    Contrairement à ``plot_star_chart`` (qui délègue à ``fs.pl.plot_stars``),
    cette fonction construit le graphique entièrement avec matplotlib à partir
    des données exposées par ``FlowSOMClusterer`` :
      - ``clusterer.get_layout_coords()``  → coordonnées MST (n_nodes, 2)
      - ``clusterer.get_node_sizes()``     → tailles des nœuds (n_nodes,)
      - ``clusterer.metacluster_map_``     → métacluster par nœud (n_nodes,)
      - ``clusterer._fsom_model.codes``    → codebook (n_nodes, n_markers)

    Args:
        clusterer: Instance ``FlowSOMClusterer`` après ``fit()``.
        output_path: Chemin PNG de sauvegarde.
        marker_names: Noms des marqueurs (ordonnés comme le codebook).
            Si None, utilise M0, M1, …
        title: Titre de la figure.
        dpi: Résolution en sortie.
        figsize: Taille de la figure en pouces.
        star_scale: Rayon maximal d'une branche étoile (unités layout).
        max_bubble: Taille matplotlib maximale du disque central.
        min_bubble: Taille matplotlib minimale du disque central.

    Returns:
        Figure matplotlib ou None si la génération échoue.
    """
    if not _MPL_AVAILABLE:
        _logger.warning("matplotlib requis pour plot_star_chart_custom")
        return None

    try:
        # ── 1. Récupération du codebook ───────────────────────────────────────
        fsom_model = getattr(clusterer, "_fsom_model", None)
        codebook: Optional[np.ndarray] = None

        if fsom_model is not None:
            if hasattr(fsom_model, "codes"):
                codebook = np.asarray(fsom_model.codes, dtype=float)
            elif hasattr(fsom_model, "get_cluster_data"):
                # CPU fs.FlowSOM : codebook = .X de cluster_data
                try:
                    codebook = np.asarray(fsom_model.get_cluster_data().X, dtype=float)
                except Exception:
                    pass

        if codebook is None or codebook.shape[0] == 0:
            _logger.warning(
                "plot_star_chart_custom : codebook indisponible — star chart ignoré."
            )
            return None

        n_nodes, n_markers = codebook.shape

        # ── 2. Noms des marqueurs ─────────────────────────────────────────────
        if marker_names is None or len(marker_names) != n_markers:
            marker_names = [f"M{i}" for i in range(n_markers)]

        # ── 3. Coordonnées MST (layout Kamada-Kawai) ──────────────────────────
        layout = clusterer.get_layout_coords()  # (n_nodes, 2)

        # ── 4. Arêtes MST (optionnel) ─────────────────────────────────────────
        mst_edges: List[Tuple[int, int]] = []
        try:
            import igraph as ig
            from scipy.spatial.distance import cdist

            adj = cdist(codebook, codebook, metric="euclidean")
            g_full = ig.Graph.Weighted_Adjacency(adj, mode="undirected", loops=False)
            g_mst = ig.Graph.spanning_tree(g_full, weights=g_full.es["weight"])
            mst_edges = [(e.source, e.target) for e in g_mst.es]
        except Exception:
            pass  # Arêtes non disponibles — star chart sans lignes MST

        # ── 5. Tailles et métaclusters ────────────────────────────────────────
        node_sizes = clusterer.get_node_sizes()  # (n_nodes,)
        mc_map = getattr(clusterer, "metacluster_map_", None)
        if mc_map is None:
            mc_map = np.zeros(n_nodes, dtype=int)
        mc_map = np.asarray(mc_map, dtype=int)

        # Nombre de métaclusters
        n_meta = int(mc_map.max()) + 1
        cmap_mc = plt.cm.tab20 if n_meta <= 20 else plt.cm.turbo

        # ── 6. Normalisation MFI par marqueur (min-max sur tous les nœuds) ───
        col_min = codebook.min(axis=0)
        col_max = codebook.max(axis=0)
        col_range = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
        codebook_norm = (codebook - col_min) / col_range  # ∈ [0, 1]

        # ── 7. Construction des angles des branches ───────────────────────────
        angles = np.linspace(0, 2 * np.pi, n_markers, endpoint=False)

        # ── 8. Taille des bulles ──────────────────────────────────────────────
        max_sz = float(node_sizes.max()) if node_sizes.max() > 0 else 1.0
        bubble_sz = min_bubble + (max_bubble - min_bubble) * (node_sizes / max_sz)

        # ── 9. Mise en page ───────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("#1e1e2e")
        fig.patch.set_facecolor("#1e1e2e")

        x_vals = layout[:, 0]
        y_vals = layout[:, 1]

        # ── 10. Tracé des arêtes MST ──────────────────────────────────────────
        for s, t in mst_edges:
            ax.plot(
                [x_vals[s], x_vals[t]],
                [y_vals[s], y_vals[t]],
                color="rgba(200,200,200,0.3)" if False else "#666688",
                lw=1.2,
                alpha=0.45,
                zorder=1,
            )

        # ── 11. Tracé des étoiles et disques par nœud ────────────────────────
        for node_id in range(n_nodes):
            xc, yc = x_vals[node_id], y_vals[node_id]
            mc_id = int(mc_map[node_id])
            color_mc = cmap_mc(mc_id / max(n_meta - 1, 1))

            # Disque central (taille ∝ cellules)
            ax.scatter(
                xc,
                yc,
                s=bubble_sz[node_id],
                color=color_mc,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
                alpha=0.92,
            )

            # Branches en étoile — une par marqueur
            profile = codebook_norm[node_id]  # ∈ [0, 1]

            # Polygone étoile (fermeture du contour)
            star_x = [
                xc + star_scale * profile[j] * np.cos(angles[j])
                for j in range(n_markers)
            ]
            star_y = [
                yc + star_scale * profile[j] * np.sin(angles[j])
                for j in range(n_markers)
            ]
            star_x.append(star_x[0])
            star_y.append(star_y[0])

            ax.plot(star_x, star_y, color=color_mc, lw=0.7, alpha=0.65, zorder=2)
            ax.fill(star_x, star_y, color=color_mc, alpha=0.12, zorder=2)

            # Label métacluster (numéro, centré sur le disque)
            if node_sizes[node_id] > 0:
                ax.text(
                    xc,
                    yc,
                    str(mc_id),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                    fontweight="bold",
                    zorder=4,
                )

        # ── 12. Légende des marqueurs (guide des angles) ─────────────────────
        # Afficher les 8 marqueurs les plus variables (pour ne pas surcharger)
        marker_variance = codebook.var(axis=0)
        top_k = min(8, n_markers)
        top_idx = np.argsort(marker_variance)[::-1][:top_k]

        for j in top_idx:
            ax.annotate(
                marker_names[j],
                xy=(
                    layout[:, 0].mean() + (star_scale * 1.6) * np.cos(angles[j]),
                    layout[:, 1].mean() + (star_scale * 1.6) * np.sin(angles[j]),
                ),
                ha="center",
                va="center",
                fontsize=7,
                color="#cccccc",
                zorder=5,
            )

        # ── 13. Légende des métaclusters ──────────────────────────────────────
        from matplotlib.patches import Patch  # noqa: PLC0415

        legend_elements = [
            Patch(
                facecolor=cmap_mc(mc / max(n_meta - 1, 1)),
                edgecolor="white",
                label=f"MC{mc}",
            )
            for mc in range(n_meta)
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=8,
            framealpha=0.6,
            facecolor="#2a2a3e",
            edgecolor="#555",
            labelcolor="white",
            ncol=max(1, n_meta // 8),
            title="Métacluster",
            title_fontsize=8,
        )

        ax.set_title(title, fontsize=13, fontweight="bold", color="#e2e8f0", pad=10)

        # ── 14. Sauvegarde ────────────────────────────────────────────────────
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            str(output_path),
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close("all")
        _logger.info("Star Chart custom sauvegardé: %s", output_path)
        return fig

    except Exception as exc:
        _logger.warning("Échec plot_star_chart_custom: %s", exc)
        return None


# =============================================================================
# SECTION §14 — SOM Grid statique matplotlib (2 panneaux: MC + Condition)
# =============================================================================


def plot_som_grid_static(
    clustering: np.ndarray,
    metaclustering: np.ndarray,
    grid_coords: np.ndarray,
    condition_labels: Optional[np.ndarray],
    xdim: int,
    ydim: int,
    output_path: Path | str,
    *,
    seed: int = 42,
    max_radius: float = 0.45,
    min_radius: float = 0.10,
    dpi: int = 150,
    title_prefix: str = "Grille FlowSOM",
) -> Optional["_mpl_figure.Figure"]:
    """
    Grille SOM statique matplotlib — 2 panneaux côte-à-côte.

    Reproduit exactement la section §14 du pipeline monolithique :
      - Panneau gauche  : cellules colorées par métacluster (tab20)
        avec labels de métacluster annotés au centre de chaque nœud.
      - Panneau droite  : cellules colorées par condition (Sain=vert / Patho=rouge)
        avec légende.

    Le jitter circulaire (style FlowSOM R) positionne les cellules autour
    de leur nœud avec un rayon proportionnel à √(taille_nœud).

    Args:
        clustering: Assignation de nœud par cellule (n_cells,), dtype int.
        metaclustering: Assignation de métacluster par nœud (n_nodes,).
        grid_coords: Coordonnées (n_nodes, 2) dans la grille SOM.
        condition_labels: Condition par cellule (n_cells,) — ``None`` désactive
            le panneau droit (figure 1×1).
        xdim: Largeur de la grille SOM.
        ydim: Hauteur de la grille SOM.
        output_path: Chemin PNG de sauvegarde.
        seed: Graine pour le jitter circulaire.
        max_radius: Rayon jitter max.
        min_radius: Rayon jitter min.
        dpi: Résolution de sortie.
        title_prefix: Préfixe pour les titres des panneaux.

    Returns:
        Figure matplotlib ou None si matplotlib absent.
    """
    if not _MPL_AVAILABLE:
        _logger.warning("matplotlib requis pour plot_som_grid_static")
        return None

    try:
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        n_cells = len(clustering)
        n_nodes = len(metaclustering)
        cluster_ids_int = clustering.astype(int)

        # ── Vectorisé : plus de list comprehension sur les 787k cellules ──────
        # Coordonnées de grille par cellule
        xGrid_base = grid_coords[cluster_ids_int, 0].astype(np.float32)
        yGrid_base = grid_coords[cluster_ids_int, 1].astype(np.float32)
        xGrid_shifted = xGrid_base - xGrid_base.min() + 1
        yGrid_shifted = yGrid_base - yGrid_base.min() + 1

        # Métacluster par cellule
        metaclustering_cells = metaclustering[cluster_ids_int]

        # Taille de chaque nœud (vectorisé via bincount)
        node_sizes = np.bincount(cluster_ids_int, minlength=n_nodes).astype(np.float32)

        # Jitter circulaire vectorisé (reproductible)
        np.random.seed(seed)
        jitter_x, jitter_y = circular_jitter_viz(
            n_cells,
            cluster_ids_int,
            node_sizes,
            max_radius=max_radius,
            min_radius=min_radius,
        )

        n_panels = 2 if condition_labels is not None else 1
        fig, axes = plt.subplots(
            1, n_panels, figsize=(8 * n_panels, 7), facecolor="white"
        )
        if n_panels == 1:
            axes = [axes]

        # ── Panneau 1 : Métaclusters ─────────────────────────────────────────
        ax1 = axes[0]
        n_meta = len(np.unique(metaclustering))
        cmap_mc = plt.cm.tab20 if n_meta <= 20 else plt.cm.turbo

        scatter1 = ax1.scatter(
            xGrid_shifted + jitter_x,
            yGrid_shifted + jitter_y,
            c=metaclustering_cells,
            cmap=cmap_mc,
            s=5,
            alpha=0.5,
            edgecolors="none",
        )
        for node_id in range(n_nodes):
            if node_sizes[node_id] > 0:
                x_pos = grid_coords[node_id, 0] - grid_coords[:, 0].min() + 1
                y_pos = grid_coords[node_id, 1] - grid_coords[:, 1].min() + 1
                meta_id = int(metaclustering[node_id])
                ax1.annotate(
                    str(meta_id + 1),
                    (x_pos, y_pos),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        boxstyle="circle,pad=0.2",
                        facecolor=cmap_mc(meta_id / max(n_meta - 1, 1)),
                        edgecolor="white",
                        alpha=0.9,
                    ),
                )
        ax1.set_xlabel("xGrid", fontsize=12, fontweight="bold")
        ax1.set_ylabel("yGrid", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"{title_prefix} — {xdim}×{ydim} nœuds\nColoré par Métacluster (style FlowSOM R)",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_xlim(0.5, xdim + 1.5)
        ax1.set_ylim(0.5, ydim + 1.5)
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3, linestyle="--")
        plt.colorbar(scatter1, ax=ax1, label="Métacluster")

        # ── Panneau 2 : Conditions ────────────────────────────────────────────
        if condition_labels is not None:
            ax2 = axes[1]
            # Vectorisé : numpy string ops plutôt qu'une list comprehension sur 787k cellules
            cond_arr = np.asarray(condition_labels, dtype=str)
            cond_lower = np.char.lower(cond_arr)
            condition_num = np.where(
                np.isin(cond_lower, ["sain", "healthy", "nbm", "normal"]), 0, 1
            ).astype(int)
            cmap_cond = ListedColormap(["#a6e3a1", "#f38ba8"])
            ax2.scatter(
                xGrid_shifted + jitter_x,
                yGrid_shifted + jitter_y,
                c=condition_num,
                cmap=cmap_cond,
                s=5,
                alpha=0.5,
                edgecolors="none",
            )
            ax2.set_xlabel("xGrid", fontsize=12, fontweight="bold")
            ax2.set_ylabel("yGrid", fontsize=12, fontweight="bold")
            ax2.set_title(
                f"{title_prefix} — {xdim}×{ydim} nœuds\nColoré par Condition (style FlowSOM R)",
                fontsize=12,
                fontweight="bold",
            )
            ax2.set_xlim(0.5, xdim + 1.5)
            ax2.set_ylim(0.5, ydim + 1.5)
            ax2.set_aspect("equal")
            ax2.grid(True, alpha=0.3, linestyle="--")
            legend_elements = [
                Patch(facecolor="#a6e3a1", edgecolor="white", label="Sain (NBM)"),
                Patch(facecolor="#f38ba8", edgecolor="white", label="Pathologique"),
            ]
            ax2.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        plt.close("all")
        _logger.info("Grille SOM statique sauvegardée: %s", output_path)
        return fig

    except Exception as exc:
        _logger.error("Échec plot_som_grid_static: %s", exc)
        return None


# =============================================================================
# SECTION §15 — Radar / Spider Chart interactif Plotly (MFI par métacluster)
# =============================================================================


def plot_metacluster_radar(
    mfi_matrix: np.ndarray,
    used_markers: List[str],
    metaclustering: np.ndarray,
    output_path: Path | str,
    *,
    n_metaclusters: Optional[int] = None,
    title: str = "Profil d'Expression par Métacluster (Radar Interactif)",
) -> Optional[Any]:
    """
    Spider / Radar chart interactif Plotly — tous les métaclusters.

    Reproduit la section §15 du pipeline monolithique. Pour chaque métacluster,
    les valeurs de MFI sont normalisées entre 0 et 1 (min-max par cluster) afin
    de rendre toutes les intensités comparables sur un axe radial commun.

    Chaque trace Plotly ``go.Scatterpolar`` correspond à un métacluster.
    Le hover affiche la MFI brute et la valeur normalisée par marqueur.

    Args:
        mfi_matrix: Matrice MFI (n_metaclusters × n_markers).
        used_markers: Liste des noms de marqueurs (len == n_markers).
        metaclustering: Assignation de métacluster par cellule, pour
            indiquer les effectifs dans la légende.
        output_path: Chemin HTML de sauvegarde.
        n_metaclusters: Nombre de métaclusters ; inféré depuis ``mfi_matrix``
            si None.
        title: Titre de la figure.

    Returns:
        ``plotly.graph_objects.Figure`` ou None si plotly absent.
    """
    try:
        import plotly.graph_objects as go
        import plotly.colors as pc
    except ImportError:
        _logger.warning("plotly requis pour plot_metacluster_radar")
        return None

    try:
        if n_metaclusters is None:
            n_metaclusters = mfi_matrix.shape[0]

        if n_metaclusters <= 10:
            _palette = pc.qualitative.Set3
        elif n_metaclusters <= 20:
            _palette = pc.qualitative.Alphabet
        else:
            _palette = [
                f"hsl({int(i * 360 / n_metaclusters)},70%,55%)"
                for i in range(n_metaclusters)
            ]

        fig = go.Figure()

        for cluster_id in range(n_metaclusters):
            values = mfi_matrix[cluster_id].copy()
            v_min, v_max = float(values.min()), float(values.max())
            values_norm = (values - v_min) / (v_max - v_min + 1e-10)

            _c = _palette[cluster_id % len(_palette)]
            _n_cells = int((metaclustering == cluster_id).sum())

            if "rgb" in str(_c):
                _fill = _c.replace(")", ",0.08)").replace("rgb", "rgba")
            else:
                _fill = "rgba(128,128,128,0.05)"

            fig.add_trace(
                go.Scatterpolar(
                    r=np.append(values_norm, values_norm[0]),
                    theta=list(used_markers) + [used_markers[0]],
                    fill="toself",
                    fillcolor=_fill,
                    opacity=0.85,
                    name=f"MC{cluster_id}  ({_n_cells:,} cells)",
                    line=dict(color=_c, width=2),
                    marker=dict(size=5),
                    customdata=np.stack(
                        [
                            np.append(
                                mfi_matrix[cluster_id], mfi_matrix[cluster_id][0]
                            ),
                            np.append(values_norm, values_norm[0]),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        f"<b>MC{cluster_id}</b><br>"
                        "Marqueur: %{theta}<br>"
                        "MFI brute: %{customdata[0]:.2f}<br>"
                        "Normalisé: %{customdata[1]:.3f}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            polar=dict(
                bgcolor="#1e1e2e",
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.05],
                    tickfont=dict(size=9, color="white"),
                    gridcolor="rgba(255,255,255,0.15)",
                    linecolor="rgba(255,255,255,0.3)",
                ),
                angularaxis=dict(
                    tickfont=dict(size=10, color="white"),
                    gridcolor="rgba(255,255,255,0.15)",
                    linecolor="rgba(255,255,255,0.3)",
                ),
            ),
            showlegend=True,
            title=dict(
                text=title,
                font=dict(size=16, color="white"),
                x=0.5,
            ),
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            font=dict(color="white"),
            legend=dict(
                bgcolor="#313244",
                bordercolor="#45475a",
                font=dict(color="white"),
            ),
            height=700,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        _logger.info("Radar métaclusters sauvegardé: %s", output_path)
        return fig

    except Exception as exc:
        _logger.error("Échec plot_metacluster_radar: %s", exc)
        return None


# =============================================================================
# SECTION §15 — Analyse des clusters exclusifs (100% Patho / 100% Sain)
# =============================================================================


def compute_exclusive_clusters(
    metaclustering: np.ndarray,
    condition_labels: np.ndarray,
    n_metaclusters: int,
    *,
    patho_label: str = "Pathologique",
    sain_label: str = "Sain",
) -> Dict[str, Any]:
    """
    Identifie les métaclusters exclusivement pathologiques ou exclusivement sains.

    Reproduit la section §15 du pipeline monolithique (lignes ~6840–6931).
    Un cluster est ``exclusif`` si 100 % de ses cellules proviennent d'une
    seule condition. Ces clusters représentent des populations absentes ou
    aberrantes par rapport à la moelle normale de référence.

    Args:
        metaclustering: Assignation de métacluster par cellule (n_cells,).
        condition_labels: Label de condition par cellule (n_cells,).
        n_metaclusters: Nombre total de métaclusters.
        patho_label: Valeur de la condition pathologique.
        sain_label: Valeur de la condition saine.

    Returns:
        Dict avec les clés:
          ``patho_only``  : List[Tuple[int, int]] — (cluster_id, n_cells) 100% patho
          ``sain_only``   : List[Tuple[int, int]] — (cluster_id, n_cells) 100% sain
          ``mixed``       : List[int]             — cluster_ids partagés
          ``summary_lines``: List[str]            — lignes de rapport texte
    """
    patho_only: List[Tuple[int, int]] = []
    sain_only: List[Tuple[int, int]] = []
    mixed: List[int] = []

    cond_arr = np.asarray(condition_labels)

    for cluster_id in range(n_metaclusters):
        mask = metaclustering == cluster_id
        total = int(mask.sum())
        if total == 0:
            continue
        n_patho = int((mask & (cond_arr == patho_label)).sum())
        n_sain = int((mask & (cond_arr == sain_label)).sum())
        if n_patho == total:
            patho_only.append((cluster_id, total))
        elif n_sain == total:
            sain_only.append((cluster_id, total))
        else:
            mixed.append(cluster_id)

    summary_lines: List[str] = []
    summary_lines.append("=" * 70)
    summary_lines.append("ANALYSE DES CLUSTERS EXCLUSIFS")
    summary_lines.append("=" * 70)

    if patho_only:
        total_patho = sum(c[1] for c in patho_only)
        summary_lines.append(f"\n[!] CLUSTERS 100% PATHOLOGIQUES: {len(patho_only)}")
        summary_lines.append(f"    Métaclusters : {[c[0] for c in patho_only]}")
        summary_lines.append(f"    Total cellules: {total_patho:,}")
        summary_lines.append(
            "    → Ces clusters représentent des populations UNIQUEMENT présentes chez le patient"
        )
    else:
        summary_lines.append("\n    Aucun cluster exclusivement pathologique détecté")

    if sain_only:
        total_sain = sum(c[1] for c in sain_only)
        summary_lines.append(f"\n[!] CLUSTERS 100% SAINS: {len(sain_only)}")
        summary_lines.append(f"    Métaclusters : {[c[0] for c in sain_only]}")
        summary_lines.append(f"    Total cellules: {total_sain:,}")
        summary_lines.append(
            "    → Ces clusters représentent des populations ABSENTES chez le patient"
        )
    else:
        summary_lines.append("\n    Aucun cluster exclusivement sain détecté")

    summary_lines.append(f"\n    Clusters mixtes (partagés): {len(mixed)}")

    for line in summary_lines:
        _logger.info(line)

    return {
        "patho_only": patho_only,
        "sain_only": sain_only,
        "mixed": mixed,
        "summary_lines": summary_lines,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Post-clustering bar charts Plotly
# ─────────────────────────────────────────────────────────────────────────────


def plot_patho_pct_per_cluster(
    metaclustering: np.ndarray,
    condition_labels: np.ndarray,
    output_html: Optional[Path | str] = None,
    output_jpg: Optional[Path | str] = None,
    *,
    patho_label: str = "Pathologique",
    title: str = "% Cellules pathologiques par cluster (post-FlowSOM)",
) -> Optional[Any]:
    """
    Bar chart Plotly : % de cellules provenant de la moelle pathologique dans chaque cluster.

    Pour chaque métacluster, calcule :
        pct_patho = n_cells_patho_in_cluster / n_cells_total_in_cluster × 100

    Utile pour identifier les clusters enrichis en cellules pathologiques (populations LAIP).

    Args:
        metaclustering: Assignation métacluster par cellule (n_cells,).
        condition_labels: Étiquette de condition par cellule (n_cells,).
        output_html: Chemin HTML de sortie (optionnel).
        output_jpg: Chemin JPG de sortie (optionnel).
        patho_label: Valeur dans condition_labels désignant la condition pathologique.
        title: Titre du graphique.

    Returns:
        Figure Plotly ou None si plotly absent.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        _logger.warning("plotly requis pour plot_patho_pct_per_cluster")
        return None

    try:
        cond_arr = np.asarray(condition_labels)
        cluster_ids_sorted = np.sort(np.unique(metaclustering))
        labels = [f"MC{int(c)}" for c in cluster_ids_sorted]

        pct_patho: List[float] = []
        hover_texts: List[str] = []

        for cid in cluster_ids_sorted:
            mask = metaclustering == cid
            total = int(mask.sum())
            n_patho = int((mask & (cond_arr == patho_label)).sum())
            pct = (n_patho / total * 100) if total > 0 else 0.0
            pct_patho.append(round(pct, 2))
            hover_texts.append(
                f"<b>MC{int(cid)}</b><br>"
                f"Cellules patho : {n_patho:,}<br>"
                f"Total cluster  : {total:,}<br>"
                f"% Patho        : {pct:.2f}%"
            )

        # Gradient rouge (fort % patho) → bleu (faible % patho)
        bar_colors = [
            f"rgba({int(243 * p / 100 + 137 * (1 - p / 100))}, "
            f"{int(139 * (1 - p / 100))}, "
            f"{int(168 * (1 - p / 100) + 250 * (1 - p / 100))}, 0.85)"
            for p in pct_patho
        ]

        fig = go.Figure(
            go.Bar(
                x=labels,
                y=pct_patho,
                marker=dict(
                    color=bar_colors,
                    line=dict(color="#585b70", width=0.8),
                ),
                hovertext=hover_texts,
                hoverinfo="text",
                text=[f"{p:.1f}%" for p in pct_patho],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=11),
            )
        )

        max_y = max(pct_patho) if pct_patho else 10.0
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=16, color="#e2e8f0"),
                x=0.5,
            ),
            xaxis=dict(
                title="Métacluster",
                color="#e2e8f0",
                gridcolor="#313244",
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title="% Cellules pathologiques",
                color="#e2e8f0",
                gridcolor="#313244",
                range=[0, min(max_y * 1.18, 105)],
                ticksuffix="%",
            ),
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            font=dict(color="#e2e8f0"),
            height=520,
            margin=dict(l=60, r=40, t=80, b=60),
            bargap=0.25,
        )

        # Ligne de référence à 100 %
        fig.add_hline(
            y=100,
            line=dict(color="#f38ba8", dash="dash", width=1.5),
            annotation_text="100 % patho",
            annotation_font=dict(color="#f38ba8", size=10),
        )

        if output_html is not None:
            out_html = Path(output_html)
            out_html.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out_html), include_plotlyjs="cdn")
            _logger.info("Patho%% par cluster (HTML) sauvegardé: %s", out_html.name)

        if output_jpg is not None:
            out_jpg = Path(output_jpg)
            out_jpg.parent.mkdir(parents=True, exist_ok=True)
            try:
                fig.write_image(
                    str(out_jpg), format="jpg", width=1400, height=620, scale=2
                )
                _logger.info("Patho%% par cluster (JPG) sauvegardé: %s", out_jpg.name)
            except Exception as _img_err:
                _logger.warning(
                    "Export JPG échoué (kaleido requis — pip install kaleido): %s",
                    _img_err,
                )

        return fig

    except Exception as exc:
        _logger.error("Échec plot_patho_pct_per_cluster: %s", exc)
        return None


def plot_cells_pct_per_cluster(
    metaclustering: np.ndarray,
    output_html: Optional[Path | str] = None,
    output_jpg: Optional[Path | str] = None,
    *,
    condition_labels: Optional[np.ndarray] = None,
    title: str = "% Cellules par cluster (post-FlowSOM)",
) -> Optional[Any]:
    """
    Bar chart Plotly : distribution en % des cellules par métacluster.

    Pour chaque métacluster :
        pct = n_cells_in_cluster / n_cells_total × 100

    Si ``condition_labels`` est fourni, les barres sont empilées par condition
    (Sain / Pathologique), permettant de visualiser la composition de chaque cluster.

    Args:
        metaclustering: Assignation métacluster par cellule (n_cells,).
        output_html: Chemin HTML de sortie (optionnel).
        output_jpg: Chemin JPG de sortie (optionnel).
        condition_labels: Étiquette de condition par cellule (optionnel).
        title: Titre du graphique.

    Returns:
        Figure Plotly ou None si plotly absent.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        _logger.warning("plotly requis pour plot_cells_pct_per_cluster")
        return None

    try:
        total_cells = len(metaclustering)
        cluster_ids_sorted = np.sort(np.unique(metaclustering))
        labels = [f"MC{int(c)}" for c in cluster_ids_sorted]

        _COND_COLORS = ["#89b4fa", "#f38ba8", "#a6e3a1", "#f9e2af", "#cba6f7"]

        if condition_labels is not None:
            cond_arr = np.asarray(condition_labels)
            unique_conds = sorted(set(cond_arr.tolist()))
            traces: List[Any] = []
            for ci, cond in enumerate(unique_conds):
                cond_mask = cond_arr == cond
                pcts: List[float] = []
                hover_texts: List[str] = []
                for cid in cluster_ids_sorted:
                    mask = metaclustering == cid
                    n_in_cond = int((mask & cond_mask).sum())
                    pct = n_in_cond / total_cells * 100
                    pcts.append(round(pct, 3))
                    hover_texts.append(
                        f"<b>MC{int(cid)}</b> — {cond}<br>"
                        f"Cellules : {n_in_cond:,}<br>"
                        f"% global : {pct:.2f}%"
                    )
                traces.append(
                    go.Bar(
                        name=str(cond),
                        x=labels,
                        y=pcts,
                        marker=dict(
                            color=_COND_COLORS[ci % len(_COND_COLORS)],
                            line=dict(color="#585b70", width=0.6),
                        ),
                        hovertext=hover_texts,
                        hoverinfo="text",
                        text=[f"{p:.1f}%" for p in pcts],
                        textposition="inside",
                        textfont=dict(color="white", size=9),
                    )
                )
            barmode = "stack"
        else:
            counts = np.array(
                [int((metaclustering == c).sum()) for c in cluster_ids_sorted]
            )
            pcts = [round(int(cnt) / total_cells * 100, 3) for cnt in counts]
            hover_texts = [
                f"<b>MC{int(cid)}</b><br>"
                f"Cellules : {int(cnt):,}<br>"
                f"% Total  : {pct:.2f}%"
                for cid, cnt, pct in zip(cluster_ids_sorted, counts, pcts)
            ]
            n = len(cluster_ids_sorted)
            color_scale = [
                f"hsl({int(240 - 240 * i / max(n - 1, 1))}, 70%, 60%)" for i in range(n)
            ]
            traces = [
                go.Bar(
                    x=labels,
                    y=pcts,
                    marker=dict(
                        color=color_scale, line=dict(color="#585b70", width=0.8)
                    ),
                    hovertext=hover_texts,
                    hoverinfo="text",
                    text=[f"{p:.1f}%" for p in pcts],
                    textposition="outside",
                    textfont=dict(color="#e2e8f0", size=11),
                )
            ]
            barmode = "group"

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=16, color="#e2e8f0"),
                x=0.5,
            ),
            barmode=barmode,
            xaxis=dict(
                title="Métacluster",
                color="#e2e8f0",
                gridcolor="#313244",
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title="% Cellules",
                color="#e2e8f0",
                gridcolor="#313244",
                ticksuffix="%",
            ),
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            font=dict(color="#e2e8f0"),
            height=520,
            margin=dict(l=60, r=40, t=80, b=60),
            bargap=0.25,
            legend=dict(bgcolor="#313244", bordercolor="#585b70"),
        )

        if output_html is not None:
            out_html = Path(output_html)
            out_html.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out_html), include_plotlyjs="cdn")
            _logger.info("%%Cellules par cluster (HTML) sauvegardé: %s", out_html.name)

        if output_jpg is not None:
            out_jpg = Path(output_jpg)
            out_jpg.parent.mkdir(parents=True, exist_ok=True)
            try:
                fig.write_image(
                    str(out_jpg), format="jpg", width=1400, height=620, scale=2
                )
                _logger.info(
                    "%%Cellules par cluster (JPG) sauvegardé: %s", out_jpg.name
                )
            except Exception as _img_err:
                _logger.warning(
                    "Export JPG échoué (kaleido requis — pip install kaleido): %s",
                    _img_err,
                )

        return fig

    except Exception as exc:
        _logger.error("Échec plot_cells_pct_per_cluster: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Bar charts POST-FlowSOM par nœud SOM (100 clusters pour grille 10×10)
# ─────────────────────────────────────────────────────────────────────────────


def plot_patho_pct_per_som_node(
    clustering: np.ndarray,
    condition_labels: np.ndarray,
    output_html: Optional[Path | str] = None,
    output_jpg: Optional[Path | str] = None,
    *,
    patho_label: str = "Pathologique",
    title: str = "% Cellules pathologiques par cluster SOM (post-FlowSOM)",
) -> Optional[Any]:
    """
    Bar chart Plotly : % de cellules provenant de la moelle pathologique dans chaque
    nœud SOM (cluster fin, ex. 100 nœuds pour une grille 10×10).

    Pour chaque nœud SOM, calcule :
        pct_patho = n_cells_patho_in_node / n_cells_total_in_node × 100

    Permet d'identifier les nœuds enrichis en cellules pathologiques (populations LAIP)
    à une résolution plus fine que les métaclusters.

    Args:
        clustering: Assignation de nœud SOM par cellule (n_cells,), entiers 0…N-1.
        condition_labels: Étiquette de condition par cellule (n_cells,).
        output_html: Chemin HTML de sortie (optionnel).
        output_jpg: Chemin JPG de sortie (optionnel).
        patho_label: Valeur dans condition_labels désignant la condition pathologique.
        title: Titre du graphique.

    Returns:
        Figure Plotly ou None si plotly absent.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        _logger.warning("plotly requis pour plot_patho_pct_per_som_node")
        return None

    try:
        cond_arr = np.asarray(condition_labels)
        node_ids = np.sort(np.unique(clustering))
        n_nodes = len(node_ids)
        labels = [f"C{int(n)}" for n in node_ids]

        pct_patho: List[float] = []
        cell_counts: List[int] = []
        hover_texts: List[str] = []

        for nid in node_ids:
            mask = clustering == nid
            total = int(mask.sum())
            n_patho = int((mask & (cond_arr == patho_label)).sum())
            pct = (n_patho / total * 100) if total > 0 else 0.0
            pct_patho.append(round(pct, 2))
            cell_counts.append(total)
            hover_texts.append(
                f"<b>Cluster {int(nid)}</b><br>"
                f"Cellules patho : {n_patho:,}<br>"
                f"Total cluster  : {total:,}<br>"
                f"% Patho        : {pct:.2f}%"
            )

        # Gradient rouge (fort % patho) → bleu (faible % patho)
        bar_colors = [
            f"rgba({int(243 * p / 100 + 137 * (1 - p / 100))}, "
            f"{int(139 * (1 - p / 100))}, "
            f"{int(168 * (1 - p / 100) + 250 * (1 - p / 100))}, 0.85)"
            for p in pct_patho
        ]

        fig = go.Figure(
            go.Bar(
                x=labels,
                y=pct_patho,
                marker=dict(
                    color=bar_colors,
                    line=dict(color="#585b70", width=0.5),
                ),
                hovertext=hover_texts,
                hoverinfo="text",
                text=[f"{p:.1f}%" for p in pct_patho],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=8),
                width=0.7,
            )
        )

        max_y = max(pct_patho) if pct_patho else 10.0
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><sup>{n_nodes} nœuds SOM — {len(clustering):,} cellules</sup>",
                font=dict(size=15, color="#e2e8f0"),
                x=0.5,
            ),
            xaxis=dict(
                title="Nœud SOM (cluster)",
                color="#e2e8f0",
                gridcolor="#313244",
                tickfont=dict(size=9),
                tickangle=90,
                tickmode="array",
                tickvals=labels,
                ticktext=labels,
            ),
            yaxis=dict(
                title="% Cellules pathologiques",
                color="#e2e8f0",
                gridcolor="#313244",
                range=[0, min(max_y * 1.18, 105)],
                ticksuffix="%",
            ),
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            font=dict(color="#e2e8f0"),
            height=580,
            width=max(1400, n_nodes * 22),
            margin=dict(l=70, r=40, t=100, b=110),
            bargap=0.15,
        )

        # Ligne de référence à 100 %
        fig.add_hline(
            y=100,
            line=dict(color="#f38ba8", dash="dash", width=1.5),
            annotation_text="100 % patho",
            annotation_font=dict(color="#f38ba8", size=10),
        )

        # Ligne de référence à 50 %
        fig.add_hline(
            y=50,
            line=dict(color="#f9e2af", dash="dot", width=1.0),
            annotation_text="50%",
            annotation_font=dict(color="#f9e2af", size=9),
        )

        if output_html is not None:
            out_html = Path(output_html)
            out_html.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out_html), include_plotlyjs="cdn")
            _logger.info("Patho%% par nœud SOM (HTML) sauvegardé: %s", out_html.name)

        if output_jpg is not None:
            out_jpg = Path(output_jpg)
            out_jpg.parent.mkdir(parents=True, exist_ok=True)
            try:
                fig.write_image(
                    str(out_jpg),
                    format="jpg",
                    width=max(1800, n_nodes * 22),
                    height=680,
                    scale=2,
                )
                _logger.info("Patho%% par nœud SOM (JPG) sauvegardé: %s", out_jpg.name)
            except Exception as _img_err:
                _logger.warning(
                    "Export JPG échoué (kaleido requis — pip install kaleido): %s",
                    _img_err,
                )

        return fig

    except Exception as exc:
        _logger.error("Échec plot_patho_pct_per_som_node: %s", exc)
        return None


def plot_cells_pct_per_som_node(
    clustering: np.ndarray,
    output_html: Optional[Path | str] = None,
    output_jpg: Optional[Path | str] = None,
    *,
    condition_labels: Optional[np.ndarray] = None,
    title: str = "% Cellules par cluster SOM (post-FlowSOM)",
) -> Optional[Any]:
    """
    Bar chart Plotly : distribution en % des cellules par nœud SOM (cluster fin).

    Pour chaque nœud SOM :
        pct = n_cells_in_node / n_cells_total × 100

    Si ``condition_labels`` est fourni, les barres sont empilées par condition
    (Sain / Pathologique), permettant de visualiser la composition de chaque nœud.

    Args:
        clustering: Assignation de nœud SOM par cellule (n_cells,), entiers 0…N-1.
        output_html: Chemin HTML de sortie (optionnel).
        output_jpg: Chemin JPG de sortie (optionnel).
        condition_labels: Étiquette de condition par cellule (optionnel).
        title: Titre du graphique.

    Returns:
        Figure Plotly ou None si plotly absent.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        _logger.warning("plotly requis pour plot_cells_pct_per_som_node")
        return None

    try:
        total_cells = len(clustering)
        node_ids = np.sort(np.unique(clustering))
        n_nodes = len(node_ids)
        labels = [f"C{int(n)}" for n in node_ids]

        _COND_COLORS = ["#89b4fa", "#f38ba8", "#a6e3a1", "#f9e2af", "#cba6f7"]

        if condition_labels is not None:
            cond_arr = np.asarray(condition_labels)
            unique_conds = sorted(set(cond_arr.tolist()))
            traces: List[Any] = []
            for ci, cond in enumerate(unique_conds):
                cond_mask = cond_arr == cond
                pcts: List[float] = []
                hover_texts: List[str] = []
                for nid in node_ids:
                    mask = clustering == nid
                    n_in_cond = int((mask & cond_mask).sum())
                    pct = n_in_cond / total_cells * 100
                    pcts.append(round(pct, 3))
                    hover_texts.append(
                        f"<b>Cluster {int(nid)}</b> — {cond}<br>"
                        f"Cellules : {n_in_cond:,}<br>"
                        f"% global : {pct:.3f}%"
                    )
                traces.append(
                    go.Bar(
                        name=str(cond),
                        x=labels,
                        y=pcts,
                        marker=dict(
                            color=_COND_COLORS[ci % len(_COND_COLORS)],
                            line=dict(color="#585b70", width=0.4),
                        ),
                        hovertext=hover_texts,
                        hoverinfo="text",
                        text=[f"{p:.2f}%" if p >= 0.05 else "" for p in pcts],
                        textposition="inside",
                        textfont=dict(color="white", size=7),
                    )
                )
            barmode = "stack"
        else:
            counts = np.array([int((clustering == n).sum()) for n in node_ids])
            pcts_list = [round(int(cnt) / total_cells * 100, 3) for cnt in counts]
            hover_texts_simple = [
                f"<b>Cluster {int(nid)}</b><br>"
                f"Cellules : {int(cnt):,}<br>"
                f"% Total  : {pct:.3f}%"
                for nid, cnt, pct in zip(node_ids, counts, pcts_list)
            ]
            n = len(node_ids)
            color_scale = [
                f"hsl({int(240 - 240 * i / max(n - 1, 1))}, 70%, 60%)" for i in range(n)
            ]
            traces = [
                go.Bar(
                    x=labels,
                    y=pcts_list,
                    marker=dict(
                        color=color_scale,
                        line=dict(color="#585b70", width=0.5),
                    ),
                    hovertext=hover_texts_simple,
                    hoverinfo="text",
                    text=[f"{p:.2f}%" if p >= 0.05 else "" for p in pcts_list],
                    textposition="outside",
                    textfont=dict(color="#e2e8f0", size=8),
                )
            ]
            barmode = "group"

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><sup>{n_nodes} nœuds SOM — {total_cells:,} cellules</sup>",
                font=dict(size=15, color="#e2e8f0"),
                x=0.5,
            ),
            barmode=barmode,
            xaxis=dict(
                title="Nœud SOM (cluster)",
                color="#e2e8f0",
                gridcolor="#313244",
                tickfont=dict(size=9),
                tickangle=90,
                tickmode="array",
                tickvals=labels,
                ticktext=labels,
            ),
            yaxis=dict(
                title="% Cellules",
                color="#e2e8f0",
                gridcolor="#313244",
                ticksuffix="%",
            ),
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            font=dict(color="#e2e8f0"),
            height=580,
            width=max(1400, n_nodes * 22),
            margin=dict(l=70, r=40, t=100, b=110),
            bargap=0.15,
            legend=dict(bgcolor="#313244", bordercolor="#585b70"),
        )

        if output_html is not None:
            out_html = Path(output_html)
            out_html.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out_html), include_plotlyjs="cdn")
            _logger.info("%%Cellules par nœud SOM (HTML) sauvegardé: %s", out_html.name)

        if output_jpg is not None:
            out_jpg = Path(output_jpg)
            out_jpg.parent.mkdir(parents=True, exist_ok=True)
            try:
                fig.write_image(
                    str(out_jpg),
                    format="jpg",
                    width=max(1800, n_nodes * 22),
                    height=680,
                    scale=2,
                )
                _logger.info(
                    "%%Cellules par nœud SOM (JPG) sauvegardé: %s", out_jpg.name
                )
            except Exception as _img_err:
                _logger.warning(
                    "Export JPG échoué (kaleido requis — pip install kaleido): %s",
                    _img_err,
                )

        return fig

    except Exception as exc:
        _logger.error("Échec plot_cells_pct_per_som_node: %s", exc)
        return None


def plot_combined_som_node_html(
    clustering: np.ndarray,
    condition_labels: np.ndarray,
    output_html: Path | str,
    *,
    patho_label: str = "Pathologique",
    title: str = "Analyse post-FlowSOM — Clusters SOM",
) -> Optional[Any]:
    """
    HTML combiné : deux bar charts partageant le même axe X, triés par % patho décroissant.

    Panneau supérieur  : % de cellules pathologiques par nœud SOM.
    Panneau inférieur  : % de cellules (empilé Sain/Patho) par nœud SOM.

    Les deux panneaux utilisent **exactement le même ordre de clusters** sur l'axe X :
    trié par % de cellules pathologiques décroissant.  Le cluster le plus enrichi en
    cellules patho apparaît à gauche dans les deux vues, ce qui permet de lire
    en un coup d'œil quels nœuds SOM concentrent les cellules tumorales.

    Args:
        clustering: Assignation de nœud SOM par cellule (n_cells,), entiers 0…N-1.
        condition_labels: Étiquette de condition par cellule (n_cells,).
        output_html: Chemin HTML de sortie.
        patho_label: Valeur désignant la condition pathologique.
        title: Titre général de la figure.

    Returns:
        Figure Plotly ou None si plotly absent / condition_labels manquant.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        _logger.warning("plotly requis pour plot_combined_som_node_html")
        return None

    if condition_labels is None:
        _logger.warning("plot_combined_som_node_html ignoré : condition_labels requis")
        return None

    try:
        cond_arr = np.asarray(condition_labels)
        node_ids = np.sort(np.unique(clustering))
        n_nodes = len(node_ids)
        total_cells = len(clustering)

        # ── Calcul des métriques brutes dans l'ordre naturel ─────────────────
        pct_patho_raw: List[float] = []
        hover_patho_raw: List[str] = []
        for nid in node_ids:
            mask = clustering == nid
            total = int(mask.sum())
            n_patho = int((mask & (cond_arr == patho_label)).sum())
            pct = (n_patho / total * 100) if total > 0 else 0.0
            pct_patho_raw.append(round(pct, 2))
            hover_patho_raw.append(
                f"<b>Cluster {int(nid)}</b><br>"
                f"Cellules patho : {n_patho:,}<br>"
                f"Total cluster  : {total:,}<br>"
                f"% Patho        : {pct:.2f}%"
            )

        # ── Tri par % patho décroissant ───────────────────────────────────────
        sort_idx = np.argsort(pct_patho_raw)[::-1]  # indices triés
        sorted_node_ids = node_ids[sort_idx]
        sorted_pct_patho = [pct_patho_raw[i] for i in sort_idx]
        sorted_hover_patho = [hover_patho_raw[i] for i in sort_idx]
        x_labels = [f"C{int(n)}" for n in sorted_node_ids]

        # ── Gradient de couleur rouge→bleu selon % patho ─────────────────────
        bar_colors_patho = [
            f"rgba({int(243 * p / 100 + 137 * (1 - p / 100))}, "
            f"{int(139 * (1 - p / 100))}, "
            f"{int(168 * (1 - p / 100) + 250 * (1 - p / 100))}, 0.85)"
            for p in sorted_pct_patho
        ]

        # ── Métriques % cellules (panneau bas) dans le même ordre trié ───────
        _COND_COLORS = ["#89b4fa", "#f38ba8", "#a6e3a1", "#f9e2af", "#cba6f7"]
        unique_conds = sorted(set(cond_arr.tolist()))
        bottom_traces: List[go.Bar] = []
        for ci, cond in enumerate(unique_conds):
            cond_mask = cond_arr == cond
            pcts_cond: List[float] = []
            hover_cond: List[str] = []
            for nid in sorted_node_ids:
                mask = clustering == nid
                n_in_cond = int((mask & cond_mask).sum())
                pct_c = n_in_cond / total_cells * 100
                pcts_cond.append(round(pct_c, 3))
                hover_cond.append(
                    f"<b>Cluster {int(nid)}</b> — {cond}<br>"
                    f"Cellules : {n_in_cond:,}<br>"
                    f"% global : {pct_c:.3f}%"
                )
            bottom_traces.append(
                go.Bar(
                    name=str(cond),
                    x=x_labels,
                    y=pcts_cond,
                    marker=dict(
                        color=_COND_COLORS[ci % len(_COND_COLORS)],
                        line=dict(color="#585b70", width=0.4),
                    ),
                    hovertext=hover_cond,
                    hoverinfo="text",
                    text=[f"{p:.2f}%" if p >= 0.05 else "" for p in pcts_cond],
                    textposition="inside",
                    textfont=dict(color="white", size=7),
                    showlegend=True,
                    legendgroup=str(cond),
                )
            )

        # ── Figure à deux sous-graphiques empilés, axe X partagé ─────────────
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                "% Cellules pathologiques par cluster SOM "
                "(trié par % patho décroissant)",
                "% Cellules par cluster SOM — distribution Sain / Patho (même ordre)",
            ),
        )

        # Panneau 1 — % patho
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=sorted_pct_patho,
                marker=dict(
                    color=bar_colors_patho,
                    line=dict(color="#585b70", width=0.5),
                ),
                hovertext=sorted_hover_patho,
                hoverinfo="text",
                text=[f"{p:.1f}%" if p >= 1.0 else "" for p in sorted_pct_patho],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=8),
                showlegend=False,
                name="% Patho",
            ),
            row=1,
            col=1,
        )

        # Ligne 100 % et 50 %
        max_y1 = max(sorted_pct_patho) if sorted_pct_patho else 10.0
        fig.add_hline(
            y=100,
            line=dict(color="#f38ba8", dash="dash", width=1.5),
            annotation_text="100%",
            annotation_font=dict(color="#f38ba8", size=9),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=50,
            line=dict(color="#f9e2af", dash="dot", width=1.0),
            annotation_text="50%",
            annotation_font=dict(color="#f9e2af", size=9),
            row=1,
            col=1,
        )

        # Panneau 2 — % cellules par condition
        for trace in bottom_traces:
            fig.add_trace(trace, row=2, col=1)

        # ── Mise en page globale ──────────────────────────────────────────────
        chart_width = max(1600, n_nodes * 25)
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>{title}</b><br>"
                    f"<sup>{n_nodes} nœuds SOM — {total_cells:,} cellules — "
                    f"trié par % pathologique décroissant</sup>"
                ),
                font=dict(size=16, color="#e2e8f0"),
                x=0.5,
            ),
            barmode="stack",
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            font=dict(color="#e2e8f0"),
            height=900,
            width=chart_width,
            margin=dict(l=70, r=40, t=110, b=120),
            bargap=0.12,
            legend=dict(
                bgcolor="#313244",
                bordercolor="#585b70",
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.12,
            ),
        )

        # X partagé — labels en bas seulement (panneau 2), rotation 90°
        fig.update_xaxes(
            tickfont=dict(size=9),
            tickangle=90,
            color="#e2e8f0",
            gridcolor="#313244",
            tickmode="array",
            tickvals=x_labels,
            ticktext=x_labels,
        )

        # Axes Y panneau 1
        fig.update_yaxes(
            title_text="% Cellules pathologiques",
            color="#e2e8f0",
            gridcolor="#313244",
            ticksuffix="%",
            range=[0, min(max_y1 * 1.18, 105)],
            row=1,
            col=1,
        )
        # Axes Y panneau 2
        fig.update_yaxes(
            title_text="% Cellules",
            color="#e2e8f0",
            gridcolor="#313244",
            ticksuffix="%",
            row=2,
            col=1,
        )

        # Titres des sous-graphiques en blanc
        for ann in fig.layout.annotations:
            ann.font.color = "#cdd6f4"
            ann.font.size = 13

        # ── Export HTML ───────────────────────────────────────────────────────
        out_html = Path(output_html)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_html), include_plotlyjs="cdn")
        _logger.info("HTML combiné nœuds SOM sauvegardé: %s", out_html.name)

        return fig

    except Exception as exc:
        _logger.error("Échec plot_combined_som_node_html: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  MRD Résiduelle — Visualisations
# ─────────────────────────────────────────────────────────────────────────────


def plot_mrd_summary(
    mrd_result: "Any",
    output_html: Optional[Path | str] = None,
    output_png: Optional[Path | str] = None,
    *,
    title: str = "MRD Résiduelle — Analyse par Nœuds SOM",
) -> Optional["Any"]:
    """
    Figure Plotly scrollable (style fig_som_combined) montrant la MRD :

      Panneau haut (grand) : % patho vs % sain par nœud SOM, trié par
                              % patho décroissant. Bordure colorée selon
                              le statut MRD (JF / Flo / les deux).
      Panneau bas :           MRD% global — JF et Flo uniquement.

    Les contrôles ELN (LOQ ≥50 events, seuil clinique 0.1%) sont appliqués
    en amont dans le calcul mais ne sont PAS affichés comme barre séparée.
    Le statut ELN reste visible dans le titre et les infobulles.

    Args:
        mrd_result: MRDResult du module mrd_calculator.
        output_html: Chemin HTML interactif (optionnel).
        output_png: Chemin PNG statique (optionnel).
        title: Titre principal.

    Returns:
        Figure Plotly ou None.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        _logger.warning("plotly requis pour plot_mrd_summary")
        return None

    try:
        nodes = mrd_result.per_node
        if not nodes:
            _logger.warning("Aucun nœud SOM dans MRDResult — pas de figure.")
            return None

        n_nodes = len(nodes)
        total_cells = mrd_result.total_cells

        # ── Tri par % patho décroissant (comme fig_som_combined) ──────────
        sorted_nodes = sorted(nodes, key=lambda c: c.pct_patho, reverse=True)

        x_labels = [f"N{c.cluster_id}" for c in sorted_nodes]
        pct_patho = [round(c.pct_patho, 2) for c in sorted_nodes]
        pct_sain = [round(c.pct_sain, 2) for c in sorted_nodes]

        # Gradient rouge→bleu selon % patho
        bar_colors_patho = [
            f"rgba({int(243 * p / 100 + 137 * (1 - p / 100))}, "
            f"{int(139 * (1 - p / 100))}, "
            f"{int(168 * (1 - p / 100) + 250 * (1 - p / 100))}, 0.85)"
            for p in pct_patho
        ]

        # Bordure colorée par statut MRD (JF + Flo uniquement)
        border_colors = []
        for c in sorted_nodes:
            if c.is_mrd_jf and c.is_mrd_flo:
                border_colors.append("#fab387")  # orange = les deux
            elif c.is_mrd_jf:
                border_colors.append("#f9e2af")  # or = JF seul
            elif c.is_mrd_flo:
                border_colors.append("#89dceb")  # cyan = Flo seul
            else:
                border_colors.append("#585b70")  # gris = non-MRD

        # Statut ELN textuel (contrôle uniquement, pas de barre)
        eln_status = ""
        if hasattr(mrd_result, "eln_positive"):
            if mrd_result.eln_positive:
                eln_status = "MRD POSITIVE"
            elif mrd_result.eln_low_level:
                eln_status = "MRD LOW-LEVEL"
            else:
                eln_status = "MRD NEGATIVE"

        hover_texts = [
            f"<b>Nœud SOM {c.cluster_id}</b><br>"
            f"Patho: {c.n_cells_patho:,} ({c.pct_patho:.2f}%)<br>"
            f"Sain: {c.n_cells_sain:,} ({c.pct_sain:.2f}%)<br>"
            f"Total: {c.n_cells_total:,}<br>"
            f"─────────────<br>"
            f"MRD JF: <b>{'OUI' if c.is_mrd_jf else 'non'}</b><br>"
            f"MRD Flo: <b>{'OUI' if c.is_mrd_flo else 'non'}</b><br>"
            f"ELN LOQ (≥50): <b>{'✓' if c.n_cells_total >= 50 else '✗'}</b>"
            for c in sorted_nodes
        ]

        # ── Figure 2 panneaux, axes X indépendants ───────────────────────
        # shared_xaxes=False : panneau 2 a ses propres labels (JF / Flo),
        # indépendants des nœuds SOM du panneau 1.
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.10,
            subplot_titles=[
                f"% Cellules Pathologiques vs Moelle Normale par Nœud SOM "
                f"(trié par % patho décroissant — {n_nodes} nœuds)",
                "MRD Résiduelle Globale (% vs cellules totales du patient)",
            ],
        )

        # ── Panneau 1 : barres patho + sain par nœud ─────────────────────
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=pct_patho,
                name="% Pathologique",
                marker=dict(
                    color=bar_colors_patho,
                    line=dict(color=border_colors, width=2.5),
                ),
                hovertext=hover_texts,
                hoverinfo="text",
                text=[f"{p:.1f}%" if p >= 1.0 else "" for p in pct_patho],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=8),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=pct_sain,
                name="% Moelle Normale",
                marker=dict(
                    color="rgba(137, 180, 250, 0.65)",
                    line=dict(color="#585b70", width=0.4),
                ),
                text=[f"{p:.1f}%" if p >= 1.0 else "" for p in pct_sain],
                textposition="inside",
                textfont=dict(color="white", size=7),
            ),
            row=1,
            col=1,
        )

        # Lignes de référence
        fig.add_hline(
            y=50,
            line=dict(color="#f9e2af", dash="dot", width=1.0),
            annotation_text="50%",
            annotation_font=dict(color="#f9e2af", size=9),
            row=1,
            col=1,
        )

        # ── Panneau 2 : MRD% global — JF et Flo uniquement ───────────────
        method_names = []
        mrd_pcts = []
        mrd_cells_list = []
        bar_cols = []
        n_nodes_list = []

        m = mrd_result.method_used
        if m in ("jf", "both", "all"):
            method_names.append("Méthode JF")
            mrd_pcts.append(round(mrd_result.mrd_pct_jf, 4))
            mrd_cells_list.append(mrd_result.mrd_cells_jf)
            bar_cols.append("rgba(249, 226, 175, 0.9)")
            n_nodes_list.append(mrd_result.n_nodes_mrd_jf)

        if m in ("flo", "both", "all"):
            method_names.append("Méthode Flo")
            mrd_pcts.append(round(mrd_result.mrd_pct_flo, 4))
            mrd_cells_list.append(mrd_result.mrd_cells_flo)
            bar_cols.append("rgba(137, 220, 235, 0.9)")
            n_nodes_list.append(mrd_result.n_nodes_mrd_flo)

        hover_mrd = [
            f"<b>{name}</b><br>"
            f"MRD: {pct:.4f}%<br>"
            f"Cellules MRD: {cells:,} dans {nn} nœuds<br>"
            f"Cellules totales: {total_cells:,}"
            for name, pct, cells, nn in zip(
                method_names, mrd_pcts, mrd_cells_list, n_nodes_list
            )
        ]

        fig.add_trace(
            go.Bar(
                x=method_names,
                y=mrd_pcts,
                name="MRD %",
                marker=dict(
                    color=bar_cols,
                    line=dict(color="#cdd6f4", width=1.5),
                ),
                hovertext=hover_mrd,
                hoverinfo="text",
                text=[f"<b>{p:.4f}%</b>" for p in mrd_pcts],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=14, family="monospace"),
                width=0.35,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Annotations sous les barres
        for name, cells, nn in zip(method_names, mrd_cells_list, n_nodes_list):
            fig.add_annotation(
                x=name,
                y=0,
                text=f"{cells:,} cellules · {nn} nœuds",
                showarrow=False,
                yshift=-15,
                font=dict(color="#a6adc8", size=11),
                xref="x2",
                yref="y2",
            )

        # ── Layout — grande taille scrollable ─────────────────────────────
        chart_width = max(1600, n_nodes * 30)

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>{title}</b><br>"
                    f"<sup>{n_nodes} nœuds SOM — {total_cells:,} cellules"
                    + (f" — Contrôle ELN: {eln_status}" if eln_status else "")
                    + "</sup>"
                ),
                font=dict(size=16, color="#e2e8f0"),
                x=0.5,
            ),
            barmode="group",
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            font=dict(color="#e2e8f0"),
            height=950,
            width=chart_width,
            margin=dict(l=70, r=40, t=110, b=130),
            bargap=0.25,
            legend=dict(
                bgcolor="#313244",
                bordercolor="#585b70",
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.08,
                font=dict(color="#cdd6f4", size=11),
            ),
        )

        # X partagé — labels en bas, rotation 90°
        fig.update_xaxes(
            tickfont=dict(size=9),
            tickangle=90,
            color="#e2e8f0",
            gridcolor="#313244",
            tickmode="array",
            tickvals=x_labels,
            ticktext=x_labels,
            row=1,
            col=1,
        )
        fig.update_xaxes(color="#e2e8f0", gridcolor="#313244", row=2, col=1)

        max_y1 = max(max(pct_patho, default=1), max(pct_sain, default=1))
        max_y2 = max(mrd_pcts, default=0.01)
        fig.update_yaxes(
            title_text="% dans le nœud SOM",
            color="#e2e8f0",
            gridcolor="#313244",
            range=[0, min(max_y1 * 1.18, 105)],
            ticksuffix="%",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text="MRD (%)",
            color="#e2e8f0",
            gridcolor="#313244",
            range=[0, max(max_y2 * 1.6, 0.01)],
            ticksuffix="%",
            row=2,
            col=1,
        )

        # Titres subplots en blanc
        for ann in fig.layout.annotations:
            ann.font.color = "#cdd6f4"
            ann.font.size = 13

        # Légende bordure MRD
        fig.add_annotation(
            text=(
                "Bordure nœud: "
                "<span style='color:#f9e2af'>■</span> MRD JF · "
                "<span style='color:#89dceb'>■</span> MRD Flo · "
                "<span style='color:#fab387'>■</span> Les deux · "
                "<span style='color:#585b70'>■</span> Non-MRD"
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.13,
            showarrow=False,
            font=dict(color="#a6adc8", size=11),
        )

        # ── Export ────────────────────────────────────────────────────────
        if output_html is not None:
            out_html = Path(output_html)
            out_html.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out_html), include_plotlyjs="cdn")
            _logger.info("MRD summary (HTML): %s", out_html.name)

        if output_png is not None:
            out_png = Path(output_png)
            out_png.parent.mkdir(parents=True, exist_ok=True)
            try:
                fig.write_image(
                    str(out_png), format="png", width=1800, height=950, scale=2
                )
                _logger.info("MRD summary (PNG): %s", out_png.name)
            except Exception as _img_err:
                _logger.warning("Export PNG MRD échoué (kaleido requis): %s", _img_err)

        return fig

    except Exception as exc:
        _logger.error("Échec plot_mrd_summary: %s", exc)
        return None
