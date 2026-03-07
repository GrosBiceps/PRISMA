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
    Graphique en barres de la taille des métaclusters.

    Peut afficher la composition par condition (Sain/Patho) via des
    barres empilées si condition_labels est fourni.

    Args:
        metaclustering: Assignation par cellule (n_cells,).
        n_metaclusters: Nombre total de métaclusters.
        output_path: Chemin PNG de sortie.
        condition_labels: Labels de condition par cellule (optionnel).
        title: Titre du graphique.

    Returns:
        True si succès.
    """
    if not _MPL_AVAILABLE:
        return None

    try:
        fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG_COLOR)

        cluster_ids = np.arange(n_metaclusters)

        if condition_labels is None:
            counts = np.array([int((metaclustering == i).sum()) for i in cluster_ids])
            bars = ax.bar(
                cluster_ids,
                counts,
                color="#89b4fa",
                edgecolor=SPINE_COLOR,
                linewidth=0.8,
            )
            # Annotations
            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    f"{count:,}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=TEXT_COLOR,
                )
        else:
            # Barres empilées par condition
            unique_conds = sorted(set(condition_labels))
            colors_list = ["#89b4fa", "#f38ba8", "#a6e3a1", "#f9e2af"]
            bottom = np.zeros(n_metaclusters)

            for i, cond in enumerate(unique_conds):
                cond_mask = condition_labels == cond
                counts = np.array(
                    [
                        int(((metaclustering == k) & cond_mask).sum())
                        for k in cluster_ids
                    ]
                )
                ax.bar(
                    cluster_ids,
                    counts,
                    bottom=bottom,
                    color=colors_list[i % len(colors_list)],
                    edgecolor=SPINE_COLOR,
                    linewidth=0.5,
                    label=str(cond),
                )
                bottom += counts

            ax.legend(
                facecolor="#313244",
                labelcolor=TEXT_COLOR,
                edgecolor=SPINE_COLOR,
                fontsize=11,
            )

        ax.set_xlabel("Métacluster", fontsize=13, color=TEXT_COLOR)
        ax.set_ylabel("Nombre de cellules", fontsize=13, color=TEXT_COLOR)
        ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_COLOR)
        ax.set_xticks(cluster_ids)
        ax.set_xticklabels([f"MC{i}" for i in cluster_ids], rotation=45, ha="right")
        apply_dark_style(ax)

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
    max_pts: int = 100_000,
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
        from scipy.spatial.distance import cdist
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.sparse import csr_matrix
        from collections import defaultdict

        layout_coords = clusterer.get_layout_coords()  # (n_nodes, 2)
        node_sizes = clusterer.get_node_sizes()  # (n_nodes,)
        n_nodes = clusterer.n_nodes

        # ── Métacluster dominant par node ─────────────────────────────────────
        mc_per_node = np.full(n_nodes, -1, dtype=int)
        na = getattr(clusterer, "node_assignments_", None)
        ma = getattr(clusterer, "metacluster_assignments_", None)
        if na is not None and ma is not None:
            for node_id in range(n_nodes):
                cells_in_node = na == node_id
                if cells_in_node.any():
                    mc_per_node[node_id] = int(np.bincount(ma[cells_in_node]).argmax())

        # ── Arêtes MST depuis le codebook ─────────────────────────────────────
        edges: List[Tuple[int, int]] = []
        codebook = None
        fsm = getattr(clusterer, "_fsom_model", None)
        if fsm is not None:
            if hasattr(fsm, "codes"):
                codebook = np.asarray(fsm.codes, dtype=float)
            elif hasattr(fsm, "model") and hasattr(fsm.model, "codes"):
                codebook = np.asarray(fsm.model.codes, dtype=float)
        if codebook is not None:
            dist_mat = cdist(codebook, codebook, metric="euclidean")
            mst = minimum_spanning_tree(csr_matrix(dist_mat))
            coo = mst.tocoo()
            edges = list(zip(coo.row.tolist(), coo.col.tolist()))

        # ── Dessin ────────────────────────────────────────────────────────────
        n_meta = int(metaclustering.max()) + 1 if len(metaclustering) > 0 else 1
        cmap = plt.cm.get_cmap("tab20", n_meta)
        max_sz = float(node_sizes.max()) if node_sizes.max() > 0 else 1.0
        display_sizes = 80 + (node_sizes / max_sz) * 500

        fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        # Arêtes
        for i, j in edges:
            ax.plot(
                [layout_coords[i, 0], layout_coords[j, 0]],
                [layout_coords[i, 1], layout_coords[j, 1]],
                color=SPINE_COLOR,
                lw=1.2,
                zorder=1,
                alpha=0.7,
            )

        # Nodes
        node_colors = [
            cmap(mc_per_node[i] % 20)
            if mc_per_node[i] >= 0
            else (0.45, 0.45, 0.45, 1.0)
            for i in range(n_nodes)
        ]
        ax.scatter(
            layout_coords[:, 0],
            layout_coords[:, 1],
            s=display_sizes,
            c=node_colors,
            zorder=2,
            edgecolors=TEXT_COLOR,
            linewidths=0.5,
            alpha=0.9,
        )

        # Labels — centroïde de chaque métacluster
        mc_pts: dict = defaultdict(list)
        for i in range(n_nodes):
            if mc_per_node[i] >= 0:
                mc_pts[mc_per_node[i]].append(layout_coords[i])
        for mc_id, pts in mc_pts.items():
            centroid = np.mean(pts, axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                f"MC{mc_id}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                zorder=3,
            )

        ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=12)
        ax.set_xlabel("MST Dim 1", fontsize=11, color=TEXT_COLOR)
        ax.set_ylabel("MST Dim 2", fontsize=11, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(SPINE_COLOR)

        # Légende métaclusters
        legend_handles = [
            plt.scatter([], [], s=80, color=cmap(i % 20), label=f"MC{i}")
            for i in range(n_meta)
        ]
        ax.legend(
            handles=legend_handles,
            loc="best",
            facecolor="#313244",
            labelcolor=TEXT_COLOR,
            edgecolor=SPINE_COLOR,
            fontsize=9,
            title="Métacluster",
            title_fontsize=9,
        )

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

        # Métacluster dominant par node
        mc_per_node = np.full(n_nodes, -1, dtype=int)
        na = getattr(clusterer, "node_assignments_", None)
        ma = getattr(clusterer, "metacluster_assignments_", None)
        if na is not None and ma is not None:
            for node_id in range(n_nodes):
                mask_node = na == node_id
                if mask_node.any():
                    mc_per_node[node_id] = int(np.bincount(ma[mask_node]).argmax())

        # Arêtes MST
        edge_x: List[Optional[float]] = []
        edge_y: List[Optional[float]] = []
        fsm = getattr(clusterer, "_fsom_model", None)
        codebook = None
        if fsm is not None:
            if hasattr(fsm, "codes"):
                codebook = np.asarray(fsm.codes, dtype=float)
            elif hasattr(fsm, "model") and hasattr(fsm.model, "codes"):
                codebook = np.asarray(fsm.model.codes, dtype=float)
        if codebook is not None:
            dist_mat = cdist(codebook, codebook, metric="euclidean")
            mst = minimum_spanning_tree(csr_matrix(dist_mat))
            coo = mst.tocoo()
            for i, j in zip(coo.row.tolist(), coo.col.tolist()):
                edge_x += [layout_coords[i, 0], layout_coords[j, 0], None]
                edge_y += [layout_coords[i, 1], layout_coords[j, 1], None]

        palette = (
            px.colors.qualitative.Set1
            + px.colors.qualitative.Pastel
            + px.colors.qualitative.Set2
        )
        n_meta = int(metaclustering.max()) + 1 if len(metaclustering) > 0 else 1
        max_sz = float(node_sizes.max()) if node_sizes.max() > 0 else 1.0

        traces: List[Any] = []
        # Arêtes en premier
        if edge_x:
            traces.append(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(color="rgba(180,180,180,0.35)", width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Nodes par métacluster
        for mc_id in range(n_meta):
            indices = [i for i in range(n_nodes) if mc_per_node[i] == mc_id]
            if not indices:
                continue
            idx = np.array(indices)
            node_sz = 10 + (node_sizes[idx] / max_sz) * 38

            mc_key = f"MC{mc_id}"
            if mc_key in mfi_matrix.index:
                top2 = mfi_matrix.loc[mc_key].nlargest(2).index.tolist()
                hover = [
                    f"Node {i}<br><b>MC{mc_id}</b><br>Cellules: {int(node_sizes[i]):,}<br>Top: {', '.join(top2)}"
                    for i in idx
                ]
            else:
                hover = [
                    f"Node {i}<br><b>MC{mc_id}</b><br>Cellules: {int(node_sizes[i]):,}"
                    for i in idx
                ]

            traces.append(
                go.Scatter(
                    x=layout_coords[idx, 0].tolist(),
                    y=layout_coords[idx, 1].tolist(),
                    mode="markers+text",
                    marker=dict(
                        size=node_sz.tolist(),
                        color=palette[mc_id % len(palette)],
                        line=dict(color="white", width=0.8),
                        opacity=0.9,
                    ),
                    text=[f"MC{mc_id}"] * len(idx),
                    textposition="middle center",
                    textfont=dict(size=8, color="white"),
                    name=f"MC{mc_id}",
                    hovertext=hover,
                    hoverinfo="text",
                )
            )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=16)),
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            font=dict(color="#e2e8f0"),
            height=620,
            showlegend=True,
            legend=dict(bgcolor="#313244", bordercolor="#585b70"),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            margin=dict(l=20, r=20, t=60, b=20),
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
