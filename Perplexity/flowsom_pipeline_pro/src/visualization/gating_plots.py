"""
gating_plots.py — Visualisations des étapes de pré-gating.

Génère les graphiques de QC du gating en utilisant les helpers bas-niveau
de plot_helpers.py. Un graphique par gate, sauvegardé en PNG.
Inclut la génération de diagrammes Sankey Plotly (global + par fichier).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import plotly.graph_objects as go

    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

from flowsom_pipeline_pro.src.visualization.plot_helpers import (
    plot_density,
    plot_gating,
    add_gate_rectangle,
    save_figure,
    BG_COLOR,
)
from flowsom_pipeline_pro.src.core.gating import PreGating
from flowsom_pipeline_pro.src.utils.logger import get_logger

_logger = get_logger("visualization.gating_plots")


def _find_idx(var_names: List[str], patterns: List[str]) -> Optional[int]:
    """Wrapper autour de PreGating.find_marker_index."""
    return PreGating.find_marker_index(var_names, patterns)


def plot_overview(
    X: np.ndarray,
    var_names: List[str],
    output_path: Path | str,
    n_sample: int = 60_000,
    seed: int = 42,
) -> bool:
    """
    Graphique 1: Vue d'ensemble FSC-A vs SSC-A (pré-gating).

    Args:
        X: Matrice brute (n_cells, n_markers).
        var_names: Noms des marqueurs.
        output_path: Chemin PNG de sortie.
        n_sample: Nombre de cellules à afficher.
        seed: Graine pour sous-échantillonnage.

    Returns:
        True si succès.
    """
    if not _MPL_AVAILABLE:
        _logger.warning("matplotlib manquant — plot_overview ignoré")
        return None

    fsc_a = _find_idx(var_names, ["FSC-A"])
    ssc_a = _find_idx(var_names, ["SSC-A", "SSC-H", "SSC"])

    if fsc_a is None or ssc_a is None:
        _logger.warning("FSC-A ou SSC-A non trouvé — plot_overview ignoré")
        return None

    rng = np.random.default_rng(seed)
    n = min(n_sample, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    plot_density(
        ax,
        X[idx, fsc_a],
        X[idx, ssc_a],
        f"VUE D'ENSEMBLE\n{X.shape[0]:,} événements totaux",
        "FSC-A (Forward Scatter - Taille)",
        "SSC-A (Side Scatter - Granularité)",
    )
    save_figure(fig, output_path)
    _logger.info("Sauvegardé: %s", output_path)
    return fig


def plot_debris_gate(
    X: np.ndarray,
    var_names: List[str],
    mask: np.ndarray,
    output_path: Path | str,
    debris_min_pct: float = 2.0,
    debris_max_pct: float = 99.0,
    n_sample: int = 60_000,
    seed: int = 42,
) -> Optional[Any]:
    """
    Graphique 2: Gate débris (FSC-A vs SSC-A avec overlay).

    Args:
        X: Matrice brute (n_cells, n_markers).
        var_names: Noms des marqueurs.
        mask: Masque G1 — True = cellule viable.
        output_path: Chemin PNG de sortie.
        debris_min_pct: Percentile borne inférieure de la gate.
        debris_max_pct: Percentile borne supérieure de la gate.
        n_sample: Nombre de cellules à afficher.
        seed: Graine pour sous-échantillonnage.

    Returns:
        True si succès.
    """
    if not _MPL_AVAILABLE:
        return None

    fsc_a = _find_idx(var_names, ["FSC-A"])
    ssc_a = _find_idx(var_names, ["SSC-A", "SSC-H", "SSC"])

    if fsc_a is None or ssc_a is None:
        _logger.warning("FSC-A ou SSC-A non trouvé — plot_debris_gate ignoré")
        return None

    rng = np.random.default_rng(seed)
    n = min(n_sample, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    plot_gating(
        ax,
        X[idx, fsc_a],
        X[idx, ssc_a],
        mask[idx],
        "GATE 1: Exclusion des Débris",
        "FSC-A (Taille)",
        "SSC-A (Granularité)",
        "Cellules viables",
        "Débris/Bruit",
    )

    # Rectangle de gate indicatif
    fsc_lo = np.nanpercentile(X[:, fsc_a], debris_min_pct)
    fsc_hi = np.nanpercentile(X[:, fsc_a], debris_max_pct)
    ssc_lo = np.nanpercentile(X[:, ssc_a], debris_min_pct)
    ssc_hi = np.nanpercentile(X[:, ssc_a], debris_max_pct)
    add_gate_rectangle(ax, fsc_lo, fsc_hi, ssc_lo, ssc_hi)

    save_figure(fig, output_path)
    _logger.info("Sauvegardé: %s", output_path)
    return fig


def plot_singlets_gate(
    X: np.ndarray,
    var_names: List[str],
    mask: np.ndarray,
    output_path: Path | str,
    n_sample: int = 60_000,
    seed: int = 42,
) -> Optional[Any]:
    """
    Graphique 3: Gate singlets (FSC-A vs FSC-H).

    Args:
        X: Matrice brute.
        var_names: Noms des marqueurs.
        mask: Masque G2 — True = singlet.
        output_path: Chemin PNG de sortie.
        n_sample: Nombre de cellules à afficher.
        seed: Graine.

    Returns:
        True si succès.
    """
    if not _MPL_AVAILABLE:
        return None

    fsc_a = _find_idx(var_names, ["FSC-A"])
    fsc_h = _find_idx(var_names, ["FSC-H"])

    if fsc_a is None or fsc_h is None:
        _logger.warning("FSC-A ou FSC-H non trouvé — plot_singlets_gate ignoré")
        return None

    rng = np.random.default_rng(seed)
    n = min(n_sample, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    plot_gating(
        ax,
        X[idx, fsc_a],
        X[idx, fsc_h],
        mask[idx],
        "GATE 2: Exclusion des Doublets",
        "FSC-A (Area)",
        "FSC-H (Height)",
        "Singlets",
        "Doublets",
    )
    save_figure(fig, output_path)
    _logger.info("Sauvegardé: %s", output_path)
    return fig


def plot_cd45_gate(
    X: np.ndarray,
    var_names: List[str],
    mask: np.ndarray,
    output_path: Path | str,
    n_sample: int = 60_000,
    seed: int = 42,
) -> Optional[Any]:
    """
    Graphique 4: Gate CD45+ (CD45 vs SSC-A).

    Args:
        X: Matrice brute.
        var_names: Noms des marqueurs.
        mask: Masque G3 — True = CD45+.
        output_path: Chemin PNG de sortie.
        n_sample: Nombre de cellules à afficher.
        seed: Graine.

    Returns:
        True si succès.
    """
    if not _MPL_AVAILABLE:
        return None

    cd45_idx = _find_idx(var_names, ["CD45", "CD45-PECY5", "CD45-PC5"])
    ssc_a = _find_idx(var_names, ["SSC-A", "SSC-H", "SSC"])

    if cd45_idx is None or ssc_a is None:
        _logger.warning("CD45 ou SSC-A non trouvé — plot_cd45_gate ignoré")
        return None

    rng = np.random.default_rng(seed)
    n = min(n_sample, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    plot_gating(
        ax,
        X[idx, cd45_idx],
        X[idx, ssc_a],
        mask[idx],
        "GATE 3: Cellules CD45+",
        "CD45",
        "SSC-A",
        "CD45+",
        "CD45-",
    )
    save_figure(fig, output_path)
    _logger.info("Sauvegardé: %s", output_path)
    return fig


def plot_cd34_gate(
    X: np.ndarray,
    var_names: List[str],
    mask: np.ndarray,
    output_path: Path | str,
    n_sample: int = 60_000,
    seed: int = 42,
) -> Optional[Any]:
    """
    Graphique 5: Gate CD34+ blastes (CD34 vs SSC-A).

    Args:
        X: Matrice brute.
        var_names: Noms des marqueurs.
        mask: Masque G4 — True = blast CD34+.
        output_path: Chemin PNG de sortie.
        n_sample: Nombre de cellules à afficher.
        seed: Graine.

    Returns:
        True si succès.
    """
    if not _MPL_AVAILABLE:
        return None

    cd34_idx = _find_idx(var_names, ["CD34", "CD34-PE", "CD34-APC"])
    ssc_a = _find_idx(var_names, ["SSC-A", "SSC-H", "SSC"])

    if cd34_idx is None or ssc_a is None:
        _logger.warning("CD34 ou SSC-A non trouvé — plot_cd34_gate ignoré")
        return None

    rng = np.random.default_rng(seed)
    n = min(n_sample, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    plot_gating(
        ax,
        X[idx, cd34_idx],
        X[idx, ssc_a],
        mask[idx],
        "GATE 4: CD34+ Blastes (SSC bas)",
        "CD34",
        "SSC-A",
        "CD34+ blastes",
        "Autres",
    )
    save_figure(fig, output_path)
    _logger.info("Sauvegardé: %s", output_path)
    return fig


def generate_all_gating_plots(
    X_raw: np.ndarray,
    var_names: List[str],
    masks: Dict[str, np.ndarray],
    output_dir: Path | str,
    sample_name: str = "sample",
    n_sample: int = 60_000,
    seed: int = 42,
) -> Dict[str, bool]:
    """
    Génère tous les graphiques de gating pour un échantillon.

    Args:
        X_raw: Matrice brute (pré-transformation, pré-gating).
        var_names: Noms des marqueurs.
        masks: Dict {gate_name: mask_array} — ex: {"G1": ..., "G2": ..., ...}.
        output_dir: Dossier de sortie.
        sample_name: Identifiant de l'échantillon pour les noms de fichiers.
        n_sample: Cellules max à afficher.
        seed: Graine.

    Returns:
        Dict {plot_name: figure_object_or_None}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = sample_name.replace(".fcs", "").replace(".FCS", "")

    results: Dict[str, Any] = {}

    results["01_overview"] = plot_overview(
        X_raw,
        var_names,
        output_dir / f"{name}_01_overview.png",
        n_sample=n_sample,
        seed=seed,
    )

    # Accept both short keys ("G1", "debris") and full keys ("G1_debris")
    def _get_mask(keys: List[str]) -> Optional[np.ndarray]:
        for k in keys:
            if k in masks:
                return masks[k]
        return None

    mask_g1 = _get_mask(["G1", "debris", "G1_debris"])
    if mask_g1 is not None:
        results["02_debris"] = plot_debris_gate(
            X_raw,
            var_names,
            mask_g1,
            output_dir / f"{name}_02_debris.png",
            n_sample=n_sample,
            seed=seed,
        )

    mask_g2 = _get_mask(["G2", "singlets", "G2_singlets"])
    if mask_g2 is not None:
        results["03_singlets"] = plot_singlets_gate(
            X_raw,
            var_names,
            mask_g2,
            output_dir / f"{name}_03_singlets.png",
            n_sample=n_sample,
            seed=seed,
        )

    mask_g3 = _get_mask(["G3", "cd45", "G3_cd45"])
    if mask_g3 is not None:
        results["04_cd45"] = plot_cd45_gate(
            X_raw,
            var_names,
            mask_g3,
            output_dir / f"{name}_04_cd45.png",
            n_sample=n_sample,
            seed=seed,
        )

    mask_g4 = _get_mask(["G4", "cd34", "G4_cd34"])
    if mask_g4 is not None:
        results["05_cd34"] = plot_cd34_gate(
            X_raw,
            var_names,
            mask_g4,
            output_dir / f"{name}_05_cd34.png",
            n_sample=n_sample,
            seed=seed,
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Sankey Diagram — Flux des événements à travers le gating hiérarchique
# ─────────────────────────────────────────────────────────────────────────────


def _pct_of(value: int, parent: int) -> str:
    """Pourcentage relatif à la gate parente."""
    return f"{value / max(parent, 1) * 100:.1f}%"


def generate_sankey_diagram(
    gate_counts: Dict[str, int],
    output_path: Path | str,
    *,
    filter_blasts: bool = False,
    title: str = "Gating Hierarchy — Flux des Événements",
) -> Optional[Any]:
    """
    Génère un diagramme Sankey Plotly du flux de gating hiérarchique.

    Représente le parcours : Total → G1 (débris) → G2 (singlets) → G3 (CD45)
    → optionnellement G4 (CD34) → Population finale.

    Lien vert = conservé, gris/orange = exclu.

    Args:
        gate_counts: Dict avec les clés:
            - n_total: événements totaux
            - n_g1_pass: après gate débris
            - n_g2_pass: après gate singlets
            - n_g3_pass: après gate CD45
            - n_final: population finale (après G4 si filter_blasts)
        output_path: Chemin du fichier HTML de sortie.
        filter_blasts: Si True, ajoute la gate G4 (CD34+).
        title: Titre du diagramme.

    Returns:
        True si succès.
    """
    if not _PLOTLY_AVAILABLE:
        _logger.warning("plotly manquant — Sankey ignoré")
        return None

    n_total = gate_counts["n_total"]
    n_g1_pass = gate_counts["n_g1_pass"]
    n_g1_fail = n_total - n_g1_pass
    n_g2_pass = gate_counts["n_g2_pass"]
    n_g2_fail = n_g1_pass - n_g2_pass
    n_g3_pass = gate_counts["n_g3_pass"]
    n_g3_fail = n_g2_pass - n_g3_pass
    n_final = gate_counts["n_final"]
    n_g4_fail = n_g3_pass - n_final

    labels = [
        f"Événements<br>totaux<br>{n_total:,}",
        f"Gate 1<br>Viables<br>{n_g1_pass:,}<br>({_pct_of(n_g1_pass, n_total)} of total)",
        f"Débris<br>exclus<br>{n_g1_fail:,}<br>({_pct_of(n_g1_fail, n_total)})",
        f"Gate 2<br>Singlets<br>{n_g2_pass:,}<br>({_pct_of(n_g2_pass, n_g1_pass)} of G1)",
        f"Doublets<br>exclus<br>{n_g2_fail:,}<br>({_pct_of(n_g2_fail, n_g1_pass)})",
        f"Gate 3<br>CD45+<br>{n_g3_pass:,}<br>({_pct_of(n_g3_pass, n_g2_pass)} of G2)",
        f"CD45-<br>exclus<br>{n_g3_fail:,}<br>({_pct_of(n_g3_fail, n_g2_pass)})",
    ]

    src = [0, 0, 1, 1, 3, 3]
    tgt = [1, 2, 3, 4, 5, 6]
    vals = [n_g1_pass, n_g1_fail, n_g2_pass, n_g2_fail, n_g3_pass, n_g3_fail]
    link_colors = [
        "rgba(49,163,84,0.4)",
        "rgba(99,99,99,0.3)",
        "rgba(49,163,84,0.4)",
        "rgba(230,85,13,0.3)",
        "rgba(49,163,84,0.4)",
        "rgba(253,141,60,0.3)",
    ]

    node_colors = [
        "#4a90d9",
        "#31a354",
        "#636363",
        "#31a354",
        "#e6550d",
        "#31a354",
        "#fd8d3c",
    ]

    if filter_blasts:
        labels.append(
            f"Gate 4<br>CD34+<br>{n_final:,}<br>({_pct_of(n_final, n_g3_pass)} of G3)"
        )
        labels.append(
            f"Non-blastes<br>exclus<br>{n_g4_fail:,}<br>({_pct_of(n_g4_fail, n_g3_pass)})"
        )
        final_label = f"Population<br>finale<br>{n_final:,}<br>({_pct_of(n_final, n_total)} of total)"
        labels.append(final_label)
        src += [5, 5, 7]
        tgt += [7, 8, 9]
        vals += [n_final, max(n_g4_fail, 1), n_final]
        link_colors += [
            "rgba(49,163,84,0.4)",
            "rgba(253,174,107,0.3)",
            "rgba(49,163,84,0.6)",
        ]
        node_colors += ["#31a354", "#fdae6b", "#2ca02c"]
    else:
        final_label = f"Population<br>finale<br>{n_g3_pass:,}<br>({_pct_of(n_g3_pass, n_total)} of total)"
        labels.append(final_label)
        src.append(5)
        tgt.append(7)
        vals.append(n_g3_pass)
        link_colors.append("rgba(49,163,84,0.6)")
        node_colors.append("#2ca02c")

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="#333", width=1),
                label=labels,
                color=node_colors[: len(labels)],
            ),
            link=dict(source=src, target=tgt, value=vals, color=link_colors),
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=18)),
        font=dict(size=13, color="#222"),
        paper_bgcolor="#fafafa",
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    _logger.info("Sankey global sauvegardé: %s", output_path.name)
    return fig


def generate_per_file_sankey(
    per_file_counts: Dict[str, Dict[str, int]],
    output_dir: Path | str,
    *,
    filter_blasts: bool = False,
    max_files: int = 6,
    timestamp: str = "",
) -> Dict[str, bool]:
    """
    Génère un mini-Sankey par fichier FCS (max 6 fichiers).

    Args:
        per_file_counts: Dict {nom_fichier: {n_total, n_g1_pass, n_g2_pass,
                         n_g3_pass, n_final}}.
        output_dir: Dossier de sortie.
        filter_blasts: Si True, ajoute la gate G4.
        max_files: Nombre max de mini-Sankey à générer.
        timestamp: Suffixe optionnel.

    Returns:
        Dict {nom_fichier: succès}.
    """
    if not _PLOTLY_AVAILABLE:
        _logger.warning("plotly manquant — mini-Sankey ignorés")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, bool] = {}

    files_to_show = list(per_file_counts.keys())[:max_files]

    for f_name in files_to_show:
        counts = per_file_counts[f_name]
        n_total = counts["n_total"]
        if n_total < 10:
            results[f_name] = False
            continue

        n_g1 = counts["n_g1_pass"]
        n_g1_fail = n_total - n_g1
        n_g2 = counts["n_g2_pass"]
        n_g2_fail = n_g1 - n_g2
        n_g3 = counts["n_g3_pass"]
        n_g3_fail = n_g2 - n_g3
        n_final = counts["n_final"]
        n_g4_fail = n_g3 - n_final

        short_name = str(f_name) if len(str(f_name)) <= 30 else str(f_name)[:27] + "..."

        labels = [
            f"Total<br>{n_total:,}",
            f"G1<br>{n_g1:,}<br>({_pct_of(n_g1, n_total)})",
            f"Débris<br>{n_g1_fail:,}",
            f"G2<br>{n_g2:,}<br>({_pct_of(n_g2, n_g1)})",
            f"Doubl.<br>{n_g2_fail:,}",
            f"G3<br>{n_g3:,}<br>({_pct_of(n_g3, n_g2)})",
            f"CD45-<br>{n_g3_fail:,}",
        ]
        src = [0, 0, 1, 1, 3, 3]
        tgt = [1, 2, 3, 4, 5, 6]
        vals = [
            n_g1,
            max(n_g1_fail, 1),
            n_g2,
            max(n_g2_fail, 1),
            n_g3,
            max(n_g3_fail, 1),
        ]
        link_col = [
            "rgba(49,163,84,0.4)",
            "rgba(99,99,99,0.3)",
            "rgba(49,163,84,0.4)",
            "rgba(230,85,13,0.3)",
            "rgba(49,163,84,0.4)",
            "rgba(253,141,60,0.3)",
        ]
        node_col = [
            "#4a90d9",
            "#31a354",
            "#636363",
            "#31a354",
            "#e6550d",
            "#31a354",
            "#fd8d3c",
        ]

        if filter_blasts:
            labels += [
                f"G4<br>{n_final:,}<br>({_pct_of(n_final, n_g3)})",
                f"Excl.<br>{n_g4_fail:,}",
                f"Final<br>{n_final:,}",
            ]
            src += [5, 5, 7]
            tgt += [7, 8, 9]
            vals += [n_final, max(n_g4_fail, 1), n_final]
            link_col += [
                "rgba(49,163,84,0.4)",
                "rgba(253,174,107,0.3)",
                "rgba(49,163,84,0.6)",
            ]
            node_col += ["#31a354", "#fdae6b", "#2ca02c"]
        else:
            labels.append(f"Final<br>{n_g3:,}")
            src.append(5)
            tgt.append(7)
            vals.append(n_g3)
            link_col.append("rgba(49,163,84,0.6)")
            node_col.append("#2ca02c")

        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="#333", width=0.5),
                    label=labels,
                    color=node_col[: len(labels)],
                ),
                link=dict(source=src, target=tgt, value=vals, color=link_col),
            )
        )
        fig.update_layout(
            title=dict(text=f"<b>Gating — {short_name}</b>", font=dict(size=14)),
            font=dict(size=11, color="#222"),
            paper_bgcolor="#fafafa",
            height=300,
            margin=dict(l=10, r=10, t=45, b=10),
        )

        safe = f_name.replace(" ", "_").replace(".fcs", "").replace(".FCS", "")
        suffix = f"_{timestamp}" if timestamp else ""
        out_path = output_dir / f"sankey_{safe}{suffix}.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        results[f_name] = True
        _logger.info("Mini-Sankey sauvegardé: %s", out_path.name)

    return results
