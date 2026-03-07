"""
blast_detection.py — Scoring et classification des nœuds Unknown en blastes.

Implémente un scoring heuristique basé sur les poids de marqueurs cytométriques
(ELN 2022) pour discriminer les blastes LAM parmi les nœuds "Unknown" du mapping.

Critères biologiques:
  CD34 bright   → marqueur de progéniteur         (+3.0)
  CD117 bright  → c-Kit, progéniteur myéloïde     (+2.5)
  CD45 dim      → discriminant LAIP               (-2.0 → contribution si bas)
  HLA-DR pos    → blastes myéloïdes               (+1.5)
  CD33 var      → engagement myéloïde             (+1.0)
  CD13 var      → engagement myéloïde             (+0.5)
  CD19/CD3 pos  → signature lymphoïde (anti-blaste) (-1.5)
  SSC dim       → morphologie de blaste           (-1.0)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from flowsom_pipeline_pro.src.utils.logger import get_logger

_logger = get_logger("analysis.blast_detection")

# Seuils de catégorisation (score /10)
BLAST_HIGH_THRESHOLD = 6.0
BLAST_MODERATE_THRESHOLD = 3.0
BLAST_WEAK_THRESHOLD = 0.0


def build_blast_weights(marker_names: List[str]) -> Dict[str, float]:
    """
    Construit le dictionnaire de poids pour le scoring blast.

    Chaque poids est assigné selon le pattern du nom de marqueur.
    Cf. ELN 2022 Section 4.3 pour le rationnel biologique.

    Args:
        marker_names: Liste des noms de marqueurs à scorer.

    Returns:
        Dict {marqueur: poids}.
    """
    weights: Dict[str, float] = {}

    for name in marker_names:
        upper = name.upper()

        if "CD34" in upper:
            weights[name] = +3.0
        elif "CD117" in upper or "CKIT" in upper:
            weights[name] = +2.5
        elif "CD45" in upper:
            weights[name] = -2.0  # négatif: contribution si DIM
        elif "HLAD" in upper or "HLA-DR" in upper:
            weights[name] = +1.5
        elif "CD33" in upper:
            weights[name] = +1.0
        elif "CD13" in upper:
            weights[name] = +0.5
        elif "CD19" in upper or ("CD3" in upper and "CD34" not in upper):
            weights[name] = -1.5  # marqueur lymphoïde → anti-blast
        elif "SSC" in upper:
            weights[name] = -1.0  # SSC faible = morphologie de blaste
        else:
            weights[name] = 0.0

    return weights


def score_nodes_for_blasts(
    X_norm: np.ndarray,
    marker_names: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Calcule le score blast /10 pour chaque nœud.

    Logique de scoring:
      - Marqueur à poids positif (CD34, CD117): contribution quand valeur > 1.0
        (au-dessus du plafond de la population de référence)
      - Marqueur à poids négatif (CD45, SSC): contribution quand valeur < 0.0
        (en-dessous du plancher de la population de référence)

    Args:
        X_norm: Matrice normalisée [n_nodes, n_markers] avec les valeurs dans
                l'espace de référence (0 = min_ref, 1 = max_ref).
        marker_names: Noms des marqueurs (colonnes de X_norm).
        weights: Poids pré-calculés (si None, auto-calculés via build_blast_weights).

    Returns:
        Vecteur (n_nodes,) avec scores dans [0, 10].
    """
    if weights is None:
        weights = build_blast_weights(marker_names)

    W = np.array([weights.get(m, 0.0) for m in marker_names])
    max_theoretical = max(sum(abs(w) for w in weights.values() if w != 0), 1e-6)

    scores_raw = np.zeros(X_norm.shape[0])

    for j, (marker, w) in enumerate(zip(marker_names, W)):
        v = X_norm[:, j]
        if w > 0:
            # Contribution quand valeur AU-DESSUS du plafond référence
            scores_raw += w * np.maximum(0.0, v - 1.0)
        elif w < 0:
            # Contribution quand valeur EN-DESSOUS du plancher référence
            scores_raw += (-w) * np.maximum(0.0, -v)

    scores_10 = np.clip(scores_raw / max_theoretical * 10.0, 0.0, 10.0)
    return scores_10


def categorize_blast_score(score: float) -> str:
    """
    Classifie un score blast /10 en catégorie clinique.

    Args:
        score: Score blast dans [0, 10].

    Returns:
        Catégorie: "BLAST_HIGH", "BLAST_MODERATE", "BLAST_WEAK", "NON_BLAST_UNK".
    """
    if score >= BLAST_HIGH_THRESHOLD:
        return "BLAST_HIGH"  # Signature forte → suspecter LAM
    elif score >= BLAST_MODERATE_THRESHOLD:
        return "BLAST_MODERATE"  # Intermédiaire → confirmation nécessaire
    elif score > BLAST_WEAK_THRESHOLD:
        return "BLAST_WEAK"  # Signal léger → population atypique
    return "NON_BLAST_UNK"  # Aucune signature blastique


def build_blast_score_dataframe(
    node_ids: np.ndarray,
    X_norm: np.ndarray,
    marker_names: List[str],
    cell_counts_per_node: Optional[Dict[int, int]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Construit le DataFrame complet de scoring blast pour les nœuds Unknown.

    Args:
        node_ids: IDs des nœuds à scorer (n_unknown_nodes,).
        X_norm: Valeurs normalisées [n_unknown_nodes, n_markers].
        marker_names: Noms des marqueurs.
        cell_counts_per_node: {node_id: n_cells} pour enrichir le DataFrame.
        weights: Poids pré-calculés (optionnel).

    Returns:
        DataFrame trié par blast_score décroissant.
    """
    if weights is None:
        weights = build_blast_weights(marker_names)

    scores = score_nodes_for_blasts(X_norm, marker_names, weights)
    categories = [categorize_blast_score(float(s)) for s in scores]

    records: Dict = {
        "node_id": node_ids,
        "blast_score": np.round(scores, 2),
        "blast_category": categories,
    }

    if cell_counts_per_node:
        records["n_cells"] = [cell_counts_per_node.get(int(nid), 0) for nid in node_ids]

    df = pd.DataFrame(records)

    # Ajouter les valeurs normalisées par marqueur
    for j, m in enumerate(marker_names):
        df[f"{m}_M8"] = np.round(X_norm[:, j], 3)

    df = df.sort_values("blast_score", ascending=False).reset_index(drop=True)

    # Log du résumé
    for cat in ["BLAST_HIGH", "BLAST_MODERATE", "BLAST_WEAK", "NON_BLAST_UNK"]:
        n = int((df["blast_category"] == cat).sum())
        if n > 0:
            _logger.info("  %s: %d nœud(s)", cat, n)

    return df


def compute_reference_normalization(
    X_unknown: np.ndarray,
    X_reference: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalise X_unknown dans l'espace de référence.

    X_norm[i,j] = 0 si valeur = min_ref[j], 1 si valeur = max_ref[j].
    Les blastes ont typiquement CD34 >> 1 et CD45 << 0.

    Args:
        X_unknown: Matrice des nœuds Unknown [n_unknown, n_markers].
        X_reference: Matrice de toutes les populations référence [n_ref, n_markers].

    Returns:
        Tuple (X_norm, ref_min, ref_range).
    """
    ref_min = X_reference.min(axis=0)
    ref_max = X_reference.max(axis=0)
    ref_range = ref_max - ref_min
    ref_range[ref_range == 0] = 1.0  # éviter div/0

    X_norm = (X_unknown - ref_min) / ref_range
    return X_norm, ref_min, ref_range
