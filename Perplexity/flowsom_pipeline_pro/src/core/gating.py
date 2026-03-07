"""
gating.py — Pre-gating séquentiel basé sur percentiles fixes (mode manuel).

PreGating implémente 4 gates hiérarchiques :
  G1: Débris      (SSC-A vs FSC-A, exclusion polygonale)
  G2: Doublets    (FSC-H vs FSC-A, ratio ou polygone)
  G3: Leucocytes  (CD45+, percentile bas)
  G4: Blastes     (CD34+ bright, percentile haut + SSC low optionnel)

Toutes les méthodes sont statiques et retournent un masque booléen numpy.
Pour le gating adaptatif par GMM, voir auto_gating.py.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class PreGating:
    """
    Pre-gating standard par percentiles fixes (mode 'manual' dans la config).

    Toutes les méthodes acceptent la matrice brute X et la liste var_names.
    Aucun état n'est stocké dans la classe.
    """

    # ------------------------------------------------------------------
    # Utilitaire de recherche de marqueur
    # ------------------------------------------------------------------

    @staticmethod
    def find_marker_index(
        var_names: List[str],
        patterns: List[str],
    ) -> Optional[int]:
        """
        Trouve l'index d'un marqueur parmi une liste de patterns.

        Les patterns sont testés dans l'ordre ; le premier qui correspond
        à un nom de marqueur (recherche par `in`, insensible à la casse)
        est retourné.

        Args:
            var_names: Liste des noms de marqueurs.
            patterns: Patterns à rechercher, par ordre de priorité
                      (ex: ['FSC-A', 'FSC-H', 'FSC']).

        Returns:
            Index du premier marqueur correspondant, ou None si absent.
        """
        upper = [v.upper() for v in var_names]
        for pattern in patterns:
            for i, name in enumerate(upper):
                if pattern.upper() in name:
                    return i
        return None

    # ------------------------------------------------------------------
    # Gate 1 — Débris
    # ------------------------------------------------------------------

    @staticmethod
    def gate_viable_cells(
        X: np.ndarray,
        var_names: List[str],
        min_percentile: float = 1.0,
        max_percentile: float = 99.0,
    ) -> np.ndarray:
        """
        Exclut les débris et événements saturés par fenêtre percentile sur FSC et SSC.

        Les débris sont typiquement : FSC-A bas, SSC-A bas.
        Les cellules mortes / débris cellulaires forment une population
        distincte dans le coin inférieur gauche du plot FSC-A vs SSC-A.

        Args:
            X: Matrice de données brutes (n_cells, n_markers).
            var_names: Noms des marqueurs correspondant aux colonnes de X.
            min_percentile: Percentile inférieur de coupure (défaut 1%).
            max_percentile: Percentile supérieur de coupure (défaut 99%).

        Returns:
            Masque booléen (True = cellule viable retenue).
        """
        n_cells = X.shape[0]
        mask = np.ones(n_cells, dtype=bool)

        for patterns, name in (
            (["FSC-A", "FSC-H", "FSC"], "FSC"),
            (["SSC-A", "SSC-H", "SSC"], "SSC"),
        ):
            idx = PreGating.find_marker_index(var_names, patterns)
            if idx is not None:
                vals = X[:, idx].astype(np.float64)
                vals = np.where(np.isfinite(vals), vals, np.nan)
                lo = np.nanpercentile(vals, min_percentile)
                hi = np.nanpercentile(vals, max_percentile)
                mask &= np.isfinite(vals) & (vals >= lo) & (vals <= hi)

        return mask

    # Alias sémantique exposé dans la doc utilisateur
    gate_debris_polygon = gate_viable_cells

    # ------------------------------------------------------------------
    # Gate 2 — Doublets
    # ------------------------------------------------------------------

    @staticmethod
    def gate_singlets(
        X: np.ndarray,
        var_names: List[str],
        ratio_min: float = 0.6,
        ratio_max: float = 1.5,
    ) -> np.ndarray:
        """
        Exclut les doublets par ratio FSC-A / FSC-H.

        Les singlets forment une bande diagonale serrée (ratio ≈ 1).
        Les doublets ont un FSC-A trop grand par rapport au FSC-H
        (deux cellules fusionnées).

        Args:
            X: Matrice de données.
            var_names: Noms des marqueurs.
            ratio_min: Ratio minimum FSC-A/FSC-H pour un singlet.
            ratio_max: Ratio maximum FSC-A/FSC-H pour un singlet.

        Returns:
            Masque booléen (True = singlet).
        """
        n_cells = X.shape[0]
        fsc_a_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
        fsc_h_idx = PreGating.find_marker_index(var_names, ["FSC-H"])

        if fsc_a_idx is None or fsc_h_idx is None:
            return np.ones(n_cells, dtype=bool)

        fsc_a = X[:, fsc_a_idx].astype(np.float64)
        fsc_h = X[:, fsc_h_idx].astype(np.float64)

        valid_h = fsc_h > 100  # éviter division par zéro
        ratio = np.full(n_cells, np.nan)
        ratio[valid_h] = fsc_a[valid_h] / fsc_h[valid_h]

        return np.isfinite(ratio) & (ratio >= ratio_min) & (ratio <= ratio_max)

    # ------------------------------------------------------------------
    # Gate 3 — Leucocytes CD45+
    # ------------------------------------------------------------------

    @staticmethod
    def gate_cd45_positive(
        X: np.ndarray,
        var_names: List[str],
        threshold_percentile: float = 10.0,
    ) -> np.ndarray:
        """
        Sélectionne les leucocytes CD45+ par seuil de percentile bas.

        Les leucocytes expriment CD45 (pan-leucocyte) à des niveaux variables.
        Les débris, cellules épithéliales et érythrocytes sont CD45-.

        Args:
            X: Matrice de données.
            var_names: Noms des marqueurs.
            threshold_percentile: Percentile de coupure bas (défaut 10).
                Les cellules avec CD45 < ce percentile sont exclues.

        Returns:
            Masque booléen (True = CD45+).
        """
        n_cells = X.shape[0]
        cd45_idx = PreGating.find_marker_index(
            var_names, ["CD45", "CD45-PECY5", "CD45-PC5"]
        )
        if cd45_idx is None:
            return np.ones(n_cells, dtype=bool)

        cd45 = X[:, cd45_idx].astype(np.float64)
        cd45 = np.where(np.isfinite(cd45), cd45, np.nan)
        threshold = np.nanpercentile(cd45, threshold_percentile)

        return np.where(np.isnan(cd45), False, cd45 > threshold)

    # ------------------------------------------------------------------
    # Gate 4 — Blastes CD34+
    # ------------------------------------------------------------------

    @staticmethod
    def gate_cd34_blasts(
        X: np.ndarray,
        var_names: List[str],
        threshold_percentile: float = 85.0,
        use_ssc_filter: bool = True,
        ssc_max_percentile: float = 70.0,
    ) -> np.ndarray:
        """
        Sélectionne les blastes CD34+ bright avec option filtre SSC low.

        Les blastes leucémiques sont caractérisés par :
        - CD34 bright (forte expression, top 15% typiquement)
        - SSC-A low (faible granularité interne)

        Args:
            X: Matrice de données.
            var_names: Noms des marqueurs.
            threshold_percentile: Percentile de coupure pour CD34+ (ex: 85 → top 15%).
            use_ssc_filter: Si True, combine avec SSC-A low pour enrichir en blastes.
            ssc_max_percentile: Percentile max SSC-A pour blastes (faible granularité).

        Returns:
            Masque booléen (True = blaste CD34+).
        """
        n_cells = X.shape[0]
        cd34_idx = PreGating.find_marker_index(
            var_names, ["CD34", "CD34-PE", "CD34-APC", "CD34-PECY7"]
        )
        if cd34_idx is None:
            return np.ones(n_cells, dtype=bool)

        cd34 = X[:, cd34_idx].astype(np.float64)
        cd34 = np.where(np.isfinite(cd34), cd34, np.nan)
        thr = np.nanpercentile(cd34, threshold_percentile)
        mask = np.where(np.isnan(cd34), False, cd34 >= thr)

        if use_ssc_filter:
            ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A", "SSC-H", "SSC"])
            if ssc_idx is not None:
                ssc = X[:, ssc_idx].astype(np.float64)
                ssc = np.where(np.isfinite(ssc), ssc, np.nan)
                thr_ssc = np.nanpercentile(ssc, ssc_max_percentile)
                mask_ssc = np.where(np.isnan(ssc), False, ssc <= thr_ssc)
                return mask & mask_ssc

        return mask
