"""
gate_result.py — Structure de données pour les résultats de gating.

GateResult est le type de retour standard de chaque opération de gating.
Il est conçu pour être sérialisable en JSON (sans le masque numpy) afin
de permettre un audit complet des décisions de gating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class GateResult:
    """
    Résultat structuré d'une opération de gating.

    Attributes:
        mask: Masque booléen numpy (True = cellule conservée).
        n_kept: Nombre de cellules conservées.
        n_total: Nombre total de cellules avant ce gate.
        method: Méthode utilisée ('auto_gmm_debris', 'ransac_singlets', …).
        gate_name: Identifiant du gate (ex: 'G1_debris', 'G3_cd45').
        details: Paramètres et métriques de la méthode.
        warnings: Messages d'avertissement levés pendant le gate.
    """

    mask: np.ndarray
    n_kept: int
    n_total: int
    method: str
    gate_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Propriétés calculées
    # ------------------------------------------------------------------

    @property
    def pct_kept(self) -> float:
        """Pourcentage de cellules conservées (0–100)."""
        return (self.n_kept / max(self.n_total, 1)) * 100

    @property
    def n_excluded(self) -> int:
        """Nombre de cellules exclues par ce gate."""
        return self.n_total - self.n_kept

    @property
    def pct_excluded(self) -> float:
        """Pourcentage de cellules exclues (0–100)."""
        return 100.0 - self.pct_kept

    @property
    def is_good_quality(self) -> bool:
        """
        Qualité basique du gate : True si > 20% des cellules ont été conservées.
        Un gate qui élimine > 80% des cellules mérite une vérification manuelle.
        """
        return self.pct_kept > 20.0

    # ------------------------------------------------------------------
    # Sérialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Sérialisation JSON-safe (sans le masque numpy).

        Returns:
            Dictionnaire sérialisable.
        """
        return {
            "gate_name": self.gate_name,
            "method": self.method,
            "n_kept": self.n_kept,
            "n_total": self.n_total,
            "n_excluded": self.n_excluded,
            "pct_kept": round(self.pct_kept, 2),
            "pct_excluded": round(self.pct_excluded, 2),
            "is_good_quality": self.is_good_quality,
            "details": self.details,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        """Sérialise en JSON formaté."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __repr__(self) -> str:
        return (
            f"GateResult({self.gate_name!r}, method={self.method!r}, "
            f"kept={self.n_kept}/{self.n_total} ({self.pct_kept:.1f}%))"
        )
