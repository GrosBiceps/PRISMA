"""
sample.py — Modèle d'un échantillon FCS chargé.

FlowSample encapsule un fichier FCS avec ses métadonnées et fournit
un accès structuré à la matrice d'expression et aux annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FlowSample:
    """
    Représente un échantillon FCS après chargement.

    Attributes:
        name: Nom court du fichier (sans chemin).
        path: Chemin complet vers le fichier FCS d'origine.
        condition: Label de condition ('Sain', 'Pathologique', …).
        data: DataFrame pandas (cellules × marqueurs).
        metadata: Métadonnées FCS brutes ($PnN, $DATE, …).
        n_cells_raw: Nombre de cellules avant tout gating.
    """

    name: str
    path: str
    condition: str
    data: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    n_cells_raw: int = 0
    # Données brutes pré-transformation (valeurs linéaires pour export FCS Kaluza)
    raw_data: Optional[pd.DataFrame] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Constructeur alternatif
    # ------------------------------------------------------------------

    @classmethod
    def from_anndata(
        cls,
        adata: Any,
        path: str,
        condition: str,
    ) -> "FlowSample":
        """
        Construit un FlowSample depuis un objet AnnData (flowsom.io.read_FCS).

        Args:
            adata: AnnData renvoyé par fs.io.read_FCS().
            path: Chemin du fichier FCS source.
            condition: Label de condition.

        Returns:
            FlowSample initialisé.
        """
        import pandas as pd

        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        df = pd.DataFrame(X, columns=list(adata.var_names))
        n_cells = df.shape[0]

        meta = dict(adata.uns) if hasattr(adata, "uns") else {}

        return cls(
            name=Path(path).name,
            path=path,
            condition=condition,
            data=df,
            metadata=meta,
            n_cells_raw=n_cells,
        )

    # ------------------------------------------------------------------
    # Accès rapides
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Nombre de cellules dans le DataFrame courant (après filtrage éventuel)."""
        return len(self.data)

    @property
    def markers(self) -> List[str]:
        """Liste des noms de marqueurs (colonnes du DataFrame)."""
        return list(self.data.columns)

    @property
    def var_names(self) -> List[str]:
        """Alias de markers — noms des marqueurs (colonnes du DataFrame)."""
        return list(self.data.columns)

    @property
    def matrix(self) -> np.ndarray:
        """Matrice numpy (n_cells × n_markers)."""
        return self.data.values

    def get_marker_index(self, name: str) -> Optional[int]:
        """Retourne l'index d'un marqueur par son nom (insensible à la casse)."""
        upper = [m.upper() for m in self.markers]
        try:
            return upper.index(name.upper())
        except ValueError:
            return None

    def filter(self, mask: np.ndarray) -> "FlowSample":
        """
        Retourne un nouveau FlowSample filtré par le masque booléen.

        Args:
            mask: Masque booléen de longueur n_cells.

        Returns:
            FlowSample avec uniquement les cellules sélectionnées.
        """
        filtered_df = self.data[mask].reset_index(drop=True)
        return FlowSample(
            name=self.name,
            path=self.path,
            condition=self.condition,
            data=filtered_df,
            metadata=self.metadata,
            n_cells_raw=self.n_cells_raw,
        )

    def downsample(self, n: int, seed: int = 42) -> "FlowSample":
        """
        Sous-échantillonne aléatoirement à n cellules.

        Args:
            n: Nombre cible de cellules.
            seed: Graine aléatoire.

        Returns:
            FlowSample sous-échantillonné.
        """
        if n >= self.n_cells:
            return self
        sampled_df = self.data.sample(n=n, random_state=seed).reset_index(drop=True)
        return FlowSample(
            name=self.name,
            path=self.path,
            condition=self.condition,
            data=sampled_df,
            metadata=self.metadata,
            n_cells_raw=self.n_cells_raw,
        )

    def summary(self) -> str:
        """Résumé court de l'échantillon."""
        return (
            f"FlowSample({self.name!r}, condition={self.condition!r}, "
            f"cells={self.n_cells:,}/{self.n_cells_raw:,}, "
            f"markers={len(self.markers)})"
        )

    def __repr__(self) -> str:
        return self.summary()


# =======================================================================
# FlowCytometrySample — Copie conforme du monolithe (alias simplifié)
# =======================================================================
try:
    import flowsom as fs
    _FS_AVAILABLE = True
except ImportError:
    _FS_AVAILABLE = False

class FlowCytometrySample:
    """
    Classe utilitaire pour le chargement d'un fichier FCS en DataFrame pandas.
    Utilisée par run_flowsom_pipeline() comme interface simplifiée.
    """

    @classmethod
    def from_fcs(cls, fcs_file: str, verbose: bool = True) -> pd.DataFrame:
        """Charge un fichier FCS et retourne un DataFrame pandas."""
        adata = fs.io.read_FCS(fcs_file)
        df = pd.DataFrame(adata.X, columns=list(adata.var_names))
        if verbose:
            print(f"    [OK] {Path(fcs_file).name}: {len(df):,} cellules")
        return df
