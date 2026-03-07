"""
transformers.py — Transformations cytométriques et normalisation.

Toutes les méthodes sont statiques et sans état : elles transforment
un ndarray et le retournent. La sélection de la méthode à appliquer
est déléguée à PreprocessingService.

Méthodes disponibles:
    - arcsinh_transform  : Standard pour cytométrie en flux (cofacteur 5)
    - logicle_transform  : Biexponentielle précise (FlowKit si disponible)
    - log10_transform    : Log base 10 sur valeurs positives
    - zscore_normalize   : Centrage-réduction (moyenne=0, std=1)
    - min_max_normalize  : Mise à l'échelle [0, 1]
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np

# FlowKit pour la transformation Logicle précise (comportement dégradé si absent)
try:
    import flowkit as fk

    _FLOWKIT_AVAILABLE = True
except ImportError:
    _FLOWKIT_AVAILABLE = False


class DataTransformer:
    """
    Transformations de données de cytométrie.

    Toutes les méthodes sont statiques : pas besoin d'instancier la classe.
    Aucun état global n'est modifié.
    """

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    @staticmethod
    def arcsinh_transform(
        data: np.ndarray,
        cofactor: float = 5.0,
    ) -> np.ndarray:
        """
        Transformation Arcsinh (inverse sinus hyperbolique).

        C'est la transformation de référence pour la cytométrie en flux :
        elle gère les valeurs négatives (artefacts de compensation) et
        comprime les grandes valeurs sans saturation.

        Args:
            data: Matrice (n_cells, n_markers) ou vecteur (n_cells,).
            cofactor: Facteur de division avant l'arcsinh.
                - 5   = standard cytométrie en flux (immunophénotypage)
                - 150 = cytométrie de masse (CyTOF)

        Returns:
            Données transformées, même shape que l'entrée.
        """
        return np.arcsinh(data / cofactor)

    @staticmethod
    def arcsinh_inverse(
        data: np.ndarray,
        cofactor: float = 5.0,
    ) -> np.ndarray:
        """
        Inverse de la transformation Arcsinh (retour à l'espace d'intensité).

        Args:
            data: Données transformées.
            cofactor: Même cofacteur que pour arcsinh_transform.

        Returns:
            Données en espace d'intensité original.
        """
        return np.sinh(data) * cofactor

    @staticmethod
    def logicle_transform(
        data: np.ndarray,
        T: float = 262144.0,
        M: float = 4.5,
        W: float = 0.5,
        A: float = 0.0,
    ) -> np.ndarray:
        """
        Transformation Logicle (biexponentielle).

        Utilise FlowKit si disponible (plus précis scientifiquement).
        Sinon, applique une approximation par arcsinh modifié.

        L'avantage de la Logicle sur l'arcsinh est la linéarisation
        naturelle de la zone négative/proche de zéro selon les paramètres
        d'acquisition (T, M, W définis dans le fichier FCS).

        Args:
            data: Matrice ou vecteur de données.
            T: Maximum de l'échelle (262144 = 2^18 standard).
            M: Largeur en décades de la partie log.
            W: Partie linéaire (linéarisation proche de zéro).
            A: Décades additionnelles pour valeurs négatives.

        Returns:
            Données transformées.
        """
        if _FLOWKIT_AVAILABLE:
            try:
                xform = fk.transforms.LogicleTransform(
                    param_t=T, param_w=W, param_m=M, param_a=A
                )
                return xform.apply(data)
            except Exception as e:
                warnings.warn(
                    f"FlowKit LogicleTransform échoué ({e}), "
                    "utilisation de l'approximation arcsinh modifié."
                )

        # Approximation sans FlowKit: arcsinh modifié
        # Source: Parks et al. (2006) Cytometry A
        return np.arcsinh(data / (T / (10**M))) * (M / np.log(10))

    @staticmethod
    def log10_transform(
        data: np.ndarray,
        min_val: float = 1.0,
    ) -> np.ndarray:
        """
        Transformation log base 10.

        Les valeurs <= 0 sont clippées à min_val avant transformation
        pour éviter les NaN/Inf.

        Args:
            data: Matrice ou vecteur de données.
            min_val: Valeur minimum avant log (évite log(0)).

        Returns:
            Données transformées.
        """
        return np.log10(np.maximum(data, min_val))

    @staticmethod
    def apply(
        data: np.ndarray,
        method: str,
        cofactor: float = 5.0,
        var_names: Optional[List[str]] = None,
        apply_to_scatter: bool = False,
    ) -> np.ndarray:
        """
        Applique la transformation spécifiée, avec gestion optionnelle
        des canaux scatter (FSC/SSC/Time).

        Args:
            data: Matrice (n_cells, n_markers).
            method: 'arcsinh' | 'logicle' | 'log10' | 'none'.
            cofactor: Cofacteur pour arcsinh.
            var_names: Noms des colonnes (même ordre que data).
                Si fourni et apply_to_scatter=False, les colonnes
                FSC/SSC/Time sont exclues de la transformation.
            apply_to_scatter: Si True, transforme TOUS les canaux
                y compris FSC/SSC/Time.

        Returns:
            Données transformées (même shape que data).

        Raises:
            ValueError: Si la méthode est inconnue.
        """
        from flowsom_pipeline_pro.config.constants import SCATTER_PATTERNS
        import re

        method_lc = method.lower()
        if method_lc == "none":
            return data.copy() if isinstance(data, np.ndarray) else np.array(data)

        # Déterminer les indices à transformer
        if var_names is not None and not apply_to_scatter:
            scatter_patterns = SCATTER_PATTERNS
            fluorescence_idx = [
                i
                for i, name in enumerate(var_names)
                if not any(re.search(p, name, re.IGNORECASE) for p in scatter_patterns)
            ]
            scatter_idx = [i for i in range(data.shape[1]) if i not in fluorescence_idx]
        else:
            fluorescence_idx = list(range(data.shape[1]))
            scatter_idx = []

        result = (
            data.copy() if isinstance(data, np.ndarray) else np.array(data, dtype=float)
        )

        if not fluorescence_idx:
            return result

        sub = result[:, fluorescence_idx]

        if method_lc == "arcsinh":
            transformed = DataTransformer.arcsinh_transform(sub, cofactor=cofactor)
        elif method_lc == "logicle":
            transformed = DataTransformer.logicle_transform(sub)
        elif method_lc == "log10":
            transformed = DataTransformer.log10_transform(sub)
        else:
            raise ValueError(
                f"Méthode de transformation inconnue: '{method}'. "
                "Valeurs acceptées: 'arcsinh', 'logicle', 'log10', 'none'."
            )

        result[:, fluorescence_idx] = transformed
        return result
