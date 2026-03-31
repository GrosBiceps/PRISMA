"""
mrd_calculator.py — Calcul de la MRD résiduelle post-FlowSOM.

Trois méthodes de détection des nœuds SOM MRD :

  Méthode JF :
    Nœud MRD si % moelle normale < seuil ET % patho > seuil.

  Méthode Flo :
    Nœud MRD si % patho > N × % moelle normale dans ce nœud.

  Méthode ELN (European LeukemiaNet 2018/2021/2025 — DfN) :
    1. Filtre LOQ : le nœud doit contenir >= min_cluster_events cellules.
    2. Critère DfN : % patho > % sain dans le nœud.
    3. Positivité globale : MRD% >= clinical_positivity_pct (0.1%).

Les seuils sont paramétrables via config/mrd_config.yaml.

Usage:
    from flowsom_pipeline_pro.src.analysis.mrd_calculator import (
        load_mrd_config, compute_mrd,
    )
    mrd_cfg = load_mrd_config()
    results = compute_mrd(df_cells, clustering, mrd_cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from flowsom_pipeline_pro.src.utils.logger import get_logger

_logger = get_logger("analysis.mrd_calculator")

# ─────────────────────────────────────────────────────────────────────────────
#  Config dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MRDMethodJF:
    """Seuils pour la méthode JF."""
    max_normal_marrow_pct: float = 0.1
    min_patho_cells_pct: float = 10.0


@dataclass
class MRDMethodFlo:
    """Seuils pour la méthode Flo."""
    normal_marrow_multiplier: float = 2.0


@dataclass
class ELNStandards:
    """Recommandations ELN 2025 pour la MRD par cytométrie en flux."""
    min_cluster_events: int = 50       # LOQ — minimum d'événements par nœud
    clinical_positivity_pct: float = 0.1  # Seuil global de positivité MRD (%)


@dataclass
class MRDConfig:
    """Configuration complète du calcul MRD."""
    enabled: bool = True
    method: str = "all"  # "jf", "flo", "eln", "all"
    method_jf: MRDMethodJF = field(default_factory=MRDMethodJF)
    method_flo: MRDMethodFlo = field(default_factory=MRDMethodFlo)
    eln_standards: ELNStandards = field(default_factory=ELNStandards)
    condition_sain: str = "Sain"
    condition_patho: str = "Pathologique"


def load_mrd_config(config_path: Optional[Path | str] = None) -> MRDConfig:
    """
    Charge la configuration MRD depuis un fichier YAML.

    Si aucun chemin n'est donné, cherche config/mrd_config.yaml à côté du
    package flowsom_pipeline_pro.
    """
    cfg = MRDConfig()

    if config_path is None:
        candidates = [
            Path(__file__).parent.parent.parent / "config" / "mrd_config.yaml",
        ]
        for p in candidates:
            if p.exists():
                config_path = p
                break

    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            try:
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                params = raw.get("mrd_parameters", {})

                cfg.enabled = params.get("enabled", cfg.enabled)
                cfg.method = params.get("method", cfg.method)
                cfg.condition_sain = params.get("condition_sain", cfg.condition_sain)
                cfg.condition_patho = params.get("condition_patho", cfg.condition_patho)

                jf = params.get("method_jf", {})
                cfg.method_jf.max_normal_marrow_pct = jf.get(
                    "max_normal_marrow_pct", cfg.method_jf.max_normal_marrow_pct)
                cfg.method_jf.min_patho_cells_pct = jf.get(
                    "min_patho_cells_pct", cfg.method_jf.min_patho_cells_pct)

                flo = params.get("method_flo", {})
                cfg.method_flo.normal_marrow_multiplier = flo.get(
                    "normal_marrow_multiplier", cfg.method_flo.normal_marrow_multiplier)

                eln = params.get("eln_standards", {})
                cfg.eln_standards.min_cluster_events = eln.get(
                    "min_cluster_events", cfg.eln_standards.min_cluster_events)
                cfg.eln_standards.clinical_positivity_pct = eln.get(
                    "clinical_positivity_pct", cfg.eln_standards.clinical_positivity_pct)

                _logger.info("MRD config chargée depuis %s", config_path.name)
            except Exception as e:
                _logger.warning("Erreur lecture mrd_config.yaml (%s) — valeurs par défaut.", e)

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
#  Résultats
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MRDClusterResult:
    """Résultat MRD pour un nœud SOM individuel."""
    cluster_id: int      # ID du nœud SOM
    n_cells_total: int
    n_cells_sain: int
    n_cells_patho: int
    pct_sain: float        # % de cellules saines DANS ce nœud (par rapport au total du nœud)
    pct_patho: float       # % de cellules pathologiques DANS ce nœud
    pct_sain_global: float  # % des cellules saines de ce nœud / total moelle normale (méthode JF)
    is_mrd_jf: bool      # qualifié MRD par méthode JF
    is_mrd_flo: bool     # qualifié MRD par méthode Flo
    is_mrd_eln: bool     # qualifié MRD par méthode ELN


@dataclass
class MRDResult:
    """Résultat global du calcul MRD."""
    method_used: str
    total_cells: int
    total_cells_patho: int
    total_cells_sain: int

    # Méthode JF
    mrd_cells_jf: int = 0
    mrd_pct_jf: float = 0.0
    n_nodes_mrd_jf: int = 0

    # Méthode Flo
    mrd_cells_flo: int = 0
    mrd_pct_flo: float = 0.0
    n_nodes_mrd_flo: int = 0

    # Méthode ELN
    mrd_cells_eln: int = 0
    mrd_pct_eln: float = 0.0
    n_nodes_mrd_eln: int = 0
    eln_positive: bool = False     # MRD% >= seuil clinique ELN
    eln_low_level: bool = False    # MRD détectable mais < seuil clinique

    # Détail par nœud SOM
    per_node: List[MRDClusterResult] = field(default_factory=list)

    # Config utilisée (traçabilité)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le résultat pour export JSON / rapport HTML."""
        return {
            "method_used": self.method_used,
            "total_cells": self.total_cells,
            "total_cells_patho": self.total_cells_patho,
            "total_cells_sain": self.total_cells_sain,
            "mrd_jf": {
                "mrd_cells": self.mrd_cells_jf,
                "mrd_pct": round(self.mrd_pct_jf, 6),
                "n_nodes_mrd": self.n_nodes_mrd_jf,
            },
            "mrd_flo": {
                "mrd_cells": self.mrd_cells_flo,
                "mrd_pct": round(self.mrd_pct_flo, 6),
                "n_nodes_mrd": self.n_nodes_mrd_flo,
            },
            "mrd_eln": {
                "mrd_cells": self.mrd_cells_eln,
                "mrd_pct": round(self.mrd_pct_eln, 6),
                "n_nodes_mrd": self.n_nodes_mrd_eln,
                "eln_positive": self.eln_positive,
                "eln_low_level": self.eln_low_level,
            },
            "per_node": [
                {
                    "som_node_id": c.cluster_id,
                    "n_cells_total": c.n_cells_total,
                    "n_cells_sain": c.n_cells_sain,
                    "n_cells_patho": c.n_cells_patho,
                    "pct_sain": round(c.pct_sain, 4),
                    "pct_sain_global": round(c.pct_sain_global, 6),
                    "pct_patho": round(c.pct_patho, 4),
                    "is_mrd_jf": c.is_mrd_jf,
                    "is_mrd_flo": c.is_mrd_flo,
                    "is_mrd_eln": c.is_mrd_eln,
                }
                for c in self.per_node
            ],
            "config": self.config_snapshot,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Calcul MRD
# ─────────────────────────────────────────────────────────────────────────────


def compute_mrd(
    df_cells: pd.DataFrame,
    clustering: np.ndarray,
    mrd_cfg: MRDConfig,
    condition_column: str = "condition",
) -> MRDResult:
    """
    Calcule la MRD résiduelle selon les méthodes JF, Flo et/ou ELN.

    Opère au niveau des **nœuds SOM** (clustering), pas des métaclusters.

    Args:
        df_cells: DataFrame cellulaire avec colonne condition.
        clustering: Array (n_cells,) d'assignation nœud SOM (0-based).
        mrd_cfg: Configuration MRD (seuils, méthodes).
        condition_column: Nom de la colonne condition dans df_cells.

    Returns:
        MRDResult avec le détail par nœud SOM et les totaux MRD.
    """
    _logger.info("Calcul MRD (nœuds SOM) — méthode(s): %s", mrd_cfg.method)

    condition = df_cells[condition_column].values if condition_column in df_cells.columns else None
    if condition is None:
        _logger.warning("Colonne '%s' absente — MRD impossible.", condition_column)
        return MRDResult(
            method_used=mrd_cfg.method,
            total_cells=len(df_cells),
            total_cells_patho=0,
            total_cells_sain=0,
        )

    unique_nodes = np.unique(clustering)
    n_total = len(clustering)
    total_patho = int((condition == mrd_cfg.condition_patho).sum())
    total_sain = int((condition == mrd_cfg.condition_sain).sum())

    run_jf = mrd_cfg.method in ("jf", "both", "all")
    run_flo = mrd_cfg.method in ("flo", "both", "all")
    run_eln = mrd_cfg.method in ("eln", "all")

    per_node: List[MRDClusterResult] = []
    mrd_cells_jf = 0
    mrd_cells_flo = 0
    mrd_cells_eln = 0
    n_nodes_jf = 0
    n_nodes_flo = 0
    n_nodes_eln = 0

    for node_id in unique_nodes:
        mask = clustering == node_id
        n_in_node = int(mask.sum())
        if n_in_node == 0:
            continue

        cond_in_node = condition[mask]
        n_sain = int((cond_in_node == mrd_cfg.condition_sain).sum())
        n_patho = int((cond_in_node == mrd_cfg.condition_patho).sum())

        # pct_sain / pct_patho : % AU SEIN du cluster (utilisé par Flo et ELN)
        pct_sain = (n_sain / n_in_node) * 100.0
        pct_patho = (n_patho / n_in_node) * 100.0

        # pct_sain_global : % des cellules saines du cluster par rapport à la
        # TOTALITÉ de la moelle normale (dénominateur = total_sain).
        # C'est ce qu'utilise la méthode JF : un cluster MRD ne doit contenir
        # qu'une infime fraction de la moelle normale totale (< max_normal_marrow_pct).
        _denom_sain = total_sain if total_sain > 0 else 1
        pct_sain_global = (n_sain / _denom_sain) * 100.0

        # pct_patho_in_cluster : % de cellules pathologiques dans le cluster.
        # Utilisé par la méthode JF pour vérifier que le cluster est
        # massivement envahi (> min_patho_cells_pct).
        # (pct_patho ci-dessus est identique, on l'utilise directement.)

        # ── Méthode JF ────────────────────────────────────────────────
        is_mrd_jf = False
        if run_jf:
            # Critère 1 : la fraction de moelle normale dans ce cluster doit être
            #             < max_normal_marrow_pct (% GLOBAL de la moelle normale)
            # Critère 2 : le cluster doit être composé à > min_patho_cells_pct
            #             de cellules pathologiques (% DANS le cluster)
            if (pct_sain_global < mrd_cfg.method_jf.max_normal_marrow_pct
                    and pct_patho > mrd_cfg.method_jf.min_patho_cells_pct):
                is_mrd_jf = True
                mrd_cells_jf += n_patho
                n_nodes_jf += 1

        # ── Méthode Flo ───────────────────────────────────────────────
        is_mrd_flo = False
        if run_flo:
            threshold_flo = pct_sain * mrd_cfg.method_flo.normal_marrow_multiplier
            if pct_patho > threshold_flo:
                is_mrd_flo = True
                mrd_cells_flo += n_patho
                n_nodes_flo += 1

        # ── Méthode ELN (DfN + LOQ) ──────────────────────────────────
        is_mrd_eln = False
        if run_eln:
            # Filtre 1 : LOQ — au moins N événements
            if n_in_node >= mrd_cfg.eln_standards.min_cluster_events:
                # Filtre 2 : DfN — le nœud est enrichi en patho (% patho > % sain)
                if pct_patho > pct_sain:
                    is_mrd_eln = True
                    mrd_cells_eln += n_patho
                    n_nodes_eln += 1

        per_node.append(MRDClusterResult(
            cluster_id=int(node_id),
            n_cells_total=n_in_node,
            n_cells_sain=n_sain,
            n_cells_patho=n_patho,
            pct_sain=pct_sain,
            pct_patho=pct_patho,
            pct_sain_global=pct_sain_global,
            is_mrd_jf=is_mrd_jf,
            is_mrd_flo=is_mrd_flo,
            is_mrd_eln=is_mrd_eln,
        ))

    # MRD % = cellules MRD / total cellules pathologiques de la patiente
    # (dénominateur = fichier patho uniquement, pas le total sain+patho)
    _denom_patho = total_patho if total_patho > 0 else 1
    mrd_pct_jf = (mrd_cells_jf / _denom_patho * 100.0) if total_patho > 0 else 0.0
    mrd_pct_flo = (mrd_cells_flo / _denom_patho * 100.0) if total_patho > 0 else 0.0
    mrd_pct_eln = (mrd_cells_eln / _denom_patho * 100.0) if total_patho > 0 else 0.0

    # Statut clinique ELN
    eln_positive = mrd_pct_eln >= mrd_cfg.eln_standards.clinical_positivity_pct
    eln_low_level = (mrd_cells_eln > 0) and (not eln_positive)

    result = MRDResult(
        method_used=mrd_cfg.method,
        total_cells=n_total,
        total_cells_patho=total_patho,
        total_cells_sain=total_sain,
        mrd_cells_jf=mrd_cells_jf,
        mrd_pct_jf=mrd_pct_jf,
        n_nodes_mrd_jf=n_nodes_jf,
        mrd_cells_flo=mrd_cells_flo,
        mrd_pct_flo=mrd_pct_flo,
        n_nodes_mrd_flo=n_nodes_flo,
        mrd_cells_eln=mrd_cells_eln,
        mrd_pct_eln=mrd_pct_eln,
        n_nodes_mrd_eln=n_nodes_eln,
        eln_positive=eln_positive,
        eln_low_level=eln_low_level,
        per_node=per_node,
        config_snapshot={
            "method": mrd_cfg.method,
            "jf_max_normal_pct": mrd_cfg.method_jf.max_normal_marrow_pct,
            "jf_min_patho_pct": mrd_cfg.method_jf.min_patho_cells_pct,
            "flo_multiplier": mrd_cfg.method_flo.normal_marrow_multiplier,
            "eln_min_events": mrd_cfg.eln_standards.min_cluster_events,
            "eln_positivity_pct": mrd_cfg.eln_standards.clinical_positivity_pct,
        },
    )

    _logger.info(
        "MRD JF : %d cellules patho dans %d nœuds SOM → MRD = %.4f%%",
        mrd_cells_jf, n_nodes_jf, mrd_pct_jf,
    )
    _logger.info(
        "MRD Flo: %d cellules patho dans %d nœuds SOM → MRD = %.4f%%",
        mrd_cells_flo, n_nodes_flo, mrd_pct_flo,
    )
    _logger.info(
        "MRD ELN: %d cellules patho dans %d nœuds SOM → MRD = %.4f%% — %s",
        mrd_cells_eln, n_nodes_eln, mrd_pct_eln,
        "POSITIVE" if eln_positive else ("LOW-LEVEL" if eln_low_level else "NEGATIVE"),
    )

    return result
