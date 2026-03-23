"""
patho_fcs_exporter.py — Export FCS restreint à la moelle pathologique avec annotation MRD.

Génère un fichier FCS contenant uniquement les cellules de la moelle pathologique,
enrichi de toutes les métriques FlowSOM (cluster, métacluster, coordonnées xGrid/yGrid,
xNodes/yNodes, size) et d'une colonne Is_MRD binaire (1 = cellule MRD, 0 = non-MRD).

La méthode utilisée pour Is_MRD est configurable : "jf" (méthode JF) ou "flo" (méthode Flo).
Par défaut : "flo".

Usage:
    from flowsom_pipeline_pro.src.io.patho_fcs_exporter import export_patho_mrd_fcs
    export_patho_mrd_fcs(df_fcs, mrd_result, output_path, mrd_method="flo")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flowsom_pipeline_pro.src.analysis.mrd_calculator import MRDResult
from flowsom_pipeline_pro.src.io.fcs_writer import export_to_fcs_kaluza
from flowsom_pipeline_pro.src.utils.logger import get_logger

_logger = get_logger("io.patho_fcs_exporter")


def export_patho_mrd_fcs(
    df_fcs: pd.DataFrame,
    mrd_result: "MRDResult",
    output_path: Path | str,
    mrd_method: str = "flo",
    condition_patho: str = "Pathologique",
) -> bool:
    """
    Exporte un FCS restreint aux cellules de la moelle pathologique avec colonne Is_MRD.

    Le DataFrame ``df_fcs`` doit contenir les colonnes suivantes (produites par
    FlowSOMPipeline._build_fcs_dataframe) :
      - Tous les marqueurs bruts (pré-transformation)
      - FlowSOM_metacluster  (1-based, float32)
      - FlowSOM_cluster      (1-based, float32)
      - xGrid, yGrid         (coordonnées grille SOM avec jitter)
      - xNodes, yNodes       (coordonnées MST avec jitter)
      - size                 (nombre de cellules par nœud SOM)
      - Condition            (string "Sain" / "Pathologique")
      - Condition_Num        (float32  1.0 = Sain, 2.0 = Pathologique)
      - File_Origin          (string)

    La colonne ``Is_MRD`` est ajoutée :
      - 1.0  →  cellule appartenant à un nœud SOM classé MRD selon la méthode choisie
      - 0.0  →  cellule non-MRD

    Seules les colonnes numériques sont écrites dans le FCS (``fcswrite`` requis).
    ``Condition`` et ``File_Origin`` (strings) sont implicitement exclues.

    Args:
        df_fcs:          DataFrame complet produit par _build_fcs_dataframe.
        mrd_result:      Objet MRDResult issu de compute_mrd().
                         Peut être None — dans ce cas l'export est annulé.
        output_path:     Chemin de sortie du fichier .fcs.
        mrd_method:      Méthode Is_MRD : "flo" (défaut) ou "jf".
        condition_patho: Label de la condition pathologique dans la colonne Condition.

    Returns:
        True si succès, False sinon.
    """
    if mrd_result is None:
        _logger.warning(
            "export_patho_mrd_fcs: mrd_result est None — export annulé."
        )
        return False

    if df_fcs is None or df_fcs.empty:
        _logger.warning(
            "export_patho_mrd_fcs: df_fcs vide ou None — export annulé."
        )
        return False

    mrd_method = mrd_method.lower().strip()
    if mrd_method not in ("jf", "flo"):
        _logger.warning(
            "export_patho_mrd_fcs: méthode '%s' inconnue — utilisation de 'flo'.",
            mrd_method,
        )
        mrd_method = "flo"

    # ── 1. Filtrer les cellules pathologiques ─────────────────────────────────
    if "Condition" not in df_fcs.columns:
        _logger.error(
            "export_patho_mrd_fcs: colonne 'Condition' absente du df_fcs — export annulé."
        )
        return False

    mask_patho = df_fcs["Condition"] == condition_patho
    n_patho = int(mask_patho.sum())

    if n_patho == 0:
        _logger.warning(
            "export_patho_mrd_fcs: aucune cellule '%s' dans df_fcs — export annulé.",
            condition_patho,
        )
        return False

    _logger.info(
        "export_patho_mrd_fcs: %d cellules pathologiques sélectionnées / %d total",
        n_patho,
        len(df_fcs),
    )

    df_patho = df_fcs[mask_patho].copy()

    # ── 2. Construire le mapping nœud SOM → Is_MRD ───────────────────────────
    # FlowSOM_cluster est 1-based → convertir en 0-based pour rejoindre per_node
    per_node = getattr(mrd_result, "per_node", [])
    if not per_node:
        _logger.warning(
            "export_patho_mrd_fcs: mrd_result.per_node vide — Is_MRD = 0 pour toutes les cellules."
        )
        df_patho["Is_MRD"] = np.float32(0.0)
    else:
        # Choisir le flag selon la méthode
        flag_attr = "is_mrd_flo" if mrd_method == "flo" else "is_mrd_jf"
        node_to_mrd: dict = {
            node.cluster_id: float(getattr(node, flag_attr, False))
            for node in per_node
        }

        # FlowSOM_cluster est 1-based → node_id 0-based = cluster - 1
        cluster_col = df_patho["FlowSOM_cluster"].values.astype(np.int32) - 1
        n_nodes = int(cluster_col.max()) + 1 if len(cluster_col) > 0 else 0
        lookup = np.array(
            [node_to_mrd.get(nid, 0.0) for nid in range(n_nodes)],
            dtype=np.float32,
        )
        is_mrd_arr = lookup[cluster_col]
        df_patho["Is_MRD"] = is_mrd_arr

        n_mrd_cells = int((is_mrd_arr > 0).sum())
        pct_mrd = 100.0 * n_mrd_cells / n_patho if n_patho > 0 else 0.0
        _logger.info(
            "  Is_MRD (%s) : %d cellules MRD / %d patho (%.2f%%)",
            mrd_method.upper(),
            n_mrd_cells,
            n_patho,
            pct_mrd,
        )

    # ── 3. Export FCS ─────────────────────────────────────────────────────────
    output_path = Path(output_path)
    ok = export_to_fcs_kaluza(df_patho, output_path)
    if ok:
        _logger.info(
            "FCS pathologique exporté : %s (%d events)",
            output_path.name,
            n_patho,
        )
    return ok
