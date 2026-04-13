"""
mrd_calculator.py — Calcul de la MRD résiduelle post-FlowSOM.

═══════════════════════════════════════════════════════════════════════════════
  TROIS MÉTHODES DE DÉTECTION DES NŒUDS SOM MRD
═══════════════════════════════════════════════════════════════════════════════

  Méthode JF :
    Nœud MRD si % moelle normale GLOBAL < seuil ET % patho DANS LE NŒUD > seuil.
    Critère conservateur qui exige que le cluster soit quasi-exclusivement
    pathologique ET ne contienne qu'une infime fraction de la moelle normale.

  Méthode Flo :
    Nœud MRD si % patho > N × % sain dans le même nœud.
    Mesure le rapport de déséquilibre intra-cluster. Tolère des clusters mixtes
    à condition que le ratio patho/sain soit supérieur au multiplicateur N.

  Méthode ELN (European LeukemiaNet 2022 — DfN, « Different from Normal ») :
    Recommandations Schuurhuis et al., Blood 2018 ; Heuser et al., Leukemia 2022.
    1. Filtre LOQ (Limit Of Quantification) : >= min_cluster_events cellules
       dans le nœud — garantit la robustesse statistique du ratio.
    2. Critère DfN : % patho > % sain dans le nœud — l'enrichissement relatif
       est la signature topologique d'une population anormale.
    3. Positivité globale : MRD% >= clinical_positivity_pct (0.1% ELN standard)
       — seuil clinique de décision thérapeutique.

═══════════════════════════════════════════════════════════════════════════════
  APPROCHE HYBRIDE (optionnelle) — Entonnoir à Deux Portes
═══════════════════════════════════════════════════════════════════════════════

  Problème résolu : l'effet batch provoque l'isolation de cellules saines
  atypiques dans des nœuds SOM dédiés. Ces nœuds franchissent la porte
  mathématique (ratio patho/sain élevé) mais ne sont pas des blastes.

  Solution — double porte (ET logique) :
    1. Porte Topologique/Mathématique : critère de la méthode (JF / Flo / ELN).
    2. Porte Phénotypique/Biologique  : blast_category IN allowed_categories.
       Calculée via blast_detection.py (scoring ELN 2022 / score d'Ogata).
       Un nœud doit présenter une signature blastique (CD34/CD117 bright,
       CD45-dim, SSC-bas) pour être validé comme nœud MRD.

  Activé via blast_phenotype_filter.enabled dans config/mrd_config.yaml.

  Données requises pour la porte biologique :
    - X_norm         : médianes SOM normalisées dans l'espace de référence
                       (valeurs produites par compute_reference_normalization)
    - marker_names   : noms des marqueurs correspondants
    OU (si X_norm non disponible) :
    - node_medians   : médianes SOM brutes (normalisation intra-dataset appliquée
                       automatiquement — moins précise sans référence externe)

Les seuils sont paramétrables via config/mrd_config.yaml.

Usage:
    from flowsom_pipeline_pro.src.analysis.mrd_calculator import (
        load_mrd_config, compute_mrd,
    )
    mrd_cfg = load_mrd_config()
    results = compute_mrd(df_cells, clustering, mrd_cfg)
    # Avec porte biologique :
    results = compute_mrd(
        df_cells, clustering, mrd_cfg,
        X_norm=node_medians_normalized,   # médianes dans l'espace de référence
        marker_names=selected_markers,
    )
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
    """
    Seuils pour la méthode JF (Jabbour-Faderl, adaptée cytométrie clinique).

    Logique : un nœud SOM est MRD si et seulement si :
      1. Il ne contient qu'une fraction infime de la moelle normale TOTALE
         (pct_sain_global < max_normal_marrow_pct). Cela garantit que le cluster
         n'est pas un compartiment normal sous-représenté.
      2. Il est majoritairement composé de cellules pathologiques
         (pct_patho > min_patho_cells_pct au sein du cluster).

    Les deux critères ensemble évitent les nœuds "mixtes" qui contiennent des
    cellules normales et pathologiques en proportions comparables.
    """
    max_normal_marrow_pct: float = 0.1   # % max de moelle normale dans ce cluster / total sain
    min_patho_cells_pct: float = 10.0    # % min de cellules patho DANS le cluster


@dataclass
class MRDMethodFlo:
    """
    Seuils pour la méthode Flo (ratio intra-cluster).

    Logique : un nœud SOM est MRD si le % de cellules patho est supérieur à
    normal_marrow_multiplier × % de cellules saines dans le même nœud.

    Avec normal_marrow_multiplier=2.0 : pct_patho > 2 × pct_sain.
    Plus permissive que JF sur les clusters mixtes, mais exige un déséquilibre
    significatif en faveur des cellules pathologiques.
    """
    normal_marrow_multiplier: float = 2.0   # ratio pct_patho / pct_sain minimum


@dataclass
class ELNStandards:
    """
    Recommandations ELN 2022 pour la MRD par cytométrie en flux multiparamétrique.

    Référence : Schuurhuis G.J. et al. (2018) Blood 131(12):1275–1291.
               Heuser M. et al. (2022) Leukemia 36:5–22.

    min_cluster_events (LOQ — Limit Of Quantification) :
      Nombre minimum d'événements dans un nœud SOM pour que son ratio
      patho/sain soit statistiquement robuste. L'ELN recommande un minimum
      de 50 événements pour éviter les ratios artefactuels sur petits clusters.

    clinical_positivity_pct :
      Seuil de positivité clinique MRD : 0.1% des cellules totales.
      Valeur de référence ELN pour la décision thérapeutique (rechute précoce).
      En dessous : MRD détectable mais non positive (MRD low-level).
    """
    min_cluster_events: int = 50          # LOQ ELN 2022 : min d'événements/nœud
    clinical_positivity_pct: float = 0.1  # Seuil de positivité clinique (%)


@dataclass
class BlastPhenotypeFilter:
    """
    Porte biologique hybride — filtre phénotypique ELN 2022 / score d'Ogata.

    Rôle dans l'entonnoir à deux portes :
      Quand enabled=True, un nœud ne peut être classé MRD que si sa
      blast_category figure dans allowed_categories. Cela exige que le nœud
      présente une signature phénotypique blastique conforme aux critères
      ELN 2022 (LAIP) et au score d'Ogata (CD45-dim, SSC-bas, CD34/CD117 bright).

    Réduit fortement les faux positifs liés à :
      - L'effet batch (cellules saines atypiques isolées dans un nœud dédié)
      - Les clusters de transition (progéniteurs normaux CD34+)
      - Les débris non exclus par le gating

    Granularité : applicable indépendamment à chaque méthode (apply_to_jf,
    apply_to_flo, apply_to_eln) pour une configuration fine.

    Seuils configurables (high_threshold, moderate_threshold, weak_threshold) :
      Permettent d'adapter la sensibilité à la présentation clinique.
      Ex : abaisser moderate_threshold à 2.0 pour capturer les blastes matures
      ayant perdu le CD34 dans les leucémies massives avancées.

    Référence : blast_detection.py — build_blast_weights(), score_nodes_for_blasts(),
               categorize_blast_score() — basés sur ELN 2022 + score Ogata.
    """
    enabled: bool = False
    allowed_categories: List[str] = field(
        default_factory=lambda: ["BLAST_HIGH", "BLAST_MODERATE"]
    )
    apply_to_jf: bool = True
    apply_to_flo: bool = True
    apply_to_eln: bool = True
    high_threshold: float = 6.0
    moderate_threshold: float = 2.0
    weak_threshold: float = 0.0


@dataclass
class MRDConfig:
    """Configuration complète du calcul MRD."""
    enabled: bool = True
    method: str = "all"  # "jf", "flo", "eln", "all"
    method_jf: MRDMethodJF = field(default_factory=MRDMethodJF)
    method_flo: MRDMethodFlo = field(default_factory=MRDMethodFlo)
    eln_standards: ELNStandards = field(default_factory=ELNStandards)
    blast_phenotype_filter: BlastPhenotypeFilter = field(
        default_factory=BlastPhenotypeFilter
    )
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

                bpf = params.get("blast_phenotype_filter", {})
                cfg.blast_phenotype_filter.enabled = bpf.get(
                    "enabled", cfg.blast_phenotype_filter.enabled)
                cfg.blast_phenotype_filter.allowed_categories = bpf.get(
                    "allowed_categories", cfg.blast_phenotype_filter.allowed_categories)
                cfg.blast_phenotype_filter.apply_to_jf = bpf.get(
                    "apply_to_jf", cfg.blast_phenotype_filter.apply_to_jf)
                cfg.blast_phenotype_filter.apply_to_flo = bpf.get(
                    "apply_to_flo", cfg.blast_phenotype_filter.apply_to_flo)
                cfg.blast_phenotype_filter.apply_to_eln = bpf.get(
                    "apply_to_eln", cfg.blast_phenotype_filter.apply_to_eln)
                cfg.blast_phenotype_filter.high_threshold = float(bpf.get(
                    "high_threshold", cfg.blast_phenotype_filter.high_threshold))
                cfg.blast_phenotype_filter.moderate_threshold = float(bpf.get(
                    "moderate_threshold", cfg.blast_phenotype_filter.moderate_threshold))
                cfg.blast_phenotype_filter.weak_threshold = float(bpf.get(
                    "weak_threshold", cfg.blast_phenotype_filter.weak_threshold))

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
    # Porte biologique (None si filtre désactivé ou données absentes)
    blast_score: Optional[float] = None
    blast_category: Optional[str] = None


@dataclass
class MRDResult:
    """Résultat global du calcul MRD."""
    method_used: str
    total_cells: int
    total_cells_patho: int
    total_cells_sain: int
    # Dénominateur effectif utilisé pour le calcul MRD%
    # = total_cells_patho si cd45_autogating_mode="none"
    # = cellules patho CD45+ si cd45_autogating_mode in ("cd45", "cd45_dim")
    mrd_denominator: int = 0
    mrd_denominator_mode: str = "none"  # "none" | "cd45" | "cd45_dim"
    # Nombre de cellules patho CD45+ — toujours calculé si cd45_mask fourni,
    # indépendamment du dénominateur effectif. Permet le toggle UI.
    n_patho_cd45pos: int = 0
    # Nombre de cellules patho AVANT la gate CD45 (CD45+ + CD45-).
    # = total_cells_patho si gate CD45 inactive.
    # Stocké pour le toggle UI : dénominateur "toutes cellules patho".
    n_patho_pre_cd45: int = 0

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

    # Filtre phénotypique hybride — état pour la traçabilité
    blast_filter_active: bool = False  # True si le filtre a été appliqué

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
            "mrd_denominator": self.mrd_denominator,
            "mrd_denominator_mode": self.mrd_denominator_mode,
            "n_patho_cd45pos": self.n_patho_cd45pos,
            "n_patho_pre_cd45": self.n_patho_pre_cd45,
            "blast_filter_active": self.blast_filter_active,
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
                    "blast_score": round(c.blast_score, 2) if c.blast_score is not None else None,
                    "blast_category": c.blast_category,
                }
                for c in self.per_node
            ],
            "config": self.config_snapshot,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Calcul MRD
# ─────────────────────────────────────────────────────────────────────────────


def _build_node_blast_scores(
    marker_names: List[str],
    X_norm: Optional[np.ndarray] = None,
    node_medians: Optional[np.ndarray] = None,
    nbm_center: Optional[np.ndarray] = None,
    nbm_scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calcule le blast_score /10 pour chaque nœud SOM, vectorisé sur toute la grille.

    ── Trois modes d'entrée (par ordre de priorité) ─────────────────────────────

    Mode 1 — X_norm fourni (z-scores pré-calculés) :
      Les médianes SOM sont déjà en z-scores par rapport à la moelle normale,
      produits par compute_reference_normalization() de blast_detection.py.
      C'est le mode le plus précis : les blastes ont z_CD34 ≈ +2, z_SSC ≈ −2.

    Mode 2 — node_medians + nbm_center + nbm_scale fournis :
      Z-scoring à la volée via les statistiques NBM passées explicitement.
      Equivalent au Mode 1 mais sans pré-calcul de X_norm.
      nbm_center = médiane NBM par marqueur (après transformation arcsinh).
      nbm_scale  = IQR/1.35 NBM par marqueur (pseudo-std robuste).

    Mode 3 — node_medians seul (fallback dégradé) :
      Z-scoring intra-dataset : centre = médiane des médianes de nœuds,
      scale = std des médianes de nœuds.
      MOINS PRÉCIS : fonctionne si le dataset contient un mélange sain/patho
      (les nœuds de blastes s'éloignent de la moyenne des nœuds normaux).
      NE PAS UTILISER en production — uniquement en l'absence totale de NBM.

    Args:
        marker_names: Noms des marqueurs (colonnes de X_norm ou node_medians).
        X_norm: Matrice [n_nodes, n_markers] des z-scores SOM vs NBM.
                Prioritaire sur tous les autres modes si fournie.
        node_medians: Matrice [n_nodes, n_markers] des médianes brutes par nœud
                      (dans l'espace transformé, ex: arcsinh/5).
        nbm_center: Vecteur [n_markers] des médianes NBM par marqueur.
                    Requis avec node_medians pour le Mode 2.
        nbm_scale: Vecteur [n_markers] des échelles NBM (IQR/1.35 ou std).
                   Requis avec node_medians pour le Mode 2.

    Returns:
        np.ndarray de forme (n_nodes,) avec scores dans [0.0, 10.0].

    Raises:
        ValueError: Si aucun des modes ne peut être utilisé.
    """
    from flowsom_pipeline_pro.src.analysis.blast_detection import (
        build_blast_weights,
        score_nodes_for_blasts,
        compute_reference_stats,
    )

    if X_norm is not None:
        # Mode 1 : z-scores pré-calculés dans l'espace NBM
        _X = np.asarray(X_norm, dtype=float)

    elif node_medians is not None and nbm_center is not None and nbm_scale is not None:
        # Mode 2 : z-scoring à la volée avec stats NBM explicites
        _raw    = np.asarray(node_medians, dtype=float)
        _center = np.asarray(nbm_center, dtype=float)
        _scale  = np.asarray(nbm_scale, dtype=float)
        _scale  = np.where(_scale < 0.01, 0.01, _scale)  # éviter div/0
        _X = (_raw - _center) / _scale

    elif node_medians is not None:
        # Mode 3 : z-scoring intra-dataset (fallback dégradé — voir docstring)
        _raw = np.asarray(node_medians, dtype=float)
        _logger.warning(
            "Blast scoring : pas de stats NBM disponibles — z-scoring intra-dataset "
            "(Mode 3 dégradé). Les scores seront moins discriminants. "
            "Fournir nbm_center + nbm_scale pour un scoring optimal."
        )
        _center, _scale = compute_reference_stats(_raw, robust=True)
        _X = (_raw - _center) / _scale

    else:
        raise ValueError(
            "_build_node_blast_scores : X_norm ou node_medians requis."
        )

    weights = build_blast_weights(marker_names)
    return score_nodes_for_blasts(_X, marker_names, weights)


def compute_mrd(
    df_cells: pd.DataFrame,
    clustering: np.ndarray,
    mrd_cfg: MRDConfig,
    condition_column: str = "condition",
    cd45_autogating_mode: str = "none",
    cd45_mask: Optional[np.ndarray] = None,
    X_norm: Optional[np.ndarray] = None,
    node_medians: Optional[np.ndarray] = None,
    marker_names: Optional[List[str]] = None,
    nbm_center: Optional[np.ndarray] = None,
    nbm_scale: Optional[np.ndarray] = None,
) -> MRDResult:
    """
    Calcule la MRD résiduelle selon les méthodes JF, Flo et/ou ELN.

    Opère au niveau des **nœuds SOM** (clustering), pas des métaclusters.

    ── Approche Hybride (si blast_phenotype_filter.enabled=True) ───────────────

    Un nœud est classé MRD seulement s'il franchit deux portes successives :

      Porte 1 — Topologique/Mathématique (critère de la méthode) :
        JF  : pct_sain_global < seuil ET pct_patho > seuil dans le nœud
        Flo : pct_patho > N × pct_sain dans le nœud
        ELN : n_cells >= LOQ ET pct_patho > pct_sain (critère DfN)

      Porte 2 — Phénotypique/Biologique (scoring ELN 2022 / Ogata) :
        blast_category IN allowed_categories (défaut: BLAST_HIGH, BLAST_MODERATE)
        Calculée vectoriellement sur toute la grille SOM AVANT la boucle,
        via score_nodes_for_blasts() de blast_detection.py.

    Données pour la porte biologique (par ordre de priorité) :
      X_norm        : médianes SOM normalisées dans l'espace de référence —
                      produit de compute_reference_normalization() (blast_detection.py).
                      Recommandé : la normalisation relative à la moelle normale est
                      biologiquement plus significative que l'intra-dataset.
      node_medians  : médianes SOM brutes — normalisation intra-dataset appliquée
                      automatiquement en fallback (moins précis, mais fonctionnel
                      sans population de référence externe).

    ── Dénominateur MRD ─────────────────────────────────────────────────────────

    Le pourcentage MRD (blastes / cellules totales) utilise différents
    dénominateurs selon le mode d'autogating CD45 :
      "none"     → dénominateur = toutes cellules patho (comportement historique)
      "cd45"     → dénominateur = cellules patho CD45+ uniquement
      "cd45_dim" → idem "cd45" (inclut blastes CD45-dim passés par la gate)

    Ce design permet de refléter la pratique clinique ELN 2022 qui mesure
    la MRD comme % des cellules CD45+ leucocytaires totales.

    Args:
        df_cells: DataFrame cellulaire avec colonne condition (shape: n_cells).
        clustering: Array (n_cells,) d'assignation nœud SOM (entiers 0-based).
                    Aligné ligne-à-ligne avec df_cells.
        mrd_cfg: Configuration MRD chargée via load_mrd_config().
        condition_column: Nom de la colonne condition dans df_cells.
                          Doit contenir mrd_cfg.condition_sain et condition_patho.
        cd45_autogating_mode: Mode de dénominateur CD45.
            "none"     → toutes cellules patho (comportement historique).
            "cd45"     → cellules patho CD45+ seulement.
            "cd45_dim" → idem (inclut blastes CD45-dim).
        cd45_mask: Masque booléen (n_cells,) — True = cellule CD45+.
            Nécessaire si cd45_autogating_mode != "none".
        X_norm: Matrice [n_nodes, n_markers] des z-scores des médianes SOM
            par rapport à la moelle normale (NBM). Prioritaire sur node_medians.
            Produit par compute_reference_normalization(node_medians, X_nbm).
            Les blastes ont z_CD34 ≈ +2, z_SSC ≈ −2 dans cet espace.
        node_medians: Matrice [n_nodes, n_markers] des médianes brutes par nœud
            (dans l'espace transformé, ex: arcsinh/5). Utilisé si X_norm=None.
            Combiné avec nbm_center+nbm_scale pour le z-scoring (Mode 2).
            Z-scoring intra-dataset en fallback si stats NBM absentes (Mode 3 dégradé).
        marker_names: Noms des marqueurs (colonnes de X_norm ou node_medians).
            Requis si X_norm ou node_medians est fourni.
        nbm_center: Vecteur [n_markers] des médianes NBM par marqueur (arcsinh/5).
            Calcule par compute_reference_stats() sur les données NBM.
        nbm_scale: Vecteur [n_markers] des IQR/1.35 NBM par marqueur.
            Combiné avec nbm_center pour le z-scoring du Mode 2.

    Returns:
        MRDResult avec :
          - Totaux MRD par méthode (mrd_cells_*, mrd_pct_*, n_nodes_mrd_*)
          - Statut ELN (eln_positive, eln_low_level)
          - Détail par nœud SOM (per_node : List[MRDClusterResult])
            incluant blast_score et blast_category si porte biologique active
          - blast_filter_active : True si la porte biologique a été appliquée
    """
    _logger.info(
        "Calcul MRD (nœuds SOM) — méthode(s): %s | dénominateur CD45: %s",
        mrd_cfg.method,
        cd45_autogating_mode,
    )

    bpf = mrd_cfg.blast_phenotype_filter
    blast_filter_active = False

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
    is_patho = condition == mrd_cfg.condition_patho
    total_patho = int(is_patho.sum())
    total_sain = int((condition == mrd_cfg.condition_sain).sum())

    # ── Comptage cellules patho CD45+ (toujours calculé si masque disponible) ──
    n_patho_cd45pos = 0
    if cd45_mask is not None:
        cd45_arr = np.asarray(cd45_mask, dtype=bool)
        if len(cd45_arr) == len(condition):
            n_patho_cd45pos = int((is_patho & cd45_arr).sum())
        else:
            _logger.warning(
                "cd45_mask longueur %d ≠ n_cells %d — n_patho_cd45pos non calculé.",
                len(cd45_arr),
                len(condition),
            )

    # ── Dénominateur effectif pour le calcul MRD% ─────────────────────────────
    use_cd45_denom = cd45_autogating_mode in ("cd45", "cd45_dim")
    if use_cd45_denom and n_patho_cd45pos > 0:
        total_patho_cd45pos = n_patho_cd45pos
        _logger.info(
            "Dénominateur MRD CD45+ : %d cellules patho CD45+ (sur %d patho totales)",
            total_patho_cd45pos,
            total_patho,
        )
    else:
        total_patho_cd45pos = total_patho

    run_jf = mrd_cfg.method in ("jf", "both", "all")
    run_flo = mrd_cfg.method in ("flo", "both", "all")
    run_eln = mrd_cfg.method in ("eln", "all")

    # ── Porte Biologique : pré-calcul des blast scores par nœud ──────────────
    # node_blast_scores[i] = blast_score du i-ème nœud dans unique_nodes.
    # node_blast_cats[i]   = blast_category correspondante.
    node_blast_scores: Optional[np.ndarray] = None
    node_blast_cats: Optional[List[str]] = None

    if bpf.enabled:
        # Détermine le mode d'entrée disponible pour le scoring blast vectorisé
        _has_xnorm = X_norm is not None and marker_names is not None
        _has_medians = node_medians is not None and marker_names is not None

        if _has_xnorm or _has_medians:
            try:
                from flowsom_pipeline_pro.src.analysis.blast_detection import (
                    categorize_blast_score,
                )

                # Choix du mode de z-scoring :
                #   Mode 1 : X_norm déjà en z-scores (prioritaire)
                #   Mode 2 : node_medians + stats NBM explicites
                #   Mode 3 : node_medians seul, z-scoring intra-dataset (dégradé)
                _has_nbm_stats = nbm_center is not None and nbm_scale is not None
                if _has_xnorm:
                    _mode_label = "Mode 1 — X_norm (z-scores pré-calculés)"
                elif _has_medians and _has_nbm_stats:
                    _mode_label = "Mode 2 — node_medians + stats NBM"
                else:
                    _mode_label = "Mode 3 — z-scoring intra-dataset (dégradé)"

                node_blast_scores = _build_node_blast_scores(
                    marker_names=list(marker_names),
                    X_norm=np.asarray(X_norm, dtype=float) if _has_xnorm else None,
                    node_medians=np.asarray(node_medians, dtype=float) if not _has_xnorm else None,
                    nbm_center=np.asarray(nbm_center, dtype=float) if _has_nbm_stats and not _has_xnorm else None,
                    nbm_scale=np.asarray(nbm_scale, dtype=float) if _has_nbm_stats and not _has_xnorm else None,
                )
                node_blast_cats = [
                    categorize_blast_score(
                        float(s),
                        high_thresh=bpf.high_threshold,
                        mod_thresh=bpf.moderate_threshold,
                        weak_thresh=bpf.weak_threshold,
                    )
                    for s in node_blast_scores
                ]
                blast_filter_active = True
                _n_high = sum(1 for c in node_blast_cats if c == "BLAST_HIGH")
                _n_mod  = sum(1 for c in node_blast_cats if c == "BLAST_MODERATE")
                _logger.info(
                    "Filtre phénotypique hybride ACTIVE — %s — %d noeuds scores : "
                    "%d BLAST_HIGH, %d BLAST_MODERATE (categories acceptees : %s)",
                    _mode_label,
                    len(node_blast_scores),
                    _n_high, _n_mod,
                    bpf.allowed_categories,
                )
            except Exception as exc:
                _logger.warning(
                    "Filtre phénotypique : erreur lors du scoring blast (%s) — "
                    "porte biologique désactivée pour cette analyse.",
                    exc,
                )
        else:
            _logger.warning(
                "blast_phenotype_filter.enabled=True mais ni X_norm ni node_medians "
                "n'est fourni avec marker_names — porte biologique désactivée."
            )

    per_node: List[MRDClusterResult] = []
    mrd_cells_jf = 0
    mrd_cells_flo = 0
    mrd_cells_eln = 0
    n_nodes_jf = 0
    n_nodes_flo = 0
    n_nodes_eln = 0

    for node_idx, node_id in enumerate(unique_nodes):
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

        # pct_sain_global : % des cellules saines du cluster / totalité moelle normale
        _denom_sain = total_sain if total_sain > 0 else 1
        pct_sain_global = (n_sain / _denom_sain) * 100.0

        # ── Porte Biologique pour ce nœud ─────────────────────────────────
        _blast_score: Optional[float] = None
        _blast_cat: Optional[str] = None
        if node_blast_scores is not None and node_blast_cats is not None:
            _blast_score = float(node_blast_scores[node_idx])
            _blast_cat = node_blast_cats[node_idx]

        def _passes_blast_gate(apply_flag: bool) -> bool:
            """Vérifie la porte biologique pour une méthode donnée."""
            if not blast_filter_active or not apply_flag:
                return True  # filtre inactif → porte toujours ouverte
            if _blast_cat is None:
                return True  # pas de score calculé → ne pas bloquer
            return _blast_cat in bpf.allowed_categories

        # ── Méthode JF ────────────────────────────────────────────────
        is_mrd_jf = False
        if run_jf:
            if (pct_sain_global < mrd_cfg.method_jf.max_normal_marrow_pct
                    and pct_patho > mrd_cfg.method_jf.min_patho_cells_pct):
                if _passes_blast_gate(bpf.apply_to_jf):
                    is_mrd_jf = True
                    mrd_cells_jf += n_patho
                    n_nodes_jf += 1

        # ── Méthode Flo ───────────────────────────────────────────────
        is_mrd_flo = False
        if run_flo:
            threshold_flo = pct_sain * mrd_cfg.method_flo.normal_marrow_multiplier
            if pct_patho > threshold_flo:
                if _passes_blast_gate(bpf.apply_to_flo):
                    is_mrd_flo = True
                    mrd_cells_flo += n_patho
                    n_nodes_flo += 1

        # ── Méthode ELN (DfN + LOQ) ──────────────────────────────────
        is_mrd_eln = False
        if run_eln:
            if n_in_node >= mrd_cfg.eln_standards.min_cluster_events:
                if pct_patho > pct_sain:
                    if _passes_blast_gate(bpf.apply_to_eln):
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
            blast_score=_blast_score,
            blast_category=_blast_cat,
        ))

    # MRD % = cellules MRD / dénominateur patho
    _denom_patho = total_patho_cd45pos if total_patho_cd45pos > 0 else 1
    _has_patho = total_patho_cd45pos > 0
    mrd_pct_jf = (mrd_cells_jf / _denom_patho * 100.0) if _has_patho else 0.0
    mrd_pct_flo = (mrd_cells_flo / _denom_patho * 100.0) if _has_patho else 0.0
    mrd_pct_eln = (mrd_cells_eln / _denom_patho * 100.0) if _has_patho else 0.0

    # Statut clinique ELN
    eln_positive = mrd_pct_eln >= mrd_cfg.eln_standards.clinical_positivity_pct
    eln_low_level = (mrd_cells_eln > 0) and (not eln_positive)

    result = MRDResult(
        method_used=mrd_cfg.method,
        total_cells=n_total,
        total_cells_patho=total_patho,
        total_cells_sain=total_sain,
        mrd_denominator=total_patho_cd45pos,
        mrd_denominator_mode=cd45_autogating_mode,
        n_patho_cd45pos=n_patho_cd45pos,
        n_patho_pre_cd45=total_patho,
        blast_filter_active=blast_filter_active,
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
            "blast_filter_enabled": bpf.enabled,
            "blast_filter_categories": bpf.allowed_categories,
        },
    )

    _logger.info(
        "MRD JF : %d cellules patho dans %d nœuds SOM → MRD = %.4f%%%s",
        mrd_cells_jf, n_nodes_jf, mrd_pct_jf,
        " [+porte biologique]" if blast_filter_active and bpf.apply_to_jf else "",
    )
    _logger.info(
        "MRD Flo: %d cellules patho dans %d nœuds SOM → MRD = %.4f%%%s",
        mrd_cells_flo, n_nodes_flo, mrd_pct_flo,
        " [+porte biologique]" if blast_filter_active and bpf.apply_to_flo else "",
    )
    _logger.info(
        "MRD ELN: %d cellules patho dans %d nœuds SOM → MRD = %.4f%% — %s%s",
        mrd_cells_eln, n_nodes_eln, mrd_pct_eln,
        "POSITIVE" if eln_positive else ("LOW-LEVEL" if eln_low_level else "NEGATIVE"),
        " [+porte biologique]" if blast_filter_active and bpf.apply_to_eln else "",
    )

    return result
