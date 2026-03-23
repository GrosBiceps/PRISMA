"""
cli/parsers.py — Construction de l'interface en ligne de commande.

Fournit :
  - build_argument_parser() : construit le parser argparse
  - detect_config_file()    : auto-détection du fichier YAML
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construit et retourne le parser d'arguments CLI du pipeline FlowSOM.

    Returns:
        ArgumentParser configuré avec tous les groupes d'arguments.
    """
    parser = argparse.ArgumentParser(
        description="FlowSOM Analysis Pipeline Pro — Analyse de cytométrie en flux MRD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Analyse avec auto-détection du fichier de configuration
  python -m flowsom_pipeline_pro

  # Spécifier un fichier de configuration explicitement
  python -m flowsom_pipeline_pro --config config_flowsom.yaml

  # Analyse simple d'un dossier NBM
  python -m flowsom_pipeline_pro --healthy-folder "Data/NBM" --output Results

  # Mode comparaison Sain vs Pathologique
  python -m flowsom_pipeline_pro \\
      --healthy-folder "Data/NBM" \\
      --patho-folder "Data/Patho" \\
      --compare-mode \\
      --output Results_Comparison

  # Personnaliser les paramètres FlowSOM
  python -m flowsom_pipeline_pro \\
      --healthy-folder "Data/NBM" \\
      --xdim 15 --ydim 15 \\
      --n-metaclusters 20 \\
      --n-iterations 20

  # Désactiver l'accélération GPU
  python -m flowsom_pipeline_pro --healthy-folder "Data/NBM" --no-gpu

  # Mode autoclustering (détection automatique du nombre de clusters)
  python -m flowsom_pipeline_pro \\
      --healthy-folder "Data/NBM" \\
      --auto-clustering
        """,
    )

    # ── Fichier de configuration ──────────────────────────────────────────────
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        metavar="PATH",
        help="Fichier de configuration YAML (défaut: auto-détection de config_flowsom.yaml)",
    )

    # ── Chemins ───────────────────────────────────────────────────────────────
    paths = parser.add_argument_group("Chemins et fichiers")
    paths.add_argument(
        "--healthy-folder",
        type=str,
        default=None,
        metavar="DIR",
        help="Dossier contenant les fichiers FCS sains (NBM)",
    )
    paths.add_argument(
        "--patho-folder",
        type=str,
        default=None,
        metavar="DIR",
        help="Dossier contenant les fichiers FCS pathologiques (optionnel)",
    )
    paths.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="DIR",
        help="Dossier de sortie (défaut: Results)",
    )

    # ── Mode d'analyse ────────────────────────────────────────────────────────
    mode = parser.add_argument_group("Mode d'analyse")
    mode_grp = mode.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--compare-mode",
        dest="compare_mode",
        action="store_true",
        default=None,
        help="Activer le mode comparaison Sain vs Pathologique",
    )
    mode_grp.add_argument(
        "--no-compare-mode",
        dest="compare_mode",
        action="store_false",
        help="Désactiver le mode comparaison",
    )

    # ── Pre-gating ────────────────────────────────────────────────────────────
    gating = parser.add_argument_group("Pre-gating")
    gating.add_argument(
        "--no-pregate-viable",
        dest="pregate_viable",
        action="store_false",
        default=None,
        help="Désactiver le gating des cellules viables",
    )
    gating.add_argument(
        "--no-pregate-singlets",
        dest="pregate_singlets",
        action="store_false",
        default=None,
        help="Désactiver le gating des singlets",
    )
    gating.add_argument(
        "--no-pregate-cd45",
        dest="pregate_cd45",
        action="store_false",
        default=None,
        help="Désactiver le gating CD45+",
    )
    gating.add_argument(
        "--pregate-cd34",
        dest="pregate_cd34",
        action="store_true",
        default=None,
        help="Activer le gating CD34+ (désactivé par défaut)",
    )
    gating.add_argument(
        "--pregate-mode",
        type=str,
        choices=["auto", "manual"],
        default=None,
        help="Mode de gating: 'auto' (GMM) ou 'manual' (percentiles fixes) [défaut: auto]",
    )

    # ── Paramètres FlowSOM ────────────────────────────────────────────────────
    flowsom = parser.add_argument_group("Paramètres FlowSOM")
    flowsom.add_argument(
        "--xdim",
        type=int,
        default=None,
        metavar="N",
        help="Dimension X de la grille SOM (défaut: 10)",
    )
    flowsom.add_argument(
        "--ydim",
        type=int,
        default=None,
        metavar="N",
        help="Dimension Y de la grille SOM (défaut: 10)",
    )
    flowsom.add_argument(
        "--n-metaclusters",
        type=int,
        default=None,
        metavar="N",
        help="Nombre de métaclusters (défaut: 15)",
    )
    flowsom.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        metavar="F",
        help="Taux d'apprentissage du SOM (défaut: 0.05)",
    )
    flowsom.add_argument(
        "--sigma",
        type=float,
        default=None,
        metavar="F",
        help="Sigma pour le voisinage du SOM (défaut: 1.5)",
    )
    flowsom.add_argument(
        "--n-iterations",
        type=int,
        default=None,
        metavar="N",
        help="Nombre d'itérations du SOM (défaut: 10)",
    )
    flowsom.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Graine aléatoire pour la reproductibilité (défaut: 42)",
    )

    # ── Auto-clustering ───────────────────────────────────────────────────────
    autocl = parser.add_argument_group("Auto-clustering")
    autocl.add_argument(
        "--auto-clustering",
        dest="auto_clustering",
        action="store_true",
        default=None,
        help="Détecter automatiquement le nombre optimal de clusters",
    )
    autocl.add_argument(
        "--min-clusters",
        type=int,
        default=None,
        metavar="N",
        help="Nombre minimum de clusters pour la recherche automatique (défaut: 5)",
    )
    autocl.add_argument(
        "--max-clusters",
        type=int,
        default=None,
        metavar="N",
        help="Nombre maximum de clusters pour la recherche automatique (défaut: 40)",
    )

    # ── Transformation et normalisation ───────────────────────────────────────
    transform = parser.add_argument_group("Transformation et normalisation")
    transform.add_argument(
        "--transform",
        type=str,
        default=None,
        choices=["arcsinh", "logicle", "log10", "none"],
        help="Méthode de transformation (défaut: logicle)",
    )
    transform.add_argument(
        "--cofactor",
        type=float,
        default=None,
        metavar="F",
        help="Cofacteur pour transformation arcsinh (défaut: 150.0)",
    )
    transform.add_argument(
        "--normalize",
        type=str,
        default=None,
        choices=["zscore", "minmax", "none"],
        help="Méthode de normalisation (défaut: zscore)",
    )

    # ── Downsampling ──────────────────────────────────────────────────────────
    sampling = parser.add_argument_group("Downsampling")
    sampling.add_argument(
        "--no-downsample",
        dest="downsample",
        action="store_false",
        default=None,
        help="Désactiver le downsampling",
    )
    sampling.add_argument(
        "--max-cells-per-file",
        type=int,
        default=None,
        metavar="N",
        help="Nombre max de cellules par fichier (défaut: 50000)",
    )
    sampling.add_argument(
        "--max-cells-total",
        type=int,
        default=None,
        metavar="N",
        help="Nombre max de cellules total (défaut: 1000000)",
    )

    # ── Visualisation ─────────────────────────────────────────────────────────
    viz = parser.add_argument_group("Visualisation")
    viz.add_argument(
        "--no-save-plots",
        dest="save_plots",
        action="store_false",
        default=None,
        help="Ne pas sauvegarder les graphiques",
    )
    viz.add_argument(
        "--plot-format",
        type=str,
        default=None,
        choices=["png", "pdf", "svg"],
        help="Format des graphiques (défaut: png)",
    )
    viz.add_argument(
        "--dpi",
        type=int,
        default=None,
        metavar="N",
        help="Résolution des graphiques en DPI (défaut: 300)",
    )

    # ── GPU ───────────────────────────────────────────────────────────────────
    gpu = parser.add_argument_group("GPU")
    gpu.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        default=None,
        help="Désactiver l'accélération GPU (CuPy/RAPIDS)",
    )

    # ── Verbosité ─────────────────────────────────────────────────────────────
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Mode verbeux (DEBUG)",
    )
    verbosity.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Mode silencieux (WARNING uniquement)",
    )

    return parser


def detect_config_file(script_dir: Optional[Path] = None) -> Optional[str]:
    """
    Recherche automatiquement un fichier de configuration YAML dans l'ordre suivant :
      1. <répertoire du script>/config_flowsom.yaml
      2. ./config_flowsom.yaml (CWD)
      3. <répertoire du script>/config.yaml
      4. ./config.yaml (CWD)

    Args:
        script_dir: Répertoire où chercher en priorité (optionnel).

    Returns:
        Chemin absolu du fichier trouvé, ou None si aucun.
    """
    import os

    candidates = []
    if script_dir:
        candidates += [
            Path(script_dir) / "config_flowsom.yaml",
            Path(script_dir) / "config.yaml",
        ]
    candidates += [
        Path(os.getcwd()) / "config_flowsom.yaml",
        Path(os.getcwd()) / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def parse_arguments() -> "argparse.Namespace":
    """
    Construit le parser et parse sys.argv en une seule opération.

    Alias de commodité équivalent à ``build_argument_parser().parse_args()``
    utilisé par le point d'entrée CLI principal de ``flowsom_pipeline.py``.

    Returns:
        argparse.Namespace avec tous les arguments CLI résolus.
    """
    return build_argument_parser().parse_args()
