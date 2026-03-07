"""
cli/main.py — Point d'entrée de la CLI FlowSOM Pipeline Pro.

Orchestre :
  1. Parsing des arguments (argparse via parsers.py)
  2. Configuration du logging
  3. Construction du PipelineConfig (YAML + CLI args en surcharge)
  4. Exécution du pipeline via FlowSOMPipeline.execute()
  5. Affichage du résumé et gestion des codes de sortie
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import NoReturn


def main() -> None:
    """
    Point d'entrée principal de la CLI.

    Charge la configuration, exécute le pipeline et affiche le résumé.
    Sort avec le code 0 en cas de succès, 1 en cas d'échec.
    """
    from flowsom_pipeline_pro.cli.parsers import (
        build_argument_parser,
        detect_config_file,
    )
    from flowsom_pipeline_pro.config.pipeline_config import PipelineConfig
    from flowsom_pipeline_pro.src.pipeline.pipeline_executor import FlowSOMPipeline

    parser = build_argument_parser()
    args = parser.parse_args()

    # ── Configuration du logging ──────────────────────────────────────────────
    if args.quiet:
        log_level = logging.WARNING
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("flowsom_pipeline_pro")

    # ── Banner ────────────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("  FlowSOM Analysis Pipeline Pro — v1.0.0")
    logger.info("=" * 72)

    # ── Résolution du fichier YAML ────────────────────────────────────────────
    # Priorité : --config CLI > auto-détection dans le répertoire courant
    config_path = args.config
    if config_path is None:
        config_path = detect_config_file(script_dir=Path(__file__).parent)
        if config_path:
            logger.info(f"Configuration auto-détectée : {config_path}")
        else:
            logger.info(
                "Aucun fichier YAML détecté — utilisation des valeurs par défaut."
            )
    else:
        logger.info(f"Configuration chargée depuis : {config_path}")

    # ── Construction du PipelineConfig ────────────────────────────────────────
    try:
        config = _build_config(args, config_path, logger)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    # ── Validation des chemins obligatoires ───────────────────────────────────
    if not config.paths.healthy_folder:
        logger.error(
            "Aucun dossier sain (NBM) spécifié.\n"
            "  → Utilisez : --healthy-folder 'chemin/vers/NBM'\n"
            "  → Ou renseignez 'paths.healthy_folder' dans le fichier YAML."
        )
        sys.exit(1)

    if not Path(config.paths.healthy_folder).exists():
        logger.error(f"Le dossier sain n'existe pas : {config.paths.healthy_folder}")
        sys.exit(1)

    if (
        config.analysis.compare_mode
        and config.paths.patho_folder
        and not Path(config.paths.patho_folder).exists()
    ):
        logger.error(
            f"Le dossier pathologique n'existe pas : {config.paths.patho_folder}"
        )
        sys.exit(1)

    # ── Résumé de la configuration ────────────────────────────────────────────
    logger.info("Configuration active :")
    logger.info(f"  Dossier sain          : {config.paths.healthy_folder}")
    if config.analysis.compare_mode and config.paths.patho_folder:
        logger.info(f"  Dossier pathologique  : {config.paths.patho_folder}")
    logger.info(f"  Mode comparaison      : {config.analysis.compare_mode}")
    logger.info(
        f"  Pre-gating            : viable={config.pregate.viable}, "
        f"singlets={config.pregate.singlets}, "
        f"CD45={config.pregate.cd45}, CD34={config.pregate.cd34}"
    )
    logger.info(
        f"  FlowSOM               : grille {config.flowsom.xdim}×{config.flowsom.ydim}, "
        f"{config.flowsom.n_metaclusters} métaclusters, "
        f"{config.flowsom.n_iterations} itérations"
    )
    logger.info(
        f"  Transformation        : {config.transform.method} "
        f"(cofacteur={config.transform.cofactor})"
    )
    logger.info(f"  Normalisation         : {config.normalize.method}")
    logger.info(f"  GPU                   : {config.gpu.enabled}")
    logger.info(f"  Dossier de sortie     : {config.paths.output_dir}")

    # ── Exécution ─────────────────────────────────────────────────────────────
    logger.info("\nDémarrage de l'analyse…")
    pipeline = FlowSOMPipeline(config=config)

    try:
        result = pipeline.execute()
    except Exception as exc:
        logger.error(
            f"Erreur inattendue lors de l'exécution : {exc}", exc_info=args.verbose
        )
        sys.exit(1)

    # ── Résultat ──────────────────────────────────────────────────────────────
    print(result.summary())

    if not result.success:
        logger.error("Le pipeline a échoué. Consultez le résumé ci-dessus.")
        sys.exit(1)

    logger.info(f"Résultats disponibles dans : {config.paths.output_dir}")
    # Code de sortie 0 implicite


# ── Helpers internes ──────────────────────────────────────────────────────────


def _build_config(args, config_path, logger):
    """Construit le PipelineConfig, puis applique les surcharges CLI."""
    from flowsom_pipeline_pro.config.pipeline_config import PipelineConfig

    cfg = PipelineConfig.from_args(args, yaml_path=config_path)

    # Surcharges spécifiques non prises en charge par from_args
    if getattr(args, "pregate_viable", None) is not None:
        cfg.pregate.viable = args.pregate_viable
    if getattr(args, "pregate_singlets", None) is not None:
        cfg.pregate.singlets = args.pregate_singlets
    if getattr(args, "pregate_cd45", None) is not None:
        cfg.pregate.cd45 = args.pregate_cd45
    if getattr(args, "pregate_cd34", None) is not None:
        cfg.pregate.cd34 = args.pregate_cd34
    if getattr(args, "pregate_mode", None):
        cfg.pregate.mode = args.pregate_mode

    if getattr(args, "auto_clustering", None) is not None:
        cfg.auto_clustering.enabled = args.auto_clustering
    if getattr(args, "min_clusters", None) is not None:
        cfg.auto_clustering.min_clusters = args.min_clusters
    if getattr(args, "max_clusters", None) is not None:
        cfg.auto_clustering.max_clusters = args.max_clusters

    return cfg


if __name__ == "__main__":
    main()
