"""
pipeline_config.py — Dataclass centralisée de configuration du pipeline.

Chargement en cascade :
  1. Valeurs par défaut (constants.py)
  2. default_config.yaml (fourni avec le package)
  3. Fichier YAML utilisateur (chemin explicite ou auto-détecté)
  4. Arguments CLI (priorité maximale)

Usage:
    config = PipelineConfig.from_yaml("my_config.yaml")
    config = PipelineConfig.from_args(args)   # args = argparse.Namespace
    config = PipelineConfig()                  # valeurs par défaut
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any

from .constants import (
    DEFAULT_XDIM,
    DEFAULT_YDIM,
    DEFAULT_N_METACLUSTERS,
    DEFAULT_SEED,
    DEFAULT_RLEN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SIGMA,
    DEFAULT_N_ITERATIONS,
    DEFAULT_TRANSFORM_METHOD,
    DEFAULT_ARCSINH_COFACTOR,
    DEFAULT_NORMALIZE_METHOD,
    DEFAULT_MAX_CELLS_PER_FILE,
    DEFAULT_MAX_CELLS_TOTAL,
    DEFAULT_MIN_CLUSTERS,
    DEFAULT_MAX_CLUSTERS,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SAMPLE_SIZE_BOOTSTRAP,
    DEFAULT_MIN_STABILITY_THRESHOLD,
    DEFAULT_W_STABILITY,
    DEFAULT_W_SILHOUETTE,
    DEFAULT_DEBRIS_MIN_PCT,
    DEFAULT_DEBRIS_MAX_PCT,
    DEFAULT_SINGLETS_RATIO_MIN,
    DEFAULT_SINGLETS_RATIO_MAX,
    DEFAULT_CD45_THRESHOLD_PCT,
    DEFAULT_CD34_THRESHOLD_PCT,
    DEFAULT_CD34_SSC_MAX_PCT,
    DEFAULT_PLOT_FORMAT,
    DEFAULT_DPI,
)


@dataclass
class PathsConfig:
    healthy_folder: str = "Data/Moelle normale"
    patho_folder: str = "Data/Patho"
    output_dir: str = "Results"


@dataclass
class AnalysisConfig:
    compare_mode: bool = True


@dataclass
class PregateConfig:
    apply: bool = True
    mode: str = "auto"  # "auto" | "manual"
    mode_blastes_vs_normal: bool = True
    viable: bool = True
    singlets: bool = True
    cd45: bool = True
    cd34: bool = False
    # Paramètres avancés
    debris_min_percentile: float = DEFAULT_DEBRIS_MIN_PCT
    debris_max_percentile: float = DEFAULT_DEBRIS_MAX_PCT
    doublets_ratio_min: float = DEFAULT_SINGLETS_RATIO_MIN
    doublets_ratio_max: float = DEFAULT_SINGLETS_RATIO_MAX
    cd45_threshold_percentile: float = DEFAULT_CD45_THRESHOLD_PCT
    cd34_threshold_percentile: float = DEFAULT_CD34_THRESHOLD_PCT
    cd34_use_ssc_filter: bool = True
    cd34_ssc_max_percentile: float = DEFAULT_CD34_SSC_MAX_PCT


@dataclass
class FlowSOMConfig:
    xdim: int = DEFAULT_XDIM
    ydim: int = DEFAULT_YDIM
    rlen: Any = DEFAULT_RLEN  # "auto" ou entier
    n_metaclusters: int = DEFAULT_N_METACLUSTERS
    learning_rate: float = DEFAULT_LEARNING_RATE
    sigma: float = DEFAULT_SIGMA
    n_iterations: int = DEFAULT_N_ITERATIONS
    seed: int = DEFAULT_SEED


@dataclass
class AutoClusteringConfig:
    enabled: bool = False
    min_clusters: int = DEFAULT_MIN_CLUSTERS
    max_clusters: int = DEFAULT_MAX_CLUSTERS
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    sample_size_bootstrap: int = DEFAULT_SAMPLE_SIZE_BOOTSTRAP
    min_stability_threshold: float = DEFAULT_MIN_STABILITY_THRESHOLD
    weight_stability: float = DEFAULT_W_STABILITY
    weight_silhouette: float = DEFAULT_W_SILHOUETTE


@dataclass
class TransformConfig:
    method: str = DEFAULT_TRANSFORM_METHOD
    cofactor: float = DEFAULT_ARCSINH_COFACTOR
    apply_to_scatter: bool = False


@dataclass
class NormalizeConfig:
    method: str = DEFAULT_NORMALIZE_METHOD


@dataclass
class MarkersConfig:
    exclude_scatter: bool = True
    exclude_additional: List[str] = field(default_factory=list)
    # Supprime les marqueurs -H (Height) quand le doublon -A (Area) existe.
    # Recommandé : réduit la colinéarité et accélère le SOM.
    keep_area_only: bool = True


@dataclass
class DownsamplingConfig:
    enabled: bool = True
    max_cells_per_file: int = DEFAULT_MAX_CELLS_PER_FILE
    max_cells_total: int = DEFAULT_MAX_CELLS_TOTAL


@dataclass
class VisualizationConfig:
    save_plots: bool = True
    umap_enabled: bool = True  # Calculer UMAP après clustering
    plot_format: str = DEFAULT_PLOT_FORMAT
    dpi: int = DEFAULT_DPI
    figures: dict = field(default_factory=dict)  # Paramètres par figure


@dataclass
class GPUConfig:
    enabled: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class PopulationMappingConfig:
    """Configuration Section 10 — Mapping des populations via MFI de référence."""

    enabled: bool = True
    ref_mfi_dir: str = "Data/Ref MFI"
    cache_dir: str = "output/ref_mfi_parquet_cache"
    # Distance et mapping
    distance_percentile: int = 60
    include_scatter: bool = True
    normalization_method: str = "range"  # range | zscore
    mapping_method: str = "cosine_prior"  # M12 recommandé ELN 2022
    unknown_threshold_mode: str = "auto_otsu"  # auto_otsu | percentile | mean_std
    hard_limit_factor: float = 5.0
    prior_mode: str = "log10_cubed"
    # Transformation des CSV de référence
    transform_method: str = "arcsinh"  # arcsinh | logicle | none
    arcsinh_cofactor: float = 5.0
    apply_to_scatter: bool = False
    # Stats biologiques (Mahalanobis / KNN)
    knn_sample_size: int = 2000
    knn_k: int = 15
    cov_reg_alpha: float = 1e-4
    total_knn_points: int = 15000
    # Blast detection
    blast_enabled: bool = True
    blast_suspect_categories: List[str] = field(
        default_factory=lambda: ["BLAST_HIGH", "BLAST_MODERATE"]
    )
    # Visualisation interactive (Plotly)
    viz_interactive: bool = True
    viz_max_points: int = 50_000
    # Couleurs population
    population_colors: dict = field(
        default_factory=lambda: {
            "Granulo": "#e26f1a",
            "Granulocytes": "#e26f1a",
            "Hématogone 34+": "#9467bd",
            "Hematogones19+": "#2ca02c",
            "Ly T_NK": "#17becf",
            "Lymphos B": "#1f77b4",
            "Lymphos": "#aec7e8",
            "Plasmo": "#d62728",
            "Unknown": "#7f7f7f",
        }
    )


@dataclass
class PipelineConfig:
    """Configuration complète du pipeline FlowSOM Pro."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    pregate: PregateConfig = field(default_factory=PregateConfig)
    flowsom: FlowSOMConfig = field(default_factory=FlowSOMConfig)
    auto_clustering: AutoClusteringConfig = field(default_factory=AutoClusteringConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    markers: MarkersConfig = field(default_factory=MarkersConfig)
    downsampling: DownsamplingConfig = field(default_factory=DownsamplingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    population_mapping: PopulationMappingConfig = field(
        default_factory=PopulationMappingConfig
    )

    # ------------------------------------------------------------------
    # Constructeurs alternatifs
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PipelineConfig":
        """
        Charge la configuration depuis un fichier YAML.

        Args:
            yaml_path: Chemin vers le fichier YAML.

        Returns:
            PipelineConfig initialisé à partir du YAML.

        Raises:
            FileNotFoundError: Si le fichier YAML n'existe pas.
            ImportError: Si PyYAML n'est pas installé.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML requis: pip install pyyaml") from exc

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> "PipelineConfig":
        """Construit un PipelineConfig depuis un dictionnaire YAML brut."""
        cfg = cls()
        p = raw.get("paths", {})
        if p.get("healthy_folder"):
            cfg.paths.healthy_folder = p["healthy_folder"]
        if p.get("patho_folder"):
            cfg.paths.patho_folder = p["patho_folder"]
        if p.get("output_dir"):
            cfg.paths.output_dir = p["output_dir"]

        an = raw.get("analysis", {})
        if "compare_mode" in an:
            cfg.analysis.compare_mode = bool(an["compare_mode"])

        pg = raw.get("pregate", {})
        for attr in (
            "apply",
            "mode",
            "mode_blastes_vs_normal",
            "viable",
            "singlets",
            "cd45",
            "cd34",
        ):
            if attr in pg:
                setattr(cfg.pregate, attr, pg[attr])

        pg_adv = raw.get("pregate_advanced", {})
        mapping_pregate = {
            "debris_min_percentile": "debris_min_percentile",
            "debris_max_percentile": "debris_max_percentile",
            "doublets_ratio_min": "doublets_ratio_min",
            "doublets_ratio_max": "doublets_ratio_max",
            "cd45_threshold_percentile": "cd45_threshold_percentile",
            "cd34_threshold_percentile": "cd34_threshold_percentile",
            "cd34_use_ssc_filter": "cd34_use_ssc_filter",
            "cd34_ssc_max_percentile": "cd34_ssc_max_percentile",
        }
        for yaml_key, attr in mapping_pregate.items():
            if yaml_key in pg_adv:
                setattr(cfg.pregate, attr, pg_adv[yaml_key])

        fs = raw.get("flowsom", {})
        for attr in (
            "xdim",
            "ydim",
            "rlen",
            "n_metaclusters",
            "learning_rate",
            "sigma",
            "n_iterations",
            "seed",
        ):
            if attr in fs:
                setattr(cfg.flowsom, attr, fs[attr])

        ac = raw.get("auto_clustering", {})
        for attr in (
            "enabled",
            "min_clusters",
            "max_clusters",
            "n_bootstrap",
            "sample_size_bootstrap",
            "min_stability_threshold",
            "weight_stability",
            "weight_silhouette",
        ):
            if attr in ac:
                setattr(cfg.auto_clustering, attr, ac[attr])

        tr = raw.get("transform", {})
        for attr in ("method", "cofactor", "apply_to_scatter"):
            if attr in tr:
                setattr(cfg.transform, attr, tr[attr])

        no = raw.get("normalize", {})
        if "method" in no:
            cfg.normalize.method = no["method"]

        mk = raw.get("markers", {})
        if "exclude_scatter" in mk:
            cfg.markers.exclude_scatter = bool(mk["exclude_scatter"])
        if "exclude_additional" in mk:
            cfg.markers.exclude_additional = list(mk["exclude_additional"] or [])
        if "keep_area_only" in mk:
            cfg.markers.keep_area_only = bool(mk["keep_area_only"])

        ds = raw.get("downsampling", {})
        for attr in ("enabled", "max_cells_per_file", "max_cells_total"):
            if attr in ds:
                setattr(cfg.downsampling, attr, ds[attr])

        viz = raw.get("visualization", {})
        for attr in ("save_plots", "plot_format", "dpi"):
            if attr in viz:
                setattr(cfg.visualization, attr, viz[attr])

        gp = raw.get("gpu", {})
        if "enabled" in gp:
            cfg.gpu.enabled = bool(gp["enabled"])

        lg = raw.get("logging", {})
        if "level" in lg:
            cfg.logging.level = lg["level"]

        pm = raw.get("population_mapping", {})
        if pm:
            for attr in (
                "enabled",
                "ref_mfi_dir",
                "cache_dir",
                "distance_percentile",
                "include_scatter",
                "normalization_method",
                "mapping_method",
                "unknown_threshold_mode",
                "hard_limit_factor",
                "prior_mode",
                "transform_method",
                "arcsinh_cofactor",
                "apply_to_scatter",
                "knn_sample_size",
                "knn_k",
                "cov_reg_alpha",
                "total_knn_points",
                "blast_enabled",
                "viz_interactive",
                "viz_max_points",
            ):
                if attr in pm:
                    setattr(cfg.population_mapping, attr, pm[attr])
            if "blast_suspect_categories" in pm:
                cfg.population_mapping.blast_suspect_categories = list(
                    pm["blast_suspect_categories"]
                )
            if "population_colors" in pm:
                cfg.population_mapping.population_colors = dict(pm["population_colors"])

        # Validation
        cfg._validate()
        return cfg

    @classmethod
    def from_args(cls, args: Any, yaml_path: Optional[str] = None) -> "PipelineConfig":
        """
        Construit la configuration depuis des arguments CLI (argparse.Namespace).
        Si yaml_path est fourni, charge le YAML d'abord, puis surcharge avec args.

        Args:
            args: Namespace argparse.
            yaml_path: Chemin optionnel vers un fichier YAML.

        Returns:
            PipelineConfig.
        """
        cfg = cls.from_yaml(yaml_path) if yaml_path else cls()

        # Surcharger avec les args CLI (seulement si explicitement définis)
        if getattr(args, "healthy_folder", None):
            cfg.paths.healthy_folder = args.healthy_folder
        if getattr(args, "patho_folder", None):
            cfg.paths.patho_folder = args.patho_folder
        if getattr(args, "output", None):
            cfg.paths.output_dir = args.output

        if getattr(args, "compare_mode", None) is not None:
            cfg.analysis.compare_mode = args.compare_mode

        for attr_args, attr_cfg in (
            ("xdim", "xdim"),
            ("ydim", "ydim"),
            ("n_metaclusters", "n_metaclusters"),
            ("learning_rate", "learning_rate"),
            ("sigma", "sigma"),
            ("n_iterations", "n_iterations"),
            ("seed", "seed"),
        ):
            val = getattr(args, attr_args, None)
            if val is not None:
                setattr(cfg.flowsom, attr_cfg, val)

        if getattr(args, "transform", None):
            cfg.transform.method = args.transform
        if getattr(args, "cofactor", None) is not None:
            cfg.transform.cofactor = args.cofactor
        if getattr(args, "normalize", None):
            cfg.normalize.method = args.normalize

        if getattr(args, "downsample", None) is not None:
            cfg.downsampling.enabled = args.downsample
        if getattr(args, "max_cells_per_file", None) is not None:
            cfg.downsampling.max_cells_per_file = args.max_cells_per_file
        if getattr(args, "max_cells_total", None) is not None:
            cfg.downsampling.max_cells_total = args.max_cells_total

        if getattr(args, "save_plots", None) is not None:
            cfg.visualization.save_plots = args.save_plots
        if getattr(args, "plot_format", None):
            cfg.visualization.plot_format = args.plot_format
        if getattr(args, "dpi", None) is not None:
            cfg.visualization.dpi = args.dpi

        if getattr(args, "use_gpu", None) is not None:
            cfg.gpu.enabled = args.use_gpu

        cfg._validate()
        return cfg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validation des paramètres critiques."""
        if self.pregate.mode not in ("auto", "manual"):
            warnings.warn(
                f"pregate.mode '{self.pregate.mode}' inconnu. Valeurs acceptées: 'auto', 'manual'. "
                "Basculement sur 'auto'."
            )
            self.pregate.mode = "auto"

        if self.transform.method not in ("arcsinh", "logicle", "log10", "none"):
            warnings.warn(
                f"transform.method '{self.transform.method}' inconnu. Basculement sur 'logicle'."
            )
            self.transform.method = "logicle"

        if self.normalize.method not in ("zscore", "minmax", "none"):
            warnings.warn(
                f"normalize.method '{self.normalize.method}' inconnu. Basculement sur 'zscore'."
            )
            self.normalize.method = "zscore"

        if self.flowsom.xdim < 2 or self.flowsom.ydim < 2:
            raise ValueError("flowsom.xdim et flowsom.ydim doivent être >= 2")

        if self.flowsom.n_metaclusters < 2:
            raise ValueError("flowsom.n_metaclusters doit être >= 2")

        if self.pregate.mode_blastes_vs_normal and not self.analysis.compare_mode:
            warnings.warn(
                "pregate.mode_blastes_vs_normal nécessite analysis.compare_mode=True. "
                "Désactivation du mode asymétrique."
            )
            self.pregate.mode_blastes_vs_normal = False

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Sérialise la configuration en dictionnaire (pour JSON/YAML export)."""
        from dataclasses import asdict

        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"PipelineConfig("
            f"mode={self.pregate.mode}, "
            f"transform={self.transform.method}, "
            f"grid={self.flowsom.xdim}x{self.flowsom.ydim}, "
            f"metaclusters={self.flowsom.n_metaclusters}, "
            f"compare_mode={self.analysis.compare_mode})"
        )
