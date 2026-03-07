"""
src/visualization/__init__.py — Exports publics de la couche visualisation.
"""

from .plot_helpers import (
    format_axis,
    apply_dark_style,
    plot_density,
    plot_gating,
    add_gate_rectangle,
    save_figure,
)
from .gating_plots import (
    plot_overview,
    plot_debris_gate,
    plot_singlets_gate,
    plot_cd45_gate,
    plot_cd34_gate,
    generate_all_gating_plots,
    generate_sankey_diagram,
    generate_per_file_sankey,
)
from .flowsom_plots import (
    plot_mfi_heatmap,
    plot_metacluster_sizes,
    plot_umap,
    circular_jitter,
)
from .html_report import generate_html_report

__all__ = [
    # Helpers bas-niveau
    "format_axis",
    "apply_dark_style",
    "plot_density",
    "plot_gating",
    "add_gate_rectangle",
    "save_figure",
    # Gating
    "plot_overview",
    "plot_debris_gate",
    "plot_singlets_gate",
    "plot_cd45_gate",
    "plot_cd34_gate",
    "generate_all_gating_plots",
    "generate_sankey_diagram",
    "generate_per_file_sankey",
    # FlowSOM
    "plot_mfi_heatmap",
    "plot_metacluster_sizes",
    "plot_umap",
    "circular_jitter",
    # HTML Report
    "generate_html_report",
]
