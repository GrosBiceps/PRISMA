"""
src/io/__init__.py — Exports publics de la couche IO.
"""

from .fcs_reader import (
    get_fcs_files,
    load_fcs_file,
    load_fcs_files,
    load_as_flow_samples,
)
from .fcs_writer import (
    export_to_fcs,
    export_to_fcs_kaluza,
    add_clustering_columns,
    circular_jitter,
)
from .csv_exporter import (
    export_cells_csv,
    export_statistics_csv,
    export_mfi_matrix_csv,
    export_per_file_csv,
    compute_cluster_statistics,
    extract_date_from_filename,
    add_timepoint_columns,
)
from .json_exporter import (
    export_analysis_metadata,
    build_analysis_metadata,
    export_gating_log,
)

__all__ = [
    # FCS Reader
    "get_fcs_files",
    "load_fcs_file",
    "load_fcs_files",
    "load_as_flow_samples",
    # FCS Writer
    "export_to_fcs",
    "export_to_fcs_kaluza",
    "add_clustering_columns",
    "circular_jitter",
    # CSV Exporter
    "export_cells_csv",
    "export_statistics_csv",
    "export_mfi_matrix_csv",
    "export_per_file_csv",
    "compute_cluster_statistics",
    "extract_date_from_filename",
    "add_timepoint_columns",
    # JSON Exporter
    "export_analysis_metadata",
    "build_analysis_metadata",
    "export_gating_log",
]
