"""
flowsom_pipeline_pro — Pipeline d'analyse de cytométrie en flux MRD.

Point d'entrée public du package :

    from flowsom_pipeline_pro import FlowSOMPipeline, PipelineConfig

    config = PipelineConfig.from_yaml("config_flowsom.yaml")
    result = FlowSOMPipeline(config).execute()
    print(result.summary())
"""

from __future__ import annotations

from flowsom_pipeline_pro.config.pipeline_config import PipelineConfig
from flowsom_pipeline_pro.src.pipeline.pipeline_executor import FlowSOMPipeline

__version__ = "1.0.0"
__author__ = "FlowSOM Pipeline Pro"

__all__ = [
    "FlowSOMPipeline",
    "PipelineConfig",
    "__version__",
]
