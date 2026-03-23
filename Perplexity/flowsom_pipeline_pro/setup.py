"""
setup.py — Packaging de flowsom_pipeline_pro.

Installation:
    pip install -e .                    # mode développement
    pip install -e ".[gpu]"             # avec support CuPy/RAPIDS
    pip install -e ".[fcs_export]"      # avec export FCS (fcswrite)
    pip install -e ".[full]"            # toutes les dépendances
"""

from setuptools import setup, find_packages
from pathlib import Path

# Lecture du README pour la description longue
_HERE = Path(__file__).parent
_README = (
    (_HERE / "README.md").read_text(encoding="utf-8")
    if (_HERE / "README.md").exists()
    else ""
)

setup(
    name="flowsom_pipeline_pro",
    version="1.0.0",
    description=(
        "Pipeline d'analyse de cytométrie en flux MRD (Maladie Résiduelle Détectable) "
        "basé sur FlowSOM — architecture modulaire production-ready"
    ),
    long_description=_README,
    long_description_content_type="text/markdown",
    author="Florian",
    python_requires=">=3.10",
    # Découverte automatique de tous les sous-paquets
    packages=find_packages(exclude=["tests*", "*.tests*"]),
    # Inclure les fichiers de données (YAML par défaut)
    package_data={
        "flowsom_pipeline_pro": [
            "config/default_config.yaml",
        ],
    },
    # ── Dépendances obligatoires ───────────────────────────────────────────────
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.11",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "anndata>=0.10",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
    # ── Dépendances optionnelles ───────────────────────────────────────────────
    extras_require={
        # FlowSOM principal (saeyslab)
        "flowsom": ["flowsom"],
        # Lecture/écriture FCS
        "fcs": ["flowio"],
        # Export FCS avec colonnes de clustering
        "fcs_export": ["fcswrite"],
        # Transformation logicle via FlowKit
        "logicle": ["flowkit"],
        # Transformations cytométriques (pytometry = scanpy-based)
        "pytometry": ["pytometry"],
        # Accélération GPU (nécessite CuPy + pilotes CUDA)
        "gpu": ["cupy"],
        # Génération de rapports PDF
        "reports": ["reportlab"],
        # Toutes les dépendances (hors GPU)
        "full": [
            "flowsom",
            "flowio",
            "fcswrite",
            "flowkit",
            "pytometry",
            "reportlab",
        ],
    },
    # ── Points d'entrée CLI ───────────────────────────────────────────────────
    entry_points={
        "console_scripts": [
            "flowsom-analyze=flowsom_pipeline_pro.cli.main:main",
        ],
    },
    # ── Classifieurs PyPI ─────────────────────────────────────────────────────
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "flowsom",
        "cytometry",
        "flow cytometry",
        "MRD",
        "AML",
        "bioinformatics",
        "clustering",
    ],
)
