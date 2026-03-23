"""
run_pipeline.py — Lanceur de développement (sans pip install).

Usage depuis n'importe où :
    python "C:/Users/Florian Travail/Documents/FlowSom/Perplexity/flowsom_pipeline_pro/run_pipeline.py" --verbose
    python run_pipeline.py --config config_flowsom.yaml --verbose
    python run_pipeline.py --healthy "Data/Moelle normale" --patho "Data/Patho" --verbose

Ce script injecte le répertoire parent dans sys.path afin que
`import flowsom_pipeline_pro` fonctionne sans installation préalable.
"""

import sys
from pathlib import Path

# ── Injection du répertoire parent pour que `flowsom_pipeline_pro` soit trouvable ──
_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

# ── Réparer les chemins cassés par l'espace dans "Florian Travail" ────────────
# PowerShell sans guillemets split "C:\Users\Florian Travail\..." en deux tokens.
# On réassemble: si un token suit un flag connu (--config, --healthy-folder, etc.)
# et que le token suivant ne commence pas par "--", on les colle avec un espace.
_PATH_FLAGS = {
    "--config",
    "--healthy-folder",
    "--patho-folder",
    "--output",
    "--healthy",
    "--patho",
}
_fixed: list = [sys.argv[0]]
i = 1
while i < len(sys.argv):
    token = sys.argv[i]
    _fixed.append(token)
    if token in _PATH_FLAGS and i + 1 < len(sys.argv):
        # Récupère la valeur et colle tous les fragments suivants qui ne sont
        # pas des flags (ils font partie du chemin avec espace)
        i += 1
        value = sys.argv[i]
        while i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
            i += 1
            value = value + " " + sys.argv[i]
        _fixed.append(value)
    i += 1
sys.argv = _fixed

# ── Si --config absent de la ligne de commande, injecter le default_config.yaml ──
_DEFAULT_CONFIG = (
    _PACKAGE_PARENT / "flowsom_pipeline_pro" / "config" / "default_config.yaml"
)
if "--config" not in sys.argv and _DEFAULT_CONFIG.exists():
    sys.argv.extend(["--config", str(_DEFAULT_CONFIG)])

# ── Lancement de la CLI ──────────────────────────────────────────────────────
from flowsom_pipeline_pro.cli.main import main  # noqa: E402

if __name__ == "__main__":
    main()
