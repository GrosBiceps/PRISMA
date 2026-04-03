# FlowSOM Analyzer Pro 🧬

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FlowSOM Analyzer Pro** est une boîte à outils Desktop (GUI) et Ligne de Commande (CLI) ultra-optimisée pour l'analyse de cytométrie en flux, orientée vers les workflows MRD (Maladie Résiduelle Minime).

Elle couvre l'intégralité de la chaîne d'analyse : de l'importation de millions de cellules (fichiers FCS), au pré-gating automatique, au clustering FlowSOM, jusqu'à l'interprétation MRD et l'exportation de rapports interactifs.

Récemment refondue pour des performances "Entreprise", l'application est capable de traiter un pipeline complet sur plus de **2 millions de cellules en moins de 90 secondes sur un processeur standard (CPU)**, sans figer l'interface.

![Aperçu de l'interface](https://via.placeholder.com/800x450.png?text=Insérer+une+capture+d'écran+de+l'UI+Catppuccin+ici) *(Remplacer par une capture d'écran du nouveau thème Catppuccin)*

---

## ✨ Nouveautés & Points Forts

- ⚡ **Performances Extrêmes (Pur CPU) :** Algorithmique vectorisée (NumPy) et parallélisée (`joblib`). Finis les longs temps d'attente, le pipeline s'exécute en un temps record.
- 💾 **Checkpointing Intelligent :** Mise en cache automatique des étapes lourdes (Gating, SOM) sur le disque. Modifiez un paramètre MRD et obtenez le résultat en 1 seconde.
- 🖥️ **Interface Asynchrone & Moderne :** Architecture *Producer-Consumer*. Les graphiques lourds se génèrent en arrière-plan sans bloquer l'UI. Le tout habillé d'un élégant thème sombre **Catppuccin Mocha**.
- 📊 **Sorties Multi-Formats :** FCS annotés, CSV (stats, MFI), JSON, graphiques (PNG/PDF ultra-rapides via Hexbin) et rapports HTML interactifs (Plotly).

---

## 🏗️ Structure du Projet

L'architecture est modulaire, séparant proprement la logique métier (Backend) de l'interface graphique (Frontend).

```text
flowsom_pipeline_pro/
|-- cli/                # Interface en ligne de commande (automatisation)
|   |-- main.py
|   `-- parsers.py
|-- config/             # Fichiers YAML de configuration
|   |-- constants.py
|   |-- default_config.yaml
|   |-- mrd_config.yaml
|   `-- pipeline_config.py
|-- gui/                # L'application PyQt5 (Frontend)
|   |-- main_window.py
|   |-- styles.py       # Thème Catppuccin (QSS)
|   |-- workers.py      # Threads asynchrones (PipelineWorker, PlottingWorker)
|   |-- tabs/
|   |   `-- home_tab.py # Interface Wizard 4 étapes
|   `-- widgets/        # Composants UI réutilisables
|       |-- mrd_gauge.py
|       `-- mrd_node_table.py
|-- src/                # Cœur métier / Data Science (Backend)
|   |-- analysis/       # Calculs cliniques (MRD)
|   |-- core/           # Mathématiques (Gating GMM/RANSAC, SOM)
|   |-- io/             # Lecture/Écriture des fichiers
|   |-- models/         # Modèles de données
|   |-- pipeline/       # Orchestrateur (pipeline_executor.py avec Checkpoints)
|   |-- services/
|   |-- utils/          # Outils (Logger, Kaleido persistant)
|   `-- visualization/  # Génération des graphiques (Hexbins, Plotly)
|-- launch_gui.py       # Point d'entrée GUI (sécurisé pour stdout/stderr)
|-- run_pipeline.py     # Lanceur Dev CLI
|-- flowsom_gui.spec    # Configuration PyInstaller
|-- build.bat           # Script de compilation Windows
|-- requirements.txt
`-- setup.py
```

## 🚀 Installation

### Prérequis

- Python 3.11 recommandé
- Windows 10/11 recommandé pour la GUI

### Installation Rapide (Environnement Virtuel)

```bash
python -m venv .venv
# Sous Windows :
.venv\Scripts\activate
# Sous Linux/Mac :
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Installation Editable (Pour le développement)

```bash
pip install -e .
```

## 🖥️ Modes D'exécution

### 1) Interface GUI (Recommandée)

L'interface graphique est un "Wizard" (assistant) en 4 étapes fluides : Importation, Paramétrage, Exécution, Résultats.

```bash
python launch_gui.py
```

Note : launch_gui.py inclut une gestion sécurisée de stdout/stderr pour éviter les crashs en mode fenêtre figée, et limite les logs bruyants de Chromium/Kaleido.

### 2) Interface CLI (Module installé)

Parfait pour les traitements en lot (batch) et l'automatisation sur serveur.

```bash
python -m flowsom_pipeline_pro --healthy-folder "Data/NBM" --output "Results"
```

### 3) Lanceur de Développement (CLI)

Utile si l'application n'est pas installée globalement via pip. Ce script corrige également les problèmes de découpage de chemins Windows contenant des espaces.

```bash
python run_pipeline.py --healthy-folder "Data/NBM" --output "Results"
```

## ⚙️ Exemples D'utilisation CLI

```bash
# Exécution avec auto-détection d'un fichier YAML de configuration
python -m flowsom_pipeline_pro

# Comparaison Cohorte Saine vs Patient Pathologique
python -m flowsom_pipeline_pro \
  --healthy-folder "Data/NBM" \
  --patho-folder "Data/Patho" \
  --compare-mode \
  --output "Results_Comparison"

# Personnalisation de la grille SOM (15x15) et des métaclusters
python -m flowsom_pipeline_pro \
  --healthy-folder "Data/NBM" \
  --xdim 15 --ydim 15 --n-metaclusters 20 --n-iterations 20

# Mode Auto-Clustering (Recherche du k optimal)
python -m flowsom_pipeline_pro \
  --healthy-folder "Data/NBM" \
  --auto-clustering --min-clusters 5 --max-clusters 40
```

Pour voir tous les arguments disponibles, consultez cli/parsers.py.

## 📁 Fichiers De Configuration (YAML)

Le pipeline est hautement configurable sans toucher au code via deux fichiers principaux :

- config/default_config.yaml
- config/mrd_config.yaml

(⚠️ Important : default_config.yaml peut contenir des chemins absolus spécifiques à une machine. Pensez à les adapter avant l'exécution sur un nouveau poste de travail).

## 📦 Build EXE pour Windows

Pour distribuer l'application dans des laboratoires sans nécessiter l'installation de Python, utilisez le script de build basé sur PyInstaller. Le processus génère un dossier autonome (mode --onedir) favorisant un démarrage instantané.

Via le script batch :

```dos
build.bat
```

Via l'appel PyInstaller direct :

```bash
python -m PyInstaller flowsom_gui.spec -y
```

Le résultat sera disponible dans dist/FlowSOMAnalyzer/FlowSOMAnalyzer.exe.

## 🛠️ Dépannage

- Crash au démarrage de l'exécutable (GUI) : Utilisez toujours la logique de bootstrap de launch_gui.py qui protège stdout/stderr (indispensable en mode windowed sous Windows).
- Graphiques non générés ou trop de logs Chromium : Assurez-vous que le paquet kaleido est bien installé. Le lanceur launch_gui.py s'occupe de masquer les avertissements normaux de Chromium.
- Arguments CLI non reconnus (Chemins Windows) : Les chemins contenant des espaces doivent impérativement être entourés de guillemets "C:\Mon Dossier".
- Problème d'UI (Couleurs du tableau MRD) : Les couleurs spécifiques du tableau MRD sont gérées dynamiquement dans gui/widgets/mrd_node_table.py. En cas de conflit d'affichage, vérifiez ce fichier avant la feuille de style globale (gui/styles.py).

## ⚠️ Avertissement Clinique

Ce logiciel est un outil de recherche et une aide à l'analyse et à la visualisation de données de cytométrie. Il ne remplace en aucun cas l'interprétation clinique d'un expert biologiste ou médecin, ni les procédures d'assurance qualité (AQ) du laboratoire, ni les recommandations médicales institutionnelles.
