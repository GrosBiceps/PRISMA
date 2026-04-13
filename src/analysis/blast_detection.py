"""
blast_detection.py — Scoring et classification phénotypique des nœuds SOM en blastes.

═══════════════════════════════════════════════════════════════════════════════
  FONDEMENTS BIOLOGIQUES ET CLINIQUES
═══════════════════════════════════════════════════════════════════════════════

Ce module implémente un scoring heuristique inspiré de deux référentiels
cliniques majeurs pour la détection des blastes LAM par cytométrie en flux :

  1. Score d'Ogata (Ogata et al., Blood 2006 ; Sandes et al., Cytometry B 2013)
     ─────────────────────────────────────────────────────────────────────────
     Conçu initialement pour le diagnostic des syndromes myélodysplasiques
     (SMD) et adapté à la LAM, ce score évalue la déviation phénotypique des
     blastes par rapport à une moelle normale de référence. Les critères
     biologiques retenus sont :

       • CD34 bright       → Marqueur de cellule souche hématopoïétique /
                              progéniteur immature. Surexprimé dans les LAM
                              à progéniteurs. Poids fort : +3.0

       • CD117 (c-Kit)     → Récepteur de la stem cell factor (SCF). Marqueur
                              clé des progéniteurs myéloïdes précoces (CFU-GM).
                              Complémentaire à CD34 dans les LAM CD34−. Poids : +2.5

       • CD45 dim          → Dans la moelle normale, les blastes sont CD45-
                              faibles (population "blasts gate" sur CD45/SSC).
                              Un signal CD45 très bas est discriminant pour les
                              blastes LAM, conformément au score d'Ogata
                              (critère morphologique CD45-dim). Poids : −2.0
                              (contribution quand valeur en-dessous du plancher
                              de la référence normale)

       • SSC bas           → Faible granularité cytoplasmique. Critère
                              morphologique classique des blastes (taille petite,
                              peu de granules). Aligné sur le quadrant
                              SSC-bas / CD45-dim du score d'Ogata. Poids : −1.0

  2. Recommandations ELN 2022 (Schuurhuis et al., Blood 2018 ; Heuser et al.,
     Leukemia 2022 ; ELN MRD Working Party 2022)
     ─────────────────────────────────────────────────────────────────────────
     L'ELN définit les LAIPs (Leukemia-Associated ImmunoPhénotypes) comme des
     combinaisons anormales de marqueurs permettant de traquer la MRD :

       • HLA-DR positif    → Blaste myéloïde mature (monocytaire ou granulocytaire).
                              Fortement exprimé sur les blastes LAM M4/M5. Poids : +1.5

       • CD33 variable     → Engagement myéloïde (antigène pan-myéloïde).
                              Présent à des niveaux anormaux sur les blastes.
                              Poids : +1.0

       • CD13 variable     → Engagement myéloïde (amino-peptidase N). Souvent
                              co-exprimé avec CD33. Poids : +0.5

       • CD19 / CD3 pos    → Marqueurs lymphoïdes B et T. Leur présence
                              suggère une population normale ou un LAIP
                              lympho-myéloïde rare. Considérés ici comme
                              freins biologiques anti-blaste. Poids : −1.5

═══════════════════════════════════════════════════════════════════════════════
  SCORE COMPOSITE ET SEUILS DE CATÉGORISATION
═══════════════════════════════════════════════════════════════════════════════

Le score composite /10 est calculé comme une somme pondérée de contributions
directionnelles (cf. score_nodes_for_blasts) :

  • Marqueurs positifs (CD34, CD117, HLA-DR, CD33, CD13) :
    Contribuent quand leur valeur normalisée dépasse le plafond de la
    référence (valeur > 1.0 dans l'espace normalisé), c'est-à-dire quand
    ils sont surexprimés par rapport à la moelle normale.

  • Marqueurs négatifs (CD45, SSC) :
    Contribuent quand leur valeur normalisée est sous le plancher de la
    référence (valeur < 0.0), c'est-à-dire quand ils sont sous-exprimés
    (CD45-dim, SSC-bas).

  • Marqueurs frein (CD19, CD3) :
    Contribuent négativement au score quand surexprimés (population non-blaste).

Seuils de catégorisation (heuristiques calibrables) :
──────────────────────────────────────────────────────
  BLAST_HIGH     ≥ 6.0 / 10   Équivalent à ≥2 anomalies majeures (LAIP fort).
                               Interprétation clinique : fort indice de blaste LAM.
                               Recommandation : confirmer par autre méthode.

  BLAST_MODERATE ≥ 3.0 / 10   Équivalent à 1 anomalie majeure + anomalies mineures.
                               Interprétation : phénotype atypique, surveillance requise.

  BLAST_WEAK     > 0.0 / 10   Signal léger. Population atypique mais non décisive.

  NON_BLAST_UNK  = 0.0 / 10   Aucune signature blastique décelable.

⚠  Ces seuils sont des HEURISTIQUES d'initialisation, non des valeurs ELN
   officiellement validées. Ils doivent être calibrés sur une cohorte locale
   via une analyse ROC (AUC blast vs non-blast) avant utilisation diagnostique.
   La procédure recommandée est :
     1. Annoter manuellement ≥50 nœuds (blast / non-blast) sur des cas connus.
     2. Calculer l'AUC du score composite.
     3. Choisir le seuil BLAST_HIGH maximisant sensibilité × spécificité (Youden J).
     4. Mettre à jour BLAST_HIGH_THRESHOLD et BLAST_MODERATE_THRESHOLD ci-dessous.

Références :
  - Ogata K. et al. (2006) Diagnostic utility of flow cytometry in low-grade
    myelodysplastic syndromes. Haematologica.
  - Sandes A.F. et al. (2013) Flow cytometric diagnosis of myelodysplastic
    syndromes using the Ogata score. Cytometry B Clin Cytom.
  - Schuurhuis G.J. et al. (2018) Minimal/measurable residual disease in AML:
    a consensus document from the ELN MRD Working Party. Blood.
  - Heuser M. et al. (2022) ELN MRD Working Party consensus on MRD in AML.
    Leukemia.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from flowsom_pipeline_pro.src.utils.logger import get_logger

_logger = get_logger("analysis.blast_detection")

# ─────────────────────────────────────────────────────────────────────────────
#  Seuils de catégorisation (score /10)
#
#  BLAST_HIGH_THRESHOLD     = 6.0 → ≥2 anomalies phénotypiques majeures (LAIP fort)
#  BLAST_MODERATE_THRESHOLD = 2.0 → blastes matures CD34- (CD45-dim/HLA-DR), anciennement 3.0
#  BLAST_WEAK_THRESHOLD     = 0.0 → signal positif minimal
#
#  ⚠  Valeurs heuristiques — calibrer via ROC sur cohorte locale avant usage clinique.
# ─────────────────────────────────────────────────────────────────────────────
BLAST_HIGH_THRESHOLD = 6.0
BLAST_MODERATE_THRESHOLD = 2.0
BLAST_WEAK_THRESHOLD = 0.0


def build_blast_weights(marker_names: List[str]) -> Dict[str, float]:
    """
    Construit le dictionnaire de poids pour le scoring blast ELN 2022 / Ogata.

    Chaque poids est assigné par correspondance de pattern sur le nom du marqueur.
    Les valeurs sont calibrées selon leur importance diagnostique dans la LAM
    (cf. module docstring pour le rationnel clinique complet).

    Hiérarchie des poids positifs (marqueurs progéniteurs / myéloïdes) :
      +3.0  CD34      — progéniteur hématopoïétique (Ogata critère 1)
      +2.5  CD117     — c-Kit, progéniteur myéloïde (Ogata critère 2)
      +1.5  HLA-DR    — blaste myéloïde mature (ELN 2022 LAIP)
      +1.0  CD33      — engagement myéloïde (ELN 2022 LAIP)
      +0.5  CD13      — engagement myéloïde secondaire (ELN 2022 LAIP)

    Poids négatifs (morphologie / phénotype inverse — contribution si sous-exprimé) :
      −2.0  CD45      — CD45-dim discrimine les blastes (score Ogata)
      −1.5  CD19/CD3  — marqueurs lymphoïdes → anti-blaste (frein biologique)
      −1.0  SSC       — faible granularité cytoplasmique = morphologie blaste (Ogata)

    Args:
        marker_names: Liste des noms de marqueurs présents dans le panel.

    Returns:
        Dict {nom_marqueur: poids_float} — poids = 0.0 pour les marqueurs
        non reconnus (neutres, sans contribution au score).

    Note:
        La correspondance est insensible à la casse et tolère les suffixes
        courants (ex: "CD34-FITC", "CD34_BV421", "CD117/PE").
    """
    weights: Dict[str, float] = {}

    for name in marker_names:
        upper = name.upper()

        if "CD34" in upper:
            # Progéniteur hématopoïétique — marqueur majeur LAIP (Ogata critère 1)
            weights[name] = +3.0
        elif "CD117" in upper or "CKIT" in upper:
            # c-Kit / SCF receptor — progéniteur myéloïde (Ogata critère 2)
            weights[name] = +2.5
        elif "CD45" in upper:
            # CD45-dim = signature blaste classique (score Ogata, axe CD45/SSC)
            # Poids négatif : contribue au score quand valeur < plancher référence
            weights[name] = -2.0
        elif "HLAD" in upper or "HLA-DR" in upper:
            # HLA-DR positif sur blaste myéloïde = LAIP ELN 2022
            weights[name] = +1.5
        elif "CD33" in upper:
            # Antigène pan-myéloïde — LAIP ELN 2022 (co-expression anormale)
            weights[name] = +1.0
        elif "CD13" in upper:
            # Amino-peptidase N — marqueur myéloïde secondaire (LAIP ELN 2022)
            weights[name] = +0.5
        elif "CD19" in upper or ("CD3" in upper and "CD34" not in upper):
            # Marqueurs lymphoïdes B (CD19) et T (CD3) — frein anti-blaste
            # Leur présence dans un nœud oriente vers une population normale
            weights[name] = -1.5
        elif "SSC" in upper:
            # Faible granularité (SSC-bas) = morphologie de blaste (Ogata)
            # Poids négatif : contribue quand valeur < plancher référence
            weights[name] = -1.0
        else:
            # Marqueur non reconnu — contribution nulle (neutre)
            weights[name] = 0.0

    return weights


def score_nodes_for_blasts(
    X_norm: np.ndarray,
    marker_names: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Calcule le score blast /10 pour chaque nœud SOM à partir de médianes z-scorées.

    ── Espace de représentation : z-score par rapport à la moelle normale ───────

    X_norm contient des **z-scores** calculés par rapport aux statistiques de la
    population de référence (moelle normale / NBM) :

        z[j] = (mediane_nœud[j] − median_ref[j]) / std_ref[j]

    Dans cet espace :
      • z ≈ 0   → le marqueur est au niveau médian de la moelle normale
      • z > +0.5 → surexpression au-dessus du NBM (seuil de sensibilité)
      • z < −0.5 → sous-expression en-dessous du NBM (CD45-dim, SSC-bas)

    ── Logique de contribution directionnelle (3 règles) ───────────────────────

    Seuil de déclenchement : ±0.5 SD (abaissé vs l'ancienne valeur de 1.0 SD
    pour capturer les blastes matures ayant perdu une partie de leur LAIP).

      1. Marqueurs blastiques à poids positif (CD34, CD117, HLA-DR, CD33, CD13) :
           → +points si z > +0.5 (surexpression dès 0.5 SD au-dessus du NBM).
           → Contribution = w × max(0, z − 0.5)
           → Interprétation ELN 2022 : CD34++ ou CD117++ = LAIP positif.

      2. Marqueurs de maturation à poids négatif (CD45, SSC) — récompense :
           → +points si z < −0.5 (sous-expression caractéristique des blastes).
           → Contribution = |w| × max(0, −z − 0.5)
           → Interprétation Ogata : CD45-dim / SSC-bas = zone « blasts gate ».

      3. Marqueurs de maturation à poids négatif (CD45, SSC) — PÉNALITÉ :
           → −points si z > 0 (surexpression = cellule mature normale).
           → Pénalité = |w| × max(0, z)
           → Élimine les faux positifs : lymphocytes CD45-bright, granulocytes
             SSC-high, monocytes accumuleraient sinon des points sur d'autres
             marqueurs sans être punis par leur CD45/SSC normal.

    ── Calibration attendue (arcsinh/5, référence NBM médiane+std) ─────────────

      Blaste typique  (CD34++, CD117+, CD45-dim, SSC-bas) → score ≈ 6–8 / 10
      Lymphocyte sain (CD45-bright, SSC-bas, CD34−)       → score ≈ 0–1 / 10
      Granulocyte sain (SSC-high, CD45+, CD34−)           → score ≈ 0   / 10

    ── Normalisation du score final ────────────────────────────────────────────

    Score brut clampé à 0 (pas de score négatif), puis normalisé par le maximum
    théorique (somme des |poids| non nuls) × 10, et clampé dans [0, 10].

    Args:
        X_norm: Matrice **z-scorée** [n_nodes, n_markers].
                Axe 0 = nœuds SOM, axe 1 = marqueurs.
                Produite par compute_reference_normalization() avec mode="zscore".
                z ≈ 0 = niveau NBM ; z > 0.5 = surexpression ; z < −0.5 = sous-expr.
        marker_names: Noms des marqueurs (colonnes de X_norm, dans le même ordre).
        weights: Poids pré-calculés (optionnel).
                 Si None, calculés automatiquement via build_blast_weights().

    Returns:
        np.ndarray de forme (n_nodes,), valeurs dans [0.0, 10.0].
        Score = 0.0 → aucune déviation blastique détectée (ou cellule mature pénalisée).
        Score ≈ 6–8 → signature blastique typique (CD34++ + CD117++ + CD45-dim).

    Example:
        >>> ref_med, ref_std = compute_reference_stats(X_nbm, marker_names)
        >>> X_zscore = (node_medians - ref_med) / ref_std
        >>> scores = score_nodes_for_blasts(X_zscore, marker_names)
    """
    if weights is None:
        weights = build_blast_weights(marker_names)

    # Vecteur de poids aligné sur marker_names
    W = np.array([weights.get(m, 0.0) for m in marker_names])

    # Dénominateur = somme des poids absolus (pour normaliser en /10)
    max_theoretical = max(sum(abs(w) for w in weights.values() if w != 0), 1e-6)

    scores_raw = np.zeros(X_norm.shape[0])

    for j, (marker, w) in enumerate(zip(marker_names, W)):
        z = X_norm[:, j]
        if w > 0:
            # Baisse du seuil à 0.5 SD pour capter les blastes même s'ils sont peu brillants
            scores_raw += w * np.maximum(0.0, z - 0.5)

        elif w < 0:
            # CD45-dim ou SSC-bas (z < -0.5) DONNE des points
            scores_raw += (-w) * np.maximum(0.0, -z - 0.5)

            # PÉNALITÉ : CD45-bright ou SSC-haut (z > 0) RETIRE des points !
            # Élimine les cellules matures (lymphocytes, monocytes, granulos normaux)
            scores_raw -= (-w) * np.maximum(0.0, z)

    # Normalisation stricte pour éviter les scores négatifs
    scores_raw = np.maximum(0.0, scores_raw)
    scores_10 = np.clip(scores_raw / max_theoretical * 10.0, 0.0, 10.0)
    return scores_10


def categorize_blast_score(
    score: float,
    high_thresh: float = BLAST_HIGH_THRESHOLD,
    mod_thresh: float = BLAST_MODERATE_THRESHOLD,
    weak_thresh: float = BLAST_WEAK_THRESHOLD,
) -> str:
    """
    Classifie un score blast /10 en catégorie clinique ELN 2022 / Ogata.

    ── Correspondance avec les critères cliniques ───────────────────────────────

    BLAST_HIGH (≥ high_thresh, défaut 6.0) :
      Correspond à ≥2 anomalies phénotypiques majeures simultanées (ex: CD34++
      ET CD117++, ou CD34++ ET CD45-dim). Équivalent à un LAIP fort selon ELN 2022
      (Schuurhuis et al., Blood 2018, Table 2). Recommande une confirmation par
      morphologie ou biologie moléculaire.

    BLAST_MODERATE (≥ mod_thresh, défaut 2.0) :
      1 anomalie majeure ou combinaison d'anomalies mineures. Abaissé à 2.0 pour
      capturer les blastes matures ayant perdu le CD34 mais conservant CD45-dim
      (+2.0 pts) ou HLA-DR (+1.5 pts) — typique des leucémies massives avancées.
      Configurable via blast_phenotype_filter.moderate_threshold dans mrd_config.yaml.

    BLAST_WEAK (> weak_thresh, défaut 0.0) :
      Signal faible — population atypique sans seuil d'alarme clinique. Utile
      pour la traçabilité et la détection précoce sur séries longitudinales.

    NON_BLAST_UNK (= 0.0) :
      Aucune signature détectable. Nœud très probablement composé de cellules
      hématopoïétiques matures normales (granulocytes, lymphocytes, monocytes).

    ⚠  Rappel : ces seuils sont heuristiques. Calibrer via ROC avant usage
       diagnostique (cf. module docstring).

    Args:
        score: Score blast dans [0.0, 10.0] produit par score_nodes_for_blasts().
        high_thresh: Seuil BLAST_HIGH (défaut : constante globale BLAST_HIGH_THRESHOLD).
        mod_thresh: Seuil BLAST_MODERATE (défaut : constante globale BLAST_MODERATE_THRESHOLD).
        weak_thresh: Seuil BLAST_WEAK (défaut : constante globale BLAST_WEAK_THRESHOLD).

    Returns:
        Catégorie : "BLAST_HIGH" | "BLAST_MODERATE" | "BLAST_WEAK" | "NON_BLAST_UNK".
    """
    if score >= high_thresh:
        # ≥2 anomalies majeures — LAIP fort (ELN 2022)
        return "BLAST_HIGH"
    elif score >= mod_thresh:
        # 1 anomalie majeure ou combinaison mineures — phénotype intermédiaire
        return "BLAST_MODERATE"
    elif score > weak_thresh:
        # Signal léger — population atypique à surveiller
        return "BLAST_WEAK"
    # Score nul — aucune déviation phénotypique par rapport à la moelle normale
    return "NON_BLAST_UNK"


def build_blast_score_dataframe(
    node_ids: np.ndarray,
    X_norm: np.ndarray,
    marker_names: List[str],
    cell_counts_per_node: Optional[Dict[int, int]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Construit le DataFrame complet de scoring blast ELN 2022 / Ogata pour un
    ensemble de nœuds SOM.

    Wrapper de haut niveau combinant score_nodes_for_blasts() et
    categorize_blast_score(). Utilisé principalement pour l'export JSON,
    les rapports HTML/PDF et la visualisation des candidats blastiques.

    Le DataFrame résultant est trié par blast_score décroissant pour faciliter
    l'inspection manuelle des nœuds les plus suspects.

    Args:
        node_ids: IDs des nœuds SOM à scorer, shape (n_nodes,).
                  Doit correspondre en ordre aux lignes de X_norm.
        X_norm: Matrice normalisée [n_nodes, n_markers] dans l'espace de référence.
        marker_names: Noms des marqueurs (colonnes de X_norm).
        cell_counts_per_node: {node_id: n_cells} pour enrichir le rapport.
                               Permet d'évaluer la robustesse statistique du score
                               (un nœud avec 5 cellules et score élevé est moins
                               fiable qu'un nœud avec 500 cellules).
        weights: Poids pré-calculés (optionnel — auto-calculés si None).

    Returns:
        pd.DataFrame trié par blast_score décroissant, avec colonnes :
          - node_id, blast_score, blast_category
          - n_cells (si cell_counts_per_node fourni)
          - {marqueur}_M8 : valeur normalisée par marqueur (suffixe _M8 = espace
            de normalisation, où M8 = matrice de référence 8 populations)
    """
    if weights is None:
        weights = build_blast_weights(marker_names)

    scores = score_nodes_for_blasts(X_norm, marker_names, weights)
    categories = [categorize_blast_score(float(s)) for s in scores]

    records: Dict = {
        "node_id": node_ids,
        "blast_score": np.round(scores, 2),
        "blast_category": categories,
    }

    if cell_counts_per_node:
        records["n_cells"] = [cell_counts_per_node.get(int(nid), 0) for nid in node_ids]

    df = pd.DataFrame(records)

    # Ajouter les valeurs normalisées par marqueur (suffixe _M8 = espace référence)
    for j, m in enumerate(marker_names):
        df[f"{m}_M8"] = np.round(X_norm[:, j], 3)

    df = df.sort_values("blast_score", ascending=False).reset_index(drop=True)

    # Log du résumé par catégorie
    for cat in ["BLAST_HIGH", "BLAST_MODERATE", "BLAST_WEAK", "NON_BLAST_UNK"]:
        n = int((df["blast_category"] == cat).sum())
        if n > 0:
            _logger.info("  %s: %d nœud(s)", cat, n)

    return df


def compute_reference_stats(
    X_reference: np.ndarray,
    robust: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule les statistiques de la population de référence (moelle normale / NBM)
    nécessaires au z-scoring des nœuds SOM.

    ── Pourquoi le z-score plutôt que min-max ? ─────────────────────────────────

    La normalisation min-max [0, 1] place TOUTES les populations (y compris les
    blastes les plus extrêmes) dans l'intervalle [0, 1]. Dans cet espace,
    score_nodes_for_blasts() ne peut jamais trouver de valeurs > 1 ou < 0,
    donc le score est systématiquement 0 pour tous les nœuds.

    Le z-score par rapport à la médiane + IQR/1.35 de la moelle normale place
    les blastes À L'EXTÉRIEUR de l'espace de référence (z > +1 ou z < −1),
    permettant à la logique directionnelle de score_nodes_for_blasts() de
    détecter les déviations phénotypiques ELN 2022 / Ogata.

    ── Validation sur BLAST110 ───────────────────────────────────────────────────

    Sur BLAST110_100_P1 (arcsinh/5, NBM T1) :
      CD34 Cy55 : médiane_blast = 7.93, médiane_NBM = 4.72, std_NBM = 1.58
                  → z_blast = +2.07  (surexpression > 2σ) ✓
      CD45 KO   : médiane_blast = 6.82, médiane_NBM = 7.88, std_NBM = 1.22
                  → z_blast = −0.87  (CD45-dim modéré)
      SSC-A     : médiane_blast = 8.65, médiane_NBM = 10.63, std_NBM = 0.82
                  → z_blast = −2.42  (SSC-bas significatif) ✓

    Args:
        X_reference: Matrice des cellules/médianes de référence [n_ref, n_markers].
                     Typiquement les nœuds SOM issus des fichiers NBM (moelle normale).
                     Doit être dans le même espace transformé que X_unknown
                     (ex: arcsinh cofactor=5).
        robust: Si True (défaut), utilise la médiane + IQR/1.35 (pseudo-std robuste).
                Plus résistant aux outliers que la moyenne + std.
                Si False, utilise la moyenne + std standard.

    Returns:
        Tuple (ref_center, ref_scale) :
          - ref_center : vecteur [n_markers] — médiane (ou moyenne) par marqueur.
          - ref_scale  : vecteur [n_markers] — IQR/1.35 (ou std), clampé à 0.01
                         pour éviter les divisions par zéro.
    """
    # Utiliser les versions nan-safe pour gérer les marqueurs absents dans
    # certains fichiers NBM (ex: CD34 absent d'un tube → NaN après concat)
    if robust:
        ref_center = np.nanmedian(X_reference, axis=0)
        q75 = np.nanpercentile(X_reference, 75, axis=0)
        q25 = np.nanpercentile(X_reference, 25, axis=0)
        ref_scale = (q75 - q25) / 1.35  # pseudo-std robuste (IQR / 1.35 ≈ σ pour gaussienne)
    else:
        ref_center = np.nanmean(X_reference, axis=0)
        ref_scale  = np.nanstd(X_reference, axis=0)

    # Eviter div/0 :
    #   - marqueurs constants dans la référence (scale ≈ 0)
    #   - marqueurs absents de tous les fichiers NBM (scale = NaN)
    ref_scale = np.where(
        np.isnan(ref_scale) | (ref_scale < 0.01),
        0.01,
        ref_scale,
    )
    # Centre NaN → remplacer par 0 (marqueur absent = pas de shift)
    ref_center = np.where(np.isnan(ref_center), 0.0, ref_center)
    return ref_center, ref_scale


def compute_reference_normalization(
    X_unknown: np.ndarray,
    X_reference: np.ndarray,
    robust: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalise X_unknown en z-scores par rapport à la population de référence (NBM).

    ── Changement vs version min-max ────────────────────────────────────────────

    L'ancienne normalisation min-max plaçait tout dans [0, 1], rendant
    score_nodes_for_blasts() aveugle (score = 0 systématique).

    Cette version produit des z-scores :
        z[j] = (X_unknown[j] − median_ref[j]) / scale_ref[j]

    Les blastes se retrouvent EN DEHORS de [−1, +1] sur les marqueurs
    discriminants (CD34 z ≈ +2, SSC z ≈ −2), déclenchant les contributions
    directionnelles de score_nodes_for_blasts().

    Args:
        X_unknown: Matrice des nœuds à scorer [n_unknown, n_markers].
        X_reference: Matrice de référence (cellules NBM ou médianes NBM)
                     [n_ref, n_markers], dans le même espace transformé.
        robust: Utiliser médiane+IQR/1.35 (True, défaut) ou moyenne+std (False).

    Returns:
        Tuple (X_zscore, ref_center, ref_scale) :
          - X_zscore   : z-scores [n_unknown, n_markers].
          - ref_center : vecteur des centres de référence par marqueur.
          - ref_scale  : vecteur des échelles de référence par marqueur.
    """
    ref_center, ref_scale = compute_reference_stats(X_reference, robust=robust)
    X_zscore = (X_unknown - ref_center) / ref_scale
    return X_zscore, ref_center, ref_scale


# ─────────────────────────────────────────────────────────────────────────────
#  §10.4d — Traçabilité FCS source des cellules blast
# ─────────────────────────────────────────────────────────────────────────────


def trace_blast_cells_to_fcs_source(
    blast_candidates_df: pd.DataFrame,
    cell_data: "Any",  # anndata.AnnData
    source_priority_cols: Optional[List[str]] = None,
    condition_col: Optional[str] = "Condition",
    blast_categories_to_trace: Optional[List[str]] = None,
    alert_patho_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Retrace les cellules des nœuds candidats blast vers leurs fichiers FCS sources.

    Utilisation clinique ELN 2022 : si >50% des cellules d'un nœud BLAST_HIGH
    proviennent d'un fichier "Patho" (diagnostic ou suivi), déclencher une
    ALERTE CLINIQUE.

    Cela permet de distinguer :
      - Un nœud BLAST_HIGH peuplé de cellules Saines → faux positif batch
      - Un nœud BLAST_HIGH peuplé de cellules Pathologiques → vrai signal MRD

    Args:
        blast_candidates_df: DataFrame de build_blast_score_dataframe (colonnes:
                              node_id, blast_score, blast_category, ...).
        cell_data: AnnData avec .obs['clustering'] (nœud SOM, 0-indexé) et
                   .obs[condition_col] (condition de la cellule).
        source_priority_cols: Colonnes .obs à inspecter pour retrouver le fichier
                               source (ex: ['File_Origin', 'filename']).
        condition_col: Colonne de condition dans cell_data.obs.
        blast_categories_to_trace: Catégories à inclure dans la traçabilité.
                                    Défaut: ['BLAST_HIGH', 'BLAST_MODERATE'].
        alert_patho_threshold: Fraction de cellules Pathologiques déclenchant
                                l'ALERTE CLINIQUE (défaut 0.50 = 50%).

    Returns:
        DataFrame avec colonnes :
          node_id, blast_score, blast_category,
          n_cells_total, n_cells_patho, n_cells_sain, pct_patho,
          source_files (str), clinical_alert (bool).
    """
    if blast_categories_to_trace is None:
        blast_categories_to_trace = ["BLAST_HIGH", "BLAST_MODERATE"]

    mask_trace = blast_candidates_df["blast_category"].isin(blast_categories_to_trace)
    df_trace = blast_candidates_df[mask_trace].copy()

    if df_trace.empty:
        _logger.info("Aucun nœud blast à tracer pour ces catégories.")
        return df_trace

    # ── Résoudre la colonne de clustering dans cell_data ─────────────────────
    obs_df = None
    try:
        obs_df = cell_data.obs.copy()
    except AttributeError:
        if isinstance(cell_data, pd.DataFrame):
            obs_df = cell_data.copy()

    if obs_df is None:
        _logger.warning("cell_data.obs inaccessible — traçabilité impossible.")
        df_trace["n_cells_total"] = 0
        df_trace["clinical_alert"] = False
        return df_trace

    clustering_col = None
    for candidate in ["clustering", "FlowSOM_cluster", "cluster", "node_id"]:
        if candidate in obs_df.columns:
            clustering_col = candidate
            break

    if clustering_col is None:
        _logger.warning("Colonne clustering introuvable dans cell_data.obs.")
        df_trace["n_cells_total"] = 0
        df_trace["clinical_alert"] = False
        return df_trace

    # ── Résoudre la colonne source de fichier ─────────────────────────────────
    source_col = None
    if source_priority_cols:
        for col in source_priority_cols:
            if col in obs_df.columns:
                source_col = col
                break
    if source_col is None:
        for candidate in ["File_Origin", "filename", "Filename", "sample_id", "SampleID"]:
            if candidate in obs_df.columns:
                source_col = candidate
                break

    # ── Traçabilité nœud par nœud ─────────────────────────────────────────────
    records: List[Dict] = []

    for _, row in df_trace.iterrows():
        node_id = int(row["node_id"])
        blast_score = float(row["blast_score"])
        blast_category = str(row["blast_category"])

        mask_node = obs_df[clustering_col].astype(int) == node_id
        cells_in_node = obs_df[mask_node]
        n_total = len(cells_in_node)

        if n_total == 0:
            records.append({
                "node_id": node_id, "blast_score": blast_score,
                "blast_category": blast_category, "n_cells_total": 0,
                "n_cells_patho": 0, "n_cells_sain": 0,
                "pct_patho": 0.0, "source_files": "", "clinical_alert": False,
            })
            continue

        n_patho = 0
        n_sain = 0
        if condition_col and condition_col in cells_in_node.columns:
            cond_vals = cells_in_node[condition_col].astype(str).str.upper()
            n_patho = int((cond_vals.str.contains("PATHO|DIAG|DX|LAM|AML", na=False)).sum())
            n_sain  = int((cond_vals.str.contains("SAIN|NORMAL|NBM|HEALTHY", na=False)).sum())

        pct_patho = n_patho / max(n_total, 1)
        clinical_alert = (blast_category == "BLAST_HIGH") and (pct_patho >= alert_patho_threshold)

        if clinical_alert:
            _logger.warning(
                "ALERTE CLINIQUE — Nœud %d (%s) : %.1f%% cellules Pathologiques (%d/%d)",
                node_id, blast_category, 100.0 * pct_patho, n_patho, n_total,
            )

        source_files_str = ""
        if source_col:
            src_counts = cells_in_node[source_col].value_counts()
            source_files_str = " | ".join(
                f"{fname}({cnt})" for fname, cnt in src_counts.items()
            )

        records.append({
            "node_id": node_id, "blast_score": blast_score,
            "blast_category": blast_category, "n_cells_total": n_total,
            "n_cells_patho": n_patho, "n_cells_sain": n_sain,
            "pct_patho": round(pct_patho * 100.0, 1),
            "source_files": source_files_str, "clinical_alert": clinical_alert,
        })

    result_df = (
        pd.DataFrame(records)
        .sort_values(["blast_category", "blast_score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    n_alerts = int(result_df["clinical_alert"].sum())
    _logger.info(
        "Traçabilité terminée: %d nœuds tracés, %d ALERTE(S) CLINIQUE(S)",
        len(result_df), n_alerts,
    )
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitaire — vecteur de poids depuis colonnes + dict
# ─────────────────────────────────────────────────────────────────────────────


def build_weight_vector(
    cols: List[str],
    weights: Dict[str, float],
    default: float = 1.0,
) -> np.ndarray:
    """
    Construit un vecteur numpy de poids aligné sur une liste de colonnes.

    Utilisé pour les distances pondérées dans le mapping de populations :
    chaque marqueur reçoit un poids depuis le dict ``weights``, ou
    ``default`` s'il n'est pas présent.

    Args:
        cols: Liste ordonnée de noms de marqueurs.
        weights: Dict {marqueur: poids}.
        default: Poids par défaut si le marqueur est absent du dict (défaut 1.0).

    Returns:
        np.ndarray de shape (len(cols),) et dtype float64.

    Example::

        w = build_weight_vector(["CD34", "CD45", "SSC-A"],
                                {"CD34": 3.0, "CD45": -2.0},
                                default=0.5)
        # array([ 3. , -2. ,  0.5])
    """
    return np.array([weights.get(c, default) for c in cols], dtype=np.float64)


# Alias privé — compatibilité avec flowsom_pipeline.py (même logique que categorize_blast_score)
def _categorize_blast(score: float) -> str:
    """Alias interne de categorize_blast_score (mêmes seuils ELN 2022 / Ogata)."""
    return categorize_blast_score(score)
