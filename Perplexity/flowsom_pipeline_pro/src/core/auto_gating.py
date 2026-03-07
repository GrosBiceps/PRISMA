"""
auto_gating.py — Gating adaptatif par GMM et régression RANSAC (mode 'auto').

AutoGating utilise scikit-learn (GaussianMixture, RANSACRegressor) pour
détecter automatiquement les seuils de gating en modélisant les populations
naturelles dans les données, sans percentiles fixes.

Avantages vs PreGating (percentiles):
    - Si un échantillon a 10% de débris → la porte s'adapte automatiquement
    - Si un échantillon est propre → moins de perte de données
    - Pour les doublets: modélise la diagonale FSC-A/FSC-H statistiquement
    - Pour CD45+: trouve le creux bimodal entre CD45- et CD45+

Chaque méthode:
    - Retourne un masque numpy booléen
    - Enregistre un GateResult dans le registre global
    - Log structuré l'événement (audit JSON)
    - Gère les fallbacks (GMM → percentile si échec)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from ..models.gate_result import (
    GateResult,
    gating_reports,
    gating_log_entries,
    log_gating_event,
)
from .gating import PreGating
from flowsom_pipeline_pro.config.constants import (
    GMM_MAX_SAMPLES,
    RANSAC_R2_THRESHOLD,
    RANSAC_MAD_FACTOR,
)


# =============================================================================
# CLASSE AutoGating — Gating adaptatif par GMM/KDE
# =============================================================================
# Inspiré de CytoPy AutonomousGate (sans dépendance MongoDB)
# Utilise scikit-learn GaussianMixture pour trouver les "creux" réels
# entre les populations au lieu de couper à des percentiles fixes.
#
# Avantages vs PreGating (percentiles):
#   - Si un échantillon a 10% de débris → la porte s'adapte automatiquement
#   - Si un échantillon est propre → moins de perte de données
#   - Pour les doublets: modélise la diagonale FSC-A/FSC-H statistiquement
#   - Pour CD45+: trouve le creux bimodal entre CD45- et CD45+
#
# [V2 AMÉLIORATIONS]:
#   - safe_fit_gmm: sous-échantillonnage à 200k points max avant fit
#   - auto_gate_singlets: contrôle R² RANSAC + fallback ratio si R² < 0.85
#   - Toutes les fonctions retournent un GateResult structuré
#   - Scatter FSC-A vs FSC-H par fichier + tableau % singlets stockés
#   - Log structuré JSON pour audit automatique des runs
# =============================================================================
# Stockage global des scatter data RANSAC par fichier (pour le rapport HTML)
ransac_scatter_data = {}  # {file_name: {fsc_h, fsc_a, pred, inlier_mask, r2, slope, intercept, pct_singlets}}
singlets_summary_per_file = []  # Liste de dicts pour tableau "% singlets par fichier"


class AutoGating:
    """
    Gating automatique adaptatif basé sur des modèles de mélange gaussien (GMM)
    et estimation de densité. Inspiré de CytoPy AutonomousGate.

    Chaque méthode utilise un GMM pour identifier les populations naturelles
    dans les données, au lieu de seuils fixes basés sur des percentiles.

    Dépendances: scikit-learn (GaussianMixture, StandardScaler)
    """

    # Seuil R² minimal pour la régression RANSAC (en dessous → fallback ratio)
    RANSAC_R2_THRESHOLD = 0.85
    # Sous-échantillonnage max avant GMM (convergence + performance)
    GMM_MAX_SAMPLES = 200_000

    @staticmethod
    def _subsample_for_gmm(data: np.ndarray, max_samples: int = None) -> np.ndarray:
        """
        Sous-échantillonne les données si elles dépassent max_samples.
        Améliore la convergence et évite les timeouts implicites sur gros jeux.

        Args:
            data: Données (n_samples, n_features)
            max_samples: Nombre max de points (défaut: GMM_MAX_SAMPLES)

        Returns:
            data_subsampled: Données sous-échantillonnées (ou originales si < max)
        """
        if max_samples is None:
            max_samples = AutoGating.GMM_MAX_SAMPLES
        if data.shape[0] > max_samples:
            idx = np.random.choice(data.shape[0], size=max_samples, replace=False)
            print(
                f"      [GMM] Sous-echantillonnage: {data.shape[0]:,} -> {max_samples:,} points"
            )
            return data[idx]
        return data

    @staticmethod
    def safe_fit_gmm(
        data: np.ndarray,
        n_components: int = 2,
        n_init: int = 3,
        max_retries: int = 5,
        random_state: int = 42,
        covariance_type: str = "full",
        max_iter: int = 200,
        subsample: bool = True,
    ) -> Any:
        """
        Wrapper robuste pour le fitting GMM avec gestion d'erreurs.

        Tente le fit plusieurs fois avec différentes initialisations.
        En cas d'échec total sur n_components > 1, fallback sur 1 composante.
        Vérifie la convergence et émet des warnings si nécessaire.

        [V2] Sous-échantillonnage automatique à 200k points max avant fit.

        Args:
            data: Données à fitter (n_samples, n_features) ou (n_samples, 1)
            n_components: Nombre de composantes GMM
            n_init: Nombre d'initialisations par tentative
            max_retries: Nombre max de tentatives avant fallback
            random_state: Seed pour reproductibilité
            covariance_type: Type de covariance ('full', 'diag', 'spherical', 'tied')
            max_iter: Nombre max d'itérations EM
            subsample: Si True, sous-échantillonne avant fit (défaut True)

        Returns:
            GaussianMixture fitté

        Raises:
            RuntimeError: Si le fit échoue après toutes les tentatives (y compris fallback)
        """
        from sklearn.mixture import GaussianMixture

        # Sous-échantillonnage pour convergence rapide sur gros jeux de données
        if subsample:
            data_fit = AutoGating._subsample_for_gmm(data)
        else:
            data_fit = data

        last_error = None
        for attempt in range(max_retries):
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    random_state=random_state + attempt,
                    n_init=n_init,
                    covariance_type=covariance_type,
                    max_iter=max_iter,
                )
                gmm.fit(data_fit)
                if not gmm.converged_:
                    warnings.warn(
                        f"GMM non-convergé (n={n_components}, tentative {attempt + 1}/{max_retries})"
                    )
                    log_gating_event(
                        "GMM",
                        f"n_components={n_components}",
                        "warning",
                        {"attempt": attempt + 1},
                        f"Non-convergé tentative {attempt + 1}/{max_retries}",
                    )
                    continue
                return gmm
            except Exception as e:
                last_error = e
                log_gating_event(
                    "GMM",
                    f"n_components={n_components}",
                    "error",
                    {"attempt": attempt + 1, "error": str(e)},
                )
                continue

        # Fallback: tenter avec 1 composante si n_components > 1
        if n_components > 1:
            warn_msg = f"GMM fallback unimodal après {max_retries} échecs (dernière erreur: {last_error})"
            warnings.warn(warn_msg)
            log_gating_event(
                "GMM",
                "fallback_unimodal",
                "fallback",
                {"original_n_components": n_components, "error": str(last_error)},
                warn_msg,
            )
            try:
                gmm = GaussianMixture(
                    n_components=1,
                    random_state=random_state,
                    n_init=1,
                    covariance_type=covariance_type,
                    max_iter=max_iter,
                )
                gmm.fit(data_fit)
                return gmm
            except Exception as e:
                raise RuntimeError(
                    f"GMM fit échoué après {max_retries} tentatives + fallback unimodal: {e}"
                )

        raise RuntimeError(
            f"GMM fit échoué après {max_retries} tentatives: {last_error}"
        )

    @staticmethod
    def auto_gate_debris(
        X: np.ndarray,
        var_names: List[str],
        n_components: int = 3,
        min_cluster_fraction: float = 0.02,
    ) -> np.ndarray:
        """
        Gate débris adaptatif par GMM 2D sur FSC-A / SSC-A.

        L'algorithme identifie les clusters naturels dans l'espace FSC/SSC:
        - Débris: événements bas en FSC-A (petites particules)
        - Cellules: population principale (cluster dominant)
        - Saturés: événements très hauts (optionnel, détecté par BIC)

        Sélection automatique du nombre de composantes par BIC (2 ou 3).

        Args:
            X: Matrice des données (n_cells, n_markers)
            var_names: Noms des marqueurs
            n_components: Nombre max de composantes GMM à tester
            min_cluster_fraction: Fraction min d'événements pour inclure un cluster

        Returns:
            Masque booléen (True = cellule viable, False = débris/saturé)
        """
        from sklearn.preprocessing import StandardScaler

        n_cells = X.shape[0]
        fsc_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
        ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A"])

        if fsc_idx is None or ssc_idx is None:
            print("[!] FSC-A ou SSC-A non trouvé pour auto-gate débris")
            log_gating_event(
                "debris", "auto_gmm", "error", warning_msg="FSC-A ou SSC-A non trouvé"
            )
            return np.ones(n_cells, dtype=bool)

        fsc = X[:, fsc_idx].astype(np.float64)
        ssc = X[:, ssc_idx].astype(np.float64)

        # Filtrer les NaN/Inf
        valid = np.isfinite(fsc) & np.isfinite(ssc)
        data_2d = np.column_stack([fsc[valid], ssc[valid]])

        if valid.sum() < 200:
            print("[!] Pas assez de données valides pour auto-gate débris")
            return np.ones(n_cells, dtype=bool)

        # Standardiser avant GMM pour meilleure convergence
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_2d)

        # Sélection automatique du nombre de composantes par BIC
        best_bic = np.inf
        best_gmm = None
        for n_comp in [2, 3]:
            try:
                gmm_test = AutoGating.safe_fit_gmm(
                    data_scaled,
                    n_components=n_comp,
                    covariance_type="full",
                    n_init=3,
                    max_iter=200,
                )
                bic = gmm_test.bic(
                    data_scaled
                    if data_scaled.shape[0] <= AutoGating.GMM_MAX_SAMPLES
                    else AutoGating._subsample_for_gmm(data_scaled)
                )
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm_test
            except RuntimeError as e:
                print(f"   [!] GMM {n_comp} composantes échoué: {e}")
                continue

        if best_gmm is None:
            print("   [!] Aucun GMM n'a convergé, conservation de tous les événements")
            log_gating_event(
                "debris",
                "auto_gmm",
                "fallback",
                warning_msg="Aucun GMM convergé, toutes cellules conservées",
            )
            return np.ones(n_cells, dtype=bool)

        labels = best_gmm.predict(data_scaled)
        n_comp = best_gmm.n_components

        # Statistiques par cluster (en espace original)
        cluster_sizes = np.bincount(labels, minlength=n_comp)
        cluster_fsc_means = np.array(
            [data_2d[labels == i, 0].mean() for i in range(n_comp)]
        )

        # Population principale = plus grand cluster
        main_cluster = np.argmax(cluster_sizes)

        # Inclure les clusters avec assez d'événements et un FSC raisonnable
        # (exclure les débris = FSC très bas)
        mask_valid = np.zeros(valid.sum(), dtype=bool)
        fsc_threshold = cluster_fsc_means[main_cluster] * 0.25

        for i in range(n_comp):
            fraction = cluster_sizes[i] / len(labels)
            if (
                fraction >= min_cluster_fraction
                and cluster_fsc_means[i] >= fsc_threshold
            ):
                mask_valid |= labels == i

        # Sécurité: si aucun cluster sélectionné, garder le principal
        if not mask_valid.any():
            mask_valid = labels == main_cluster

        mask = np.zeros(n_cells, dtype=bool)
        mask[valid] = mask_valid

        n_kept = mask.sum()
        print(
            f"   [Auto-GMM] {best_gmm.n_components} composantes détectées (BIC={best_bic:.0f})"
        )
        for i in range(n_comp):
            status = "[OK]" if mask_valid[labels == i].any() else "[--]"
            print(
                f"     {status} Cluster {i}: {cluster_sizes[i]:,} evt, FSC-A moy={cluster_fsc_means[i]:.0f}"
            )
        print(f"   [Auto-GMM] Conserves: {n_kept:,} evenements")

        # Log structuré
        log_gating_event(
            "debris",
            "auto_gmm",
            "success",
            {
                "n_components": int(n_comp),
                "bic": float(best_bic),
                "n_kept": int(n_kept),
                "n_total": int(n_cells),
                "cluster_sizes": cluster_sizes.tolist(),
            },
        )

        # Construire GateResult
        gate_result = GateResult(
            mask=mask,
            n_kept=int(n_kept),
            n_total=int(n_cells),
            method="auto_gmm_debris",
            gate_name="G1_debris",
            details={
                "n_components": int(n_comp),
                "bic": float(best_bic),
                "cluster_fsc_means": cluster_fsc_means.tolist(),
            },
        )
        gating_reports.append(gate_result)

        return mask

    @staticmethod
    def auto_gate_singlets(
        X: np.ndarray,
        var_names: List[str],
        file_origin: Optional[np.ndarray] = None,
        per_file: bool = True,
        r2_threshold: float = 0.85,
    ) -> np.ndarray:
        """
        Gate singlets adaptatif par régression linéaire robuste (RANSAC).

        Les singlets forment une diagonale sur le plot FSC-A vs FSC-H.
        Les doublets se situent au-dessus de cette diagonale (FSC-A augmente mais pas FSC-H).

        Méthode améliorée (V2):
        1. Pré-filtre viable (FSC/SSC 1-99%) pour exclure les outliers extrêmes
        2. Régression linéaire robuste RANSAC sur FSC-A vs FSC-H
        3. Contrôle qualité R² sur les inliers RANSAC
        4. Si R² < seuil (0.85): fallback vers gating ratio FSC-A/FSC-H simple
        5. Stockage des scatter data par fichier pour le rapport HTML

        Args:
            X: Matrice des données (n_cells, n_markers)
            var_names: Noms des marqueurs
            file_origin: Array contenant l'origine de chaque cellule (pour gating par fichier)
            per_file: Si True, applique le gating séparément par fichier
            r2_threshold: Seuil R² minimum (défaut 0.85). En dessous → fallback ratio

        Returns:
            Masque booléen (True = singlet, False = doublet)
        """
        from sklearn.linear_model import RANSACRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        n_cells = X.shape[0]
        fsc_a_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
        fsc_h_idx = PreGating.find_marker_index(var_names, ["FSC-H"])

        if fsc_a_idx is None or fsc_h_idx is None:
            print("[!] FSC-A ou FSC-H non trouvé pour auto-gate singlets")
            return np.ones(n_cells, dtype=bool)

        fsc_a = X[:, fsc_a_idx].astype(np.float64)
        fsc_h = X[:, fsc_h_idx].astype(np.float64)

        # Pré-filtre viable (FSC/SSC 1-99%) pour réduire l'impact des outliers
        # extrêmes (blastes matures, granulocytes agrégés) sur la régression RANSAC
        viable = PreGating.gate_viable_cells(
            X, var_names, min_percentile=1.0, max_percentile=99.0
        )

        # Filtrer: valeurs valides avec FSC > seuil minimal + viabilité
        valid = (
            viable
            & np.isfinite(fsc_a)
            & np.isfinite(fsc_h)
            & (fsc_h > 100)
            & (fsc_a > 100)
        )

        if valid.sum() < 200:
            print("[!] Pas assez de données valides pour auto-gate singlets")
            return np.ones(n_cells, dtype=bool)

        mask = np.zeros(n_cells, dtype=bool)

        # ─── Helper interne: fallback ratio FSC-A/FSC-H ───
        def _fallback_ratio_gating(
            fsc_a_local, fsc_h_local, ratio_min=0.6, ratio_max=1.5
        ):
            """Gating simple par ratio FSC-A/FSC-H (ancienne méthode)."""
            ratio = fsc_a_local.ravel() / np.maximum(fsc_h_local.ravel(), 1.0)
            return (ratio >= ratio_min) & (ratio <= ratio_max)

        # Gating par fichier si demandé et si file_origin fourni
        if per_file and file_origin is not None:
            unique_files = np.unique(file_origin)
            print(f"   [Auto-RANSAC] Gating par fichier ({len(unique_files)} fichiers)")

            total_singlets = 0
            total_doublets = 0

            for file_name in unique_files:
                # Sélectionner les cellules de ce fichier
                file_mask = (file_origin == file_name) & valid

                if file_mask.sum() < 50:
                    # Trop peu de cellules, garder toutes
                    mask[file_mask] = True
                    singlets_summary_per_file.append(
                        {
                            "file": str(file_name),
                            "n_total": int(file_mask.sum()),
                            "n_singlets": int(file_mask.sum()),
                            "pct_singlets": 100.0,
                            "method": "skip_too_few",
                            "r2": None,
                        }
                    )
                    continue

                fsc_a_file = fsc_a[file_mask].reshape(-1, 1)
                fsc_h_file = fsc_h[file_mask].reshape(-1, 1)

                # Régression RANSAC pour trouver la diagonale des singlets
                try:
                    ransac = RANSACRegressor(
                        estimator=LinearRegression(),
                        min_samples=50,
                        residual_threshold=None,  # Auto (MAD)
                        random_state=42,
                        max_trials=100,
                    )
                    ransac.fit(fsc_h_file, fsc_a_file.ravel())

                    # ─── CONTRÔLE QUALITÉ R² SUR INLIERS RANSAC ───
                    inlier_mask = ransac.inlier_mask_
                    r2_val = None
                    used_method = "ransac"

                    if inlier_mask is not None and inlier_mask.sum() > 50:
                        r2_val = r2_score(
                            fsc_a_file[inlier_mask].ravel(),
                            ransac.predict(fsc_h_file[inlier_mask]),
                        )

                        if r2_val < r2_threshold:
                            # ─── FALLBACK: gating ratio simple ───
                            warn_msg = f"R² faible pour {file_name} (R²={r2_val:.2f} < {r2_threshold}), fallback gating ratio"
                            print(f"      [!] {warn_msg}")
                            log_gating_event(
                                "singlets",
                                "ransac_fallback_ratio",
                                "fallback",
                                {"file": str(file_name), "r2": float(r2_val)},
                                warn_msg,
                            )

                            singlets_file = _fallback_ratio_gating(
                                fsc_a_file, fsc_h_file
                            )
                            used_method = "ratio_fallback"

                            # Appliquer
                            file_indices = np.where(file_mask)[0]
                            mask[file_indices] = singlets_file

                            n_sing = int(singlets_file.sum())
                            n_doub = len(singlets_file) - n_sing
                            total_singlets += n_sing
                            total_doublets += n_doub

                            file_short = (
                                file_name
                                if len(file_name) <= 25
                                else file_name[:22] + "..."
                            )
                            print(
                                f"      • {file_short}: {n_sing:,} singlets / {n_sing + n_doub:,} ({n_sing / (n_sing + n_doub) * 100:.1f}%) - RATIO FALLBACK (R²={r2_val:.2f})"
                            )

                            # Stocker les scatter data (même si fallback, pour diagnostic)
                            n_sample_pts = min(2000, len(fsc_a_file))
                            sample_idx = np.random.choice(
                                len(fsc_a_file), n_sample_pts, replace=False
                            )
                            ransac_scatter_data[str(file_name)] = {
                                "fsc_h": fsc_h_file[sample_idx].ravel().tolist(),
                                "fsc_a": fsc_a_file[sample_idx].ravel().tolist(),
                                "pred": ransac.predict(fsc_h_file[sample_idx]).tolist(),
                                "r2": float(r2_val),
                                "method": "ratio_fallback",
                                "slope": float(ransac.estimator_.coef_[0]),
                                "intercept": float(ransac.estimator_.intercept_),
                            }
                            singlets_summary_per_file.append(
                                {
                                    "file": str(file_name),
                                    "n_total": int(len(singlets_file)),
                                    "n_singlets": n_sing,
                                    "pct_singlets": round(
                                        n_sing / (n_sing + n_doub) * 100, 1
                                    ),
                                    "method": "ratio_fallback",
                                    "r2": round(float(r2_val), 3),
                                }
                            )
                            continue

                    # ─── R² OK (ou pas de inlier_mask): utiliser RANSAC normal ───
                    # Prédiction sur la droite
                    fsc_a_pred = ransac.predict(fsc_h_file)

                    # Distance verticale (résidus) - doublets au-dessus de la ligne
                    residuals = fsc_a_file.ravel() - fsc_a_pred

                    # Seuil adaptatif basé sur MAD (Median Absolute Deviation)
                    median_residual = np.median(residuals)
                    mad = np.median(np.abs(residuals - median_residual))

                    # Seuil: médiane + 3 * MAD
                    threshold_upper = median_residual + 3.0 * mad

                    # Singlets: points près de la diagonale (pas trop au-dessus)
                    singlets_file = residuals <= threshold_upper

                    # Appliquer le masque local
                    file_indices = np.where(file_mask)[0]
                    mask[file_indices] = singlets_file

                    n_sing = int(singlets_file.sum())
                    n_doub = len(singlets_file) - n_sing
                    total_singlets += n_sing
                    total_doublets += n_doub

                    # Affichage compact par fichier
                    slope = ransac.estimator_.coef_[0]
                    intercept = ransac.estimator_.intercept_
                    file_short = (
                        file_name if len(file_name) <= 25 else file_name[:22] + "..."
                    )
                    r2_str = f", R²={r2_val:.3f}" if r2_val is not None else ""
                    print(
                        f"      • {file_short}: {n_sing:,} singlets / {n_sing + n_doub:,} ({n_sing / (n_sing + n_doub) * 100:.1f}%) - y={slope:.3f}x+{intercept:.0f}{r2_str}"
                    )

                    # Stocker scatter data pour le rapport HTML (échantillonné)
                    n_sample_pts = min(2000, len(fsc_a_file))
                    sample_idx = np.random.choice(
                        len(fsc_a_file), n_sample_pts, replace=False
                    )
                    ransac_scatter_data[str(file_name)] = {
                        "fsc_h": fsc_h_file[sample_idx].ravel().tolist(),
                        "fsc_a": fsc_a_file[sample_idx].ravel().tolist(),
                        "pred": ransac.predict(fsc_h_file[sample_idx]).tolist(),
                        "r2": float(r2_val) if r2_val is not None else None,
                        "method": "ransac",
                        "slope": float(slope),
                        "intercept": float(intercept),
                    }
                    singlets_summary_per_file.append(
                        {
                            "file": str(file_name),
                            "n_total": int(len(singlets_file)),
                            "n_singlets": n_sing,
                            "pct_singlets": round(n_sing / (n_sing + n_doub) * 100, 1),
                            "method": "ransac",
                            "r2": round(float(r2_val), 3)
                            if r2_val is not None
                            else None,
                        }
                    )

                    # Log structuré
                    log_gating_event(
                        "singlets",
                        "ransac",
                        "success",
                        {
                            "file": str(file_name),
                            "r2": float(r2_val) if r2_val else None,
                            "slope": float(slope),
                            "intercept": float(intercept),
                            "n_singlets": n_sing,
                            "n_doublets": n_doub,
                        },
                    )

                except Exception as e:
                    print(f"      [!] Échec RANSAC pour {file_name}: {e}")
                    log_gating_event(
                        "singlets",
                        "ransac",
                        "error",
                        {"file": str(file_name), "error": str(e)},
                        f"Échec RANSAC pour {file_name}: {e}",
                    )
                    # En cas d'échec, garder toutes les cellules du fichier
                    mask[file_mask] = True
                    total_singlets += file_mask.sum()
                    singlets_summary_per_file.append(
                        {
                            "file": str(file_name),
                            "n_total": int(file_mask.sum()),
                            "n_singlets": int(file_mask.sum()),
                            "pct_singlets": 100.0,
                            "method": "error_keep_all",
                            "r2": None,
                        }
                    )

            print(
                f"   [Auto-RANSAC] Total: {total_singlets:,} singlets, {total_doublets:,} doublets exclus"
            )

            # Résumé tableau % singlets par fichier
            if singlets_summary_per_file:
                print(
                    f"\n   {'Fichier':<30} {'Méthode':<18} {'R²':>6} {'% Singlets':>12}"
                )
                print(f"   {'─' * 30} {'─' * 18} {'─' * 6} {'─' * 12}")
                for row in singlets_summary_per_file:
                    r2_disp = f"{row['r2']:.3f}" if row["r2"] is not None else "N/A"
                    fname_short = (
                        row["file"]
                        if len(row["file"]) <= 30
                        else row["file"][:27] + "..."
                    )
                    print(
                        f"   {fname_short:<30} {row['method']:<18} {r2_disp:>6} {row['pct_singlets']:>10.1f}%"
                    )

        else:
            # Gating global (ancien comportement)
            print(f"   [Auto-RANSAC] Gating global sur toutes les données")

            fsc_a_valid = fsc_a[valid].reshape(-1, 1)
            fsc_h_valid = fsc_h[valid].reshape(-1, 1)

            # Régression RANSAC
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=100,
                residual_threshold=None,
                random_state=42,
                max_trials=100,
            )
            ransac.fit(fsc_h_valid, fsc_a_valid.ravel())

            # ─── CONTRÔLE QUALITÉ R² GLOBAL ───
            inlier_mask = ransac.inlier_mask_
            r2_val = None
            if inlier_mask is not None and inlier_mask.sum() > 50:
                r2_val = r2_score(
                    fsc_a_valid[inlier_mask].ravel(),
                    ransac.predict(fsc_h_valid[inlier_mask]),
                )
                if r2_val < r2_threshold:
                    warn_msg = f"R² faible global (R²={r2_val:.2f} < {r2_threshold}), fallback gating ratio"
                    print(f"   [!] {warn_msg}")
                    log_gating_event(
                        "singlets",
                        "ransac_fallback_ratio",
                        "fallback",
                        {"r2": float(r2_val)},
                        warn_msg,
                    )

                    singlets_mask = _fallback_ratio_gating(fsc_a_valid, fsc_h_valid)
                    mask[valid] = singlets_mask

                    n_singlets = mask.sum()
                    n_doublets = valid.sum() - n_singlets
                    print(
                        f"   [RATIO FALLBACK] Singlets: {n_singlets:,} ({n_singlets / valid.sum() * 100:.1f}%)"
                    )

                    gate_result = GateResult(
                        mask=mask,
                        n_kept=int(n_singlets),
                        n_total=int(n_cells),
                        method="ratio_fallback_global",
                        gate_name="G2_singlets",
                        details={"r2": float(r2_val)},
                        warnings=[warn_msg],
                    )
                    gating_reports.append(gate_result)
                    return mask

            # Prédiction et résidus
            fsc_a_pred = ransac.predict(fsc_h_valid)
            residuals = fsc_a_valid.ravel() - fsc_a_pred

            # Seuil adaptatif MAD
            median_residual = np.median(residuals)
            mad = np.median(np.abs(residuals - median_residual))
            threshold_upper = median_residual + 3.0 * mad

            # Masque singlets
            singlets_mask = residuals <= threshold_upper
            mask[valid] = singlets_mask

            n_singlets = mask.sum()
            n_doublets = valid.sum() - n_singlets
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_

            r2_str = f", R²={r2_val:.3f}" if r2_val is not None else ""
            print(
                f"   [Auto-RANSAC] Droite: y = {slope:.3f}x + {intercept:.0f}{r2_str}"
            )
            print(
                f"   [Auto-RANSAC] Seuil MAD: médiane + {3.0:.1f}×MAD = {threshold_upper:.0f}"
            )
            print(
                f"   [Auto-RANSAC] Singlets: {n_singlets:,} ({n_singlets / valid.sum() * 100:.1f}%)"
            )
            print(
                f"   [Auto-RANSAC] Doublets rejetés: {n_doublets:,} ({n_doublets / valid.sum() * 100:.1f}%)"
            )

        # GateResult structuré
        gate_result = GateResult(
            mask=mask,
            n_kept=int(mask.sum()),
            n_total=int(n_cells),
            method="ransac_singlets",
            gate_name="G2_singlets",
            details={
                "per_file": per_file,
                "n_files": len(singlets_summary_per_file) if per_file else 1,
                "files_summary": singlets_summary_per_file if per_file else [],
            },
        )
        gating_reports.append(gate_result)

        return mask

    @staticmethod
    def auto_gate_cd45(
        X: np.ndarray,
        var_names: List[str],
        n_components: int = 2,
        uniform_gating: bool = False,
        threshold_percentile: float = 5.0,
    ) -> np.ndarray:
        """
        Gate CD45+ adaptatif par GMM 1D.

        Trouve automatiquement le creux bimodal entre CD45- et CD45+
        au lieu d'un percentile fixe. Le GMM modélise la distribution
        bimodale et assigne chaque événement à la population la plus probable.

        Args:
            X: Matrice des données
            var_names: Noms des marqueurs
            n_components: Nombre de composantes GMM (2 = CD45- / CD45+)
            uniform_gating: Si True, applique un seuil soft (percentile)
            threshold_percentile: Percentile pour le seuil soft CD45

        Returns:
            Masque booléen (True = CD45+, False = CD45-)
        """
        n_cells = X.shape[0]
        cd45_idx = PreGating.find_marker_index(
            var_names, ["CD45", "CD45-PECY5", "CD45-PC5"]
        )

        if cd45_idx is None:
            print("[!] CD45 non trouvé pour auto-gate CD45+")
            return np.ones(n_cells, dtype=bool)

        cd45 = X[:, cd45_idx].astype(np.float64)
        valid = np.isfinite(cd45)

        if valid.sum() < 200:
            print("[!] Pas assez de données valides pour auto-gate CD45+")
            return np.ones(n_cells, dtype=bool)

        # Mode uniform_gating: seuil soft par percentile (pas de GMM)
        if uniform_gating:
            threshold = np.nanpercentile(cd45[valid], threshold_percentile)
            mask = np.zeros(n_cells, dtype=bool)
            mask[valid] = cd45[valid] > threshold
            n_pos = mask.sum()
            print(
                f"   [Uniform-CD45] Seuil soft: {threshold:.0f} (percentile {threshold_percentile}%)"
            )
            print(
                f"   [Uniform-CD45] CD45+ identifiés: {n_pos:,} ({n_pos / valid.sum() * 100:.1f}%)"
            )

            gate_result = GateResult(
                mask=mask,
                n_kept=int(n_pos),
                n_total=int(valid.sum()),
                method="gmm_cd45_uniform",
                gate_name="G3_cd45",
                details={
                    "threshold": float(threshold),
                    "percentile": threshold_percentile,
                    "fallback": False,
                },
            )
            gating_reports.append(gate_result)
            log_gating_event(
                "cd45",
                "uniform_percentile",
                "success",
                {"threshold": float(threshold), "n_pos": int(n_pos)},
            )
            return mask

        # GMM pour séparer CD45- et CD45+
        try:
            gmm = AutoGating.safe_fit_gmm(
                cd45[valid].reshape(-1, 1), n_components=n_components, n_init=3
            )
        except RuntimeError as e:
            warn_msg = f"GMM CD45 échoué: {e} — fallback percentile"
            print(f"   [!] {warn_msg}")
            log_gating_event(
                "cd45",
                "gmm_fallback_percentile",
                "fallback",
                {"error": str(e)},
                warn_msg,
            )
            threshold = np.nanpercentile(cd45[valid], threshold_percentile)
            mask = np.zeros(n_cells, dtype=bool)
            mask[valid] = cd45[valid] > threshold

            gate_result = GateResult(
                mask=mask,
                n_kept=int(mask.sum()),
                n_total=int(valid.sum()),
                method="gmm_cd45_fallback_percentile",
                gate_name="G3_cd45",
                details={"threshold": float(threshold), "fallback": True},
                warnings=[warn_msg],
            )
            gating_reports.append(gate_result)
            return mask

        labels = gmm.predict(cd45[valid].reshape(-1, 1))
        means = gmm.means_.flatten()

        # CD45+ = composant avec la moyenne la plus élevée
        pos_component = np.argmax(means)

        # Calculer le seuil approximatif (intersection des 2 gaussiennes)
        sorted_means = np.sort(means)
        stds = np.sqrt(gmm.covariances_.flatten())
        sorted_stds = stds[np.argsort(means)]
        threshold_approx = (
            sorted_means[0] * sorted_stds[1] + sorted_means[1] * sorted_stds[0]
        ) / (sorted_stds[0] + sorted_stds[1])

        mask = np.zeros(n_cells, dtype=bool)
        mask[valid] = labels == pos_component

        n_pos = mask.sum()
        print(f"   [Auto-GMM] CD45: {n_components} composantes, μ={means.round(0)}")
        print(
            f"   [Auto-GMM] Seuil adaptatif ≈ {threshold_approx:.0f} (creux entre populations)"
        )
        print(
            f"   [Auto-GMM] CD45+ identifiés: {n_pos:,} ({n_pos / valid.sum() * 100:.1f}%)"
        )

        gate_result = GateResult(
            mask=mask,
            n_kept=int(n_pos),
            n_total=int(valid.sum()),
            method="gmm_cd45",
            gate_name="G3_cd45",
            details={
                "means": means.tolist(),
                "threshold": float(threshold_approx),
                "n_components": int(n_components),
                "fallback": False,
            },
        )
        gating_reports.append(gate_result)
        log_gating_event(
            "cd45",
            "gmm",
            "success",
            {
                "means": means.tolist(),
                "threshold": float(threshold_approx),
                "n_pos": int(n_pos),
            },
        )

        return mask

    @staticmethod
    def auto_gate_cd34(
        X: np.ndarray,
        var_names: List[str],
        use_ssc_filter: bool = True,
        n_components: int = 2,
    ) -> np.ndarray:
        """
        Gate CD34+ blastes adaptatif par GMM.

        Identifie la population CD34 bright (blastes) par GMM au lieu d'un
        percentile fixe. Optionnel: combine avec SSC low (blastes = faible granularité).

        Args:
            X: Matrice des données
            var_names: Noms des marqueurs
            use_ssc_filter: Combiner avec filtre GMM SSC low
            n_components: Nombre de composantes GMM

        Returns:
            Masque booléen (True = blaste CD34+, False = autre)
        """
        n_cells = X.shape[0]
        cd34_idx = PreGating.find_marker_index(
            var_names, ["CD34", "CD34-PE", "CD34-APC", "CD34-PECY7"]
        )

        if cd34_idx is None:
            print("[!] CD34 non trouvé pour auto-gate blastes")
            return np.ones(n_cells, dtype=bool)

        cd34 = X[:, cd34_idx].astype(np.float64)
        valid = np.isfinite(cd34)

        if valid.sum() < 200:
            print("[!] Pas assez de données valides pour auto-gate CD34")
            return np.ones(n_cells, dtype=bool)

        # GMM pour séparer CD34- et CD34+
        try:
            gmm = AutoGating.safe_fit_gmm(
                cd34[valid].reshape(-1, 1), n_components=n_components, n_init=3
            )
        except RuntimeError as e:
            warn_msg = f"GMM CD34 échoué: {e} — conservation de toutes les cellules"
            print(f"   [!] {warn_msg}")
            log_gating_event("cd34", "gmm", "error", {"error": str(e)}, warn_msg)
            return np.ones(n_cells, dtype=bool)

        labels = gmm.predict(cd34[valid].reshape(-1, 1))
        means = gmm.means_.flatten()
        pos_component = np.argmax(means)

        mask_cd34 = np.zeros(n_cells, dtype=bool)
        mask_cd34[valid] = labels == pos_component

        n_cd34_pos = mask_cd34.sum()
        print(
            f"   [Auto-GMM] CD34: μ={means.round(0)}, CD34+ cluster = μ={means[pos_component]:.0f}"
        )

        # Filtre SSC low optionnel (blastes = faible granularité)
        if use_ssc_filter:
            ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A", "SSC-H", "SSC"])
            if ssc_idx is not None:
                ssc = X[:, ssc_idx].astype(np.float64)
                valid_ssc = np.isfinite(ssc)

                if valid_ssc.sum() >= 200:
                    try:
                        gmm_ssc = AutoGating.safe_fit_gmm(
                            ssc[valid_ssc].reshape(-1, 1), n_components=2, n_init=3
                        )
                    except RuntimeError as e:
                        print(f"   [!] GMM SSC échoué: {e} — filtre SSC ignoré")
                        print(f"   [Auto-GMM] CD34+ blastes: {n_cd34_pos:,}")
                        gate_result = GateResult(
                            mask=mask_cd34,
                            n_kept=int(n_cd34_pos),
                            n_total=int(valid.sum()),
                            method="gmm_cd34_no_ssc",
                            gate_name="G4_cd34",
                            details={"means": means.tolist(), "ssc_filter": False},
                            warnings=[f"GMM SSC échoué: {e}"],
                        )
                        gating_reports.append(gate_result)
                        return mask_cd34

                    labels_ssc = gmm_ssc.predict(ssc[valid_ssc].reshape(-1, 1))
                    ssc_means = gmm_ssc.means_.flatten()
                    low_ssc_component = np.argmin(ssc_means)

                    mask_ssc = np.zeros(n_cells, dtype=bool)
                    mask_ssc[valid_ssc] = labels_ssc == low_ssc_component

                    combined = mask_cd34 & mask_ssc
                    print(
                        f"   [Auto-GMM] + Filtre SSC low (μ={ssc_means[low_ssc_component]:.0f}): {combined.sum():,} blastes purs"
                    )

                    gate_result = GateResult(
                        mask=combined,
                        n_kept=int(combined.sum()),
                        n_total=int(valid.sum()),
                        method="gmm_cd34_ssc",
                        gate_name="G4_cd34",
                        details={
                            "cd34_means": means.tolist(),
                            "ssc_means": ssc_means.tolist(),
                            "ssc_filter": True,
                        },
                    )
                    gating_reports.append(gate_result)
                    log_gating_event(
                        "cd34",
                        "gmm+ssc",
                        "success",
                        {
                            "cd34_means": means.tolist(),
                            "ssc_means": ssc_means.tolist(),
                            "n_blastes": int(combined.sum()),
                        },
                    )
                    return combined

        print(f"   [Auto-GMM] CD34+ blastes: {n_cd34_pos:,}")
        gate_result = GateResult(
            mask=mask_cd34,
            n_kept=int(n_cd34_pos),
            n_total=int(valid.sum()),
            method="gmm_cd34",
            gate_name="G4_cd34",
            details={"means": means.tolist()},
        )
        gating_reports.append(gate_result)
        return mask_cd34
