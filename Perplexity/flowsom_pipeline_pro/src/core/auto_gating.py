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

from ..models.gate_result import GateResult
from .gating import PreGating
from flowsom_pipeline_pro.config.constants import (
    GMM_MAX_SAMPLES,
    RANSAC_R2_THRESHOLD,
    RANSAC_MAD_FACTOR,
)


class AutoGating:
    """
    Gating automatique adaptatif (mode 'auto' dans la config).

    Dépendances: scikit-learn (GaussianMixture, RANSACRegressor, StandardScaler).
    Si scikit-learn est absent, les méthodes lèvent ImportError.
    """

    GMM_MAX_SAMPLES: int = GMM_MAX_SAMPLES
    RANSAC_R2_THRESHOLD: float = RANSAC_R2_THRESHOLD

    # ------------------------------------------------------------------
    # Utilitaires GMM internes
    # ------------------------------------------------------------------

    @staticmethod
    def _subsample(
        data: np.ndarray,
        max_samples: int = GMM_MAX_SAMPLES,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Sous-échantillonne les données si elles dépassent max_samples.
        Améliore la vitesse de convergence du GMM sur de grands jeux.

        Args:
            data: Données (n_samples, n_features).
            max_samples: Nombre max de points.
            seed: Graine aléatoire.

        Returns:
            Données sous-échantillonnées ou originales si < max_samples.
        """
        if data.shape[0] > max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(data.shape[0], size=max_samples, replace=False)
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
        Wrapper robuste pour le fitting GMM avec gestion d'erreurs et fallback.

        Tente le fit plusieurs fois avec différentes initialisations.
        En cas d'échec total sur n_components > 1, bascule sur 1 composante.
        Un sous-échantillonnage est appliqué par défaut pour accélérer la convergence.

        Args:
            data: Données de forme (n_samples, n_features) ou (n_samples, 1).
            n_components: Nombre de composantes GMM.
            n_init: Nombre d'initialisations par tentative.
            max_retries: Tentatives avant fallback.
            random_state: Graine de reproductibilité.
            covariance_type: Type de matrice de covariance.
            max_iter: Nombre max d'itérations EM.
            subsample: Activer le sous-échantillonnage automatique.

        Returns:
            GaussianMixture fitté et convergé.

        Raises:
            RuntimeError: Si toutes les tentatives échouent.
        """
        from sklearn.mixture import GaussianMixture

        data_fit = AutoGating._subsample(data) if subsample else data
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
                if gmm.converged_:
                    return gmm
                warnings.warn(
                    f"GMM non-convergé (n={n_components}, tentative {attempt + 1}/{max_retries})"
                )
            except Exception as e:
                last_error = e

        # Fallback sur 1 composante
        if n_components > 1:
            warnings.warn(f"GMM fallback unimodal après {max_retries} échecs")
            try:
                gmm = GaussianMixture(
                    n_components=1,
                    random_state=random_state,
                    covariance_type=covariance_type,
                    max_iter=max_iter,
                )
                gmm.fit(data_fit)
                return gmm
            except Exception as e:
                raise RuntimeError(
                    f"GMM fit échoué après {max_retries} tentatives + fallback: {e}"
                ) from e

        raise RuntimeError(
            f"GMM fit échoué après {max_retries} tentatives: {last_error}"
        )

    # ------------------------------------------------------------------
    # Gate 1 — Débris (GMM 2D FSC-A/SSC-A)
    # ------------------------------------------------------------------

    @staticmethod
    def auto_gate_debris(
        X: np.ndarray,
        var_names: List[str],
        n_components: int = 3,
        min_cluster_fraction: float = 0.02,
    ) -> GateResult:
        """
        Gate débris adaptatif par GMM 2D sur FSC-A / SSC-A.

        Le GMM identifie les clusters naturels dans l'espace FSC/SSC.
        Les clusters avec FSC-A moyen < 25% du cluster principal ET/OU
        représentant < min_cluster_fraction des événements sont exclus.

        Le nombre de composantes est automatiquement sélectionné par BIC.

        Args:
            X: Matrice de données brutes.
            var_names: Noms des marqueurs.
            n_components: Nombre max de composantes GMM à tester.
            min_cluster_fraction: Fraction min pour inclure un cluster.

        Returns:
            GateResult avec masque booléen.
        """
        from sklearn.preprocessing import StandardScaler

        n_cells = X.shape[0]
        fsc_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
        ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A"])

        if fsc_idx is None or ssc_idx is None:
            warnings.warn("FSC-A ou SSC-A non trouvé pour auto-gate débris")
            return GateResult(
                mask=np.ones(n_cells, dtype=bool),
                n_kept=n_cells,
                n_total=n_cells,
                method="skip_no_fsc_ssc",
                gate_name="G1_debris",
                warnings=["FSC-A ou SSC-A non trouvé"],
            )

        fsc = X[:, fsc_idx].astype(np.float64)
        ssc = X[:, ssc_idx].astype(np.float64)
        valid = np.isfinite(fsc) & np.isfinite(ssc)
        data_2d = np.column_stack([fsc[valid], ssc[valid]])

        if valid.sum() < 200:
            return GateResult(
                mask=np.ones(n_cells, dtype=bool),
                n_kept=n_cells,
                n_total=n_cells,
                method="skip_insufficient_data",
                gate_name="G1_debris",
                warnings=["Données insuffisantes pour GMM débris"],
            )

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_2d)

        # Sélection du meilleur modèle par BIC
        best_bic, best_gmm = np.inf, None
        for n in [2, 3]:
            try:
                g = AutoGating.safe_fit_gmm(data_scaled, n_components=n)
                bic_sample = AutoGating._subsample(data_scaled)
                bic = g.bic(bic_sample)
                if bic < best_bic:
                    best_bic, best_gmm = bic, g
            except RuntimeError:
                continue

        if best_gmm is None:
            warnings.warn("Aucun GMM débris convergé — toutes cellules conservées")
            return GateResult(
                mask=np.ones(n_cells, dtype=bool),
                n_kept=n_cells,
                n_total=n_cells,
                method="fallback_no_gmm",
                gate_name="G1_debris",
                warnings=["Aucun GMM convergé"],
            )

        labels = best_gmm.predict(data_scaled)
        n_comp = best_gmm.n_components
        sizes = np.bincount(labels, minlength=n_comp)
        fsc_means = np.array([data_2d[labels == i, 0].mean() for i in range(n_comp)])
        main_cluster = np.argmax(sizes)
        fsc_thr = fsc_means[main_cluster] * 0.25

        mask_valid = np.zeros(valid.sum(), dtype=bool)
        for i in range(n_comp):
            if (
                sizes[i] / len(labels) >= min_cluster_fraction
                and fsc_means[i] >= fsc_thr
            ):
                mask_valid |= labels == i

        if not mask_valid.any():
            mask_valid = labels == main_cluster

        mask = np.zeros(n_cells, dtype=bool)
        mask[valid] = mask_valid

        return GateResult(
            mask=mask,
            n_kept=int(mask.sum()),
            n_total=int(n_cells),
            method="auto_gmm_debris",
            gate_name="G1_debris",
            details={
                "n_components": int(n_comp),
                "bic": float(best_bic),
                "cluster_fsc_means": fsc_means.tolist(),
                "cluster_sizes": sizes.tolist(),
            },
        )

    # ------------------------------------------------------------------
    # Gate 2 — Singlets (RANSAC FSC-A vs FSC-H)
    # ------------------------------------------------------------------

    @staticmethod
    def auto_gate_singlets(
        X: np.ndarray,
        var_names: List[str],
        file_origin: Optional[np.ndarray] = None,
        per_file: bool = True,
        r2_threshold: float = RANSAC_R2_THRESHOLD,
    ) -> GateResult:
        """
        Gate singlets adaptatif par régression RANSAC (FSC-A vs FSC-H).

        Les singlets forment une diagonale robuste dans l'espace FSC-A/FSC-H.
        RANSAC identifie cette droite tout en ignorant les doublets (outliers).

        Qualité: R² > r2_threshold → RANSAC valide.
                  R² < r2_threshold → fallback ratio FSC-A/FSC-H simple.

        Pour plus de détails sur la stratégie par fichier, cf. le module
        flowsom_pipeline.py original (lignes 937–1180).

        Args:
            X: Matrice de données.
            var_names: Noms des marqueurs.
            file_origin: Vecteur d'origine (un label par cellule) pour gating par fichier.
            per_file: Si True, applique le RANSAC séparément par fichier.
            r2_threshold: Seuil R² minimum (0.85 par défaut).

        Returns:
            GateResult avec masque booléen.
        """
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        from sklearn.metrics import r2_score

        n_cells = X.shape[0]
        fsc_a_idx = PreGating.find_marker_index(var_names, ["FSC-A"])
        fsc_h_idx = PreGating.find_marker_index(var_names, ["FSC-H"])

        if fsc_a_idx is None or fsc_h_idx is None:
            return GateResult(
                mask=np.ones(n_cells, dtype=bool),
                n_kept=n_cells,
                n_total=n_cells,
                method="skip_no_fsc",
                gate_name="G2_singlets",
                warnings=["FSC-A ou FSC-H non trouvé"],
            )

        fsc_a = X[:, fsc_a_idx].astype(np.float64)
        fsc_h = X[:, fsc_h_idx].astype(np.float64)

        # Pré-filtre viable pour robustesse RANSAC
        viable = PreGating.gate_viable_cells(X, var_names, 1.0, 99.0)
        valid = (
            viable
            & np.isfinite(fsc_a)
            & np.isfinite(fsc_h)
            & (fsc_h > 100)
            & (fsc_a > 100)
        )

        if valid.sum() < 200:
            return GateResult(
                mask=np.ones(n_cells, dtype=bool),
                n_kept=n_cells,
                n_total=n_cells,
                method="skip_insufficient_data",
                gate_name="G2_singlets",
                warnings=["Données insuffisantes pour RANSAC"],
            )

        mask = np.zeros(n_cells, dtype=bool)
        files_summary: List[Dict] = []
        ransac_scatter_data: Dict[
            str, Dict
        ] = {}  # Per-file RANSAC metrics for QC plots

        def _ransac_one_block(fsc_a_block, fsc_h_block):
            """Applique RANSAC sur un bloc de données, retourne (masque, r2, method, metrics)."""
            metrics: Dict = {}
            try:
                ransac = RANSACRegressor(
                    estimator=LinearRegression(),
                    min_samples=50,
                    residual_threshold=None,
                    random_state=42,
                    max_trials=100,
                )
                ransac.fit(fsc_h_block.reshape(-1, 1), fsc_a_block)
                inliers = ransac.inlier_mask_
                r2_val = None
                if inliers is not None and inliers.sum() > 50:
                    r2_val = r2_score(
                        fsc_a_block[inliers],
                        ransac.predict(fsc_h_block[inliers].reshape(-1, 1)),
                    )

                pred = ransac.predict(fsc_h_block.reshape(-1, 1))
                slope = float(ransac.estimator_.coef_[0])
                intercept = float(ransac.estimator_.intercept_)

                metrics = {
                    "slope": slope,
                    "intercept": intercept,
                    "r2": float(r2_val) if r2_val is not None else None,
                    "inlier_mask": inliers,
                    "pred": pred,
                }

                if r2_val is not None and r2_val < r2_threshold:
                    # Fallback ratio
                    ratio = fsc_a_block / np.maximum(fsc_h_block, 1.0)
                    return (
                        (ratio >= 0.6) & (ratio <= 1.5),
                        r2_val,
                        "ratio_fallback",
                        metrics,
                    )

                residuals = fsc_a_block - pred
                med = np.median(residuals)
                mad = np.median(np.abs(residuals - med))
                singlets = residuals <= (med + RANSAC_MAD_FACTOR * mad)
                return singlets, r2_val, "ransac", metrics

            except Exception as e:
                warnings.warn(
                    f"RANSAC échoué: {e} — conservation de tous les événements"
                )
                return (
                    np.ones(len(fsc_a_block), dtype=bool),
                    None,
                    "error_keep_all",
                    metrics,
                )

        if per_file and file_origin is not None:
            for fname in np.unique(file_origin):
                file_mask = (file_origin == fname) & valid
                if file_mask.sum() < 50:
                    mask[file_mask] = True
                    continue
                fa = fsc_a[file_mask]
                fh = fsc_h[file_mask]
                result_mask, r2_val, method_used, metrics = _ransac_one_block(fa, fh)
                file_indices = np.where(file_mask)[0]
                mask[file_indices] = result_mask
                files_summary.append(
                    {
                        "file": str(fname),
                        "n_total": int(len(fa)),
                        "n_singlets": int(result_mask.sum()),
                        "pct_singlets": round(result_mask.mean() * 100, 1),
                        "method": method_used,
                        "r2": round(float(r2_val), 3) if r2_val is not None else None,
                    }
                )
                # Stocker les données scatter RANSAC pour les QC plots
                ransac_scatter_data[str(fname)] = {
                    "fsc_h": fh,
                    "fsc_a": fa,
                    "r2": float(r2_val) if r2_val is not None else None,
                    "method": method_used,
                    "pct_singlets": round(result_mask.mean() * 100, 1),
                    "slope": metrics.get("slope"),
                    "intercept": metrics.get("intercept"),
                }
        else:
            fa = fsc_a[valid]
            fh = fsc_h[valid]
            result_mask, r2_val, _, metrics = _ransac_one_block(fa, fh)
            mask[valid] = result_mask
            ransac_scatter_data["global"] = {
                "fsc_h": fh,
                "fsc_a": fa,
                "r2": float(r2_val) if r2_val is not None else None,
                "method": "ransac",
                "slope": metrics.get("slope"),
                "intercept": metrics.get("intercept"),
            }

        return GateResult(
            mask=mask,
            n_kept=int(mask.sum()),
            n_total=int(n_cells),
            method="ransac_singlets",
            gate_name="G2_singlets",
            details={
                "per_file": per_file,
                "n_files": len(files_summary),
                "files_summary": files_summary,
                "ransac_scatter_data": ransac_scatter_data,
            },
        )

    # ------------------------------------------------------------------
    # Gate 3 — CD45+ (GMM 1D bimodal)
    # ------------------------------------------------------------------

    @staticmethod
    def auto_gate_cd45(
        X: np.ndarray,
        var_names: List[str],
        n_components: int = 2,
        threshold_percentile: float = 5.0,
    ) -> GateResult:
        """
        Gate CD45+ adaptatif par GMM 1D bimodal.

        Détecte automatiquement le creux entre la population CD45- (débris,
        globules rouges) et la population CD45+ (leucocytes).
        Fallback vers percentile si le GMM échoue.

        Args:
            X: Matrice de données.
            var_names: Noms des marqueurs.
            n_components: Nombre de composantes GMM (2 = CD45-/CD45+).
            threshold_percentile: Percentile pour le fallback si GMM échoue.

        Returns:
            GateResult avec masque booléen.
        """
        n_cells = X.shape[0]
        cd45_idx = PreGating.find_marker_index(
            var_names, ["CD45", "CD45-PECY5", "CD45-PC5"]
        )
        if cd45_idx is None:
            return GateResult(
                mask=np.ones(n_cells, dtype=bool),
                n_kept=n_cells,
                n_total=n_cells,
                method="skip_no_cd45",
                gate_name="G3_cd45",
                warnings=["CD45 non trouvé"],
            )

        cd45 = X[:, cd45_idx].astype(np.float64)
        valid = np.isfinite(cd45)

        if valid.sum() < 200:
            return GateResult(
                mask=np.ones(n_cells, dtype=bool),
                n_kept=n_cells,
                n_total=n_cells,
                method="skip_insufficient_data",
                gate_name="G3_cd45",
                warnings=["Données CD45 insuffisantes"],
            )

        def _fallback_percentile():
            thr = np.nanpercentile(cd45[valid], threshold_percentile)
            m = np.zeros(n_cells, dtype=bool)
            m[valid] = cd45[valid] > thr
            return GateResult(
                mask=m,
                n_kept=int(m.sum()),
                n_total=int(valid.sum()),
                method="fallback_percentile_cd45",
                gate_name="G3_cd45",
                details={"threshold": float(thr), "fallback": True},
            )

        try:
            gmm = AutoGating.safe_fit_gmm(
                cd45[valid].reshape(-1, 1), n_components=n_components, n_init=3
            )
        except RuntimeError as e:
            warnings.warn(f"GMM CD45 échoué: {e}")
            return _fallback_percentile()

        labels = gmm.predict(cd45[valid].reshape(-1, 1))
        means = gmm.means_.flatten()
        pos_component = int(np.argmax(means))

        # Seuil approximatif par intersection des gaussiennes pondérée
        sorted_means = np.sort(means)
        stds = np.sqrt(gmm.covariances_.flatten())
        sorted_stds = stds[np.argsort(means)]
        denom = sorted_stds[0] + sorted_stds[1]
        threshold_approx = (
            (sorted_means[0] * sorted_stds[1] + sorted_means[1] * sorted_stds[0])
            / denom
            if denom > 0
            else sorted_means.mean()
        )

        mask = np.zeros(n_cells, dtype=bool)
        mask[valid] = labels == pos_component

        return GateResult(
            mask=mask,
            n_kept=int(mask.sum()),
            n_total=int(valid.sum()),
            method="gmm_cd45",
            gate_name="G3_cd45",
            details={
                "means": means.tolist(),
                "threshold_approx": float(threshold_approx),
                "n_components": int(n_components),
            },
        )

    # ------------------------------------------------------------------
    # Gate 4 — Blastes CD34+ (GMM 1D + SSC optionnel)
    # ------------------------------------------------------------------

    @staticmethod
    def auto_gate_cd34(
        X: np.ndarray,
        var_names: List[str],
        use_ssc_filter: bool = True,
        n_components: int = 2,
    ) -> GateResult:
        """
        Gate CD34+ blastes adaptatif par GMM.

        Identifie la population CD34 bright (blastes) sans percentile fixe.
        Combine optionnellement avec un filtre SSC low (blastes = faible granularité).

        Args:
            X: Matrice de données.
            var_names: Noms des marqueurs.
            use_ssc_filter: Combiner avec GMM SSC low pour enrichir en blastes.
            n_components: Nombre de composantes GMM.

        Returns:
            GateResult avec masque booléen.
        """
        n_cells = X.shape[0]
        cd34_idx = PreGating.find_marker_index(
            var_names, ["CD34", "CD34-PE", "CD34-APC", "CD34-PECY7"]
        )
        if cd34_idx is None:
            return GateResult(
                mask=np.ones(n_cells, dtype=bool),
                n_kept=n_cells,
                n_total=n_cells,
                method="skip_no_cd34",
                gate_name="G4_cd34",
                warnings=["CD34 non trouvé"],
            )

        cd34 = X[:, cd34_idx].astype(np.float64)
        valid = np.isfinite(cd34)

        if valid.sum() < 200:
            return GateResult(
                mask=np.ones(n_cells, dtype=bool),
                n_kept=n_cells,
                n_total=n_cells,
                method="skip_insufficient_data",
                gate_name="G4_cd34",
                warnings=["Données CD34 insuffisantes"],
            )

        try:
            gmm = AutoGating.safe_fit_gmm(
                cd34[valid].reshape(-1, 1), n_components=n_components, n_init=3
            )
        except RuntimeError as e:
            warnings.warn(f"GMM CD34 échoué: {e}")
            return GateResult(
                mask=np.ones(n_cells, dtype=bool),
                n_kept=n_cells,
                n_total=n_cells,
                method="error_gmm_cd34",
                gate_name="G4_cd34",
                warnings=[str(e)],
            )

        labels = gmm.predict(cd34[valid].reshape(-1, 1))
        means = gmm.means_.flatten()
        pos_component = int(np.argmax(means))

        mask_cd34 = np.zeros(n_cells, dtype=bool)
        mask_cd34[valid] = labels == pos_component

        if not use_ssc_filter:
            return GateResult(
                mask=mask_cd34,
                n_kept=int(mask_cd34.sum()),
                n_total=int(valid.sum()),
                method="gmm_cd34",
                gate_name="G4_cd34",
                details={"means": means.tolist()},
            )

        # Filtre SSC low optionnel
        ssc_idx = PreGating.find_marker_index(var_names, ["SSC-A", "SSC-H", "SSC"])
        if ssc_idx is None:
            return GateResult(
                mask=mask_cd34,
                n_kept=int(mask_cd34.sum()),
                n_total=int(valid.sum()),
                method="gmm_cd34_no_ssc",
                gate_name="G4_cd34",
                details={"means": means.tolist(), "ssc_filter": False},
            )

        ssc = X[:, ssc_idx].astype(np.float64)
        valid_ssc = np.isfinite(ssc)

        try:
            gmm_ssc = AutoGating.safe_fit_gmm(
                ssc[valid_ssc].reshape(-1, 1), n_components=2, n_init=3
            )
        except RuntimeError as e:
            warnings.warn(f"GMM SSC échoué: {e} — filtre SSC ignoré")
            return GateResult(
                mask=mask_cd34,
                n_kept=int(mask_cd34.sum()),
                n_total=int(valid.sum()),
                method="gmm_cd34_no_ssc",
                gate_name="G4_cd34",
                warnings=[f"GMM SSC échoué: {e}"],
            )

        labels_ssc = gmm_ssc.predict(ssc[valid_ssc].reshape(-1, 1))
        ssc_means = gmm_ssc.means_.flatten()
        low_ssc = int(np.argmin(ssc_means))
        mask_ssc = np.zeros(n_cells, dtype=bool)
        mask_ssc[valid_ssc] = labels_ssc == low_ssc

        combined = mask_cd34 & mask_ssc
        return GateResult(
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
