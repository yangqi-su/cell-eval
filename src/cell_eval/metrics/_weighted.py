"""DEG-weighted pseudobulk expression metrics."""

from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.stats import norm

from .._types import CombinedMetricData, DEResults

WeightSource = Literal["p_value", "abs_log2_fold_change"]


def deg_gene_weights(
    de_real: DEResults,
    genes: NDArray[np.str_],
    perturbation: str,
    source: WeightSource = "p_value",
) -> NDArray[np.float64]:
    """Build Needles-style gene weights from real perturbation-vs-control DE."""
    if source not in ("p_value", "abs_log2_fold_change"):
        raise ValueError(f"Unknown weight source: {source}")

    frame = de_real.data.filter(pl.col(de_real.target_col) == perturbation)
    score_col = (
        de_real.pvalue_col if source == "p_value" else de_real.abs_log2_fold_change_col
    )
    values = dict(frame.select(de_real.feature_col, score_col).iter_rows())
    missing_value = 1.0 if source == "p_value" else 0.0
    raw = np.array([values.get(gene, missing_value) for gene in genes], dtype=float)

    if source == "p_value":
        # pdex uses a two-sided Mann-Whitney test. Convert p-values to an
        # absolute normal-equivalent score so a stronger DE result weighs more.
        raw = norm.isf(np.clip(raw, np.finfo(float).tiny, 1.0) / 2)
    else:
        raw = np.abs(raw)

    finite = raw[np.isfinite(raw)]
    max_finite = finite.max(initial=0.0)
    raw = np.nan_to_num(raw, nan=0.0, posinf=max_finite, neginf=0.0)
    low, high = raw.min(), raw.max()
    if high <= low:
        return np.full(len(genes), 1.0 / len(genes))

    weights = ((raw - low) / (high - low)) ** 2
    total = weights.sum()
    if total == 0:
        return np.full(len(genes), 1.0 / len(genes))
    return weights / total


def wmse(
    data: CombinedMetricData,
    weight_source: WeightSource = "p_value",
) -> dict[str, float]:
    """Weighted MSE between predicted and real perturbation pseudobulks."""
    pair = data.anndata_pair
    results = {}
    for bulk in pair.iter_bulk_arrays():
        weights = deg_gene_weights(
            data.de_comparison.real, pair.genes, bulk.key, source=weight_source
        )
        results[bulk.key] = float(
            np.sum(weights * (bulk.pert_pred - bulk.pert_real) ** 2)
        )
    return results


def weighted_pearson_delta(
    data: CombinedMetricData,
    weight_source: WeightSource = "p_value",
) -> dict[str, float]:
    """Weighted Pearson correlation of perturbation-minus-control effects."""
    pair = data.anndata_pair
    results = {}
    for bulk in pair.iter_bulk_arrays():
        weights = deg_gene_weights(
            data.de_comparison.real, pair.genes, bulk.key, source=weight_source
        )
        real = bulk.perturbation_effect(which="real", abs=False)
        pred = bulk.perturbation_effect(which="pred", abs=False)
        real_centered = real - np.sum(weights * real)
        pred_centered = pred - np.sum(weights * pred)
        denominator = np.sqrt(
            np.sum(weights * real_centered**2) * np.sum(weights * pred_centered**2)
        )
        results[bulk.key] = (
            float(np.sum(weights * real_centered * pred_centered) / denominator)
            if denominator > 0
            else float("nan")
        )
    return results
