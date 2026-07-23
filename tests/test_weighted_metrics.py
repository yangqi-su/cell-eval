import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest

from cell_eval import CombinedMetricData, MetricPipeline, PerturbationAnndataPair
from cell_eval._types import initialize_de_comparison
from cell_eval.metrics import deg_gene_weights, weighted_pearson_delta, wmse


GENES = np.array(["gene_a", "gene_b", "gene_c"])


def _de_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "target": ["pert", "pert", "pert"],
            "feature": ["gene_c", "gene_a", "gene_b"],
            "log2_fold_change": [0.0, 2.0, 1.0],
            "p_value": [1.0, 1e-8, 0.05],
            "fdr": [1.0, 3e-8, 0.075],
        }
    )


def _combined_data(perfect_prediction: bool = False) -> CombinedMetricData:
    obs = pd.DataFrame(
        {"perturbation": ["control", "control", "pert", "pert"]},
        index=pd.Index(["c1", "c2", "p1", "p2"]),
    )
    real_values = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
        ]
    )
    pred_values = real_values.copy()
    if not perfect_prediction:
        pred_values[2:, :] = np.array([[3.0, 1.0, 2.0], [3.0, 1.0, 2.0]])

    var = pd.DataFrame(index=pd.Index(GENES))
    real = ad.AnnData(X=real_values, obs=obs.copy(), var=var.copy())
    pred = ad.AnnData(X=pred_values, obs=obs.copy(), var=var.copy())

    pair = PerturbationAnndataPair(
        real=real,
        pred=pred,
        pert_col="perturbation",
        control_pert="control",
    )
    de = initialize_de_comparison(real=_de_frame(), pred=_de_frame())
    return CombinedMetricData(pair, de)


def test_deg_gene_weights_align_to_gene_order() -> None:
    data = _combined_data()
    weights = deg_gene_weights(data.de_comparison.real, GENES, "pert")

    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)
    assert weights[0] > weights[1] > weights[2]


def test_weighted_metrics_are_perfect_for_identical_prediction() -> None:
    data = _combined_data(perfect_prediction=True)

    assert wmse(data)["pert"] == 0.0
    assert np.isclose(weighted_pearson_delta(data)["pert"], 1.0)


def test_uniform_weights_when_de_scores_are_uninformative() -> None:
    frame = _de_frame().with_columns(pl.lit(1.0).alias("p_value"))
    de = initialize_de_comparison(real=frame, pred=frame)

    weights = deg_gene_weights(de.real, GENES, "pert")

    np.testing.assert_allclose(weights, np.full(3, 1 / 3))


def test_unknown_weight_source_is_rejected() -> None:
    data = _combined_data()
    with pytest.raises(ValueError, match="Unknown weight source"):
        deg_gene_weights(
            data.de_comparison.real,
            GENES,
            "pert",
            source="unknown",  # ty: ignore[invalid-argument-type]
        )


def test_weighted_metrics_run_through_pipeline() -> None:
    data = _combined_data(perfect_prediction=True)
    pipeline = MetricPipeline(profile=None)
    pipeline.add_metrics(["wmse", "weighted_pearson_delta"])

    pipeline.compute_combined_metrics(data.anndata_pair, data.de_comparison)
    results = pipeline.get_results()

    assert "wmse" in results.columns
    assert "weighted_pearson_delta" in results.columns
