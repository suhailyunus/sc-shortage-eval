from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# PSI bucketing follows the conventional industry cutoffs: below this,
# a shift is treated as noise rather than drift.
DEFAULT_PSI_THRESHOLD = 0.20
DEFAULT_KS_ALPHA = 0.05
PSI_BUCKETS = 10


@dataclass(frozen=True)
class FeatureReference:
    """Reference distribution for one feature, captured at training time."""

    feature: str
    quantile_edges: list[float]
    sample: list[float]


@dataclass(frozen=True)
class FeatureDriftResult:
    feature: str
    ks_statistic: float
    ks_pvalue: float
    psi: float
    drifted: bool
    reason: str


def _psi_bucket_edges(reference: np.ndarray, buckets: int = PSI_BUCKETS) -> np.ndarray:
    """Quantile-based bucket edges from the reference sample.

    Quantile buckets (rather than fixed-width bins) keep each reference
    bucket populated even for skewed features like ``sales_lag_1``, where
    equal-width bins would leave most of the mass in one bucket and make
    PSI meaningless.
    """

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, buckets + 1))
    )
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _population_stability_index(
    reference: np.ndarray, current: np.ndarray, buckets: int = PSI_BUCKETS
) -> float:
    """
    Population Stability Index between a reference and current sample.

    PSI sums, over each bucket, ``(current% - reference%) * ln(current% /
    reference%)``. Both tails are floored at a small epsilon so an empty
    bucket doesn't produce a divide-by-zero or a log of zero.
    """

    edges = _psi_bucket_edges(reference, buckets)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    epsilon = 1e-6
    ref_pct = ref_counts / max(len(reference), 1) + epsilon
    cur_pct = cur_counts / max(len(current), 1) + epsilon

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def build_feature_reference(
    training_features: pd.DataFrame,
    feature_names: list[str],
    *,
    sample_size: int = 5000,
    random_state: int = 42,
) -> dict[str, FeatureReference]:
    """
    Capture the training feature distribution to compare future batches against.

    Only a bounded sample is stored (not the full training set) so the
    reference artifact stays small; ``sample_size`` observations is enough
    to estimate PSI buckets and a KS statistic without shipping the whole
    training table.
    """

    references: dict[str, FeatureReference] = {}
    for feature in feature_names:
        values = training_features[feature].dropna().to_numpy(dtype=float)
        if len(values) > sample_size:
            rng = np.random.default_rng(random_state)
            values = rng.choice(values, size=sample_size, replace=False)

        edges = _psi_bucket_edges(values)
        references[feature] = FeatureReference(
            feature=feature,
            quantile_edges=[float(e) for e in edges],
            sample=[float(v) for v in values],
        )
    return references


def save_feature_reference(
    references: dict[str, FeatureReference], output_path: str | Path
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: asdict(ref) for name, ref in references.items()}
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_feature_reference(input_path: str | Path) -> dict[str, FeatureReference]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return {name: FeatureReference(**ref) for name, ref in payload.items()}


def compute_drift_report(
    references: dict[str, FeatureReference],
    current: pd.DataFrame,
    *,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_alpha: float = DEFAULT_KS_ALPHA,
    min_rows: int = 30,
) -> list[FeatureDriftResult]:
    """
    Compare an incoming scoring batch against the stored training reference.

    A feature is flagged only when *both* tests agree it's worth a look:
    the KS test alone is oversensitive on large batches (it will reject
    on trivial shifts once n is large), and PSI alone is insensitive to
    small samples. Requiring PSI over threshold AND a significant KS
    p-value avoids false alarms from either failure mode individually.
    """

    results = []
    for feature, reference in references.items():
        if feature not in current.columns:
            results.append(
                FeatureDriftResult(
                    feature=feature,
                    ks_statistic=float("nan"),
                    ks_pvalue=float("nan"),
                    psi=float("nan"),
                    drifted=False,
                    reason="feature absent from current batch",
                )
            )
            continue

        current_values = current[feature].dropna().to_numpy(dtype=float)
        if len(current_values) < min_rows:
            results.append(
                FeatureDriftResult(
                    feature=feature,
                    ks_statistic=float("nan"),
                    ks_pvalue=float("nan"),
                    psi=float("nan"),
                    drifted=False,
                    reason=f"fewer than {min_rows} rows; drift not evaluated",
                )
            )
            continue

        reference_values = np.array(reference.sample)
        ks_result = ks_2samp(reference_values, current_values)
        psi = _population_stability_index(reference_values, current_values)

        drifted = bool(psi >= psi_threshold and ks_result.pvalue < ks_alpha)
        reason = (
            f"PSI={psi:.3f} (>= {psi_threshold}) and KS p={ks_result.pvalue:.4f} "
            f"(< {ks_alpha})"
            if drifted
            else f"PSI={psi:.3f}, KS p={ks_result.pvalue:.4f} - within tolerance"
        )

        results.append(
            FeatureDriftResult(
                feature=feature,
                ks_statistic=float(ks_result.statistic),
                ks_pvalue=float(ks_result.pvalue),
                psi=float(psi),
                drifted=drifted,
                reason=reason,
            )
        )

    return results
