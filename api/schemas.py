from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HistoricalObservation(BaseModel):
    """One item-store-day observation used to reconstruct temporal features."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, examples=["HOBBIES_1_004"])
    store_id: str = Field(min_length=1, examples=["CA_3"])
    state_id: str = Field(min_length=1, examples=["CA"])
    day_num: int = Field(ge=1, examples=[1898])
    sales: float = Field(ge=0, examples=[4.0])
    weekday: str = Field(min_length=1, examples=["Saturday"])
    event_name_1: str | None = Field(default=None, examples=[None])
    sell_price: float = Field(gt=0, examples=[3.97])
    snap_CA: int = Field(ge=0, le=1, examples=[1])
    snap_TX: int = Field(ge=0, le=1, examples=[0])
    snap_WI: int = Field(ge=0, le=1, examples=[0])

    @field_validator("state_id")
    @classmethod
    def validate_state(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"CA", "TX", "WI"}:
            raise ValueError("state_id must be one of CA, TX, or WI")
        return normalized


class PredictionRequest(BaseModel):
    """Historical observations and an optional operating threshold."""

    model_config = ConfigDict(extra="forbid")

    observations: list[HistoricalObservation] = Field(
        min_length=8,
        description=(
            "Chronologically ordered history. At least eight rows per item-store "
            "series are generally required to create seven-day lag features."
        ),
    )
    threshold: float | None = Field(default=None, ge=0, le=1)


class PredictionRecord(BaseModel):
    item_id: str
    store_id: str
    state_id: str
    day_num: int
    stress_probability: float
    stress_prediction: int
    risk_label: str
    risk_level: str


class PredictionResponse(BaseModel):
    model_type: str
    threshold: float
    input_rows: int
    scored_rows: int
    predictions: list[PredictionRecord]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    service: str
    version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    positive_class_label: str
    default_threshold: float
    alternative_threshold: float | None
    feature_count: int
    features: list[str]


class ErrorResponse(BaseModel):
    detail: str | list[dict[str, Any]]


class DriftCheckRequest(BaseModel):
    """Recent historical rows to compare against the training reference."""

    model_config = ConfigDict(extra="forbid")

    observations: list[HistoricalObservation] = Field(min_length=1)


class FeatureDriftRecord(BaseModel):
    feature: str
    ks_statistic: float
    ks_pvalue: float
    psi: float
    drifted: bool
    reason: str


class DriftCheckResponse(BaseModel):
    n_rows_evaluated: int
    any_feature_drifted: bool
    features: list[FeatureDriftRecord]
    note: str = (
        "Statistical drift in an input feature does not by itself mean "
        "predictions are wrong - it means the incoming data no longer "
        "resembles what the model was trained on, and performance should "
        "be re-validated against real outcomes when they're available."
    )


class BusinessImpactResponse(BaseModel):
    """Static holdout-evaluation figures, not a live production metric.

    Computed offline by scripts/report_business_impact.py under cost
    assumptions supplied at report time. Served here for convenience;
    values are only as current as the last time that script was run.
    """

    cost_assumptions: dict[str, float]
    evaluated_on: str
    n_holdout_observations: int
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_cost: float
    cost_per_observation: float
    do_nothing_cost: float
    net_savings_vs_do_nothing: float
