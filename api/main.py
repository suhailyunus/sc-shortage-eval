from __future__ import annotations

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from api.schemas import (
    BusinessImpactResponse,
    DriftCheckRequest,
    DriftCheckResponse,
    ErrorResponse,
    FeatureDriftRecord,
    HealthResponse,
    ModelInfoResponse,
    PredictionRecord,
    PredictionRequest,
    PredictionResponse,
)
from src.features import prepare_model_input
from src.monitoring import compute_drift_report, load_feature_reference
from src.predict import load_model_artifacts

SERVICE_NAME = "Supply Chain Stress Prediction API"
SERVICE_VERSION = "1.1.0"
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MAX_INPUT_ROWS = int(os.getenv("MAX_INPUT_ROWS", "100000"))
BUSINESS_IMPACT_PATH = Path(os.getenv("BUSINESS_IMPACT_PATH", "artifacts/business_impact.json"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts once when the service starts."""

    try:
        model, features, config = load_model_artifacts(MODELS_DIR)
    except Exception as exc:  # pragma: no cover - startup failure path
        logger.exception("Unable to load model artifacts from %s", MODELS_DIR)
        app.state.model = None
        app.state.features = []
        app.state.config = {}
        app.state.load_error = str(exc)
    else:
        app.state.model = model
        app.state.features = features
        app.state.config = config
        app.state.load_error = None
        logger.info(
            "Loaded %s with %d features from %s",
            config.get("model_type", type(model).__name__),
            len(features),
            MODELS_DIR,
        )

    reference_path = MODELS_DIR / "feature_reference_stats.json"
    if reference_path.exists():
        try:
            app.state.drift_reference = load_feature_reference(reference_path)
            logger.info("Loaded drift reference distribution from %s", reference_path)
        except Exception:
            logger.exception("Found %s but failed to parse it", reference_path)
            app.state.drift_reference = None
    else:
        app.state.drift_reference = None

    yield

    app.state.model = None


app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description=(
        "Containerized inference service for ranking item-store observations by "
        "potential retail supply-stress risk. Submit recent historical rows so "
        "the service can recreate lag and rolling-window features."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


def _require_model() -> tuple[object, list[str], dict]:
    model = getattr(app.state, "model", None)
    if model is None:
        detail = getattr(app.state, "load_error", "Model is not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {detail}",
        )
    return model, app.state.features, app.state.config


def _validate_input_size(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise HTTPException(status_code=422, detail="No observations were supplied.")
    if len(frame) > MAX_INPUT_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Input exceeds MAX_INPUT_ROWS={MAX_INPUT_ROWS}.",
        )


def _risk_level(probability: float, threshold: float) -> str:
    """
    Translate a model score into a business-friendly severity band.

    Bands are derived from the configured alert threshold rather than
    from fixed constants. Hardcoded cutoffs decouple severity from the
    alert decision: at a 0.80 threshold, a fixed 0.60-0.80 "High" band
    would label items that are never actually flagged, and every flagged
    item would land in a single "Critical" bucket.

    The two bands below the threshold are advisory only; the two above
    it correspond to raised alerts, split at the midpoint of the
    remaining score range.
    """

    if probability >= threshold:
        critical_floor = threshold + (1.0 - threshold) / 2
        return "Critical" if probability >= critical_floor else "High"

    return "Moderate" if probability >= threshold / 2 else "Low"


def _score_output(frame: pd.DataFrame, threshold: float | None) -> tuple[pd.DataFrame, float, str]:
    """Score a frame once and return a tabular result for JSON or CSV delivery."""

    model, features, config = _require_model()
    _validate_input_size(frame)

    operating_threshold = (
        float(config["default_threshold"])
        if threshold is None
        else float(threshold)
    )
    if not 0 <= operating_threshold <= 1:
        raise HTTPException(status_code=422, detail="threshold must be between 0 and 1")

    try:
        scored_rows, model_input, _ = prepare_model_input(
            frame,
            expected_features=features,
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if model_input.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "No rows could be scored. Supply at least eight chronological rows "
                "per item-store series and ensure required source values are complete."
            ),
        )

    probabilities = model.predict_proba(model_input)[:, 1]
    predictions = (probabilities >= operating_threshold).astype("int8")

    output = scored_rows[
        ["item_id", "store_id", "state_id", "day_num"]
    ].copy()
    output["stress_probability"] = probabilities
    output["stress_prediction"] = predictions
    output["risk_label"] = output["stress_prediction"].map(
        {0: "No Stress", 1: "Stress Risk"}
    )
    output["risk_level"] = [
        _risk_level(float(probability), operating_threshold)
        for probability in probabilities
    ]

    return output, operating_threshold, str(
        config.get("model_type", type(model).__name__)
    )


def _score_frame(frame: pd.DataFrame, threshold: float | None) -> PredictionResponse:
    output, operating_threshold, model_type = _score_output(frame, threshold)

    records = [
        PredictionRecord(
            item_id=str(row.item_id),
            store_id=str(row.store_id),
            state_id=str(row.state_id),
            day_num=int(row.day_num),
            stress_probability=round(float(row.stress_probability), 6),
            stress_prediction=int(row.stress_prediction),
            risk_label=str(row.risk_label),
            risk_level=str(row.risk_level),
        )
        for row in output.itertuples(index=False)
    ]

    return PredictionResponse(
        model_type=model_type,
        threshold=operating_threshold,
        input_rows=len(frame),
        scored_rows=len(records),
        predictions=records,
    )


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Operations"],
)
def health() -> HealthResponse:
    loaded = getattr(app.state, "model", None) is not None
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
    )


@app.get(
    "/ready",
    response_model=HealthResponse,
    tags=["Operations"],
    responses={503: {"model": ErrorResponse}},
)
def ready() -> HealthResponse:
    _require_model()
    return HealthResponse(
        status="ready",
        model_loaded=True,
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
)
def model_info() -> ModelInfoResponse:
    _, features, config = _require_model()
    return ModelInfoResponse(
        model_type=str(config.get("model_type", "unknown")),
        positive_class_label=str(config.get("positive_class_label", "Supply Stress")),
        default_threshold=float(config["default_threshold"]),
        alternative_threshold=(
            float(config["alternative_threshold"])
            if config.get("alternative_threshold") is not None
            else None
        ),
        feature_count=len(features),
        features=features,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict(request: PredictionRequest) -> PredictionResponse:
    frame = pd.DataFrame(
        [observation.model_dump() for observation in request.observations]
    )
    return _score_frame(frame, request.threshold)


@app.post(
    "/predict-file",
    response_model=PredictionResponse,
    tags=["Inference"],
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict_file(
    file: Annotated[
        UploadFile,
        File(description="CSV containing recent historical rows"),
    ],
    threshold: float | None = None,
) -> PredictionResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Upload a .csv file.")

    content = await file.read()
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {exc}") from exc

    return _score_frame(frame, threshold)

@app.post(
    "/predict-file-csv",
    tags=["Inference"],
    summary="Upload historical rows and download predictions as CSV",
    responses={
        200: {
            "content": {"text/csv": {}},
            "description": "Downloadable predictions.csv file",
        },
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict_file_csv(
    file: Annotated[
        UploadFile,
        File(description="CSV containing recent historical rows"),
    ],
    threshold: float | None = None,
) -> StreamingResponse:
    """Return a business-ready CSV instead of a JSON response."""

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Upload a .csv file.")

    content = await file.read()
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid CSV: {exc}") from exc

    output, operating_threshold, model_type = _score_output(frame, threshold)
    output["stress_probability"] = output["stress_probability"].round(6)

    csv_buffer = io.StringIO()
    output.to_csv(csv_buffer, index=False)
    csv_bytes = io.BytesIO(csv_buffer.getvalue().encode("utf-8"))

    return StreamingResponse(
        csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="predictions.csv"',
            "X-Model-Type": model_type,
            "X-Threshold": str(operating_threshold),
            "X-Input-Rows": str(len(frame)),
            "X-Scored-Rows": str(len(output)),
        },
    )


@app.post(
    "/check-drift",
    response_model=DriftCheckResponse,
    tags=["Monitoring"],
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def check_drift(request: DriftCheckRequest) -> DriftCheckResponse:
    """
    Compare the feature distribution of a recent batch to the training reference.

    This checks input drift only - whether incoming data still looks like
    what the model was trained on. It cannot tell you whether predictions
    are still accurate, since that requires labels this endpoint doesn't
    have. Treat a drifted feature as a prompt to re-validate against real
    outcomes, not as a verdict on the model itself.
    """

    reference = getattr(app.state, "drift_reference", None)
    if reference is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No drift reference is loaded. Run scripts/retrain_and_save.py "
                "to generate models/feature_reference_stats.json."
            ),
        )

    _, features, _ = _require_model()
    frame = pd.DataFrame(
        [observation.model_dump() for observation in request.observations]
    )

    try:
        _, X_ready, _ = prepare_model_input(frame, expected_features=features)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if X_ready.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "No rows had enough history to build complete features; "
                "drift cannot be evaluated on an empty batch."
            ),
        )

    report = compute_drift_report(reference, X_ready)
    records = [FeatureDriftRecord(**vars(result)) for result in report]

    return DriftCheckResponse(
        n_rows_evaluated=len(X_ready),
        any_feature_drifted=any(r.drifted for r in records),
        features=records,
    )


@app.get(
    "/business-impact",
    response_model=BusinessImpactResponse,
    tags=["Monitoring"],
    responses={404: {"model": ErrorResponse}},
)
def business_impact() -> BusinessImpactResponse:
    """
    Serve the static cost-impact figures from the last offline evaluation.

    These numbers come from scripts/report_business_impact.py, run against
    the holdout set under explicitly stated cost assumptions - they are
    not computed live from production traffic, since real outcomes aren't
    known at prediction time.
    """

    if not BUSINESS_IMPACT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"{BUSINESS_IMPACT_PATH} not found. Run "
                "scripts/report_business_impact.py with your own cost "
                "assumptions to generate it."
            ),
        )

    payload = json.loads(BUSINESS_IMPACT_PATH.read_text(encoding="utf-8"))
    return BusinessImpactResponse(**payload)

