from __future__ import annotations

import io
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
PREDICTION_ENDPOINT = f"{API_BASE_URL}/predict-file-csv"
SAMPLE_FILE = Path(__file__).resolve().parents[1] / "examples" / "sample_input.csv"
RISK_ORDER = ["Low", "Moderate", "High", "Critical"]

st.set_page_config(
    page_title="Supply Chain Stress Prediction",
    page_icon="📦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 2rem;}
    .subtitle {color: #5f6368; margin-top: -0.5rem; margin-bottom: 1.5rem;}
    .stMetric {border: 1px solid rgba(128,128,128,.22); border-radius: 12px; padding: 10px 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📦 Supply Chain Stress Prediction")
st.markdown(
    '<p class="subtitle">Upload recent retail history, score supply-stress risk, and download business-ready predictions.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Prediction settings")
    use_custom_threshold = st.toggle(
        "Override model threshold",
        value=False,
        help="Leave this off to use the threshold saved with the model.",
    )
    threshold = st.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        disabled=not use_custom_threshold,
    )

    st.divider()
    st.caption(f"Backend: {API_BASE_URL}")
    try:
        health = requests.get(f"{API_BASE_URL}/ready", timeout=3)
        if health.ok:
            st.success("API ready")
        else:
            st.warning("API not ready")
    except requests.RequestException:
        st.error("API unavailable")

    if SAMPLE_FILE.exists():
        st.download_button(
            "Download sample CSV",
            data=SAMPLE_FILE.read_bytes(),
            file_name="sample_input.csv",
            mime="text/csv",
            use_container_width=True,
        )

uploaded_file = st.file_uploader(
    "Upload historical observations",
    type=["csv"],
    help="Each item-store series needs at least eight chronological rows so lag features can be created.",
)

if uploaded_file is None:
    st.info("Upload a CSV to preview the data and run predictions.")
    st.stop()

try:
    uploaded_bytes = uploaded_file.getvalue()
    input_frame = pd.read_csv(io.BytesIO(uploaded_bytes))
except Exception as exc:  # pragma: no cover - Streamlit display path
    st.error(f"The uploaded file could not be read as CSV: {exc}")
    st.stop()

left, right = st.columns([2, 1])
with left:
    st.subheader("Input preview")
    st.dataframe(input_frame.head(20), use_container_width=True, hide_index=True)
with right:
    st.subheader("Input summary")
    st.metric("Rows uploaded", f"{len(input_frame):,}")
    st.metric("Columns", len(input_frame.columns))
    if {"item_id", "store_id"}.issubset(input_frame.columns):
        series_count = input_frame[["item_id", "store_id"]].drop_duplicates().shape[0]
        st.metric("Item-store series", f"{series_count:,}")

run_prediction = st.button(
    "Run prediction",
    type="primary",
    use_container_width=True,
)

if not run_prediction:
    st.stop()

params: dict[str, float] = {}
if use_custom_threshold:
    params["threshold"] = threshold

files = {
    "file": (
        uploaded_file.name or "input.csv",
        uploaded_bytes,
        "text/csv",
    )
}

with st.spinner("Engineering features and scoring the model..."):
    try:
        response = requests.post(
            PREDICTION_ENDPOINT,
            params=params,
            files=files,
            timeout=120,
        )
        response.raise_for_status()
    except requests.HTTPError:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        st.error(f"Prediction failed: {detail}")
        st.stop()
    except requests.RequestException as exc:
        st.error(
            "Could not reach the prediction API. Make sure Docker Compose is running. "
            f"Details: {exc}"
        )
        st.stop()

try:
    predictions = pd.read_csv(io.BytesIO(response.content))
except Exception as exc:  # pragma: no cover - defensive display path
    st.error(f"The API response could not be read as CSV: {exc}")
    st.stop()

if predictions.empty:
    st.warning("The API returned no scored rows.")
    st.stop()

st.success(f"Scored {len(predictions):,} rows successfully.")

counts = (
    predictions["risk_level"]
    .value_counts()
    .reindex(RISK_ORDER, fill_value=0)
    .astype(int)
)
metric_columns = st.columns(4)
for column, label in zip(metric_columns, RISK_ORDER):
    column.metric(label, f"{counts[label]:,}")

chart_col, top_col = st.columns([1, 1.35])
with chart_col:
    st.subheader("Risk distribution")
    # st.bar_chart renders through Vega-Lite, which sorts nominal fields
    # alphabetically and discards the RISK_ORDER reindex above. An explicit
    # sort keeps the bars in severity order.
    chart_data = counts.rename("Rows").reset_index()
    chart_data.columns = ["Risk level", "Rows"]
    st.altair_chart(
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Risk level:N", sort=RISK_ORDER, title=None),
            y=alt.Y("Rows:Q", title="Rows"),
            tooltip=["Risk level", "Rows"],
        ),
        use_container_width=True,
    )

with top_col:
    st.subheader("Highest-risk observations")
    top_risk = predictions.sort_values(
        "stress_probability", ascending=False
    ).head(10)
    display_columns = [
        column
        for column in [
            "item_id",
            "store_id",
            "day_num",
            "stress_probability",
            "risk_level",
        ]
        if column in top_risk.columns
    ]
    st.dataframe(
        top_risk[display_columns].assign(
            # ProgressColumn's format string is applied to the raw value, so
            # "%.1f%%" appends a percent sign without rescaling and renders
            # 0.908 as "0.9%". Scale here and set max_value to match.
            stress_probability=lambda frame: frame["stress_probability"] * 100
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "stress_probability": st.column_config.ProgressColumn(
                "Stress probability",
                min_value=0.0,
                max_value=100.0,
                format="%.1f%%",
            )
        },
    )

st.subheader("Prediction results")
st.dataframe(
    predictions,
    use_container_width=True,
    hide_index=True,
    column_config={
        "stress_probability": st.column_config.NumberColumn(
            "Stress probability", format="%.3f"
        )
    },
)

st.download_button(
    "Download predictions.csv",
    data=response.content,
    file_name="predictions.csv",
    mime="text/csv",
    type="primary",
    use_container_width=True,
)

with st.expander("Run details"):
    st.write(
        {
            "model_type": response.headers.get("X-Model-Type", "Unknown"),
            "threshold": response.headers.get("X-Threshold", "Unknown"),
            "input_rows": response.headers.get("X-Input-Rows", len(input_frame)),
            "scored_rows": response.headers.get("X-Scored-Rows", len(predictions)),
        }
    )
