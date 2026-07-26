from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ANNOTATIONS_PATH = Path(
    "data/annotations/needsignal_annotation_labeled.csv"
)

PREDICTIONS_PATH = Path(
    "reports/baseline_predictions.csv"
)


st.set_page_config(
    page_title="NeedSignal",
    page_icon="🔎",
    layout="wide",
)


def normalise_label(value: object) -> str:
    """Convert different binary label formats to standard labels."""

    label = str(value).strip().lower()

    workaround_values = {
        "workaround",
        "yes",
        "true",
        "1",
        "positive",
    }

    no_workaround_values = {
        "no_workaround",
        "no workaround",
        "no",
        "false",
        "0",
        "negative",
    }

    if label in workaround_values:
        return "workaround"

    if label in no_workaround_values:
        return "no_workaround"

    return label


def classify_error(row: pd.Series) -> str:
    """Identify the prediction result for one record."""

    actual = row.get("actual", "")
    predicted = row.get("predicted", "")

    if not actual or not predicted:
        return "not_evaluated"

    if actual == predicted:
        return "correct"

    if (
        actual == "workaround"
        and predicted == "no_workaround"
    ):
        return "false_negative"

    if (
        actual == "no_workaround"
        and predicted == "workaround"
    ):
        return "false_positive"

    return "other_error"


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and combine annotations with model predictions."""

    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing annotation file: {ANNOTATIONS_PATH}"
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTIONS_PATH}"
        )

    annotations = pd.read_csv(
        ANNOTATIONS_PATH,
        keep_default_na=False,
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        keep_default_na=False,
    )

    annotations.columns = [
        str(column).strip().lstrip("\ufeff")
        for column in annotations.columns
    ]

    predictions.columns = [
        str(column).strip().lstrip("\ufeff")
        for column in predictions.columns
    ]

    prediction_aliases = {
        "actual_label": "actual",
        "model_prediction": "predicted",
        "workaround_probability": "probability",
    }

    predictions = predictions.rename(
        columns=prediction_aliases
    )

    if "actual" not in predictions.columns:
        predictions["actual"] = ""

    if "predicted" not in predictions.columns:
        predictions["predicted"] = ""

    if "probability" not in predictions.columns:
        predictions["probability"] = pd.NA

    predictions["actual"] = predictions[
        "actual"
    ].apply(normalise_label)

    predictions["predicted"] = predictions[
        "predicted"
    ].apply(normalise_label)

    predictions["probability"] = pd.to_numeric(
        predictions["probability"],
        errors="coerce",
    )

    prediction_columns = [
        "annotation_id",
        "actual",
        "predicted",
        "probability",
        "heuristic_prediction",
        "heuristic_correct",
        "model_correct",
    ]

    prediction_columns = [
        column
        for column in prediction_columns
        if column in predictions.columns
    ]

    combined = annotations.merge(
        predictions[prediction_columns],
        on="annotation_id",
        how="left",
    )

    combined["actual"] = combined[
        "actual"
    ].fillna("").apply(normalise_label)

    combined["predicted"] = combined[
        "predicted"
    ].fillna("").apply(normalise_label)

    combined["error_type"] = combined.apply(
        classify_error,
        axis=1,
    )

    if "primary_label" not in combined.columns:
        combined["primary_label"] = ""

    if "matched_patterns" not in combined.columns:
        combined["matched_patterns"] = ""

    if "thread_text" not in combined.columns:
        combined["thread_text"] = ""

    return combined


def text_value(
    row: pd.Series,
    column: str,
    fallback: str = "Not recorded",
) -> str:
    """Return readable text for a possibly empty field."""

    value = str(row.get(column, "")).strip()

    return value if value else fallback


try:
    data = load_data()
except Exception as error:
    st.error(str(error))
    st.stop()


valid_labels = {
    "workaround",
    "no_workaround",
}

evaluated = data[
    data["actual"].isin(valid_labels)
    & data["predicted"].isin(valid_labels)
].copy()

actual_workarounds = evaluated[
    evaluated["actual"] == "workaround"
]

true_positives = evaluated[
    (evaluated["actual"] == "workaround")
    & (evaluated["predicted"] == "workaround")
]

false_negatives = evaluated[
    evaluated["error_type"] == "false_negative"
]

false_positives = evaluated[
    evaluated["error_type"] == "false_positive"
]

accuracy = (
    (evaluated["actual"] == evaluated["predicted"]).mean()
    if not evaluated.empty
    else 0
)

workaround_recall = (
    len(true_positives) / len(actual_workarounds)
    if len(actual_workarounds)
    else 0
)


st.title("NeedSignal")

st.caption(
    "Behavioural intelligence for detecting user-created "
    "workarounds and translating them into product opportunities."
)


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Discussions analysed",
    f"{len(data):,}",
)

metric_2.metric(
    "Baseline accuracy",
    f"{accuracy:.0%}",
)

metric_3.metric(
    "Workaround recall",
    f"{workaround_recall:.0%}",
)

metric_4.metric(
    "Missed workarounds",
    f"{len(false_negatives):,}",
)


overview_tab, signals_tab, errors_tab, records_tab = st.tabs(
    [
        "Overview",
        "Opportunity signals",
        "Model errors",
        "All records",
    ]
)


with overview_tab:
    st.subheader("Research finding")

    st.info(
        "The baseline classifier performs reasonably on ordinary "
        "issues but misses many genuine workarounds. This suggests "
        "that compensating behaviour is often described indirectly "
        "and cannot be reliably detected from isolated keywords."
    )

    chart_1, chart_2 = st.columns(2)

    with chart_1:
        st.markdown("#### Actual labels")

        actual_counts = (
            evaluated["actual"]
            .value_counts()
            .rename_axis("label")
            .reset_index(name="records")
            .set_index("label")
        )

        st.bar_chart(actual_counts)

    with chart_2:
        st.markdown("#### Model predictions")

        prediction_counts = (
            evaluated["predicted"]
            .value_counts()
            .rename_axis("label")
            .reset_index(name="records")
            .set_index("label")
        )

        st.bar_chart(prediction_counts)

    st.markdown("#### Evaluation summary")

    evaluation_summary = pd.DataFrame(
        {
            "Result": [
                "Correct predictions",
                "False negatives",
                "False positives",
            ],
            "Records": [
                int(
                    (
                        evaluated["actual"]
                        == evaluated["predicted"]
                    ).sum()
                ),
                len(false_negatives),
                len(false_positives),
            ],
        }
    )

    st.dataframe(
        evaluation_summary,
        hide_index=True,
        use_container_width=True,
    )


with signals_tab:
    st.subheader("Confirmed workaround signals")

    signal_rows = data[
        (
            data["actual"] == "workaround"
        )
        | (
            data["primary_label"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "workaround"
        )
    ].copy()

    if signal_rows.empty:
        st.warning(
            "No confirmed workaround records are available."
        )
    else:
        signal_rows["display_name"] = (
            signal_rows["annotation_id"].astype(str)
            + " — "
            + signal_rows["title"].astype(str)
        )

        selected_name = st.selectbox(
            "Choose a workaround",
            signal_rows["display_name"].tolist(),
        )

        selected = signal_rows[
            signal_rows["display_name"] == selected_name
        ].iloc[0]

        st.markdown(
            f"### {text_value(selected, 'title')}"
        )

        if text_value(
            selected,
            "issue_url",
            "",
        ):
            st.markdown(
                f"[Open original GitHub discussion]"
                f"({selected['issue_url']})"
            )

        goal_column, obstacle_column = st.columns(2)

        with goal_column:
            st.markdown("#### User goal")
            st.write(
                text_value(selected, "user_goal")
            )

        with obstacle_column:
            st.markdown("#### Obstacle")
            st.write(
                text_value(selected, "obstacle")
            )

        workaround_column, cost_column = st.columns(2)

        with workaround_column:
            st.markdown("#### Workaround behaviour")
            st.write(
                text_value(selected, "workaround")
            )

        with cost_column:
            st.markdown("#### Human cost")
            st.write(
                text_value(selected, "human_cost")
            )

        st.markdown("#### Underlying need")

        st.success(
            text_value(selected, "underlying_need")
        )

        st.markdown("#### Supporting evidence")

        st.write(
            text_value(
                selected,
                "evidence_quote",
                "No separate evidence quote was recorded.",
            )
        )

        with st.expander("View complete discussion"):
            st.write(
                text_value(
                    selected,
                    "thread_text",
                    "Discussion text unavailable.",
                )
            )


with errors_tab:
    st.subheader("Where the baseline fails")

    false_negative_tab, false_positive_tab = st.tabs(
        [
            f"False negatives ({len(false_negatives)})",
            f"False positives ({len(false_positives)})",
        ]
    )

    error_columns = [
        "annotation_id",
        "title",
        "actual",
        "predicted",
        "probability",
        "matched_patterns",
    ]

    error_columns = [
        column
        for column in error_columns
        if column in data.columns
    ]

    with false_negative_tab:
        st.caption(
            "Real workarounds that the model classified as "
            "no workaround."
        )

        st.dataframe(
            false_negatives[error_columns],
            hide_index=True,
            use_container_width=True,
        )

    with false_positive_tab:
        st.caption(
            "Ordinary discussions that the model incorrectly "
            "classified as workarounds."
        )

        st.dataframe(
            false_positives[error_columns],
            hide_index=True,
            use_container_width=True,
        )


with records_tab:
    st.subheader("Explore the research dataset")

    filter_1, filter_2, filter_3 = st.columns(3)

    with filter_1:
        actual_options = sorted(
            label
            for label in data["actual"].unique()
            if str(label).strip()
        )

        selected_actual = st.multiselect(
            "Actual label",
            actual_options,
            default=actual_options,
        )

    with filter_2:
        prediction_options = sorted(
            label
            for label in data["predicted"].unique()
            if str(label).strip()
        )

        selected_predictions = st.multiselect(
            "Model prediction",
            prediction_options,
            default=prediction_options,
        )

    with filter_3:
        error_options = sorted(
            data["error_type"].dropna().unique()
        )

        selected_errors = st.multiselect(
            "Evaluation result",
            error_options,
            default=error_options,
        )

    filtered = data.copy()

    if selected_actual:
        filtered = filtered[
            filtered["actual"].isin(selected_actual)
        ]

    if selected_predictions:
        filtered = filtered[
            filtered["predicted"].isin(
                selected_predictions
            )
        ]

    if selected_errors:
        filtered = filtered[
            filtered["error_type"].isin(
                selected_errors
            )
        ]

    display_columns = [
        "annotation_id",
        "repository",
        "issue_number",
        "title",
        "primary_label",
        "actual",
        "predicted",
        "probability",
        "error_type",
        "heuristic_score",
        "matched_patterns",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in filtered.columns
    ]

    st.write(f"Showing {len(filtered):,} records")

    st.dataframe(
        filtered[display_columns],
        hide_index=True,
        use_container_width=True,
    )

    download_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download filtered records",
        data=download_data,
        file_name="needsignal_filtered_records.csv",
        mime="text/csv",
    )


st.divider()

st.caption(
    "Current version: manually labelled research data with "
    "a transparent TF-IDF classification baseline."
)