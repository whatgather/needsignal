from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

ANNOTATIONS_PATH = (
    ROOT
    / "data"
    / "annotations"
    / "needsignal_annotation_labeled.csv"
)

PREDICTIONS_PATH = (
    ROOT
    / "reports"
    / "baseline_predictions.csv"
)

ERRORS_PATH = (
    ROOT
    / "reports"
    / "baseline_errors.csv"
)

OPPORTUNITIES_PATH = (
    ROOT
    / "reports"
    / "opportunity_signals.csv"
)

METRICS_PATH = (
    ROOT
    / "reports"
    / "baseline_metrics.json"
)


st.set_page_config(
    page_title="NeedSignal",
    page_icon="∞",
    layout="wide",
)


@st.cache_data
def load_csv(path_string: str) -> pd.DataFrame:
    """Load a CSV if it exists."""

    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        path,
        keep_default_na=False,
    )

    dataframe.columns = [
        str(column).strip().lstrip("\ufeff")
        for column in dataframe.columns
    ]

    return dataframe


@st.cache_data
def load_json(path_string: str) -> dict:
    """Load a JSON file if it exists."""

    path = Path(path_string)

    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        return {}


def normalize_prediction_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize the prediction report column names."""

    dataframe = dataframe.copy()

    aliases = {
        "actual_label": "actual",
        "model_prediction": "predicted",
        "workaround_probability": "probability",
    }

    dataframe = dataframe.rename(
        columns={
            old_name: new_name
            for old_name, new_name in aliases.items()
            if old_name in dataframe.columns
        }
    )

    for column in ["actual", "predicted"]:
        if column in dataframe.columns:
            dataframe[column] = (
                dataframe[column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

    if "probability" in dataframe.columns:
        dataframe["probability"] = pd.to_numeric(
            dataframe["probability"],
            errors="coerce",
        )

    return dataframe


def display_value(
    value: object,
    fallback: str = "Not specified",
) -> str:
    """Return readable text for a possibly empty value."""

    if pd.isna(value):
        return fallback

    text = str(value).strip()

    return text if text else fallback


def calculate_metrics(
    predictions: pd.DataFrame,
    metrics_file: dict,
) -> dict:
    """Calculate dashboard metrics from predictions."""

    evaluated = len(predictions)

    if {
        "actual",
        "predicted",
    }.issubset(predictions.columns):
        correct = int(
            (
                predictions["actual"]
                == predictions["predicted"]
            ).sum()
        )

        accuracy = (
            correct / evaluated
            if evaluated
            else 0.0
        )

        actual_workarounds = predictions[
            predictions["actual"] == "workaround"
        ]

        detected_workarounds = actual_workarounds[
            actual_workarounds["predicted"]
            == "workaround"
        ]

        missed_workarounds = actual_workarounds[
            actual_workarounds["predicted"]
            == "no_workaround"
        ]

        workaround_recall = (
            len(detected_workarounds)
            / len(actual_workarounds)
            if len(actual_workarounds)
            else 0.0
        )
    else:
        correct = int(
            metrics_file.get(
                "correct_predictions",
                0,
            )
        )

        accuracy = float(
            metrics_file.get(
                "accuracy",
                0.0,
            )
        )

        workaround_recall = 0.0

        missed_workarounds = pd.DataFrame()

    return {
        "evaluated": int(
            metrics_file.get(
                "evaluated_records",
                evaluated,
            )
        ),
        "correct": correct,
        "accuracy": accuracy,
        "workaround_recall": workaround_recall,
        "missed_workarounds": len(
            missed_workarounds
        ),
    }


def filter_opportunities(
    dataframe: pd.DataFrame,
    search_query: str,
    confidence_filter: str,
    cost_filter: str,
) -> pd.DataFrame:
    """Apply dashboard filters to opportunity records."""

    filtered = dataframe.copy()

    if confidence_filter != "All":
        filtered = filtered[
            filtered["confidence"]
            .astype(str)
            .str.lower()
            == confidence_filter.lower()
        ]

    if cost_filter != "All":
        filtered = filtered[
            filtered["cost_type"]
            .astype(str)
            .str.contains(
                cost_filter,
                case=False,
                na=False,
            )
        ]

    if search_query.strip():
        searchable_columns = [
            column
            for column in [
                "title",
                "user_goal",
                "obstacle",
                "workaround",
                "human_cost",
                "underlying_need",
            ]
            if column in filtered.columns
        ]

        combined_text = (
            filtered[searchable_columns]
            .astype(str)
            .agg(" ".join, axis=1)
        )

        filtered = filtered[
            combined_text.str.contains(
                search_query,
                case=False,
                na=False,
            )
        ]

    return filtered


annotations = load_csv(str(ANNOTATIONS_PATH))

predictions = normalize_prediction_columns(
    load_csv(str(PREDICTIONS_PATH))
)

errors = normalize_prediction_columns(
    load_csv(str(ERRORS_PATH))
)

opportunities = load_csv(
    str(OPPORTUNITIES_PATH)
)

metrics_file = load_json(
    str(METRICS_PATH)
)

metrics = calculate_metrics(
    predictions=predictions,
    metrics_file=metrics_file,
)


st.title("NeedSignal ∞")

st.caption(
    "Behavioral intelligence for detecting "
    "user-created workarounds and translating "
    "them into product opportunities."
)


metric_columns = st.columns(4)

metric_columns[0].metric(
    "Discussions analysed",
    metrics["evaluated"],
)

metric_columns[1].metric(
    "Baseline accuracy",
    f"{metrics['accuracy']:.0%}",
)

metric_columns[2].metric(
    "Workaround recall",
    f"{metrics['workaround_recall']:.0%}",
)

metric_columns[3].metric(
    "Missed workarounds",
    metrics["missed_workarounds"],
)


overview_tab, opportunities_tab, errors_tab, records_tab = (
    st.tabs(
        [
            "Overview",
            "Opportunity Signals",
            "Model Errors",
            "All Records",
        ]
    )
)


with overview_tab:
    st.subheader("Research finding")

    st.info(
        "The baseline classifier performs reasonably "
        "on ordinary issues but misses many genuine "
        "workarounds. This suggests that compensating "
        "behaviour is often described indirectly and "
        "cannot be reliably detected from isolated "
        "keywords."
    )

    if not predictions.empty and {
        "actual",
        "predicted",
    }.issubset(predictions.columns):
        chart_columns = st.columns(2)

        actual_counts = (
            predictions["actual"]
            .value_counts()
            .rename_axis("label")
            .reset_index(name="records")
        )

        predicted_counts = (
            predictions["predicted"]
            .value_counts()
            .rename_axis("label")
            .reset_index(name="records")
        )

        with chart_columns[0]:
            st.markdown("#### Actual labels")

            st.bar_chart(
                actual_counts.set_index("label")
            )

        with chart_columns[1]:
            st.markdown("#### Model predictions")

            st.bar_chart(
                predicted_counts.set_index("label")
            )

    st.markdown("#### Evaluation summary")

    summary = pd.DataFrame(
        {
            "Metric": [
                "Evaluated records",
                "Correct predictions",
                "False negatives",
                "False positives",
            ],
            "Records": [
                metrics["evaluated"],
                metrics["correct"],
                int(
                    metrics_file.get(
                        "false_negatives",
                        metrics["missed_workarounds"],
                    )
                ),
                int(
                    metrics_file.get(
                        "false_positives",
                        0,
                    )
                ),
            ],
        }
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )


with opportunities_tab:
    st.subheader("Ranked product opportunities")

    st.write(
        "These are confirmed user-created workarounds, "
        "ranked using annotation confidence, supporting "
        "evidence, reported human cost, discussion "
        "activity, and detected behavioural cues."
    )

    if opportunities.empty:
        st.warning(
            "The opportunity report has not been "
            "generated yet. Run:\n\n"
            "`python -m needsignal.build_opportunities`"
        )
    else:
        if "opportunity_score" in opportunities.columns:
            opportunities["opportunity_score"] = (
                pd.to_numeric(
                    opportunities["opportunity_score"],
                    errors="coerce",
                )
            )

        filter_columns = st.columns(
            [2, 1, 1]
        )

        with filter_columns[0]:
            search_query = st.text_input(
                "Search opportunities",
                placeholder=(
                    "Search goals, obstacles, "
                    "workarounds, or needs"
                ),
            )

        confidence_values = [
            "All",
            *sorted(
                {
                    str(value).strip().title()
                    for value in opportunities.get(
                        "confidence",
                        pd.Series(dtype=str),
                    )
                    if str(value).strip()
                }
            ),
        ]

        with filter_columns[1]:
            confidence_filter = st.selectbox(
                "Confidence",
                confidence_values,
            )

        cost_values: set[str] = set()

        if "cost_type" in opportunities.columns:
            for value in opportunities["cost_type"]:
                parts = str(value).split("|")

                for part in parts:
                    cleaned = part.strip()

                    if cleaned:
                        cost_values.add(cleaned)

        with filter_columns[2]:
            cost_filter = st.selectbox(
                "Human cost",
                [
                    "All",
                    *sorted(cost_values),
                ],
            )

        filtered_opportunities = (
            filter_opportunities(
                dataframe=opportunities,
                search_query=search_query,
                confidence_filter=confidence_filter,
                cost_filter=cost_filter,
            )
        )

        if "opportunity_score" in (
            filtered_opportunities.columns
        ):
            filtered_opportunities = (
                filtered_opportunities.sort_values(
                    "opportunity_score",
                    ascending=False,
                )
            )

        opportunity_metrics = st.columns(4)

        opportunity_metrics[0].metric(
            "Confirmed signals",
            len(filtered_opportunities),
        )

        average_score = (
            filtered_opportunities[
                "opportunity_score"
            ].mean()
            if (
                not filtered_opportunities.empty
                and "opportunity_score"
                in filtered_opportunities.columns
            )
            else 0
        )

        opportunity_metrics[1].metric(
            "Average score",
            f"{average_score:.1f}",
        )

        high_confidence = (
            filtered_opportunities[
                "confidence"
            ]
            .astype(str)
            .str.lower()
            .eq("high")
            .sum()
            if (
                "confidence"
                in filtered_opportunities.columns
            )
            else 0
        )

        opportunity_metrics[2].metric(
            "High-confidence signals",
            int(high_confidence),
        )

        repositories = (
            filtered_opportunities[
                "repository"
            ].nunique()
            if (
                "repository"
                in filtered_opportunities.columns
            )
            else 0
        )

        opportunity_metrics[3].metric(
            "Products represented",
            int(repositories),
        )

        st.download_button(
            "Download filtered opportunities",
            data=filtered_opportunities.to_csv(
                index=False,
            ),
            file_name=(
                "needsignal_opportunities_filtered.csv"
            ),
            mime="text/csv",
        )

        st.markdown("---")

        if filtered_opportunities.empty:
            st.info(
                "No opportunities match these filters."
            )

        for _, row in (
            filtered_opportunities.iterrows()
        ):
            title = display_value(
                row.get("title"),
                "Untitled opportunity",
            )

            score = pd.to_numeric(
                row.get("opportunity_score"),
                errors="coerce",
            )

            score_text = (
                f"{score:.1f}"
                if pd.notna(score)
                else "Not scored"
            )

            expander_title = (
                f"{title} — Opportunity score: "
                f"{score_text}"
            )

            with st.expander(expander_title):
                details = st.columns(2)

                with details[0]:
                    st.markdown("**User goal**")
                    st.write(
                        display_value(
                            row.get("user_goal")
                        )
                    )

                    st.markdown("**Obstacle**")
                    st.write(
                        display_value(
                            row.get("obstacle")
                        )
                    )

                    st.markdown(
                        "**Observed workaround**"
                    )
                    st.write(
                        display_value(
                            row.get("workaround")
                        )
                    )

                with details[1]:
                    st.markdown("**Human cost**")
                    st.write(
                        display_value(
                            row.get("human_cost")
                        )
                    )

                    st.markdown("**Cost category**")
                    st.write(
                        display_value(
                            row.get("cost_type")
                        )
                    )

                    st.markdown("**Underlying need**")
                    st.write(
                        display_value(
                            row.get("underlying_need")
                        )
                    )

                evidence = display_value(
                    row.get("evidence_quote"),
                    "",
                )

                if evidence:
                    st.markdown("**Direct evidence**")
                    st.info(evidence)

                confidence = display_value(
                    row.get("confidence")
                )

                st.caption(
                    f"Annotation confidence: "
                    f"{confidence}"
                )

                issue_url = display_value(
                    row.get("issue_url"),
                    "",
                )

                if issue_url:
                    st.markdown(
                        f"[Open original discussion ↗]"
                        f"({issue_url})"
                    )


with errors_tab:
    st.subheader("Model error analysis")

    if errors.empty:
        st.warning(
            "No model-error report was found."
        )
    else:
        if "error_type" in errors.columns:
            error_options = [
                "All",
                *sorted(
                    errors["error_type"]
                    .astype(str)
                    .unique()
                    .tolist()
                ),
            ]

            selected_error = st.selectbox(
                "Error type",
                error_options,
            )

            filtered_errors = errors.copy()

            if selected_error != "All":
                filtered_errors = filtered_errors[
                    filtered_errors["error_type"]
                    == selected_error
                ]
        else:
            filtered_errors = errors

        preferred_columns = [
            "annotation_id",
            "title",
            "actual",
            "predicted",
            "probability",
            "error_type",
        ]

        visible_columns = [
            column
            for column in preferred_columns
            if column in filtered_errors.columns
        ]

        st.dataframe(
            filtered_errors[visible_columns],
            use_container_width=True,
            hide_index=True,
        )


with records_tab:
    st.subheader("Human-labelled research records")

    if annotations.empty:
        st.warning(
            "The labelled annotation dataset "
            "could not be found."
        )
    else:
        filtered_records = annotations.copy()

        if "primary_label" in annotations.columns:
            label_options = [
                "All",
                *sorted(
                    annotations["primary_label"]
                    .astype(str)
                    .unique()
                    .tolist()
                ),
            ]

            selected_label = st.selectbox(
                "Primary label",
                label_options,
            )

            if selected_label != "All":
                filtered_records = (
                    filtered_records[
                        filtered_records[
                            "primary_label"
                        ]
                        == selected_label
                    ]
                )

        preferred_columns = [
            "annotation_id",
            "repository",
            "issue_number",
            "title",
            "primary_label",
            "has_workaround",
            "confidence",
            "issue_url",
        ]

        visible_columns = [
            column
            for column in preferred_columns
            if column in filtered_records.columns
        ]

        st.dataframe(
            filtered_records[visible_columns],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download displayed records",
            data=filtered_records.to_csv(
                index=False,
            ),
            file_name=(
                "needsignal_research_records.csv"
            ),
            mime="text/csv",
        )


st.caption(
    "NeedSignal uses manually labelled research "
    "data with a transparent TF-IDF classification "
    "baseline and evidence-backed opportunity scoring."
)