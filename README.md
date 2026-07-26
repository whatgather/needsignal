# NeedSignal

**A behavioural intelligence research prototype that identifies user-created workarounds and translates them into evidence-backed product opportunities.**

NeedSignal looks for what users are forced to do when a product does not properly support what they are trying to accomplish.

Rather than treating every discussion as only a bug, complaint, or feature request, NeedSignal examines compensating behaviour such as:

- manually repeating a process,
- exporting data to another tool,
- editing files outside the product,
- creating custom scripts,
- reconnecting or rebuilding configurations,
- using temporary fixes,
- avoiding features because the outcome is unreliable.

The current version was developed and tested using public GitHub discussions from **n8n as a pilot dataset**. NeedSignal is not an n8n-specific product and is not intended to evaluate n8n as a company. The dataset provides a real-world environment for testing the research method.

## Live Project

- **Dashboard:** [Explore NeedSignal](PASTE_YOUR_STREAMLIT_LINK_HERE)
- **Portfolio:** [View the project](PASTE_YOUR_PORTFOLIO_LINK_HERE)

---

## Project Summary

NeedSignal studies the gap between:

```text
What a user is trying to accomplish
                    ↓
What the product currently allows
```

When those two things do not align, users often invent alternative behaviours.

NeedSignal structures those behaviours as:

```text
User goal
    ↓
Obstacle
    ↓
Observed workaround
    ↓
Human cost
    ↓
Underlying need
    ↓
Product opportunity
```

For example, a user may describe exporting a workflow, manually editing its data, and importing it again because the product does not support the required change directly.

NeedSignal interprets this as:

- **Goal:** modify the workflow
- **Obstacle:** the required edit is not supported in the normal interface
- **Workaround:** export, manually edit, and reimport the workflow
- **Human cost:** additional effort, time, and risk of error
- **Underlying need:** a safe method for completing the edit inside the product
- **Opportunity:** improve the product’s editing capabilities

---

## The Problem

Users do not always describe their needs directly.

A user may never say:

> “I need a better workflow recovery system.”

Instead, they may say:

> “Every time the connection fails, I delete the credentials and recreate them.”

The second statement contains stronger behavioural evidence. It shows:

- what the user was trying to do,
- what prevented them,
- what they did instead,
- and what the current experience costs them.

Traditional feedback analysis often concentrates on:

- sentiment,
- feature-request counts,
- issue volume,
- topic frequency,
- issue status,
- frequently used keywords.

NeedSignal explores a different question:

> **What behaviour have users created because the intended product experience does not meet their needs?**

---

## Research Question

**Can user-created workaround behaviour reveal actionable product opportunities that ordinary issue labels and surface-level text classification overlook?**

---

## What NeedSignal Does

The current pipeline:

1. Collects public GitHub issues and comments.
2. Combines each issue and its comments into one discussion-level record.
3. Searches for possible behavioural signals.
4. Presents selected discussions for manual review.
5. Labels each discussion using a defined annotation framework.
6. Trains a baseline workaround classifier.
7. Evaluates model predictions against human labels.
8. analyses false positives and false negatives.
9. Converts confirmed workarounds into structured opportunity signals.
10. Displays the research findings in a Streamlit dashboard.

The system preserves a link to the original discussion so each interpretation can be checked against its source evidence.

---

## Pilot Dataset

The first NeedSignal pilot uses public GitHub issue discussions from the **n8n repository**.

n8n was selected because its public repository contains detailed discussions about:

- workflow configuration,
- integrations,
- authentication,
- technical limitations,
- failed attempts,
- temporary solutions,
- product requests,
- and user-created fixes.

The project is not intended to determine whether n8n is a good or bad product. Its discussions are used as a real-world test environment for the NeedSignal research method.

The same approach could later be applied to:

- other software repositories,
- customer-support conversations,
- app reviews,
- product forums,
- user interviews,
- internal help-desk tickets,
- service-delivery records.

### Current sample

- **60** manually reviewed discussions
- **15** confirmed workaround discussions
- **45** non-workaround discussions

Each research record can contain:

- repository,
- issue number,
- issue title,
- issue description,
- discussion comments,
- issue state,
- labels,
- comment count,
- creation and update dates,
- source URL,
- human annotation,
- model prediction,
- workaround probability,
- opportunity interpretation.

---

## Annotation Framework

Each discussion receives one primary label.

| Label | Definition |
|---|---|
| `workaround` | The user performs an alternative behaviour because the normal process is missing, broken, inefficient, unreliable, or unsuitable. |
| `feature_request` | The user requests a new capability but does not describe a concrete existing workaround. |
| `bug_report` | A capability that should already work is malfunctioning. |
| `user_confusion` | The central problem involves understanding, locating, or configuring an existing capability. |
| `general_complaint` | The user expresses dissatisfaction without a clear workaround, malfunction, or actionable request. |
| `not_actionable` | The discussion is irrelevant, automated, duplicated, empty, too vague, or otherwise unusable for the study. |

A workaround is identified when there is evidence of:

```text
Goal → obstacle → compensating behaviour
```

The analysis does not invent user behaviour that is not present in the original discussion.

### Workaround fields

Confirmed workaround discussions are further structured into:

- `user_goal`
- `obstacle`
- `workaround`
- `human_cost`
- `underlying_need`
- `evidence_quote`
- `confidence`
- `annotator_notes`

---

## Baseline Model

A baseline text-classification model was trained to distinguish between:

```text
workaround
no_workaround
```

The `no_workaround` class includes feature requests, bugs, user confusion, complaints, and non-actionable discussions.

### Baseline results

| Metric | Result |
|---|---:|
| Discussions evaluated | 60 |
| Confirmed workarounds | 15 |
| Confirmed non-workarounds | 45 |
| Correct predictions | 42 |
| Incorrect predictions | 18 |
| Overall accuracy | 70% |
| False negatives | 12 |
| False positives | 6 |
| Workaround recall | 20% |

The model correctly identified only **3 of the 15 confirmed workarounds**.

### Interpretation

The 70% accuracy figure is misleading when viewed alone.

Because most discussions were non-workarounds, the model could achieve reasonable overall accuracy by predicting `no_workaround` frequently. However, it missed most of the behaviour the project was specifically designed to detect.

This produced an important research finding:

> **Surface-level text classification can appear reasonably accurate while still failing to detect most genuine workaround behaviour.**

Workarounds were often described indirectly through:

- multi-step sequences,
- technical context,
- repeated attempts,
- temporary fixes,
- compromises,
- external tools,
- actions spread across several comments,
- descriptions that did not use the word “workaround.”

An attempted feature-engineering improvement reduced performance further. This suggested that adding more keyword-style signals was not enough to solve the problem.

The current evidence points toward the need for:

- more labelled examples,
- greater class balance,
- data from multiple products,
- sequence-aware analysis,
- semantic interpretation,
- and stronger validation methods.

---

## Error Analysis

NeedSignal preserves model mistakes instead of hiding them.

### False negatives

A false negative occurs when a discussion contains a real workaround but the model predicts `no_workaround`.

These are especially important because they represent valuable behavioural signals that an automated system failed to surface.

### False positives

A false positive occurs when the model predicts a workaround even though the human annotation does not confirm one.

These cases may involve:

- bug descriptions,
- requested features,
- hypothetical solutions,
- technical instructions,
- or words such as “manual” and “instead” without actual compensating behaviour.

The dashboard includes model errors so viewers can inspect the limitations of the classifier directly.

---

## Opportunity Signals

After a workaround is manually confirmed, NeedSignal translates it into a structured opportunity record.

Each opportunity signal can include:

```text
User goal
Obstacle
Observed workaround
Human cost
Cost type
Underlying need
Opportunity statement
Evidence quote
Annotation confidence
Opportunity score
Original source link
```

### Opportunity scoring

The current opportunity score is a transparent first-pass ranking based on available evidence.

It considers factors such as:

- annotation confidence,
- whether a human cost was identified,
- clarity of the obstacle,
- clarity of the underlying need,
- presence of direct evidence,
- discussion activity,
- detected behavioural cues.

The score is not a prediction of revenue, market size, or guaranteed product value.

It is intended to help organise confirmed signals for further investigation.

A high-ranking signal would still require:

- additional user research,
- product-usage evidence,
- market analysis,
- technical feasibility analysis,
- and business validation.

---

## Dashboard

The NeedSignal Streamlit dashboard includes four main sections.

### Overview

Summarises:

- the research question,
- dataset size,
- annotation distribution,
- model performance,
- and central findings.

### Opportunity Signals

Displays confirmed workaround discussions as structured product-opportunity evidence.

Each record can show:

- what the user wanted,
- what blocked them,
- what they did instead,
- the cost of the workaround,
- the underlying need,
- and the source discussion.

### Model Errors

Shows false negatives and false positives so the model’s limitations remain visible and auditable.

### All Records

Allows the research records, labels, predictions, probabilities, and source evidence to be inspected.

---

## Project Pipeline

```text
Public GitHub discussions
            ↓
Issue collection
            ↓
Comment collection
            ↓
Discussion-level record creation
            ↓
Behavioural signal sampling
            ↓
Manual annotation
            ↓
Baseline text classification
            ↓
Model evaluation
            ↓
Error analysis
            ↓
Confirmed workaround extraction
            ↓
Opportunity scoring
            ↓
Streamlit dashboard
```

---

## Technology

- Python
- pandas
- requests
- scikit-learn
- Streamlit
- GitHub REST API
- pytest
- joblib
- CSV-based research datasets
- Git and GitHub
- Streamlit Community Cloud

---

## Project Structure

```text
needsignal/
├── data/
│   ├── raw/
│   └── annotations/
│
├── models/
│
├── reports/
│   ├── baseline_predictions.csv
│   ├── baseline_metrics.json
│   ├── baseline_errors.csv
│   ├── false_negative_analysis.csv
│   ├── false_negative_pattern_counts.csv
│   ├── error_analysis.md
│   └── opportunity_signals.csv
│
├── needsignal/
│   ├── __init__.py
│   ├── collect.py
│   ├── comments.py
│   ├── prepare_annotation.py
│   ├── annotate.py
│   ├── analyze_errors.py
│   ├── build_opportunities.py
│   ├── streamlit_app.py
│   └── streamlit_app_v2.py
│
├── tests/
├── ANNOTATION_GUIDE.md
├── requirements.txt
├── .gitignore
└── README.md
```

The structure may change as the project expands.

---

## Running the Project Locally

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd needsignal
```

### 2. Create a virtual environment

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run tests

```bash
pytest -q
```

### 5. Launch the dashboard

```bash
python -m streamlit run needsignal/streamlit_app_v2.py
```

---

## Rebuilding the Pilot Dataset

### Collect issues

```bash
python -m needsignal.collect \
  --owner n8n-io \
  --repo n8n \
  --limit 100
```

### Collect issue comments

```bash
python -m needsignal.comments \
  --issues data/raw/n8n-io_n8n_issues.csv \
  --owner n8n-io \
  --repo n8n \
  --max-issues 25
```

### Prepare discussions for annotation

```bash
python -m needsignal.prepare_annotation \
  --issues data/raw/n8n-io_n8n_issues.csv \
  --comments data/raw/n8n-io_n8n_comments.csv \
  --sample-size 60
```

### Analyse model errors

```bash
python -m needsignal.analyze_errors
```

### Build confirmed opportunity signals

```bash
python -m needsignal.build_opportunities
```

---

## Research Principles

### Behaviour before sentiment

Negative sentiment alone does not prove the existence of a valuable product opportunity.

A described behaviour can provide stronger evidence because it shows what the user is already doing to compensate for a product limitation.

### Evidence before interpretation

Every opportunity should remain connected to the original discussion and supporting evidence.

### Human validation

Automated heuristics and model predictions assist the research process. They do not replace manual review.

### Transparent limitations

Class imbalance, low workaround recall, annotation uncertainty, and model failures are reported rather than hidden.

### No invented needs

NeedSignal should not create goals, actions, costs, or opportunities that are unsupported by the source discussion.

### Research before prioritisation

A workaround signal is a reason to investigate further. It is not automatically proof that a feature should be built.

---

## Limitations

The current project is a research prototype with several important limitations:

- The labelled dataset contains only 60 discussions.
- Only 15 discussions are confirmed workarounds.
- The dataset is imbalanced.
- The pilot currently relies primarily on one product community.
- GitHub contributors may not represent the full user population.
- Technical discussions may assume missing background knowledge.
- Some workaround evidence may be distributed across several comments.
- Annotation decisions currently come from one primary researcher.
- Inter-annotator agreement has not yet been measured.
- The opportunity score does not measure revenue or market demand.
- Discussion frequency does not necessarily represent user prevalence.
- The current classifier has low workaround recall.
- The full data pipeline is not yet automatically scheduled.
- Confirmed signals still require further product and market validation.

The results should not be treated as a complete evaluation of n8n or as a production-ready product intelligence system.

---

## Next Steps

Future development will focus on:

1. Expanding the dataset to at least 200 manually reviewed discussions.
2. Increasing the number of confirmed workaround examples.
3. Collecting discussions from several products and repositories.
4. Measuring annotation consistency with a second reviewer.
5. Testing performance on products excluded from model training.
6. Comparing rule-based, traditional NLP, embedding, and language-model approaches.
7. Detecting complete goal–obstacle–behaviour sequences.
8. Clustering similar workarounds across products.
9. Measuring recurrence and unresolved persistence.
10. Improving the opportunity-ranking framework.
11. Automating data collection and report generation.
12. Validating high-ranking signals through additional user and market research.

---

## Why the Project Matters

NeedSignal is not intended to be another sentiment dashboard or issue-volume report.

Its focus is the behaviour users create when the intended product experience fails to support their goals.

Those behaviours can reveal needs that users may never express clearly as formal feature requests.

The current n8n analysis is the first pilot study of that larger idea.

---

## Simple Explanation

> **NeedSignal looks for the things users are forced to do when a product does not meet their needs. I tested the method using public n8n GitHub discussions, manually identified real workarounds, evaluated a baseline detector, analysed its failures, and translated confirmed cases into evidence-backed product opportunities.**

---

## Data and Ethics

NeedSignal uses publicly available product discussions for research and portfolio demonstration.

The project does not attempt to:

- identify or profile individual users,
- infer sensitive personal characteristics,
- evaluate individual employees,
- reproduce private customer-support information,
- treat model predictions as objective truth,
- or claim that public issue authors represent every customer.

The analysis focuses on product-level behaviour, unmet needs, and process friction.

---

## Author

**WHATGATHER**

Behavioural research, Python, analytics, and decision-support systems.

---

## Licence

This project is provided for educational, research, and portfolio purposes.
