# NeedSignal Annotation Guide

## Primary labels

### workaround

The user created or adopted an alternative behavior because the normal product process was missing, broken, inefficient, or unsuitable.

Examples:

- Exporting data to a spreadsheet to complete a task
- Using another product for one step
- Repeating a manual action
- Writing custom code to bypass a limitation
- Avoiding a feature because its consequences are unreliable

A workaround must describe something the user actually does or did.

### feature_request

The user requests a new capability, but provides no clear evidence of an existing workaround.

### bug_report

Something that should already work is malfunctioning.

A bug can include a workaround. When a concrete workaround is described, use `workaround` as the primary label.

### user_confusion

The main issue is difficulty understanding, configuring, or using an existing capability.

### general_complaint

The user expresses dissatisfaction but does not clearly describe a workaround, specific malfunction, or actionable request.

### not_actionable

The discussion is irrelevant, automated, empty, duplicated, too vague, or otherwise unusable for this research.

## Tie-breaking rules

1. Concrete workaround behavior takes priority.
2. A request without workaround evidence is a feature request.
3. An existing capability malfunctioning is a bug report.
4. Difficulty understanding an existing capability is user confusion.
5. Never infer behavior that the user did not describe.

## Workaround fields

Complete these only when `has_workaround` is `yes`.

- user_goal: What was the user trying to accomplish?
- obstacle: What prevented the normal process?
- workaround: What alternative behavior did the user perform?
- human_cost: What time, effort, risk, repetition, or frustration resulted?
- underlying_need: What capability would remove the workaround?
- evidence_quote: The shortest direct passage proving the workaround
- confidence: high, medium, or low