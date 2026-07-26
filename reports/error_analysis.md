# NeedSignal Baseline Error Analysis

## Baseline summary

- Evaluated records: 60
- Correct predictions: 42
- False negatives: 12
- False positives: 6

## Behavioral cues inside missed workarounds

| Behavioral cue | Missed cases |
|---|---:|
| manual_intervention | 10 |
| multi_step_sequence | 9 |
| constraint_language | 5 |
| repetition_or_retry | 4 |
| alternative_method | 4 |
| successful_compensation | 1 |

## Missed workaround examples

### NS-0003: Unable to use AI builder on n8n

- Model probability: 0.4265
- Behavioral cues: constraint_language | manual_intervention | repetition_or_retry | multi_step_sequence

### NS-0035: Editor UI: expanding the collapsed left sidebar renders an empty panel and pushes the main content off screen

- Model probability: 0.4722
- Behavioral cues: manual_intervention | alternative_method | repetition_or_retry | multi_step_sequence

### NS-0047: Salesforce node (OAuth2): refresh silently drops oauthTokenData.instance_url, causing "Invalid URL" (ERR_INVALID_URL) until manual reconnect

- Model probability: 0.4861
- Behavioral cues: manual_intervention | repetition_or_retry | multi_step_sequence | successful_compensation

### NS-0054: AWS S3: SignatureDoesNotMatch on object keys containing "=" or ":" (path signed un-encoded, sent percent-encoded) — regression in 2.31.5

- Model probability: 0.3834
- Behavioral cues: manual_intervention | alternative_method | multi_step_sequence

### NS-0025: The built-in LDAP module does not work

- Model probability: 0.3925
- Behavioral cues: constraint_language | manual_intervention | multi_step_sequence

### NS-0057: Touchscreen on canvas stops working after sending any keystroke

- Model probability: 0.4694
- Behavioral cues: alternative_method | repetition_or_retry | multi_step_sequence

### NS-0034: MCP server: update_workflow on an ACTIVE workflow fails with SQLITE_ERROR: no such column: distinctAlias.SharedWorkflow_projectId (2.31.5)

- Model probability: 0.3944
- Behavioral cues: manual_intervention | multi_step_sequence

### NS-0017: Imported workflow shows missing credentials until node dialog is opened

- Model probability: 0.4
- Behavioral cues: manual_intervention | multi_step_sequence

### NS-0046: Google Service Account Impersonation failing validation (but working)

- Model probability: 0.4521
- Behavioral cues: constraint_language | manual_intervention

### NS-0014: AI Builder is unavailable when creating an n8n workflow

- Model probability: 0.454
- Behavioral cues: constraint_language | manual_intervention

### NS-0033: OAuth consent fails with "Invalid or expired authorization session" - n8n-oauth-session cookie hardcoded with Secure attribute ignoring N8N_SECURE_COOKIE=false

- Model probability: 0.4604
- Behavioral cues: alternative_method | multi_step_sequence

### NS-0044: MCP tool update_workflow does not support setting availableInMCP via setWorkflowSettings

- Model probability: 0.4768
- Behavioral cues: constraint_language | manual_intervention
