# Customer-Support Ticket Triage Agent

A multi-step agentic AI pipeline that classifies, prioritizes and routes support tickets,
with a Streamlit support-operations dashboard.

## Pipeline
Intake (clean + features) → Classifier (category) → Prioritizer (priority) → Router (team + SLA + escalation) → Responder (draft reply; LLM optional)

## Run
```bash
pip install -r requirements.txt
python src/generate_data.py      # or place your own data/tickets.csv
python src/train.py
streamlit run app.py
```
Optional: `export ANTHROPIC_API_KEY=...` to enable LLM-drafted replies.

## Data schema
`ticket_id, created_at, customer_tier, subject, body, category, priority`

## Deploy
Push to GitHub, then deploy on Streamlit Community Cloud (main file: `app.py`).
Commit `models/` after training, or add a training step to your startup.
