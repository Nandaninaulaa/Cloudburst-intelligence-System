# GeoAI CloudBurst — Setup Guide

## What's New vs Your Original Code

| Feature | Before | After |
|---|---|---|
| Architecture | Single ML predictor | **5-Agent AI Pipeline** |
| AI Backend | Rule-based only | **Claude claude-sonnet-4-20250514 (Anthropic)** |
| Knowledge | Hardcoded advice | **RAG from NDMA/SDRF guidelines** |
| Evacuation | None | **Zone-by-zone AI plan with priorities** |
| Resources | None | **SDRF/medical/helicopter calculator** |
| Communications | Email/SMS only | **SMS + PA + Radio + Press + Social** |
| Explainability | SHAP charts | **SHAP + Plain English explanations** |
| Tabs | 7 tabs | **8 tabs (new: AI Agents + RAG Assistant)** |

---

## Quick Start

### 1. Copy your model files
Your existing model files must be in the same folder as `app.py`:
```
rf_cloudburst_model.pkl
xgb_cloudburst_model.pkl          (optional but recommended)
stacking_cloudburst_model.pkl     (optional but recommended)
cloudburst_lstm_model.keras       (optional)
lstm_scaler.pkl                   (required if LSTM present)
database.py                       (your existing file)
metrics_rf_xgb.json               (your existing file)
metrics_lstm.json                 (your existing file)
lstm_history.json                 (your existing file)
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Edit .env and fill in your keys
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
streamlit run app.py
```

---

## How the Multi-Agent System Works

```
User clicks "Run AI Agents"
         │
         ▼
┌─────────────────────┐
│ Agent 1             │  Reads wx data → builds situation_report JSON
│ Weather Monitor     │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ RAG Retrieval       │  Fetches relevant NDMA/SDRF knowledge chunks
│ Knowledge Base      │  based on risk level + elevation + rainfall
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Agent 2             │  Claude analyses situation + KB → risk JSON
│ Risk Assessor       │  (classification, landslide risk, onset time)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Agent 3             │  Claude → evacuation routes, priority order,
│ Evacuation Planner  │  assembly points, timeline
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Agent 4             │  Claude → SDRF teams, ambulances, helicopters,
│ Resource Planner    │  relief camps, deployment timeline, cost
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Agent 5             │  Claude → SMS, PA script, radio bulletin,
│ Communications      │  press release, WhatsApp, Twitter/X
└─────────────────────┘
```

Each agent uses the output of the previous agent as context.
All agents are grounded in the RAG knowledge base.
Fallback (no API key): rule-based responses used for all agents.

---

## RAG Knowledge Base

The knowledge base (`DISASTER_KB` dict in `app.py`) currently has 5 documents:
1. **NDMA Cloudburst Guidelines** — official definitions and triggers
2. **Evacuation SOPs** — 3-phase response protocol
3. **Resource Allocation Guidelines** — per-population resource ratios
4. **Road Closure Protocols** — infrastructure response
5. **Landslide Risk Correlation** — secondary hazard assessment

### To expand the knowledge base:
Add entries to the `DISASTER_KB` dictionary in `app.py`:
```python
DISASTER_KB["my_new_doc"] = """
Your NDMA/SDRF document text here.
Can be loaded from a file or database.
"""
```

For production, replace with a proper vector database (Pinecone, ChromaDB, FAISS).

---

## Anthropic API Key

Get your key at: https://console.anthropic.com

The system uses `claude-sonnet-4-20250514` for all 5 agents.
Estimated cost per full agent run: ~$0.02–0.05 (very low).

If no API key is provided, all agents fall back to high-quality
rule-based responses — the system still works, just without LLM reasoning.

---

## Production Deployment Notes

1. **Never commit `.env` to git** — add it to `.gitignore`
2. **For Streamlit Cloud**: add secrets in the Streamlit Cloud dashboard
3. **For Docker**: pass env vars with `-e` flags
4. **Rate limiting**: add a session-based cooldown before calling agents
   to avoid excessive API costs in multi-user deployments
