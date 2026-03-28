---
name: ai-engineering
description: >
  Use this skill for advanced AI/ML systems: experiment tracking, prompt engineering,
  RAG pipelines, LLM evaluation, MLOps, model serving, and drift monitoring.
  Triggers: "MLflow", "experiment tracking", "track experiments", "W&B", "wandb",
  "RAG", "retrieval augmented generation", "vector database", "embeddings", "FAISS",
  "fine-tune", "LLM evaluation", "BLEU", "ROUGE", "BERTScore", "prompt engineering",
  "system prompt", "prompt versioning", "FastAPI serve", "model API", "drift detection",
  "model monitoring", "production ML", "A/B test models", "Anthropic API", "OpenAI API",
  "HuggingFace", or any request to deploy or productionise a model or AI system.
---

# AI Engineering Skill
## Production-grade patterns for ML systems and LLM applications

---

## Domain Index

```
A → Experiment Tracking (MLflow)
B → Prompt Engineering & Versioning
C → LLM API Integration (Anthropic / OpenAI)
D → RAG Pipeline
E → LLM Evaluation (BLEU / ROUGE / BERTScore / LLM-as-judge)
F → MLOps: Serving (FastAPI) + Drift Monitoring
```

---

## Domain A — Experiment Tracking with MLflow

```python
import mlflow, mlflow.sklearn
import json

mlflow.set_experiment("advertising-lr")

with mlflow.start_run(run_name="ols_full_model"):

    # Log config
    mlflow.log_param("model_type",    "ols")
    mlflow.log_param("features",      "TV,Radio,Newspaper")
    mlflow.log_param("test_size",     0.2)
    mlflow.log_param("random_state",  42)

    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log metrics
    mlflow.log_metric("train_r2",  float(r2_score(y_train, model.predict(X_train))))
    mlflow.log_metric("test_rmse", float(rmse))
    mlflow.log_metric("test_r2",   float(r2_test))
    mlflow.log_metric("test_mae",  float(mae))

    # Log artefacts
    mlflow.log_artifact("outputs/figures/resid_vs_fitted.png")
    mlflow.log_artifact("outputs/figures/qq_plot.png")
    mlflow.log_artifact("config.yaml")
    mlflow.sklearn.log_model(model, "model")

    print(mlflow.active_run().info.run_id)
```

**MLflow rules:**
- One `mlflow.start_run()` per experiment configuration.
- Log ALL parameters — including ones that didn't change — for full reproducibility.
- Log BOTH train and test metrics to detect overfitting.
- Always log `config.yaml` as an artefact.

---

## Domain B — Prompt Engineering

### Principles

| Principle | Rule |
|---|---|
| Be explicit | State output format, length, and any constraints |
| Provide examples | One-shot or few-shot for structured outputs |
| Chain-of-thought | Ask to "think step by step" for reasoning tasks |
| Separate concerns | System prompt = persona; user prompt = task |
| Version control | Prompts are code — commit them, don't hardcode |

### System prompt template (statistical assistant)

```python
SYSTEM_PROMPT_V1 = """You are a statistical analysis assistant specialising in
linear regression and supervised learning (ISL framework, James et al., 2023).
You respond in clear, precise language suitable for an academic paper.

Always:
- State hypotheses formally (H₀ and H₁).
- Report all 4 elements: statistic, degrees of freedom, value, p-value.
- Flag LINE assumption violations.
- Distinguish association from causation.
- Use British English spelling.
"""
```

### Prompt versioning

```python
# src/prompts/v1/analysis.py
ANALYSIS_PROMPT_V1 = "Analyse the regression output: {summary}"

# src/prompts/v2/analysis.py — improved
ANALYSIS_PROMPT_V2 = """Given this OLS regression summary:
{summary}

1. State whether H₀ is rejected for each predictor (α = .05).
2. Interpret each significant β̂ in plain language (units: $K spend → K units).
3. Report the 4-element result for the F-test.
4. Identify any LINE assumption concerns.
"""

# Always log which version was used
mlflow.log_param("prompt_version", "v2")
```

---

## Domain C — LLM API Integration

### Anthropic Claude (claude-sonnet-4-20250514)

```python
import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

def analyse_regression(ols_summary: str) -> str:
    """Pass OLS summary to Claude for plain-language interpretation."""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT_V1,
        messages=[{
            "role": "user",
            "content": ANALYSIS_PROMPT_V2.format(summary=ols_summary)
        }]
    )
    return message.content[0].text
```

### Structured JSON output

```python
import json

STRUCTURED_PROMPT = """Given this regression coefficient table:
{table}

Respond ONLY with a valid JSON object — no preamble, no markdown fences:
{{
  "significant_predictors": ["list of predictor names with p < .05"],
  "key_finding": "one sentence summary",
  "r_squared_interpretation": "plain English interpretation of R²",
  "recommendation": "actionable marketing recommendation"
}}"""

def get_structured_insight(coef_table: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user",
                   "content": STRUCTURED_PROMPT.format(table=coef_table)}]
    )
    text = response.content[0].text.strip()
    return json.loads(text)
```

---

## Domain D — RAG Pipeline

For building a Q&A system over the ISL textbook or reference papers:

```
User query
    │
    ▼
[Embedding model] → query vector
    │
    ▼
[FAISS index search] → top-k relevant chunks
    │
    ▼
[Context assembly: chunks + query]
    │
    ▼
[Claude] → answer with citations
```

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pickle

MODEL_NAME = 'all-MiniLM-L6-v2'

# ── INDEX (run once) ──────────────────────────────
def build_index(chunks: list[str], index_path: str = "outputs/vector_index.faiss"):
    model      = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, show_progress_bar=True).astype('float32')
    index      = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open("outputs/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    return index

# ── RETRIEVE ─────────────────────────────────────
def retrieve(query: str, top_k: int = 5) -> list[str]:
    model  = SentenceTransformer(MODEL_NAME)
    index  = faiss.read_index("outputs/vector_index.faiss")
    chunks = pickle.load(open("outputs/chunks.pkl", "rb"))
    q_vec  = model.encode([query]).astype('float32')
    _, idx = index.search(q_vec, top_k)
    return [chunks[i] for i in idx[0]]

# ── GENERATE ─────────────────────────────────────
def rag_answer(query: str) -> str:
    context = "\n\n".join(retrieve(query))
    prompt  = (f"Use the following context from ISL to answer the question.\n\n"
               f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
    return analyse_regression(prompt)   # reuses Claude client from Domain C
```

---

## Domain E — LLM & Model Evaluation

### Classical metrics

```python
# BLEU (for generated text quality)
from nltk.translate.bleu_score import corpus_bleu
refs  = [[r.split()] for r in reference_texts]
hyps  = [h.split() for h in generated_texts]
print(f"BLEU-4: {corpus_bleu(refs, hyps):.4f}")

# ROUGE (for summarisation)
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
scores = scorer.score(reference, generated)
print(scores)

# BERTScore (semantic similarity)
from bert_score import score as bscore
P, R, F1 = bscore(generated_texts, reference_texts, lang="en")
print(f"BERTScore F1: {F1.mean():.4f}")
```

### LLM-as-judge

```python
JUDGE_PROMPT = """Rate this answer 1–5 for each criterion and respond ONLY
with valid JSON (no markdown):
{{"accuracy": X, "completeness": X, "clarity": X, "reasoning": "..."}}

Reference answer: {reference}
Candidate answer: {answer}"""

def llm_judge(reference: str, answer: str) -> dict:
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(
                       reference=reference, answer=answer)}]
    )
    return json.loads(resp.content[0].text.strip())
```

---

## Domain F — MLOps: Serving + Monitoring

### FastAPI model endpoint

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle, numpy as np, pandas as pd

app    = FastAPI(title="Advertising Sales Predictor")
model  = pickle.load(open("outputs/model.pkl", "rb"))

class AdvertisingInput(BaseModel):
    TV:        float = Field(..., ge=0, le=500, description="TV budget $K")
    Radio:     float = Field(..., ge=0, le=100, description="Radio budget $K")
    Newspaper: float = Field(..., ge=0, le=200, description="Newspaper budget $K")

class PredictionOutput(BaseModel):
    predicted_sales: float
    unit: str = "thousands of units"

@app.post("/predict", response_model=PredictionOutput)
def predict(req: AdvertisingInput) -> PredictionOutput:
    X    = pd.DataFrame([req.model_dump()])
    pred = model.predict(X)[0]
    return PredictionOutput(predicted_sales=round(float(pred), 4))

@app.get("/health")
def health():
    return {"status": "ok"}
```

Run: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
Test: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"TV":200,"Radio":30,"Newspaper":10}'`

### Drift detection

```python
from scipy.stats import ks_2samp
from evidently.report import Report
from evidently.metrics import DataDriftTable

# Statistical drift (KS test on predictions)
stat, p = ks_2samp(y_pred_baseline, y_pred_current)
if p < 0.05:
    print(f"ALERT: prediction drift (KS p={p:.4f})")

# Full Evidently data drift report
report = Report(metrics=[DataDriftTable()])
report.run(reference_data=X_train, current_data=X_new)
report.save_html("outputs/reports/drift_report.html")
```

---

## AI Engineering Checklist

- [ ] All experiments logged in MLflow with params, metrics, and artefacts.
- [ ] Prompts in `src/prompts/vN/` and version logged in MLflow.
- [ ] RAG retrieval quality tested: recall@5 on labelled eval set.
- [ ] Model behind validated FastAPI endpoint with Pydantic input models.
- [ ] Drift monitoring scheduled (weekly or on new batch).
- [ ] Previous model version retained in registry for rollback.
- [ ] Latency benchmarked: p99 response time < 200 ms for regression endpoint.
