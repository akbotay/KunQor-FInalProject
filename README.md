# KünQor — AI Daily Life Quality Platform
### Final Project · AI & Prompt Engineering

---

## What is KünQor?

**KünQor** ("Күн" = day, "Қор" = foundation/core in Kazakh) is a multi-module AI platform
that helps people rebuild broken daily routines by combining:

- **Personalised health analysis**
- **Daily routine / circadian optimisation**
- **Live environmental awareness (AQI)**
- **Evidence-grounded Q&A (RAG)**
- **Autonomous AI agent with tool use**
- **Goal coaching with persona prompting**
- **Multi-LLM comparison (same prompt, two models)**

All of this runs inside a single unified system powered by multiple Llama models via Groq.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key (get it free at console.groq.com)
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxx"
# Windows: set GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx

# 3. Run the server
python app.py

# 4. Open in your browser
http://127.0.0.1:8080
```

---

## File Structure

```
KunQor/
├── app.py                  ← Flask server — RUN THIS
├── llm.py                  ← Multi-LLM manager (Groq + Llama)
├── rag_engine.py           ← RAG: TF-IDF retrieval + LLM generation
├── agent.py                ← AI Agent: ReAct loop + 5 tools
├── requirements.txt
└── templates/
    └── ai_platform.html    ← Full-stack UI (served by Flask)
```

---

## Models Used

| Model | Speed | Used For |
|---|---|---|
| `llama-3.1-8b-instant` | Fast | Health analysis (few-shot prompting) |
| `llama-3.3-70b-versatile` | Smart | Routine, Ecology, RAG, Agent, Coach, LLM Compare |

Both models are accessed via **Groq API** (OpenAI-compatible SDK).
API keys are free at **console.groq.com**.

---

## Course Concepts Demonstrated

| Concept | Where / How |
|---|---|
| **NLP + LLMs** | All modules use Llama (transformer-based LLM) via Groq |
| **Large Language Models & Transformers** | Llama-3 is a transformer; explained in presentation |
| **Prompt Engineering** | 5 distinct techniques across modules (see below) |
| **RAG** | `rag_engine.py` — TF-IDF retrieval → grounded LLM generation |
| **AI Agents & Tool-using LLMs** | `agent.py` — ReAct loop with 5 callable tools |
| **LLM Automation** | Flask API routes automate multi-step AI pipelines |
| **Multiple LLMs** | `llama-3.1-8b-instant` vs `llama-3.3-70b-versatile` |

---

## Prompt Engineering Techniques

### 1. Few-Shot Prompting — Health Module (`llm.py`)
The system prompt contains **2 complete worked examples** showing exact input/output pairs.
This teaches the model the JSON schema without relying on descriptions alone.
Model: `llama-3.1-8b-instant` · Temperature: `0.2` (deterministic)

### 2. Chain-of-Thought (CoT) Prompting — Routine Module (`llm.py`)
The model is instructed to reason through **4 explicit steps** before producing output:
sleep quality → meal timing → work-life balance → synthesis.
Model: `llama-3.3-70b-versatile` · Temperature: `0.3`

### 3. Role + Task Decomposition — Ecology Module (`llm.py`)
The model is assigned a specific expert role ("EcoAdvisor") and given a numbered
task list to follow before generating structured JSON output.
Model: `llama-3.3-70b-versatile` · Temperature: `0.2`

### 4. Persona + Constraint Prompting — Goal Coach (`llm.py`)
The coach has a named persona ("Qor") with strict rules: every suggestion must be
`[SPECIFIC]`, `[TIMED]`, and `[MEASURABLE]`. Vague advice is explicitly forbidden.
Model: `llama-3.3-70b-versatile` · Temperature: `0.7` (expressive)

### 5. Structured Output with Schema Enforcement — All Modules
Every module embeds the target JSON schema directly in the system prompt.
A robust `_extract_json()` function handles models that prepend reasoning text
before the JSON block (common with CoT prompting).

---

## RAG Pipeline (`rag_engine.py`)

```
User Query
    ↓
TF-IDF Tokenisation & Vectorisation
    ↓
Cosine Similarity vs 20 curated health passages
    ↓
Top-3 passages selected  ← RETRIEVE
    ↓
Passages injected into LLM prompt  ← AUGMENT
    ↓
llama-3.3-70b generates cited answer  ← GENERATE
    ↓
Response with passage IDs as citations
```

No external vector database needed — the TF-IDF engine is built from scratch
using only Python's standard library (`re`, `math`, `collections`).

Knowledge base covers: sleep, hydration, exercise, nutrition, mental health,
ecology/AQI, and habit formation — 20 passages total.

---

## AI Agent (`agent.py`)

Uses the **ReAct (Reason + Act)** pattern — the model autonomously decides
which tools to call and chains multiple steps before answering.

### Available Tools

| Tool | Description |
|---|---|
| `get_aqi(city)` | Air Quality Index for Kazakhstan cities |
| `calculate_bmi(height_cm, weight_kg)` | BMI + WHO category |
| `habit_score(description)` | Score a habit on 4 behaviour-science criteria |
| `schedule_advice(time, activity)` | Circadian-optimal time window check |
| `rag_lookup(query)` | Search the KunQor knowledge base |

The UI shows a full **execution trace** — every tool call, its arguments,
and its result — so the reasoning is completely transparent.

---

## Multi-LLM Strategy

The **LLM Compare** page sends the **exact same prompt** to both models simultaneously.
This demonstrates live that model choice meaningfully affects output quality and depth —
a core concept in prompt engineering and LLM selection strategy.

---

## Presentation Tips

1. **Dashboard** — show the architecture diagram and concept tags
2. **Health tab** — enter bad values (0.3L water, 1k steps) for a dramatic low score
3. **RAG tab** — ask "How much sleep do I need?" — show retrieved passages + cited answer
4. **Agent tab** — use "Check AQI in Almaty and tell me if I should go running at 07:00"
5. **LLM Compare tab** — enter "I'm exhausted and can't focus" — highlight the difference
6. **Goal Coach** — enter your own name for a personal touch during the demo

---

## Technologies

- **Python 3.10+** + **Flask** — backend web server
- **Groq API** (OpenAI-compatible SDK) — fast LLM inference
- **Llama 3.1 & 3.3** — open-source transformer models by Meta
- **Custom TF-IDF** — vector retrieval with no external dependencies
- **Vanilla JS + CSS** — frontend (no React/Vue needed)

---

*KünQor — Rebuild your day, one habit at a time.*
