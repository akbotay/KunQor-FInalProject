"""
rag_engine.py  –  KunQor Retrieval-Augmented Generation
=========================================================
Demonstrates:
  • Building a lightweight in-memory vector store (no external DB required)
  • TF-IDF-style cosine similarity for document retrieval
  • RAG prompt construction: retrieved context injected into LLM prompt
  • Two-stage pipeline: RETRIEVE → GENERATE

Why RAG matters for KunQor:
  The knowledge base contains curated health / habit / wellbeing research.
  Instead of relying solely on the LLM's parametric memory (which can hallucinate),
  we ground answers in specific retrieved passages.
"""

import os
import json
import math
import re
from collections import Counter
from openai import OpenAI

client = OpenAI(
    api_key  = os.environ.get("GROQ_API_KEY", "gsk_..."),
    base_url = "https://api.groq.com/openai/v1",
)

KNOWLEDGE_BASE = [
    # ── Sleep & Circadian rhythm ──────────────────────────────────────────────
    {"id": "sleep-1", "topic": "sleep",
     "text": "Adults need 7–9 hours of sleep per night according to the National Sleep Foundation. "
             "Chronic sleep deprivation (< 6 hours) increases risk of obesity, diabetes, and cardiovascular disease."},
    {"id": "sleep-2", "topic": "sleep",
     "text": "The circadian rhythm is a 24-hour internal clock regulated by light exposure. "
             "Screens emit blue light that suppresses melatonin; avoiding screens 1 hour before bed significantly improves sleep onset."},
    {"id": "sleep-3", "topic": "sleep",
     "text": "Sleep consistency matters as much as duration. Going to bed and waking at the same time "
             "every day—even weekends—anchors the circadian rhythm and improves sleep quality."},

    # ── Hydration ─────────────────────────────────────────────────────────────
    {"id": "hydration-1", "topic": "hydration",
     "text": "The WHO recommends approximately 2.0 litres of water per day for adults in temperate climates, "
             "increasing to 2.5–3.0 L in hot weather or during physical activity."},
    {"id": "hydration-2", "topic": "hydration",
     "text": "Mild dehydration (1–2% body weight) impairs cognitive performance, working memory, and mood. "
             "Thirst is already a sign of mild dehydration—drink before feeling thirsty."},

    # ── Physical activity ─────────────────────────────────────────────────────
    {"id": "activity-1", "topic": "exercise",
     "text": "WHO global physical activity guidelines recommend 150–300 minutes of moderate-intensity aerobic "
             "activity per week, or 75–150 minutes of vigorous activity, plus muscle-strengthening on 2+ days."},
    {"id": "activity-2", "topic": "exercise",
     "text": "Walking 7,000–10,000 steps daily is associated with lower all-cause mortality. "
             "Even 4,000–5,000 steps per day provides significant health benefits over a sedentary lifestyle."},
    {"id": "activity-3", "topic": "exercise",
     "text": "Sedentary behaviour (sitting > 8 hours/day) is an independent risk factor for cardiovascular disease, "
             "even in people who meet weekly exercise guidelines. Take a 2-minute movement break every 30 minutes."},

    # ── Nutrition ─────────────────────────────────────────────────────────────
    {"id": "nutrition-1", "topic": "nutrition",
     "text": "Eating the last meal of the day 2–3 hours before bedtime improves sleep quality and supports "
             "healthy weight management by aligning food intake with circadian digestive rhythms."},
    {"id": "nutrition-2", "topic": "nutrition",
     "text": "Breakfast timing within 1–2 hours of waking helps regulate blood sugar, reduces cortisol spikes, "
             "and has been associated with better concentration and mood throughout the morning."},
    {"id": "nutrition-3", "topic": "nutrition",
     "text": "A Mediterranean-style diet—rich in vegetables, whole grains, legumes, fish, and olive oil—reduces "
             "risk of depression by up to 33% compared to a Western diet high in processed foods."},

    # ── Mental health & stress ────────────────────────────────────────────────
    {"id": "mental-1", "topic": "mental health",
     "text": "App fatigue (cognitive overload from managing many digital applications) is a real phenomenon. "
             "Research shows that reducing app notifications to only critical ones lowers cortisol and improves focus."},
    {"id": "mental-2", "topic": "mental health",
     "text": "The Pomodoro Technique (25 minutes focused work, 5 minute break) leverages ultradian rhythms "
             "and is scientifically supported for maintaining high cognitive performance over long work sessions."},
    {"id": "mental-3", "topic": "mental health",
     "text": "Journaling for 15–20 minutes about daily goals and achievements activates the prefrontal cortex "
             "and is one of the most evidence-based interventions for building self-regulation habits."},

    # ── Ecology / environment ─────────────────────────────────────────────────
    {"id": "eco-1", "topic": "ecology",
     "text": "Air Quality Index (AQI) values below 50 are considered Good. Values 51–100 are Moderate. "
             "Sensitive groups should reduce outdoor activity when AQI exceeds 100. Above 150 is Unhealthy for all."},
    {"id": "eco-2", "topic": "ecology",
     "text": "Kazakhstan launched a colour-coded waste sorting programme: yellow for plastic and metal, "
             "blue for paper and cardboard, green for glass, grey for mixed/organic waste."},
    {"id": "eco-3", "topic": "ecology",
     "text": "Almaty consistently records elevated PM2.5 particulate matter in winter due to coal heating. "
             "An N95 mask filters 95% of particles ≥ 0.3 µm and is recommended on days with AQI > 150."},

    # ── Habit formation ───────────────────────────────────────────────────────
    {"id": "habit-1", "topic": "habits",
     "text": "The habit loop (cue–routine–reward) described by Charles Duhigg shows that habits are never truly "
             "deleted—only replaced. Identify the cue and reward to change the routine effectively."},
    {"id": "habit-2", "topic": "habits",
     "text": "James Clear's Atomic Habits principle: make habits 2% better each day. "
             "A 1% improvement daily compounds to 37x better performance over a year."},
    {"id": "habit-3", "topic": "habits",
     "text": "Habit stacking—attaching a new habit to an existing one—is one of the most reliable methods "
             "for behaviour change. Example: 'After I pour my morning coffee, I will write 3 goals for the day.'"},
]


def _tokenise(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{2,}\b", text.lower())

def _tf(tokens: list[str]) -> dict[str, float]:
    c = Counter(tokens)
    total = len(tokens) or 1
    return {w: cnt / total for w, cnt in c.items()}

def _build_idf(docs: list[list[str]]) -> dict[str, float]:
    N = len(docs)
    df: dict[str, int] = {}
    for doc in docs:
        for w in set(doc):
            df[w] = df.get(w, 0) + 1
    return {w: math.log((N + 1) / (freq + 1)) + 1 for w, freq in df.items()}

def _tfidf_vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = _tf(tokens)
    return {w: tf[w] * idf.get(w, 1.0) for w in tf}

def _cosine(a: dict, b: dict) -> float:
    keys = set(a) & set(b)
    dot  = sum(a[k] * b[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values())) or 1
    norm_b = math.sqrt(sum(v * v for v in b.values())) or 1
    return dot / (norm_a * norm_b)

_corpus_tokens = [_tokenise(doc["text"]) for doc in KNOWLEDGE_BASE]
_idf            = _build_idf(_corpus_tokens)
_corpus_vecs    = [_tfidf_vec(tok, _idf) for tok in _corpus_tokens]


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve the top_k most relevant passages for the query."""
    q_tokens = _tokenise(query)
    q_vec    = _tfidf_vec(q_tokens, _idf)
    scores   = [(_cosine(q_vec, dv), i) for i, dv in enumerate(_corpus_vecs)]
    scores.sort(reverse=True)
    return [
        {**KNOWLEDGE_BASE[i], "relevance_score": round(score, 4)}
        for score, i in scores[:top_k]
    ]

RAG_SYSTEM = """\
You are KunQor's knowledge-grounded wellness advisor (RAG mode).

You will receive:
  [RETRIEVED PASSAGES] – curated facts retrieved from a trusted knowledge base
  [USER QUESTION]      – the user's actual question

STRICT RULES:
1. Base your answer ONLY on the retrieved passages. Do not invent facts.
2. If the passages do not contain enough information, say so honestly.
3. Cite which passage IDs you used (e.g., "According to [sleep-2]...").
4. Return ONLY valid JSON:
{
  "answer": "<well-structured, 3–5 sentence answer with citations>",
  "key_facts": ["<fact 1 from passages>", "<fact 2>", "<fact 3>"],
  "sources_used": ["<passage id 1>", "<passage id 2>"],
  "confidence": "<high|medium|low>"
}
"""

def query(user_query: str) -> dict:
    """
    Full RAG pipeline:
    1. RETRIEVE  – find top-3 relevant passages via TF-IDF cosine similarity
    2. AUGMENT   – inject retrieved context into the prompt
    3. GENERATE  – call llama-3.3-70b to produce a grounded answer
    """
    passages = retrieve(user_query, top_k=3)

    context_block = "\n\n".join(
        f"[{p['id']}] ({p['topic']}): {p['text']}"
        for p in passages
    )

    user_msg = (
        f"[RETRIEVED PASSAGES]\n{context_block}\n\n"
        f"[USER QUESTION]\n{user_query}"
    )

    resp = client.chat.completions.create(
        model       = "llama-3.3-70b-versatile",
        messages    = [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature = 0.2,
    )

    raw = resp.choices[0].message.content or ""
    clean = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    start = clean.find("{")
    end   = clean.rfind("}")
    if start != -1 and end != -1:
        clean = clean[start:end+1]
    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        result = {"answer": raw, "key_facts": [], "sources_used": [], "confidence": "low"}

    result["retrieved_passages"] = passages
    result["model_used"]  = "llama-3.3-70b-versatile"
    result["technique"]   = "Retrieval-Augmented Generation (TF-IDF + LLM)"
    return result

class RAGEngine:
    query    = staticmethod(query)
    retrieve = staticmethod(retrieve)
