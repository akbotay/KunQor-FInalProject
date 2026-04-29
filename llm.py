"""
llm.py  –  KunQor Multi-LLM Manager
=====================================
Demonstrates:
  • Multiple Llama models via Groq with deliberate model selection strategy
  • Structured prompt engineering (system / user / few-shot / chain-of-thought)
  • JSON-mode outputs with schema enforcement
  • Persona prompting, role-playing, constraint injection
  • Temperature tuning per task type
"""

import os
import json
import re
from openai import OpenAI

client = OpenAI(
    api_key  = os.environ.get("GROQ_API_KEY", "gsk_...."),
    base_url = "https://api.groq.com/openai/v1",
)

MODELS = {
    "fast":    "llama-3.1-8b-instant",
    "smart":   "llama-3.3-70b-versatile",
    "search":  "llama-3.3-70b-versatile",
}


def _extract_json(raw: str) -> dict:
    """
    Robustly extract JSON from model output that may contain:
    - Chain-of-thought reasoning text before the JSON
    - Markdown code fences (```json ... ```)
    - Mixed prose and JSON
    Strategy: find the first { and last } and extract that block.
    """

    clean = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    start = clean.find("{")
    end   = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(clean[start:end+1])
        except json.JSONDecodeError:
            pass

    return {"error": "JSON parse failed", "raw": raw[:300]}


def _call(model_key: str,
          system: str,
          user: str,
          tools: list | None = None,
          temperature: float = 0.4,
          json_mode: bool = True) -> dict | str:
    """
    Shared low-level wrapper around the Groq API.
    Handles JSON parsing + graceful error surfacing.
    """
    model = MODELS[model_key]
    kwargs = dict(
        model      = model,
        messages   = [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
        temperature = temperature,
    )
    if tools:
        kwargs["tools"] = tools

    resp = client.chat.completions.create(**kwargs)
    raw  = resp.choices[0].message.content or ""

    if not json_mode:
        return raw

    return _extract_json(raw)

HEALTH_SYSTEM = """\
You are HealthBot, an evidence-based personal health assistant embedded in the \
KunQor daily-life platform. You analyse daily health metrics and return \
structured JSON advice.

RULES:
- Always respond with valid JSON, no markdown, no prose outside the JSON.
- Base recommendations on WHO guidelines.
- Keep language encouraging, never alarmist.
- The "score" field must be an integer 0-100.

FEW-SHOT EXAMPLES (learn the schema from these):

User: {"water_l":2.0,"steps_k":8,"calories":2100,"walk_min":45}
Assistant: {
  "score": 88,
  "verdict": "Excellent day",
  "metrics": [
    {"name":"Hydration","value":"2.0 L","status":"ok","msg":"Perfectly on target"},
    {"name":"Steps","value":"8,000","status":"ok","msg":"Above the 7k daily goal"},
    {"name":"Calories","value":"2,100 kcal","status":"ok","msg":"Within healthy range"},
    {"name":"Walk","value":"45 min","status":"ok","msg":"Meets the 30-min guideline"}
  ],
  "top_tip": "Add a short stretching session before bed to complete your day."
}

User: {"water_l":0.8,"steps_k":2,"calories":3200,"walk_min":5}
Assistant: {
  "score": 31,
  "verdict": "Needs improvement",
  "metrics": [
    {"name":"Hydration","value":"0.8 L","status":"bad","msg":"Drink at least 2 L daily"},
    {"name":"Steps","value":"2,000","status":"bad","msg":"Aim for 7,000+ steps"},
    {"name":"Calories","value":"3,200 kcal","status":"warn","msg":"Slightly above average"},
    {"name":"Walk","value":"5 min","status":"bad","msg":"WHO recommends 30 min/day"}
  ],
  "top_tip": "Start with a 10-minute walk after lunch – it is the easiest win today."
}
"""

def analyze_health(data: dict) -> dict:
    """
    Model chosen: llama-3.1-8b-instant (fast, sufficient for structured extraction).
    Temperature 0.2 → deterministic, factual output.
    Prompt technique: Few-shot prompting with JSON schema examples.
    """
    user_msg = json.dumps({
        "water_l":   data.get("water",    1.5),
        "steps_k":   data.get("steps",    5),
        "calories":  data.get("calories", 2000),
        "walk_min":  data.get("walk",     30),
    })
    result = _call("fast", HEALTH_SYSTEM, user_msg, temperature=0.2)
    result["model_used"] = MODELS["fast"]
    result["technique"]  = "Few-shot prompting + JSON-mode"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  REGIME ANALYSIS  –  llama-3.3-70b-versatile + CHAIN-OF-THOUGHT prompting
# ═══════════════════════════════════════════════════════════════════════════════
REGIME_SYSTEM = """\
You are CircadianAI, a chronobiology-informed daily schedule optimiser inside \
the KunQor platform. You think step-by-step before producing your final answer.

CHAIN-OF-THOUGHT PROTOCOL:
1. Calculate total sleep quality (duration + timing relative to circadian norms).
2. Check meal timing relative to wakeup and bedtime (should not eat 2h before sleep).
3. Identify work-life balance: work_hours / (24 - sleep_hours).
4. Synthesise a score and actionable next-day plan.

Only after completing the reasoning above, emit EXACTLY this JSON (no other text):
{
  "score": <int 0-100>,
  "verdict": "<one short sentence>",
  "sleep_quality": "<ok|warn|bad>",
  "meal_timing": "<ok|warn|bad>",
  "balance": "<ok|warn|bad>",
  "problems": ["<issue1>", ...],
  "tips": ["<tip1>", ...],
  "tomorrow_plan": "<single sentence for tomorrow>"
}
"""

def analyze_routine(data: dict) -> dict:
    """
    Model chosen: llama-3.3-70b-versatile (better reasoning for multi-variable optimisation).
    Temperature 0.3 → allows slight creative variety in tips.
    Prompt technique: Chain-of-Thought with explicit reasoning steps.
    """
    user_msg = (
        f"Sleep: {data.get('sleep', 7)}h  |  Work: {data.get('work', 8)}h  |  "
        f"Bedtime: {data.get('bedtime', '23:00')}  |  Wake: {data.get('wakeup', '07:00')}  |  "
        f"Breakfast: {data.get('breakfast', '08:00')}  |  Dinner: {data.get('dinner', '19:00')}"
    )
    result = _call("smart", REGIME_SYSTEM, user_msg, temperature=0.3)
    result["model_used"] = MODELS["smart"]
    result["technique"]  = "Chain-of-thought (CoT) prompting"
    return result

ECOLOGY_SYSTEM = """\
You are EcoAdvisor, an environmental intelligence assistant for Kazakhstan cities \
integrated into the KunQor platform.

TASK:
1. Based on your knowledge, provide a realistic estimated AQI for the given city.
2. Provide waste-sorting guidance specific to Kazakhstan regulations.
3. Return ONLY valid JSON, no additional text:
{
  "aqi": <int>,
  "aqi_label": "<Good|Moderate|Unhealthy for Sensitive|Unhealthy|Very Unhealthy|Hazardous>",
  "aqi_color": "<green|yellow|orange|red|purple|maroon>",
  "health_advice": "<one sentence based on the AQI level>",
  "waste_container": "<colour of container for the described waste in Kazakhstan>",
  "waste_tip": "<specific recycling or disposal tip>",
  "eco_action": "<one small action the user can take today>"
}
"""

def analyze_ecology(data: dict) -> dict:
    """
    Model chosen: llama-3.3-70b-versatile for environmental knowledge.
    Prompt technique: Structured output with role + task decomposition.
    """
    city  = data.get("city",  "Almaty")
    trash = data.get("trash", "plastic bottle")

    user_msg = (
        f"City: {city}. "
        f"Waste to dispose today: '{trash}'. "
        f"Provide the full JSON response with AQI estimate and waste guidance."
    )

    result = _call("smart", ECOLOGY_SYSTEM, user_msg, temperature=0.2)
    result["model_used"] = MODELS["smart"]
    result["technique"]  = "Role + task decomposition prompting"
    return result

COMPARE_SYSTEM = """\
You are a life-coaching assistant. Given the user's concern, respond with a short \
3-bullet action plan. Be direct, practical, and empathetic.
Respond in this JSON format:
{
  "summary": "<1-sentence diagnosis>",
  "actions": ["<action 1>", "<action 2>", "<action 3>"],
  "motivation": "<one inspiring closing line>"
}
"""

def compare_models(concern: str) -> dict:
    """
    Sends the SAME prompt to both llama-3.1-8b-instant and llama-3.3-70b-versatile.
    Demonstrates that model choice meaningfully affects output quality and detail.
    """
    fast_result  = _call("fast",  COMPARE_SYSTEM, concern, temperature=0.6)
    smart_result = _call("smart", COMPARE_SYSTEM, concern, temperature=0.6)

    return {
        "concern": concern,
        "fast_model":  {"name": MODELS["fast"],  "response": fast_result},
        "smart_model": {"name": MODELS["smart"], "response": smart_result},
        "technique": "Multi-LLM comparison (same prompt, different models)",
    }

COACH_SYSTEM = """\
You are "Qor" – the friendly AI coach embedded in KunQor. You speak with warmth \
and precision. You NEVER use vague advice like "try your best". Every action you \
suggest must be:
  [SPECIFIC]  – say exactly what to do
  [TIMED]     – include when or for how long
  [MEASURABLE]– state a success criterion

PERSONA CONSTRAINTS:
- Use the user's name if provided.
- Speak in the second person ("you").
- End every response with an emoji that matches the energy.

OUTPUT FORMAT (strict JSON, no prose outside):
{
  "greeting": "<personalised opening>",
  "diagnosis": "<2-sentence analysis>",
  "week_plan": [
    {"day_range": "Day 1-2", "task": "<specific task>", "metric": "<how to measure>"},
    {"day_range": "Day 3-5", "task": "<specific task>", "metric": "<how to measure>"},
    {"day_range": "Day 6-7", "task": "<specific task>", "metric": "<how to measure>"}
  ],
  "closing": "<motivational closing with emoji>"
}
"""

def goal_coach(data: dict) -> dict:
    """
    Model chosen: llama-3.3-70b-versatile (persona + constraint following benefits from larger model).
    Temperature 0.7 → more expressive, personality-forward output.
    Prompt technique: Persona + output constraints + negative examples.
    """
    name    = data.get("name",    "friend")
    concern = data.get("concern", "I want to improve my daily routine")

    user_msg = f"Name: {name}. Concern: {concern}"
    result   = _call("smart", COACH_SYSTEM, user_msg, temperature=0.7)
    result["model_used"] = MODELS["smart"]
    result["technique"]  = "Persona + constraint + structured output prompting"
    return result


class LLMRouter:
    analyze_health  = staticmethod(analyze_health)
    analyze_routine  = staticmethod(analyze_routine)
    analyze_ecology = staticmethod(analyze_ecology)
    compare_models  = staticmethod(compare_models)
    goal_coach      = staticmethod(goal_coach)
