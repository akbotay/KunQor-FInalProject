"""
agent_orchestrator.py  –  KunQor AI Agent with Tool Use
=========================================================
Demonstrates:
  • ReAct (Reason + Act) agent loop
  • Tool-using LLM: the model decides WHICH tool to call and WHEN
  • Custom tool registry (no external framework needed)
  • Multi-step agentic reasoning
  • Transparent trace: every thought and action is logged

Tools available to the agent:
  - get_aqi          : fetch Air Quality Index for a city (simulated live data)
  - calculate_bmi    : compute BMI from height/weight
  - habit_score      : score a habit description against evidence-based criteria
  - schedule_advice  : analyse a time slot for circadian-optimality
  - rag_lookup       : search the KunQor knowledge base
"""

import os
import json
import re
import math
import datetime
from openai import OpenAI
from rag_engine import retrieve   # reuse the RAG retriever

client = OpenAI(
    api_key  = os.environ.get("GROQ_API_KEY", "gsk_..."),
    base_url = "https://api.groq.com/openai/v1",
)


def get_aqi(city: str) -> dict:
    """
    Simulated AQI lookup.
    In production: call IQAir / OpenWeatherMap API.
    """
    AQI_TABLE = {
        "almaty": 95, "astana": 42, "shymkent": 78,
        "atyrau": 55, "aktobe": 61, "karaganda": 88,
    }
    aqi = AQI_TABLE.get(city.lower(), 60)
    if   aqi <= 50:  label, advice = "Good",     "Air quality is satisfactory."
    elif aqi <= 100: label, advice = "Moderate", "Acceptable; sensitive individuals should limit prolonged outdoor exertion."
    elif aqi <= 150: label, advice = "Unhealthy for Sensitive Groups", "Sensitive groups should reduce outdoor activity."
    else:            label, advice = "Unhealthy", "Everyone should reduce outdoor activity; consider an N95 mask."
    return {"city": city, "aqi": aqi, "label": label, "advice": advice}


def calculate_bmi(height_cm: float, weight_kg: float) -> dict:
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1)
    if   bmi < 18.5: cat = "Underweight"
    elif bmi < 25.0: cat = "Normal weight"
    elif bmi < 30.0: cat = "Overweight"
    else:            cat = "Obese"
    return {"bmi": bmi, "category": cat,
            "advice": f"BMI {bmi} is classified as {cat} by WHO guidelines."}


def habit_score(habit_description: str) -> dict:
    """Score how well a described habit matches evidence-based criteria."""
    desc = habit_description.lower()
    score = 0
    feedback = []
    if any(w in desc for w in ["morning", "evening", "after", "before", "when"]):
        score += 30; feedback.append("✓ Has a clear trigger / cue")
    else:
        feedback.append("✗ Add a specific trigger (time or existing habit)")
    if any(w in desc for w in ["minute", "hour", "times", "daily", "weekly"]):
        score += 30; feedback.append("✓ Has a defined duration or frequency")
    else:
        feedback.append("✗ Specify how long or how often")
    if any(w in desc for w in ["track", "measure", "count", "record", "log"]):
        score += 20; feedback.append("✓ Has a measurement mechanism")
    else:
        feedback.append("✗ Add a way to track progress")
    if any(w in desc for w in ["reward", "feel", "enjoy", "because", "goal"]):
        score += 20; feedback.append("✓ Linked to a reward or motivation")
    else:
        feedback.append("✗ Connect to a reward or deeper motivation")
    return {"score": score, "max": 100, "feedback": feedback,
            "verdict": "Strong habit" if score >= 70 else "Needs refinement"}


def schedule_advice(time_slot: str, activity: str) -> dict:
    """Advise whether a time slot is circadian-optimal for the given activity."""
    try:
        h = int(time_slot.split(":")[0])
    except (ValueError, IndexError):
        return {"error": "Invalid time format. Use HH:MM"}

    OPTIMAL = {
        "exercise":     (6, 12),
        "deep work":    (8, 12),
        "creativity":   (10, 14),
        "meditation":   (6, 9),
        "eating":       (7, 19),
        "social":       (15, 20),
        "sleep":        (21, 6),
        "learning":     (10, 16),
    }
    act_lower = activity.lower()
    match = next((k for k in OPTIMAL if k in act_lower), None)

    if match:
        start, end = OPTIMAL[match]
        in_range = (start <= h < end) if start < end else (h >= start or h < end)
        return {
            "time": time_slot, "activity": activity,
            "optimal_window": f"{start:02d}:00–{end:02d}:00",
            "is_optimal": in_range,
            "advice": (f"✓ {time_slot} is within the circadian-optimal window for {match}."
                       if in_range else
                       f"⚠ Consider shifting to {start:02d}:00–{end:02d}:00 for better results.")
        }
    return {"time": time_slot, "activity": activity,
            "advice": "No specific circadian guideline found; apply general healthy-hour principles."}


def rag_lookup(query_text: str) -> dict:
    passages = retrieve(query_text, top_k=2)
    return {"passages": passages, "count": len(passages)}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_aqi",
            "description": "Get the current Air Quality Index (AQI) for a Kazakhstan city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name in Kazakhstan"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": "Calculate BMI from height (cm) and weight (kg) and interpret the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "height_cm": {"type": "number"},
                    "weight_kg": {"type": "number"}
                },
                "required": ["height_cm", "weight_kg"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "habit_score",
            "description": "Score and critique a habit description based on behaviour-change science.",
            "parameters": {
                "type": "object",
                "properties": {"habit_description": {"type": "string"}},
                "required": ["habit_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_advice",
            "description": "Check whether a specific time slot is circadian-optimal for an activity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_slot": {"type": "string", "description": "HH:MM format"},
                    "activity": {"type": "string", "description": "e.g. exercise, deep work, eating"}
                },
                "required": ["time_slot", "activity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_lookup",
            "description": "Search the KunQor health knowledge base for evidence-based information.",
            "parameters": {
                "type": "object",
                "properties": {"query_text": {"type": "string"}},
                "required": ["query_text"]
            }
        }
    },
]

TOOL_FN_MAP = {
    "get_aqi":       get_aqi,
    "calculate_bmi": calculate_bmi,
    "habit_score":   habit_score,
    "schedule_advice": schedule_advice,
    "rag_lookup":    rag_lookup,
}


AGENT_SYSTEM = """\
You are "Qor Agent" — an autonomous AI assistant embedded in KunQor, a daily \
life quality platform. You have access to several tools and you MUST use them \
to ground your answers in real data rather than guessing.

AGENT PROTOCOL (ReAct — Reason + Act):
1. THINK: Identify what information you need to answer fully.
2. ACT:   Call the most appropriate tool(s). You may call multiple tools.
3. OBSERVE: Process each tool result.
4. ANSWER: Synthesise a final comprehensive response.

Always prefer tool use over relying solely on your training data.
When the task involves multiple sub-questions, break it down and use tools for each part.

FINAL OUTPUT: Respond with a helpful, structured paragraph. Mention which tools you used.
"""


def run(task: str) -> dict:
    """
    ReAct agent loop: allows the model to call multiple tools before answering.
    Returns the final answer plus a full execution trace.
    """
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user",   "content": task},
    ]
    trace = []
    max_iterations = 6

    for iteration in range(max_iterations):
        resp = client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = messages,
            tools       = TOOLS,
            temperature = 0.3,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            final_answer = msg.content or ""
            trace.append({"step": "FINAL ANSWER", "content": final_answer})
            return {
                "task":         task,
                "answer":       final_answer,
                "trace":        trace,
                "iterations":   iteration + 1,
                "model_used":   "llama-3.3-70b-versatile",
                "technique":    "ReAct Agent (Reason + Act) with tool-use",
            }

        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [
                             {"id": tc.id, "type": "function",
                              "function": {"name": tc.function.name,
                                           "arguments": tc.function.arguments}}
                             for tc in msg.tool_calls
                         ]})

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            fn = TOOL_FN_MAP.get(fn_name)
            if fn:
                tool_result = fn(**fn_args)
            else:
                tool_result = {"error": f"Unknown tool: {fn_name}"}

            step_entry = {
                "step": f"TOOL CALL #{iteration+1}",
                "tool": fn_name,
                "args": fn_args,
                "result": tool_result,
            }
            trace.append(step_entry)

            messages.append({
                "role":        "tool",
                "tool_call_id": tc.id,
                "content":     json.dumps(tool_result),
            })

    return {
        "task": task,
        "answer": "Agent reached maximum iterations without a final answer.",
        "trace": trace,
        "iterations": max_iterations,
        "model_used": "llama-3.3-70b-versatile",
        "technique": "ReAct Agent",
    }

class AgentOrchestrator:
    run = staticmethod(run)
