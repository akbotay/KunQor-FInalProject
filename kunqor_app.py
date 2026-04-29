"""
KunQor – AI Daily Life Quality Platform
========================================
Entry point. Run:  python app.py
Then open http://localhost:8080
"""

from flask import Flask, request, jsonify, render_template
from llm import LLMRouter
from rag_engine import RAGEngine
from agent import AgentOrchestrator
import json

app = Flask(__name__)

llm_router      = LLMRouter()
rag_engine      = RAGEngine()
agent_orch      = AgentOrchestrator()

@app.route("/")
def index():
    return render_template("ai_platform.html")

@app.route("/api/health", methods=["POST"])
def api_health():
    data = request.json
    result = llm_router.analyze_health(data)
    return jsonify(result)

@app.route("/api/routine", methods=["POST"])
def api_routine():
    data = request.json
    result = llm_router.analyze_routine(data)
    return jsonify(result)

@app.route("/api/ecology", methods=["POST"])
def api_ecology():
    data = request.json
    result = llm_router.analyze_ecology(data)
    return jsonify(result)

@app.route("/api/rag", methods=["POST"])
def api_rag():
    query = request.json.get("query", "")
    result = rag_engine.query(query)
    return jsonify(result)

@app.route("/api/agent", methods=["POST"])
def api_agent():
    task = request.json.get("task", "")
    result = agent_orch.run(task)
    return jsonify(result)

@app.route("/api/compare", methods=["POST"])
def api_compare():
    concern = request.json.get("concern", "")
    result  = llm_router.compare_models(concern)
    return jsonify(result)

@app.route("/api/coach", methods=["POST"])
def api_coach():
    data = request.json
    result = llm_router.goal_coach(data)
    return jsonify(result)

if __name__ == "__main__":
    print("| KunQor AI Platform  –  localhost:8080   |")
    app.run(debug=True, port=8080)
