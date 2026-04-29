import streamlit as st
from groq import Groq
import json
from datetime import date

st.set_page_config(
    page_title="KunQor — Daily Life AI",
    page_icon="✦",
    layout="centered"
)

st.markdown("""
<style>
    .tip-box {
        background: #f0f7ff;
        border-left: 3px solid #7F77DD;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 12px;
    }
    .goal-done { text-decoration: line-through; color: #adb5bd; }
</style>
""", unsafe_allow_html=True)

def ask_groq(prompt: str) -> dict:
    api_key = st.session_state.get("groq_api_key", "")
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        return {}
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "Respond ONLY with valid JSON. No markdown, no explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        st.error("AI returned invalid response. Try again.")
        return {}
    except Exception as e:
        st.error(f"Error: {e}")
        return {}

if "goals" not in st.session_state:
    st.session_state.goals = [
        {"text": "Drink 8 glasses of water", "done": False},
        {"text": "Walk 30 minutes", "done": False},
        {"text": "Sleep before 11 PM", "done": False},
    ]

with st.sidebar:
    st.markdown("### ✦ KunQor")
    st.markdown("*Daily Life Quality AI*")
    st.divider()
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if api_key:
        st.session_state.groq_api_key = api_key
        st.success("API key saved ✓")
    st.divider()
    st.markdown(f"📅 **{date.today().strftime('%A, %B %d')}**")
    done = sum(1 for g in st.session_state.goals if g["done"])
    total = len(st.session_state.goals)
    pct = int(done / total * 100) if total else 0
    st.markdown(f"**Progress: {pct}%**")
    st.progress(pct / 100)

st.markdown("## ✦ KunQor — Daily Life Quality")
st.markdown(f"*{date.today().strftime('%A, %B %d, %Y')}*")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Goals", "🕐 Daily Routine", "💪 Health", "🌿 Ecology", "🤖 AI Advice"
])

with tab1:
    st.markdown("### Today's Goals")
    for i, goal in enumerate(st.session_state.goals):
        col1, col2, col3 = st.columns([0.08, 0.8, 0.12])
        with col1:
            checked = st.checkbox("", value=goal["done"], key=f"goal_{i}")
            st.session_state.goals[i]["done"] = checked
        with col2:
            if checked:
                st.markdown(f"<span class='goal-done'>{goal['text']}</span>", unsafe_allow_html=True)
            else:
                st.markdown(goal["text"])
        with col3:
            if st.button("✕", key=f"del_{i}"):
                st.session_state.goals.pop(i)
                st.rerun()
    st.divider()
    new_goal = st.text_input("➕ Add a new goal", placeholder="e.g. Read 20 pages...")
    if st.button("Add Goal", use_container_width=True):
        if new_goal.strip():
            st.session_state.goals.append({"text": new_goal.strip(), "done": False})
            st.rerun()
    st.divider()
    done = sum(1 for g in st.session_state.goals if g["done"])
    total = len(st.session_state.goals)
    pct = int(done / total * 100) if total else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Done", done)
    c2.metric("⏳ Remaining", total - done)
    c3.metric("📊 Progress", f"{pct}%")
    st.progress(pct / 100)

with tab2:
    st.markdown("### Daily Routine Analyzer")
    c1, c2 = st.columns(2)
    with c1:
        sleep_hours = st.number_input("😴 Sleep (hours)", 0, 24, 7)
        bedtime = st.time_input("🌙 Bedtime", value=None)
        breakfast = st.time_input("🍳 Breakfast", value=None)
    with c2:
        work_hours = st.number_input("💼 Work (hours)", 0, 24, 8)
        wakeup = st.time_input("☀️ Wake up", value=None)
        dinner = st.time_input("🍽️ Dinner", value=None)
    if st.button("🔍 Analyze My Routine", use_container_width=True):
        with st.spinner("AI is analyzing your routine..."):
            prompt = f"""Analyze this daily routine. Sleep: {sleep_hours}h, Bedtime: {bedtime}, Wake: {wakeup}, Work: {work_hours}h, Breakfast: {breakfast}, Dinner: {dinner}.
Respond ONLY with JSON: {{"score": 75, "verdict": "Needs improvement", "problems": ["Late dinner"], "tips": ["Sleep earlier"], "tomorrow_plan": "One sentence tip"}}"""
            r = ask_groq(prompt)
            if r:
                score = r.get("score", 0)
                icon = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                st.markdown(f"### {icon} Score: **{score}/100** — {r.get('verdict')}")
                st.progress(score / 100)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**⚠️ Problems:**")
                    for p in r.get("problems", []): st.markdown(f"- {p}")
                with col2:
                    st.markdown("**💡 Tips:**")
                    for t in r.get("tips", []): st.markdown(f"- {t}")
                st.info(f"🗓️ Tomorrow: {r.get('tomorrow_plan')}")

with tab3:
    st.markdown("### Health Tracker")
    c1, c2 = st.columns(2)
    with c1:
        water = st.number_input("💧 Water (liters)", 0.0, 10.0, 1.5, 0.1)
        steps = st.number_input("👟 Steps (thousands)", 0, 50, 5)
    with c2:
        calories = st.number_input("🔥 Calories", 0, 5000, 2000, 50)
        walk_min = st.number_input("🚶 Walking (minutes)", 0, 300, 30)
    if st.button("📊 Analyze My Health", use_container_width=True):
        with st.spinner("Analyzing health data..."):
            prompt = f"""Analyze: Water {water}L, Steps {steps}000, Calories {calories}, Walking {walk_min}min.
Respond ONLY with JSON: {{"overall_score": 70, "water_ok": false, "water_msg": "Drink more", "steps_ok": true, "steps_msg": "Good", "calories_ok": true, "calories_msg": "Normal", "walk_ok": false, "walk_msg": "Aim for 45min", "main_tip": "Key advice"}}"""
            r = ask_groq(prompt)
            if r:
                score = r.get("overall_score", 0)
                icon = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                st.markdown(f"### {icon} Health Score: **{score}/100**")
                st.progress(score / 100)
                c1, c2 = st.columns(2)
                items = [
                    ("💧 Water", f"{water}L", r.get("water_ok"), r.get("water_msg")),
                    ("👟 Steps", f"{steps}k", r.get("steps_ok"), r.get("steps_msg")),
                    ("🔥 Calories", str(calories), r.get("calories_ok"), r.get("calories_msg")),
                    ("🚶 Walking", f"{walk_min}m", r.get("walk_ok"), r.get("walk_msg")),
                ]
                for i, (label, val, ok, msg) in enumerate(items):
                    col = c1 if i % 2 == 0 else c2
                    with col:
                        st.metric(f"{label} {'✅' if ok else '⚠️'}", val, msg)
                st.info(f"💡 {r.get('main_tip')}")

with tab4:
    st.markdown("### Ecology Checker")
    city = st.selectbox("🏙️ Your city", ["Almaty", "Astana", "Shymkent", "Karaganda"])
    trash = st.text_input("♻️ What did you throw away?", placeholder="e.g. plastic bottle, cardboard...")
    if st.button("🌿 Check Eco Impact", use_container_width=True):
        with st.spinner("Checking ecological data..."):
            prompt = f"""Kazakhstan ecology expert. City: {city}. Trash: "{trash}".
Respond ONLY with JSON: {{"aqi": 85, "aqi_label": "Moderate", "container": "Yellow bin (plastic)", "decompose_years": "400-500 years", "recycle_into": "New plastic products", "drop_off": "Recycling point in {city}", "eco_tip": "Practical eco tip"}}"""
            r = ask_groq(prompt)
            if r:
                aqi = r.get("aqi", 0)
                icon = "🟢" if aqi < 50 else "🟡" if aqi < 100 else "🔴"
                c1, c2 = st.columns(2)
                c1.metric(f"{icon} Air Quality ({city})", aqi, r.get("aqi_label"))
                c2.metric("🗑️ Container", r.get("container", ""))
                st.markdown(f"♻️ **Recycled into:** {r.get('recycle_into')}")
                st.markdown(f"📍 **Drop-off:** {r.get('drop_off')}")
                st.markdown(f"⏳ **Decomposes in:** {r.get('decompose_years')}")
                st.info(f"🌱 {r.get('eco_tip')}")
with tab5:
    st.markdown("### AI Personal Coach")
    st.markdown("Tell the AI what's bothering you — get a personal action plan.")
    concern = st.text_area("What's on your mind?",
        placeholder="e.g. My sleep schedule is broken, I feel tired all the time",
        height=120)
    if st.button("🤖 Get AI Advice", use_container_width=True):
        if not concern.strip():
            st.warning("Please describe your concern first.")
        else:
            with st.spinner("Creating your personal plan..."):
                prompt = f"""You are a friendly life coach AI. User concern: "{concern}"
Respond ONLY with JSON: {{"title": "Problem title", "diagnosis": "2 sentence analysis", "plan": [{{"period": "Days 1-3", "action": "First step"}}, {{"period": "Week 1", "action": "Next step"}}, {{"period": "Month 1", "action": "Long-term goal"}}], "motivation": "Inspiring closing sentence"}}"""
                r = ask_groq(prompt)
                if r:
                    st.markdown(f"### 🎯 {r.get('title')}")
                    st.info(r.get("diagnosis", ""))
                    st.markdown("**📋 Action Plan:**")
                    for step in r.get("plan", []):
                        with st.expander(f"🗓️ {step.get('period')}"):
                            st.markdown(step.get("action"))
                    st.success(f"✨ {r.get('motivation')}")

st.divider()
st.markdown(
    "<center><small>KunQor — Daily Life Quality AI · Streamlit + Groq API</small></center>",
    unsafe_allow_html=True
)
