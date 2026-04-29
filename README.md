# ✦ KünQor — Daily Life Quality AI

A Streamlit web app powered by **Groq API (LLaMA 3 70B)** that helps you track and improve your daily life quality across 5 areas.

---

## Features

| Tab | What it does |
|-----|-------------|
| 🎯 Goals | Set daily goals, check them off |
| 🕐 Daily Routine | Analyze sleep, work & meal schedule |
| 💪 Health | Track water, steps, calories, walking |
| 🌿 Ecology | Air quality + waste sorting for Kazakhstan cities |
| 🤖 AI Advice | Personal coach that creates action plans |

---

## Run in Google Colab (Recommended)

1. Upload `KunQor_Colab.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Run cells in order: **Cell 1 → Cell 2 → Cell 3 → Cell 4**
3. Click the link that appears after Cell 4
4. Enter the password shown in the output
5. Paste your Groq API key in the sidebar

---

## Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run kunqor_app.py
```

### 3. Get your free Groq API key
Go to [console.groq.com](https://console.groq.com) → Sign up → Create API Key (free, takes 1 minute)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Main language |
| Streamlit | Web interface |
| Groq API | AI brain (LLaMA 3 70B) |
| localtunnel | Share app from Colab |

---

## Project Structure

```
kunqor/
├── kunqor_app.py        # Main Streamlit app
├── KunQor_Colab.ipynb   # Google Colab notebook
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

*Built as a final project for AI & Prompt Engineering course*
