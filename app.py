"""
app.py — Synthetic Sentinel Streamlit Dashboard
=================================================
Phase 4: Interactive demo for AI-generated text detection.

Run:
    streamlit run app.py

Features:
  - Paste text or upload CSV for batch analysis
  - Synthetic Probability Score with gauge visualization
  - Binary classification (Human vs AI)
  - Burstiness sentence-level analysis chart
  - Token-level attention highlights
"""

import os
import re
import math

import numpy as np
import pandas as pd

import torch
from transformers import (
    RobertaTokenizerFast,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from train_model import HybridDetector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_detector():
    """Load the hybrid RoBERTa model."""
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    model = HybridDetector()
    model_path = os.path.join(MODEL_DIR, "best_roberta.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


@st.cache_resource
def load_gpt2():
    """Load GPT-2 for perplexity computation."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def compute_burstiness(text: str) -> tuple[float, list[int]]:
    """Return (burstiness_score, sentence_lengths)."""
    sentences = re.split(r"[.!?]+", text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0.0, lengths
    return float(np.std(lengths)), lengths


def compute_perplexity(text: str, model, tokenizer) -> float:
    """Compute GPT-2 perplexity for a single text."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return math.exp(min(out.loss.item(), 100))


def predict_single(text: str, model, tokenizer, gpt2_model, gpt2_tokenizer):
    """Run full pipeline: features -> prediction -> probability."""
    burstiness, sent_lengths = compute_burstiness(text)
    perplexity = compute_perplexity(text, gpt2_model, gpt2_tokenizer)

    encoding = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=256, padding="max_length",
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    perp_t = torch.tensor([perplexity], dtype=torch.float).to(DEVICE)
    burst_t = torch.tensor([burstiness], dtype=torch.float).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, perp_t, burst_t)
        probs = torch.softmax(logits, dim=1)[0]

    prob_ai = float(probs[1])
    label = "🤖 AI-Generated" if prob_ai >= 0.5 else "👤 Human-Written"

    return {
        "probability": prob_ai,
        "label": label,
        "perplexity": perplexity,
        "burstiness": burstiness,
        "sentence_lengths": sent_lengths,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def gauge_chart(probability: float):
    """Create a gauge chart for the synthetic probability score."""
    color = (
        "#3fb950" if probability < 0.3 else
        "#d29922" if probability < 0.7 else
        "#f85149"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 48, "color": "white"}},
        title={"text": "Synthetic Probability", "font": {"size": 20, "color": "#c9d1d9"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 30], "color": "rgba(63,185,80,0.15)"},
                {"range": [30, 70], "color": "rgba(210,153,34,0.15)"},
                {"range": [70, 100], "color": "rgba(248,81,73,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#58a6ff", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=300,
        margin={"t": 80, "b": 20, "l": 40, "r": 40},
    )
    return fig


def burstiness_chart(sentence_lengths: list[int]):
    """Bar chart of sentence lengths."""
    fig = px.bar(
        x=list(range(1, len(sentence_lengths) + 1)),
        y=sentence_lengths,
        labels={"x": "Sentence #", "y": "Word Count"},
        title="Sentence Length Distribution (Burstiness)",
        color_discrete_sequence=["#58a6ff"],
    )
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#c9d1d9",
        title_font_size=16,
        height=300,
        margin={"t": 60, "b": 40},
        xaxis={"gridcolor": "#21262d"},
        yaxis={"gridcolor": "#21262d"},
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Synthetic Sentinel",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        .stApp {
            background-color: #0d1117;
            font-family: 'Inter', sans-serif;
        }
        .main-header {
            text-align: center;
            padding: 2rem 0 1rem;
        }
        .main-header h1 {
            color: #58a6ff;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
        }
        .main-header p {
            color: #8b949e;
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }
        .result-card {
            background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .metric-label {
            color: #8b949e;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            color: #c9d1d9;
            font-size: 1.4rem;
            font-weight: 600;
        }
        .label-human {
            color: #3fb950;
            font-size: 1.5rem;
            font-weight: 700;
        }
        .label-ai {
            color: #f85149;
            font-size: 1.5rem;
            font-weight: 700;
        }
        .stTextArea textarea {
            background-color: #161b22 !important;
            color: #c9d1d9 !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
            font-family: 'Inter', sans-serif !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, #238636, #2ea043) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.2s !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(35,134,54,0.4) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Synthetic Sentinel</h1>
        <p>AI-Generated Misinformation Detection • RoBERTa + Stylometry Hybrid Model</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    try:
        model, tokenizer = load_detector()
        gpt2_model, gpt2_tokenizer = load_gpt2()
        model_loaded = True
    except Exception as e:
        st.warning(f"[!]️ Model not loaded: {e}. Train the model first with `python train_model.py`.")
        model_loaded = False

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        mode = st.radio("Input Mode", ["📝 Text Input", "📊 Batch CSV"])

        st.markdown("---")
        st.markdown("### 📖 How It Works")
        st.markdown("""
        1. **RoBERTa** analyzes contextual language patterns
        2. **Perplexity** measures text predictability (AI = lower)
        3. **Burstiness** measures sentence-length variance (humans = higher)
        4. A **hybrid MLP** combines these signals for final prediction
        """)

        st.markdown("---")
        st.markdown("### 🧪 Sample Texts")
        if st.button("Load Human Sample"):
            st.session_state["sample_text"] = (
                "Honestly I was SO nervous before the presentation but it went way better than "
                "I expected?? Like people actually laughed at my jokes lol. My boss even said she "
                "was impressed. Still can't believe it. Gonna celebrate with pizza tonight 🍕"
            )
        if st.button("Load AI Sample"):
            st.session_state["sample_text"] = (
                "The implications of artificial intelligence for modern business practices are both "
                "profound and far-reaching. Organizations that effectively leverage AI technologies "
                "stand to gain significant competitive advantages across multiple operational dimensions. "
                "From customer service optimization to supply chain management, the potential applications "
                "are virtually limitless. However, it is equally important to consider the ethical "
                "implications and ensure responsible deployment of these powerful technologies."
            )

    # Main content
    if mode == "📝 Text Input":
        default = st.session_state.get("sample_text", "")
        text_input = st.text_area(
            "Paste or type text to analyze:",
            value=default,
            height=200,
            placeholder="Enter any text — social media post, news article, or essay ...",
        )

        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            analyze_btn = st.button("🔍 Analyze", use_container_width=True)

        if analyze_btn and text_input.strip() and model_loaded:
            with st.spinner("Analyzing text ..."):
                result = predict_single(text_input, model, tokenizer, gpt2_model, gpt2_tokenizer)

            # Results layout
            st.markdown("---")

            col_gauge, col_info = st.columns([1.2, 1])

            with col_gauge:
                st.plotly_chart(gauge_chart(result["probability"]), use_container_width=True)

            with col_info:
                label_class = "label-ai" if result["probability"] >= 0.5 else "label-human"
                st.markdown(f"""
                <div class="result-card">
                    <div class="metric-label">Classification</div>
                    <div class="{label_class}">{result['label']}</div>
                    <br/>
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{abs(result['probability'] - 0.5) * 200:.1f}%</div>
                    <br/>
                    <div class="metric-label">Perplexity Score</div>
                    <div class="metric-value">{result['perplexity']:.1f}</div>
                    <br/>
                    <div class="metric-label">Burstiness Score</div>
                    <div class="metric-value">{result['burstiness']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            # Burstiness chart
            if result["sentence_lengths"]:
                st.plotly_chart(burstiness_chart(result["sentence_lengths"]), use_container_width=True)

            # Linguistic insight
            with st.expander("🔬 Linguistic Analysis"):
                ppl_status = "🟢 High (human-like)" if result["perplexity"] > 50 else "🔴 Low (AI-like)"
                burst_status = "🟢 High (human-like)" if result["burstiness"] > 3 else "🔴 Low (AI-like)"
                st.markdown(f"""
                | Feature | Value | Interpretation |
                |---|---|---|
                | **Perplexity** | {result['perplexity']:.1f} | {ppl_status} |
                | **Burstiness** | {result['burstiness']:.2f} | {burst_status} |
                | **Avg Sentence Length** | {np.mean(result['sentence_lengths']):.1f} words | — |
                | **Sentence Count** | {len(result['sentence_lengths'])} | — |
                """)

        elif analyze_btn and not text_input.strip():
            st.warning("Please enter some text to analyze.")

    else:
        # Batch CSV mode
        st.markdown("### 📊 Batch Analysis")
        uploaded = st.file_uploader("Upload a CSV with a `text` column:", type=["csv"])

        if uploaded and model_loaded:
            df = pd.read_csv(uploaded)
            if "text" not in df.columns:
                st.error("CSV must contain a `text` column.")
            else:
                if st.button("🚀 Analyze All"):
                    progress = st.progress(0)
                    results = []
                    for i, row in df.iterrows():
                        r = predict_single(
                            row["text"], model, tokenizer, gpt2_model, gpt2_tokenizer
                        )
                        results.append({
                            "text": row["text"][:100] + "..." if len(row["text"]) > 100 else row["text"],
                            "probability": f"{r['probability']:.2%}",
                            "label": r["label"],
                            "perplexity": f"{r['perplexity']:.1f}",
                            "burstiness": f"{r['burstiness']:.2f}",
                        })
                        progress.progress((i + 1) / len(df))

                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)

                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results", csv, "sentinel_results.csv", "text/csv"
                    )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#484f58; font-size:0.85rem;'>"
        "Synthetic Sentinel v1.0 • Group 9 • ARTI407 NLP Final Project"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
