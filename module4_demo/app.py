"""
app.py - Synthetic Sentinel Streamlit Dashboard
===============================================
Phase 4: Interactive demo for AI-generated text detection.

Run:
    streamlit run app.py

Features:
  - Paste text or upload CSV for batch analysis
  - Synthetic Probability Score with gauge visualization
  - Binary classification with gray-zone interpretation
  - Burstiness sentence-level analysis chart
  - Visual explanation of how the prediction was made
"""

import math
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, RobertaTokenizerFast

from module2_modeling.train_model import HybridDetector

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_detector():
    """Load the trained hybrid detector."""
    model_path = os.path.join(MODEL_DIR, "best_roberta.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing trained weights at {model_path}. Run `python train_model.py` first."
        )

    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    model = HybridDetector()
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


def compute_burstiness(text: str) -> tuple[float, list[int]]:
    """Return burstiness plus sentence lengths."""
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
    """Run features and classification for a single text."""
    burstiness, sent_lengths = compute_burstiness(text)
    perplexity = compute_perplexity(text, gpt2_model, gpt2_tokenizer)

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    perp_t = torch.tensor([perplexity], dtype=torch.float).to(DEVICE)
    burst_t = torch.tensor([burstiness], dtype=torch.float).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, perp_t, burst_t)
        probs = torch.softmax(logits, dim=1)[0]

    prob_ai = float(probs[1])
    confidence = abs(prob_ai - 0.5) * 200
    label = "AI-Generated" if prob_ai >= 0.5 else "Human-Written"

    if 0.60 <= prob_ai <= 0.85 and confidence <= 45:
        interpretation = "Possible Human-Edited AI"
    elif prob_ai > 0.85:
        interpretation = "Strong AI Signal"
    elif prob_ai < 0.40:
        interpretation = "Likely Human-Written"
    else:
        interpretation = "Uncertain / Mixed Signal"

    return {
        "probability": prob_ai,
        "confidence": confidence,
        "label": label,
        "interpretation": interpretation,
        "perplexity": perplexity,
        "burstiness": burstiness,
        "sentence_lengths": sent_lengths,
    }


def gauge_chart(probability: float):
    """Create a gauge chart for synthetic probability."""
    color = "#3fb950" if probability < 0.3 else "#d29922" if probability < 0.7 else "#f85149"

    fig = go.Figure(
        go.Indicator(
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
        )
    )
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


def signal_profile_chart(result: dict):
    """Visual comparison of the three main decision signals."""
    confidence_norm = min(result["confidence"] / 100.0, 1.0)
    perplexity_norm = min(result["perplexity"] / 80.0, 1.0)
    burstiness_norm = min(result["burstiness"] / 6.0, 1.0)

    df = pd.DataFrame(
        {
            "Signal": [
                "Synthetic Probability",
                "Confidence",
                "Perplexity",
                "Burstiness",
            ],
            "Normalized Score": [
                result["probability"],
                confidence_norm,
                perplexity_norm,
                burstiness_norm,
            ],
            "Display": [
                f"{result['probability'] * 100:.1f}%",
                f"{result['confidence']:.1f}%",
                f"{result['perplexity']:.1f}",
                f"{result['burstiness']:.2f}",
            ],
        }
    )

    fig = px.bar(
        df,
        x="Signal",
        y="Normalized Score",
        text="Display",
        color="Signal",
        color_discrete_sequence=["#f85149", "#d29922", "#58a6ff", "#3fb950"],
        title="Decision Signal Profile",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#c9d1d9",
        showlegend=False,
        yaxis={"range": [0, 1.1], "gridcolor": "#21262d", "title": "Relative Scale"},
        xaxis={"gridcolor": "#21262d"},
        margin={"t": 60, "b": 50, "l": 40, "r": 20},
        height=340,
    )
    return fig


def sentence_rhythm_chart(sentence_lengths: list[int]):
    """Line chart showing sentence-length rhythm across the text."""
    if not sentence_lengths:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(sentence_lengths) + 1)),
            y=sentence_lengths,
            mode="lines+markers",
            line={"color": "#58a6ff", "width": 3},
            marker={"size": 8, "color": "#f0f6fc"},
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.12)",
        )
    )
    fig.update_layout(
        title="Sentence Rhythm Trace",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#c9d1d9",
        xaxis={"title": "Sentence", "gridcolor": "#21262d"},
        yaxis={"title": "Words", "gridcolor": "#21262d"},
        height=320,
        margin={"t": 60, "b": 40, "l": 40, "r": 20},
        showlegend=False,
    )
    return fig


def confidence_position_chart(result: dict):
    """Show where the current sample sits in the gray-zone interpretation scale."""
    probability_pct = result["probability"] * 100
    fig = go.Figure(
        go.Indicator(
            mode="number+gauge",
            value=probability_pct,
            number={"suffix": "%", "font": {"size": 34, "color": "white"}},
            title={"text": "Gray-Zone Position", "font": {"size": 18, "color": "#c9d1d9"}},
            gauge={
                "shape": "bullet",
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": "#d29922" if result["interpretation"] == "Possible Human-Edited AI" else "#58a6ff"},
                "bgcolor": "#161b22",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, 40], "color": "rgba(63,185,80,0.18)"},
                    {"range": [40, 60], "color": "rgba(210,153,34,0.22)"},
                    {"range": [60, 85], "color": "rgba(248,81,73,0.15)"},
                    {"range": [85, 100], "color": "rgba(248,81,73,0.3)"},
                ],
                "threshold": {
                    "line": {"color": "#f0f6fc", "width": 3},
                    "thickness": 0.9,
                    "value": probability_pct,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font_color="#c9d1d9",
        height=200,
        margin={"t": 70, "b": 20, "l": 30, "r": 30},
    )
    return fig


def describe_perplexity(perplexity: float) -> str:
    """Explain perplexity in plain language."""
    if perplexity < 40:
        return "The text is fairly predictable, which often matches polished AI-style phrasing."
    if perplexity < 65:
        return "The text has mixed predictability, so it shows both structured and natural signals."
    return "The text is less predictable overall, which is more common in natural human writing."


def describe_burstiness(burstiness: float) -> str:
    """Explain burstiness in plain language."""
    if burstiness < 2.5:
        return "Sentence lengths are fairly uniform, which is a common AI signal."
    if burstiness < 4.0:
        return "Sentence rhythm varies somewhat, which can happen in edited or mixed writing."
    return "Sentence rhythm varies a lot, which usually looks more human."


def describe_probability(probability: float, interpretation: str) -> str:
    """Explain the final decision in plain language."""
    if interpretation == "Possible Human-Edited AI":
        return "The score leans AI, but not decisively. That usually means AI-like structure with more human-sounding edits."
    if probability >= 0.85:
        return "Most signals line up strongly on the AI side, so the model is fairly certain."
    if probability <= 0.40:
        return "The signals lean more human overall, so the model is not seeing a strong synthetic pattern."
    return "The signals are mixed, so the model sees evidence on both sides."


def concise_reason(result: dict) -> str:
    """One-line reason summary for the result card."""
    if result["interpretation"] == "Possible Human-Edited AI":
        return "AI-like structure with more human-style edits."
    if result["interpretation"] == "Strong AI Signal":
        return "Highly predictable wording with a strong synthetic pattern."
    if result["interpretation"] == "Likely Human-Written":
        return "More varied and less machine-like overall."
    return "Mixed evidence from wording, predictability, and sentence rhythm."


def interpretation_style(result: dict) -> tuple[str, str]:
    """Return label class and tag class for UI styling."""
    if result["interpretation"] == "Possible Human-Edited AI":
        return "label-edited", "tag-gray"
    if result["interpretation"] == "Strong AI Signal":
        return "label-ai", "tag-ai"
    if result["interpretation"] == "Likely Human-Written":
        return "label-human", "tag-human"
    return "metric-value", "tag-mixed"


def render_styles():
    """Inject custom CSS."""
    st.markdown(
        """
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
            .workflow-card {
                background: linear-gradient(180deg, #151b23 0%, #1b2432 100%);
                border: 1px solid #30363d;
                border-radius: 14px;
                padding: 1.2rem;
                min-height: 220px;
            }
            .workflow-step {
                color: #58a6ff;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.45rem;
            }
            .workflow-title {
                color: #f0f6fc;
                font-size: 1.15rem;
                font-weight: 700;
                margin-bottom: 0.6rem;
            }
            .workflow-copy {
                color: #c9d1d9;
                font-size: 0.95rem;
                line-height: 1.55;
            }
            .signal-chip {
                display: inline-block;
                margin-top: 0.8rem;
                padding: 0.35rem 0.6rem;
                border-radius: 999px;
                border: 1px solid #30363d;
                background: rgba(88,166,255,0.12);
                color: #c9d1d9;
                font-size: 0.82rem;
            }
            .explain-shell {
                background: linear-gradient(135deg, #111821 0%, #1a2230 100%);
                border: 1px solid #30363d;
                border-radius: 16px;
                padding: 1.3rem;
                margin-top: 1rem;
            }
            .explain-head {
                color: #f0f6fc;
                font-size: 1.12rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }
            .explain-sub {
                color: #8b949e;
                font-size: 0.94rem;
                margin-bottom: 0.2rem;
            }
            .signal-box {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 1rem;
                height: 100%;
            }
            .signal-title {
                color: #8b949e;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                font-size: 0.78rem;
                margin-bottom: 0.45rem;
            }
            .signal-value {
                color: #f0f6fc;
                font-size: 1.45rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }
            .signal-text {
                color: #c9d1d9;
                font-size: 0.92rem;
                line-height: 1.45;
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
            .label-edited {
                color: #ffb86b;
                font-size: 1.5rem;
                font-weight: 700;
            }
            .tag-badge {
                display: inline-block;
                padding: 0.45rem 0.8rem;
                border-radius: 999px;
                font-size: 0.88rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                border: 1px solid #30363d;
                margin-top: 0.25rem;
            }
            .tag-gray {
                background: rgba(255,184,107,0.12);
                color: #ffb86b;
                border-color: rgba(255,184,107,0.35);
            }
            .tag-ai {
                background: rgba(248,81,73,0.12);
                color: #f85149;
                border-color: rgba(248,81,73,0.35);
            }
            .tag-human {
                background: rgba(63,185,80,0.12);
                color: #3fb950;
                border-color: rgba(63,185,80,0.35);
            }
            .tag-mixed {
                background: rgba(210,153,34,0.14);
                color: #d29922;
                border-color: rgba(210,153,34,0.35);
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
        """,
        unsafe_allow_html=True,
    )


def render_how_it_works(current_result):
    """Render the visual explanation section."""
    st.markdown("---")
    st.markdown("### How It Works")

    flow_col1, flow_col2, flow_col3 = st.columns(3)

    with flow_col1:
        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-step">Step 1</div>
                <div class="workflow-title">Read the wording</div>
                <div class="workflow-copy">
                    RoBERTa reads the text itself: wording, tone, transitions, and how ideas are connected.
                    This tells the system whether the passage sounds naturally written or overly polished and formulaic.
                </div>
                <div class="signal-chip">Context signal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with flow_col2:
        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-step">Step 2</div>
                <div class="workflow-title">Measure writing clues</div>
                <div class="workflow-copy">
                    The app adds two lightweight clues:
                    <br/><br/>
                    <strong>Perplexity</strong> asks how predictable the text is.
                    <br/>
                    <strong>Burstiness</strong> asks whether sentence lengths vary like natural human rhythm.
                </div>
                <div class="signal-chip">Perplexity + Burstiness</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with flow_col3:
        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-step">Step 3</div>
                <div class="workflow-title">Combine the evidence</div>
                <div class="workflow-copy">
                    A hybrid classifier combines the language signal with the two writing clues, then outputs
                    a synthetic probability, a main label, and a gray-zone interpretation such as
                    <strong>Possible Human-Edited AI</strong>.
                </div>
                <div class="signal-chip">Final probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if current_result is not None:
        st.markdown(
            """
            <div class="explain-shell">
                <div class="explain-head">How This Example Was Judged</div>
                <div class="explain-sub">
                    These are the same signals the model used for the text you just analyzed.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        explain_col1, explain_col2, explain_col3 = st.columns(3)

        with explain_col1:
            st.markdown(
                f"""
                <div class="signal-box">
                    <div class="signal-title">Perplexity</div>
                    <div class="signal-value">{current_result['perplexity']:.1f}</div>
                    <div class="signal-text">{describe_perplexity(current_result['perplexity'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with explain_col2:
            st.markdown(
                f"""
                <div class="signal-box">
                    <div class="signal-title">Burstiness</div>
                    <div class="signal-value">{current_result['burstiness']:.2f}</div>
                    <div class="signal-text">{describe_burstiness(current_result['burstiness'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with explain_col3:
            st.markdown(
                f"""
                <div class="signal-box">
                    <div class="signal-title">Final Decision</div>
                    <div class="signal-value">{current_result['probability'] * 100:.1f}%</div>
                    <div class="signal-text">{describe_probability(current_result['probability'], current_result['interpretation'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info(
            f"Result summary: {current_result['label']} with gray-zone tag "
            f"'{current_result['interpretation']}'."
        )

        chart_col1, chart_col2 = st.columns([1.2, 1])
        with chart_col1:
            st.plotly_chart(signal_profile_chart(current_result), use_container_width=True)
        with chart_col2:
            st.plotly_chart(confidence_position_chart(current_result), use_container_width=True)

        rhythm_fig = sentence_rhythm_chart(current_result["sentence_lengths"])
        if rhythm_fig is not None:
            st.plotly_chart(rhythm_fig, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="explain-shell">
                <div class="explain-head">Analyze A Sample To See The Reasoning</div>
                <div class="explain-sub">
                    After you run a sample, this section will explain exactly how perplexity, burstiness,
                    and the final probability interacted for that passage.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(
        page_title="Synthetic Sentinel",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_styles()

    st.markdown(
        """
        <div class="main-header">
            <h1>🛡️ Synthetic Sentinel</h1>
            <p>AI-Generated Misinformation Detection • RoBERTa + Stylometry Hybrid Model</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, tokenizer = load_detector()
        gpt2_model, gpt2_tokenizer = load_gpt2()
        model_loaded = True
    except Exception as e:
        st.warning(f"[!] Model not loaded: {e}")
        model_loaded = False

    with st.sidebar:
        st.markdown("### Settings")
        mode = st.radio("Input Mode", ["Text Input", "Batch CSV"])

        st.markdown("---")
        st.markdown("### Sample Texts")
        if st.button("Load Human Sample"):
            st.session_state["sample_text"] = (
                "Honestly I was SO nervous before the presentation but it went way better than "
                "I expected?? Like people actually laughed at my jokes lol. My boss even said she "
                "was impressed. Still can't believe it. Gonna celebrate with pizza tonight."
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
        if st.button("Load Human-Edited AI Sample"):
            st.session_state["sample_text"] = (
                "Honestly, the implications of artificial intelligence for modern business practices are "
                "pretty huge, but people don't always say it in a way that feels grounded. A lot of teams "
                "could gain real advantages from using AI across daily operations, and that part is hard "
                "to ignore. Still, in real life, it usually feels less neat than these polished summaries "
                "make it sound. You also have to think about the ethics, the tradeoffs, and whether the "
                "deployment is actually responsible instead of just efficient."
            )

    current_result = None

    if mode == "Text Input":
        default = st.session_state.get("sample_text", "")
        text_input = st.text_area(
            "Paste or type text to analyze:",
            value=default,
            height=200,
            placeholder="Enter any text - social media post, news article, or essay ...",
        )

        col1, _, _ = st.columns([1, 1, 3])
        with col1:
            analyze_btn = st.button("Analyze", use_container_width=True)

        if analyze_btn and text_input.strip() and model_loaded:
            with st.spinner("Analyzing text ..."):
                current_result = predict_single(
                    text_input, model, tokenizer, gpt2_model, gpt2_tokenizer
                )

            st.markdown("---")
            col_gauge, col_info = st.columns([1.2, 1])

            with col_gauge:
                st.plotly_chart(gauge_chart(current_result["probability"]), use_container_width=True)

            with col_info:
                label_class = "label-ai" if current_result["probability"] >= 0.5 else "label-human"
                interp_class, tag_class = interpretation_style(current_result)
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="metric-label">Why This Label?</div>
                        <div class="metric-value">{concise_reason(current_result)}</div>
                        <br/>
                        <div class="metric-label">Classification</div>
                        <div class="{label_class}">{current_result['label']}</div>
                        <br/>
                        <div class="metric-label">Interpretation</div>
                        <div class="{interp_class}">{current_result['interpretation']}</div>
                        <div class="tag-badge {tag_class}">{current_result['interpretation']}</div>
                        <br/>
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{current_result['confidence']:.1f}%</div>
                        <br/>
                        <div class="metric-label">Perplexity Score</div>
                        <div class="metric-value">{current_result['perplexity']:.1f}</div>
                        <br/>
                        <div class="metric-label">Burstiness Score</div>
                        <div class="metric-value">{current_result['burstiness']:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if current_result["sentence_lengths"]:
                st.plotly_chart(
                    burstiness_chart(current_result["sentence_lengths"]),
                    use_container_width=True,
                )

            with st.expander("Linguistic Analysis"):
                ppl_status = (
                    "High (human-like)" if current_result["perplexity"] > 50 else "Low (AI-like)"
                )
                burst_status = (
                    "High (human-like)" if current_result["burstiness"] > 3 else "Low (AI-like)"
                )
                st.markdown(
                    f"""
                    | Feature | Value | Interpretation |
                    |---|---|---|
                    | **Perplexity** | {current_result['perplexity']:.1f} | {ppl_status} |
                    | **Burstiness** | {current_result['burstiness']:.2f} | {burst_status} |
                    | **Gray-Zone Tag** | {current_result['interpretation']} | Confidence-aware heuristic |
                    | **Avg Sentence Length** | {np.mean(current_result['sentence_lengths']):.1f} words | - |
                    | **Sentence Count** | {len(current_result['sentence_lengths'])} | - |
                    """
                )

        elif analyze_btn and not text_input.strip():
            st.warning("Please enter some text to analyze.")
        elif analyze_btn and not model_loaded:
            st.error("Model weights are missing. Train the model before running inference.")

    else:
        st.markdown("### Batch Analysis")
        uploaded = st.file_uploader("Upload a CSV with a `text` column:", type=["csv"])

        if uploaded and model_loaded:
            df = pd.read_csv(uploaded)
            if "text" not in df.columns:
                st.error("CSV must contain a `text` column.")
            else:
                if st.button("Analyze All"):
                    progress = st.progress(0)
                    results = []
                    for i, row in df.iterrows():
                        result = predict_single(
                            row["text"], model, tokenizer, gpt2_model, gpt2_tokenizer
                        )
                        results.append(
                            {
                                "text": row["text"][:100] + "..." if len(row["text"]) > 100 else row["text"],
                                "probability": f"{result['probability']:.2%}",
                                "label": result["label"],
                                "interpretation": result["interpretation"],
                                "confidence": f"{result['confidence']:.1f}%",
                                "perplexity": f"{result['perplexity']:.1f}",
                                "burstiness": f"{result['burstiness']:.2f}",
                            }
                        )
                        progress.progress((i + 1) / len(df))

                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    csv = results_df.to_csv(index=False)
                    st.download_button("Download Results", csv, "sentinel_results.csv", "text/csv")
        elif uploaded and not model_loaded:
            st.error("Model weights are missing. Train the model before running batch analysis.")

    render_how_it_works(current_result)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#484f58; font-size:0.85rem;'>"
        "Synthetic Sentinel v1.0 • Group 9 • ARTI407 NLP Final Project"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
