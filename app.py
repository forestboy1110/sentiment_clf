import os
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dotenv import load_dotenv

from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# =====================================================
# Config
# =====================================================

st.set_page_config(
    page_title="Persian Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in .env")
    st.stop()


# =====================================================
# LLM
# =====================================================

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0
)


class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    is_bad: bool
    reasoning: str


structured_llm = llm.with_structured_output(
    SentimentResult
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert Persian sentiment analysis model.

Classify Persian text into:

GOOD
NEUTRAL
BAD

Rules:

GOOD:
positive experience
satisfaction
recommendation

NEUTRAL:
informational
mixed feeling
no strong emotion

BAD:
complaint
negative experience
anger
frustration
recommend against purchase

Return structured output.

confidence must be between 0 and 1.
is_bad must be true only for BAD sentiment.
reasoning must be short and in English.
"""
        ),
        (
            "human",
            "{text}"
        )
    ]
)

chain = prompt | structured_llm


# =====================================================
# Session
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []


# =====================================================
# Analysis
# =====================================================

def analyze_text(text: str):

    start = time.time()

    result = chain.invoke(
        {
            "text": text
        }
    )

    duration = round(time.time() - start, 2)

    return result, duration


# =====================================================
# Charts
# =====================================================

def probability_chart(sentiment, confidence):

    labels = ["GOOD", "NEUTRAL", "BAD"]

    values = [0.1, 0.1, 0.1]

    idx = labels.index(sentiment)

    values[idx] = confidence

    remain = (1 - confidence) / 2

    for i in range(3):
        if i != idx:
            values[i] = remain

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            text=[f"{v:.1%}" for v in values],
            textposition="auto"
        )
    )

    fig.update_layout(
        title="Sentiment Confidence",
        height=350
    )

    return fig


# =====================================================
# Header
# =====================================================

st.title("📊 Persian Sentiment Analyzer")

st.markdown(
    """
Analyze Persian reviews using:

- LangChain
- OpenRouter
- GPT-4o-mini
"""
)

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.success("✅ OpenRouter Connected")

    st.caption("Model: GPT-4o-mini")

    st.divider()

    st.subheader("Statistics")

    total = len(st.session_state.history)

    bad_count = len(
        [
            x
            for x in st.session_state.history
            if x["sentiment"] == "BAD"
        ]
    )

    st.metric("Total Analyses", total)

    st.metric("Bad Reviews", bad_count)

    if total > 0:
        st.metric(
            "Bad Rate",
            f"{(bad_count / total) * 100:.1f}%"
        )

    st.divider()

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()


# =====================================================
# Examples
# =====================================================

st.subheader("Examples")

c1, c2, c3 = st.columns(3)

if c1.button("Positive Example"):
    st.session_state.example = (
        "این محصول فوق العاده بود و کیفیت بسیار خوبی داشت"
    )

if c2.button("Neutral Example"):
    st.session_state.example = (
        "محصول را دریافت کردم و طبق مشخصات بود"
    )

if c3.button("Negative Example"):
    st.session_state.example = (
        "بدترین خرید عمرم بود و کاملا ناراضی هستم"
    )


# =====================================================
# Input
# =====================================================

default_text = st.session_state.get(
    "example",
    ""
)

text = st.text_area(
    "Persian Text",
    value=default_text,
    height=150,
    placeholder="نظر خود را وارد کنید..."
)

analyze = st.button(
    "Analyze",
    type="primary",
    use_container_width=True
)

# =====================================================
# Run
# =====================================================

if analyze:

    if not text.strip():
        st.warning("Please enter text")
        st.stop()

    with st.spinner("Analyzing..."):

        result, duration = analyze_text(text)

    st.session_state.history.append(
        {
            "text": text,
            "sentiment": result.sentiment,
            "confidence": result.confidence
        }
    )

    st.divider()

    st.subheader("Result")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Sentiment",
            result.sentiment
        )

    with col2:
        st.metric(
            "Confidence",
            f"{result.confidence:.1%}"
        )

    with col3:
        st.metric(
            "Is Bad",
            "YES" if result.is_bad else "NO"
        )

    with col4:
        st.metric(
            "Response Time",
            f"{duration}s"
        )

    if result.sentiment == "GOOD":
        st.success("✅ Positive Sentiment")

    elif result.sentiment == "NEUTRAL":
        st.info("ℹ️ Neutral Sentiment")

    else:
        st.error("⚠️ Negative Sentiment")

    st.subheader("Reasoning")

    st.write(result.reasoning)

    st.subheader("Confidence Chart")

    st.plotly_chart(
        probability_chart(
            result.sentiment,
            result.confidence
        ),
        use_container_width=True
    )

# =====================================================
# History
# =====================================================

if st.session_state.history:

    st.divider()

    st.subheader("History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )