import streamlit as st
import requests

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="AI Exam Anxiety Detector",
    page_icon="🧠",
    layout="centered"
)

# =========================
# Custom CSS
# =========================
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fb;
    }

    .main-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    }

    .title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c2c2c;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .result-low {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 12px;
        color: #2e7d32;
        font-weight: 600;
        text-align: center;
    }

    .result-medium {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 12px;
        color: #f57c00;
        font-weight: 600;
        text-align: center;
    }

    .result-high {
        background-color: #fdecea;
        padding: 1rem;
        border-radius: 12px;
        color: #c62828;
        font-weight: 600;
        text-align: center;
    }

    .tips {
        margin-top: 1rem;
        background-color: #f1f5f9;
        padding: 1rem;
        border-radius: 12px;
        color: #374151;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# UI Card Start
# =========================
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 AI Exam Anxiety Detector</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>A supportive tool to help students understand exam-related stress</div>",
    unsafe_allow_html=True
)

# =========================
# Text Input
# =========================
user_text = st.text_area(
    "✍️ Share how you’re feeling about exams",
    height=130,
    placeholder="Example: I feel extremely nervous and my mind goes blank before exams..."
)

# =========================
# Predict Button
# =========================
if st.button("🔍 Analyze Anxiety"):
    if user_text.strip() == "":
        st.warning("Please enter some text so we can analyze it.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8501/predict",
                json={"text": user_text}
            )

            result = response.json()
            anxiety_level = result["anxiety_level"]

            st.markdown("### 📊 Analysis Result")

            if anxiety_level == "Low Anxiety":
                st.markdown(
                    "<div class='result-low'>😌 Low Anxiety</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<div class='tips'>💡 You seem to be handling exam stress well. "
                    "Maintain your routine, revise consistently, and stay confident!</div>",
                    unsafe_allow_html=True
                )

            elif anxiety_level == "Moderate Anxiety":
                st.markdown(
                    "<div class='result-medium'>😟 Moderate Anxiety</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<div class='tips'>💡 Try short study sessions, deep breathing, "
                    "and realistic daily goals to stay balanced.</div>",
                    unsafe_allow_html=True
                )

            elif anxiety_level == "High Anxiety":
                st.markdown(
                    "<div class='result-high'>😰 High Anxiety</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<div class='tips'>💡 You’re not alone. Take slow breaths, "
                    "break tasks into small steps, and talk to someone you trust.</div>",
                    unsafe_allow_html=True
                )

        except Exception:
            st.error(" Backend not reachable. Please ensure FastAPI is running.")

# =========================
# UI Card End
# =========================
st.markdown("</div>", unsafe_allow_html=True)