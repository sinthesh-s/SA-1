import streamlit as st
import joblib
import base64
import os
import zipfile
from textblob import TextBlob

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="🎬 Movie Review Sentiment Analyzer", layout="centered")

# ---------------------------
# Background Setup
# ---------------------------
def set_background(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            backdrop-filter: blur(2px);
        }}
        .title-text {{
            background-color: rgba(0,0,0,0.6);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.8rem;
            font-weight: bold;
        }}
        .result {{
            background-color: rgba(255,255,255,0.85);
            padding: 0.8rem;
            margin-top: 1rem;
            border-radius: 8px;
            font-size: 1.2rem;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except:
        st.warning("⚠ Background image not found.")

set_background("background_image (1).jpg")

# ---------------------------
# Load Model Artifacts
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = None
    vectorizer = None
    label_encoder = None

    # Extract the model zip if not already done
    if os.path.exists("best_sentiment_model.zip"):
        with zipfile.ZipFile("best_sentiment_model.zip", "r") as zip_ref:
            zip_ref.extractall("artifacts")

    try:
        model = joblib.load("artifacts/best_sentiment_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer (1).pkl")  # matches your file name
        label_encoder = joblib.load("label_encoder.pkl")
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_artifacts()

# ---------------------------
# Hybrid Rule-Based + ML
# ---------------------------
def rule_based_sentiment(text: str):
    text_lower = text.lower().strip()

    # Strong positive/negative bare phrases
    short_pos = ["good movie", "great movie", "nice movie", "well done", "liked it", "amazing movie"]
    short_neg = ["bad movie", "poor movie", "boring movie", "waste movie", "terrible movie"]

    if any(p == text_lower or p in text_lower for p in short_pos):
        return "positive"
    if any(p == text_lower or p in text_lower for p in short_neg):
        return "negative"

    # Negation patterns
    negation_phrases = ["not good", "not great", "not a good", "don't like", "didn't like", "don't think", "never liked"]
    if any(p in text_lower for p in negation_phrases):
        return "negative"

    # Neutral cues
    neutral_phrases = ["neither good nor bad", "average movie", "ok movie", "mediocre"]
    if any(p in text_lower for p in neutral_phrases):
        return "neutral"

    return None


def hybrid_predict(text: str):
    # 1. Rule-based override
    rb = rule_based_sentiment(text)
    if rb:
        return rb.capitalize()

    # 2. ML model prediction
    try:
        transformed = vectorizer.transform([text])
        pred_encoded = model.predict(transformed)[0]
        pred_ml = label_encoder.inverse_transform([pred_encoded])[0]
    except Exception:
        pred_ml = "neutral"

    # 3. TextBlob fallback for short/ambiguous cases
    if len(text.split()) <= 3 or len(text) < 15:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.2:
            return "Positive"
        elif polarity < -0.2:
            return "Negative"
        else:
            return "Neutral"

    return pred_ml.capitalize()

# ---------------------------
# Streamlit UI
# ---------------------------
st.markdown('<div class="title-text">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)

review = st.text_area("✍️ Write your movie review here...", height=150)

if st.button("🔍 Analyze Sentiment"):
    if review.strip():
        prediction = hybrid_predict(review)
        st.markdown(f'<div class="result">🧠 *Predicted Sentiment:* {prediction}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠ Please enter a valid review.")
