import streamlit as st
import joblib
import base64
import os

# --- Page Configuration ---
st.set_page_config(page_title="🎬 Movie Review Sentiment Analyzer", layout="centered")

# --- Set Background Image ---
def set_background(image_path):
    if os.path.exists(image_path):
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
            background-color: rgba(255,255,255,0.8);
            padding: 0.8rem;
            margin-top: 1rem;
            border-radius: 8px;
            font-size: 1.2rem;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

set_background("background_image (1).jpg")

# --- Load Model, Vectorizer, Label Encoder ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer (1).pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_artifacts()

# --- Title ---
st.markdown('<div class="title-text">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)

# --- Text Input ---
review = st.text_area("Write your movie review here...", height=150)

# --- Predict Button ---
if st.button("Analyze Sentiment"):
    if review.strip():
        transformed_review = vectorizer.transform([review])
        pred_encoded = model.predict(transformed_review)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0].capitalize()
        st.markdown(f'<div class="result">🧠 *Predicted Sentiment:* {prediction}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠ Please enter a valid review.")
