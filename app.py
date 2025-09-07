import streamlit as st
import joblib
import base64
import zipfile
import os

# --- Page Configuration ---
st.set_page_config(page_title="🎬 Movie Review Sentiment Analyzer", layout="centered")

# --- Set Background Image ---
def set_background(image_path):
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

# --- Extract & Load Model ---
@st.cache_resource
def load_artifacts():
    model_zip_path = "best_sentiment_model.zip"
    model_dir = "model_dir"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Extract only once
    if not any(fname.endswith(".pkl") for fname in os.listdir(model_dir)):
        with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
            zip_ref.extractall(model_dir)

    # Load artifacts
    model_path = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pkl")][0]
    model = joblib.load(model_path)
    vectorizer = joblib.load("tfidf_vectorizer (1).pkl")  # adjust to your actual file name
    label_encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_artifacts()

# --- Negation-aware Prediction ---
def predict_sentiment(text, model, vectorizer, label_encoder):
    # Quick negation handling
    negation_words = ["not", "n't", "never", "no", "don't", "dont", "cannot", "can't"]
    if any(word in text.lower() for word in negation_words) and "good" in text.lower():
        return "Negative"

    # Normal ML prediction
    X = vectorizer.transform([text])
    pred = model.predict(X)
    return label_encoder.inverse_transform(pred)[0].capitalize()

# --- Title ---
st.markdown('<div class="title-text">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)

# --- Text Input ---
review = st.text_area("Write your movie review here...", height=150)

# --- Predict Button ---
if st.button("Analyze Sentiment"):
    if review.strip():
        prediction = predict_sentiment(review, model, vectorizer, label_encoder)
        st.markdown(f'<div class="result">🧠 *Predicted Sentiment:* {prediction}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠ Please enter a valid review.")
