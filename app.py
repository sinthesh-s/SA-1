import streamlit as st
import joblib
import base64
import re
import html
import os

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="🎬 Movie Review Sentiment Analyzer", layout="centered")

# ----------------------------
# Background Image
# ----------------------------
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

set_background("background_image (1).jpg")  # adjust name if needed

# ----------------------------
# Clean Text (SAME as training)
# ----------------------------
def clean_text(s):
    if not s:
        return ""
    text = html.unescape(str(s))
    text = re.sub(r"<[^>]+>", " ", text)            # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"\S+@\S+", " ", text)            # emails
    emoji_pattern = re.compile("["                  
        "\U0001F600-\U0001F64F"                     
        "\U0001F300-\U0001F5FF"                     
        "\U0001F680-\U0001F6FF"                     
        "\U0001F1E0-\U0001F1FF"                     
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(" ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)    # compress repeats
    text = re.sub(r"[^\w\s'.,!?-]+", " ", text)     # junk chars
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()

# ----------------------------
# Load Artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_artifacts()

# ----------------------------
# App Title
# ----------------------------
st.markdown('<div class="title-text">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)

# ----------------------------
# User Input
# ----------------------------
review = st.text_area("Write your movie review here...", height=150)

# ----------------------------
# Predict
# ----------------------------
if st.button("Analyze Sentiment"):
    if review.strip():
        cleaned_review = clean_text(review)
        features = vectorizer.transform([cleaned_review])
        pred_encoded = model.predict(features)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0].capitalize()

        st.markdown(f'<div class="result">🧠 *Predicted Sentiment:* {prediction}</div>', unsafe_allow_html=True)

        # Debug info (optional, can remove later)
        st.write("👉 Cleaned text:", cleaned_review)
        st.write("👉 Label classes:", list(label_encoder.classes_))

    else:
        st.warning("⚠ Please enter a valid review.")
