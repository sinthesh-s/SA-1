import streamlit as st
import joblib
import zipfile
import os
import base64

st.set_page_config(page_title="🎬 Movie Review Sentiment Analyzer", layout="centered")

# --- Background Setup ---
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
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("background_image (1).jpg")

# --- Extract & Load Model ---
model_dir = "model_dir"
model_zip = "best_sentiment_model.zip"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# unzip only once
if not any(fname.endswith(".pkl") for fname in os.listdir(model_dir)):
    with zipfile.ZipFile(model_zip, "r") as zip_ref:
        zip_ref.extractall(model_dir)

# find model.pkl inside extracted dir
model_path = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pkl")][0]

# Load artifacts
model = joblib.load(model_path)
vectorizer = joblib.load("tfidf_vectorizer (1).pkl")   # match your filename
label_encoder = joblib.load("label_encoder.pkl")

# --- UI ---
st.markdown('<h2 style="text-align:center;">🎬 Movie Review Sentiment Analyzer</h2>', unsafe_allow_html=True)

review = st.text_area("Write your movie review here...", height=150)

if st.button("Analyze Sentiment"):
    if review.strip():
        transformed_review = vectorizer.transform([review])
        pred_encoded = model.predict(transformed_review)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0].capitalize()
        st.success(f"🧠 Predicted Sentiment: {prediction}")
    else:
        st.warning("⚠ Please enter a valid review.")
