import streamlit as st
import joblib
import base64
import zipfile
import os
import glob

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

# ✅ Match background filename with or without (1)
bg_image = glob.glob("background_image*.jpg")
if bg_image:
    set_background(bg_image[0])

# --- Extract & Load Model ---
# ✅ Match any version of the zip file
zip_files = glob.glob("best_sentiment_model*.zip")
if not zip_files:
    st.error("❌ Model zip file not found in repository.")
else:
    model_zip_path = zip_files[0]
    model_dir = "model_dir"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Extract only if not already extracted
    if not any(fname.endswith(".pkl") for fname in os.listdir(model_dir)):
        with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
            zip_ref.extractall(model_dir)

    # Find extracted model
    model_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    if not model_files:
        st.error("❌ No model file found inside the zip.")
    else:
        model = joblib.load(model_files[0])

        # ✅ Match vectorizer + label encoder (with or without suffix)
        vec_files = glob.glob("tfidf_vectorizer*.pkl")
        enc_files = glob.glob("label_encoder*.pkl")

        if not vec_files or not enc_files:
            st.error("❌ Vectorizer or Label Encoder file not found.")
        else:
            vectorizer = joblib.load(vec_files[0])
            label_encoder = joblib.load(enc_files[0])

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
