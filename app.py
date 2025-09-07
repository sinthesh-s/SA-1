import streamlit as st
import joblib
import base64
import zipfile
import os
import re
import html
from typing import Optional

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="🎬 Movie Review Sentiment Analyzer", layout="centered")

# ----------------------------
# Background Image
# ----------------------------
def set_background(image_path: str):
    if not os.path.exists(image_path):
        return
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

# ⚠️ match your current filename
set_background("background_image (1).jpg")

# ----------------------------
# Clean Text (SAME as training)
# ----------------------------
def clean_text(s: Optional[str]) -> str:
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
# (Optional) simple negation patch
# ----------------------------
NEG_WORDS = r"(good|great|amazing|excellent|fun|interesting|recommend|love|like|worth|enjoy(ed)?)"
NEG_TRIGGERS = r"(not|no|never|n't|dont|cannot|can't|wont|won't|didnt|didn't|doesnt|doesn't|isnt|isn't|wasnt|wasn't|aren't|aint|ain't)"

negation_re = re.compile(rf"\b{NEG_TRIGGERS}\s+{NEG_WORDS}\b")

def negation_patch(review_cleaned: str) -> bool:
    """Return True if a simple negation-of-positive is detected."""
    return bool(negation_re.search(review_cleaned))

# ----------------------------
# Load Artifacts (zip + pkl files)
# ----------------------------
@st.cache_resource
def load_artifacts():
    # Unzip model once
    model_zip = "best_sentiment_model.zip"
    model_dir = "model_dir"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not any(name.endswith(".pkl") for name in os.listdir(model_dir)):
        with zipfile.ZipFile(model_zip, "r") as zf:
            zf.extractall(model_dir)

    # Find extracted model .pkl
    model_path = None
    for f in os.listdir(model_dir):
        if f.endswith(".pkl"):
            model_path = os.path.join(model_dir, f)
            break
    if model_path is None:
        raise FileNotFoundError("No .pkl model found inside best_sentiment_model.zip")

    # Vectorizer + encoder (match your current filenames)
    # Try exact names, then fall back to any matching file.
    vec_path = "tfidf_vectorizer (1).pkl"
    if not os.path.exists(vec_path):
        candidates = [f for f in os.listdir(".") if f.startswith("tfidf_vectorizer") and f.endswith(".pkl")]
        if not candidates:
            raise FileNotFoundError("TF-IDF vectorizer .pkl not found.")
        vec_path = candidates[0]

    enc_path = "label_encoder.pkl"
    if not os.path.exists(enc_path):
        raise FileNotFoundError("label_encoder.pkl not found.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    label_encoder = joblib.load(enc_path)

    return model, vectorizer, label_encoder, model_path, vec_path, enc_path

try:
    model, vectorizer, label_encoder, model_path, vec_path, enc_path = load_artifacts()
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# ----------------------------
# UI
# ----------------------------
st.markdown('<div class="title-text">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)

with st.expander("ℹ️ Loaded artifacts (debug)"):
    st.write("Model file:", model_path)
    st.write("Vectorizer file:", vec_path)
    st.write("Label classes:", list(label_encoder.classes_))
    try:
        # FeatureUnion or TfidfVectorizer both have this
        st.write("Vectorizer features:", getattr(vectorizer, "get_feature_names_out", lambda: ["N/A"])()[:5], "…")
    except Exception:
        st.write("Vectorizer features: (unavailable)")

apply_patch = st.checkbox("Apply quick negation fix (rule-based)", value=True)

review = st.text_area("Write your movie review here...", height=150, placeholder="e.g., not a good movie")

if st.button("Analyze Sentiment"):
    if review.strip():
        cleaned = clean_text(review)

        # Optional negation patch (only overrides clearly negated positives)
        if apply_patch and negation_patch(cleaned):
            pred_label = "negative"
        else:
            X = vectorizer.transform([cleaned])
            pred_idx = model.predict(X)[0]
            pred_label = label_encoder.inverse_transform([pred_idx])[0]

        st.markdown(
            f'<div class="result">🧠 <b>Predicted Sentiment:</b> {pred_label.capitalize()}</div>',
            unsafe_allow_html=True
        )

        with st.expander("🔎 Debug (what the model saw)"):
            st.write("Cleaned text:", cleaned)
            try:
                st.write("Feature vector shape:", vectorizer.transform([cleaned]).shape)
            except Exception as e:
                st.write("Vectorization error:", e)

    else:
        st.warning("⚠ Please enter a valid review.")
