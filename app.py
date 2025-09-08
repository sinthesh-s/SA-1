import streamlit as st
import joblib
import re
import html
import zipfile
import os

# -------------------------
# Extract zipped model if needed
# -------------------------
MODEL_ZIP = "model.zip"
MODEL_FILE = "model.pkl"

if not os.path.exists(MODEL_FILE) and os.path.exists(MODEL_ZIP):
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(".")  # extracts model.pkl

# -------------------------
# Load model + vectorizer + encoder
# -------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
encoder = joblib.load("encoder.pkl")

# -------------------------
# Text Cleaning (same as training)
# -------------------------
def clean_text(s):
    if s is None:
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
    text = text.lower().strip()

    # --- Negation handling ---
    neg_patterns = re.compile(r"\b(?:not|never|no|n't)\s+(\w+)")
    text = neg_patterns.sub(lambda m: "NOT_" + m.group(1), text)

    # --- Contrast emphasis ---
    contrast_markers = ["but", "however", "though", "although", "yet"]
    for marker in contrast_markers:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) == 2:
                text = parts[0] + f" {marker} " + parts[1] + " " + parts[1]

    text = re.sub(r"\s+", " ", text)
    return text

# -------------------------
# Prediction function
# -------------------------
def predict_sentiment(texts, debug=False):
    if isinstance(texts, str):
        texts = [texts]

    cleaned = [clean_text(t) for t in texts]
    feats = vectorizer.transform(cleaned)
    preds = model.predict(feats)
    labels = encoder.inverse_transform(preds)

    if debug:
        debug_info = []
        for raw, clean, pred, label in zip(texts, cleaned, preds, labels):
            debug_info.append({
                "raw_input": raw,
                "cleaned_text": clean,
                "predicted_label": label,
                "numeric_class": int(pred)
            })
        return labels, debug_info

    return labels

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Sentiment Classifier", page_icon="🤖")
st.title("🎭 Sentiment Analysis App")
st.write("Enter a review/comment below and get the predicted sentiment.")

# Debug toggle
debug_mode = st.sidebar.checkbox("🔎 Enable Debug Mode", value=False)

# Single input
user_input = st.text_area("Enter your review here:", "")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        if debug_mode:
            label, dbg = predict_sentiment(user_input, debug=True)
            st.success(f"**Predicted Sentiment:** {label[0]}")
            st.write("🔎 Debug Info:", dbg[0])
        else:
            label = predict_sentiment(user_input)[0]
            st.success(f"**Predicted Sentiment:** {label}")

# Batch input
st.subheader("🔍 Try Multiple Reviews")
batch_text = st.text_area("Enter multiple reviews (one per line):", "")

if st.button("Predict Batch"):
    if batch_text.strip() == "":
        st.warning("⚠️ Please enter at least one review.")
    else:
        reviews = batch_text.strip().split("\n")
        if debug_mode:
            labels, dbg = predict_sentiment(reviews, debug=True)
            for d in dbg:
                st.write(f"- **{d['raw_input']}** → `{d['predicted_label']}`")
                st.caption(f"Cleaned: {d['cleaned_text']} | ClassID: {d['numeric_class']}")
        else:
            results = predict_sentiment(reviews)
            st.write("### Results:")
            for r, l in zip(reviews, results):
                st.write(f"- **{r.strip()}** → `{l}`")
