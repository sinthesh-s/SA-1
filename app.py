# app.py — Streamlit sentiment classifier
import streamlit as st
import joblib
import re
import html
import zipfile
import os

# -------------------------
# Filenames
# -------------------------
MODEL_ZIP = "model.zip"
MODEL_PKL = "model.pkl"   # expected inside model.zip
VECT_PKL = "vectorizer.pkl"
ENC_PKL = "encoder.pkl"

# -------------------------
# Extract model if needed
# -------------------------
if not os.path.exists(MODEL_PKL):
    if os.path.exists(MODEL_ZIP):
        with zipfile.ZipFile(MODEL_ZIP, "r") as zf:
            zf.extractall(".")
    else:
        st.error("❌ model.pkl not found (and no model.zip available). Upload one of them.")
        st.stop()

# -------------------------
# Load artifacts
# -------------------------
try:
    model = joblib.load(MODEL_PKL)
    vectorizer = joblib.load(VECT_PKL)
    encoder = joblib.load(ENC_PKL)
except Exception as e:
    st.error(f"❌ Failed to load artifacts: {e}")
    st.stop()

# -------------------------
# Text cleaning (same as training)
# -------------------------
def clean_text(s: str) -> str:
    if s is None:
        return ""
    text = html.unescape(str(s))
    text = re.sub(r"<[^>]+>", " ", text)            # remove HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)            # remove emails
    text = re.sub(r"[^\w\s'.,!?-]+", " ", text)     # junk chars
    text = text.lower().strip()

    # Negation handling: mark words after not/never/no
    neg_patterns = re.compile(r"\b(?:not|never|no|n't)\s+(\w+)")
    text = neg_patterns.sub(lambda m: "NOT_" + m.group(1), text)

    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------
# Prediction function
# -------------------------
def predict_sentiment(text, debug=False):
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)
    labels = encoder.inverse_transform(preds)

    if debug:
        debug_info = []
        for raw, cln, pred, lab in zip(texts, cleaned, preds, labels):
            debug_info.append({
                "raw_input": raw,
                "cleaned_text": cln,
                "numeric_class": int(pred),
                "predicted_label": lab
            })
        return labels, debug_info
    return labels, None

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Sentiment Classifier", page_icon="🤖")

st.title("🎭 Sentiment Analysis App")
st.write("Enter a review/comment below and get the predicted sentiment.")

# Sidebar toggle
debug_mode = st.sidebar.checkbox("🔎 Enable Debug Mode", value=False)

# Single input
user_input = st.text_area("Enter your review here:", "")

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a review.")
    else:
        labels, dbg = predict_sentiment(user_input, debug=debug_mode)
        st.success(f"**Predicted Sentiment:** {labels[0]}")
        if debug_mode and dbg:
            st.json(dbg[0])

st.markdown("---")

# Batch input
st.subheader("🔍 Predict Multiple Reviews")
batch_text = st.text_area("Enter multiple reviews (one per line):", "")

if st.button("Predict Batch"):
    if not batch_text.strip():
        st.warning("⚠️ Please enter at least one review.")
    else:
        reviews = [r.strip() for r in batch_text.splitlines() if r.strip()]
        labels, dbg = predict_sentiment(reviews, debug=debug_mode)
        st.write("### Results:")
        for i, (r, l) in enumerate(zip(reviews, labels)):
            st.write(f"- **{r}** → `{l}`")
            if debug_mode and dbg:
                st.caption(f"Cleaned: {dbg[i]['cleaned_text']} | ClassID: {dbg[i]['numeric_class']}")
