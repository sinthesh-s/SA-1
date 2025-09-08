# app.py — fixed, robust Streamlit app
import streamlit as st
import joblib
import re
import html
import zipfile
import os
import base64
from typing import List, Tuple, Dict, Any

# -------------------------
# Configuration / filenames
# -------------------------
MODEL_ZIP = "model.zip"
MODEL_PKL = "model.pkl"        # expected inside model.zip
VECT_PKL = "vectorizer.pkl"
ENC_PKL = "encoder.pkl"
BG_IMAGE = "background_image (1).jpg"  # optional

# -------------------------
# Helper: set background (optional)
# -------------------------
def set_background(image_path: str) -> None:
    if not os.path.exists(image_path):
        return
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stApp .block-container {{
        background: rgba(255,255,255,0.85);
        padding: 1.2rem;
        border-radius: 8px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------------------
# Load artifacts (extract zip if needed) with caching
# -------------------------
@st.cache_resource
def load_artifacts() -> Tuple[Any, Any, Any]:
    # extract model.pkl if missing and .zip exists
    if not os.path.exists(MODEL_PKL):
        if os.path.exists(MODEL_ZIP):
            try:
                with zipfile.ZipFile(MODEL_ZIP, "r") as zf:
                    zf.extractall(".")
            except zipfile.BadZipFile as e:
                raise RuntimeError(f"model.zip is not a valid zip file: {e}")
            except Exception as e:
                raise RuntimeError(f"Failed to extract {MODEL_ZIP}: {e}")
        else:
            raise FileNotFoundError(f"Neither {MODEL_PKL} nor {MODEL_ZIP} were found in the app root.")

    # check presence of other artifacts
    missing = [name for name in (MODEL_PKL, VECT_PKL, ENC_PKL) if not os.path.exists(name)]
    if missing:
        raise FileNotFoundError(f"Missing artifact(s): {', '.join(missing)}. "
                                f"Make sure these files are uploaded or inside the zip.")

    # load with joblib
    try:
        model = joblib.load(MODEL_PKL)
    except Exception as e:
        raise RuntimeError(f"Failed to load {MODEL_PKL}: {e}")

    try:
        vectorizer = joblib.load(VECT_PKL)
    except Exception as e:
        raise RuntimeError(f"Failed to load {VECT_PKL}: {e}")

    try:
        encoder = joblib.load(ENC_PKL)
    except Exception as e:
        raise RuntimeError(f"Failed to load {ENC_PKL}: {e}")

    return model, vectorizer, encoder

# -------------------------
# Text cleaning (match training): negation handling only (no clause duplication)
# -------------------------
def clean_text(s: str) -> str:
    if s is None:
        return ""
    text = html.unescape(str(s))
    text = re.sub(r"<[^>]+>", " ", text)            # remove HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)            # remove emails
    # remove non-text junk but keep punctuation useful for splitting
    text = re.sub(r"[^\w\s'.,!?-]+", " ", text)
    text = text.lower().strip()

    # Negation handling: replace "not X" -> "NOT_x"
    # This tags single-word negation targets. It's simple and robust.
    neg_patterns = re.compile(r"\b(?:not|never|no|n't)\s+(\w+)")
    text = neg_patterns.sub(lambda m: "NOT_" + m.group(1), text)

    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------
# Prediction function with debug output
# -------------------------
def predict_sentiment(texts: List[str],
                      model,
                      vectorizer,
                      encoder,
                      debug: bool = False) -> Tuple[List[str], List[Dict[str, Any]]]:
    if isinstance(texts, str):
        texts = [texts]

    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)
    labels = list(encoder.inverse_transform(preds))

    debug_info = []
    if debug:
        # try to get probabilities if available
        prob_lists = None
        if hasattr(model, "predict_proba"):
            try:
                prob_arr = model.predict_proba(X)
                prob_lists = [dict(zip(encoder.classes_, map(float, row))) for row in prob_arr]
            except Exception:
                prob_lists = None

        for raw, cln, p, lab, probs in zip(texts, cleaned, preds, labels, (prob_lists or [None]*len(texts))):
            info = {
                "raw_input": raw,
                "cleaned_text": cln,
                "predicted_label": lab,
                "numeric_label": int(p)
            }
            if probs is not None:
                info["probs"] = probs
            debug_info.append(info)

    return labels, debug_info

# -------------------------
# Streamlit app layout
# -------------------------
def main():
    st.set_page_config(page_title="Sentiment Classifier", page_icon="🤖", layout="centered")
    # optional background
    set_background(BG_IMAGE)

    st.title("🎭 Sentiment Analysis App")
    st.write("Paste a review or multiple reviews (one per line). Toggle Debug Mode to inspect cleaning and numeric classes.")

    # Load artifacts with spinner and graceful error handling
    with st.spinner("Loading model and artifacts..."):
        try:
            model, vectorizer, encoder = load_artifacts()
        except Exception as e:
            st.error(f"❌ Could not load artifacts: {e}")
            st.stop()

    # Sidebar options
    debug_mode = st.sidebar.checkbox("🔎 Enable Debug Mode", value=False)
    show_cleaned = st.sidebar.checkbox("Show cleaned text in results", value=False)

    # Single review input
    st.subheader("Single review")
    review = st.text_area("Enter your review here:", height=150, key="single_input")

    if st.button("Predict Sentiment", key="predict_single"):
        if not review or not review.strip():
            st.warning("⚠️ Please enter a review.")
        else:
            labels, dbg = predict_sentiment([review], model, vectorizer, encoder, debug=debug_mode)
            st.success(f"**Predicted Sentiment:** {labels[0]}")
            if debug_mode and dbg:
                st.write("🔎 Debug info:")
                st.json(dbg[0])
            elif show_cleaned:
                st.write("Cleaned text:")
                st.code(clean_text(review))

    st.markdown("---")

    # Batch input
    st.subheader("Batch reviews (one per line)")
    batch = st.text_area("Enter multiple reviews (one per line):", height=200, key="batch_input")

    if st.button("Predict Batch", key="predict_batch"):
        if not batch or not batch.strip():
            st.warning("⚠️ Please enter at least one review.")
        else:
            reviews = [r.strip() for r in batch.splitlines() if r.strip()]
            labels, dbg = predict_sentiment(reviews, model, vectorizer, encoder, debug=debug_mode)
            st.write("### Results")
            for r, lbl, info in zip(reviews, labels, (dbg or [None]*len(reviews))):
                st.write(f"- **{r}** → `{lbl}`")
                if show_cleaned:
                    st.caption(f"Cleaned: `{clean_text(r)}`")
                if debug_mode and info:
                    st.caption(f"numeric: {info['numeric_label']}" + (f", probs: {info['probs']}" if "probs" in info else ""))

    st.markdown("---")
    st.info("Tip: If predictions are still poor, enable Debug Mode and collect several misclassified examples (cleaned_text + raw_input). Use those to augment training data and retrain.")

if __name__ == "__main__":
    main()
