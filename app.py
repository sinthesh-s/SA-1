import streamlit as st
import joblib, re, html
from textblob import TextBlob

# -------------------------
# Load artifacts
# -------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_sentiment_model (1).pkl")
    vectorizer = joblib.load("tfidf_vectorizer (2).pkl")
    label_encoder = joblib.load("label_encoder (1).pkl")
    return model, vectorizer, label_encoder

# -------------------------
# Negation Handling (improved scope)
# -------------------------
NEGATION_WORDS = {
    "not","no","never","none","nobody","nothing","neither",
    "isn't","wasn't","aren't","weren't","don't","didn't",
    "doesn't","can't","couldn't","won't","wouldn't","shouldn't"
}

def handle_negation(text):
    tokens = text.split()
    new_tokens, negate = [], False
    for word in tokens:
        lw = word.lower()
        if lw in NEGATION_WORDS:
            negate = True
            new_tokens.append(lw)
        elif negate:
            new_tokens.append(word + "_NEG")
            if any(p in word for p in [".", ",", ";", "!", "?"]):
                negate = False
        else:
            new_tokens.append(word)
    return " ".join(new_tokens)

# -------------------------
# Cleaning
# -------------------------
def clean_text(s):
    if s is None: return ""
    text = html.unescape(str(s))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s'.,!?-]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return handle_negation(text)

# -------------------------
# TextBlob fallback
# -------------------------
def textblob_label(text, thresh=0.1):
    p = TextBlob(text).sentiment.polarity
    if p > thresh: return "positive"
    elif p < -thresh: return "negative"
    else: return "neutral"

# -------------------------
# Neutral / Mixed Adjustments
# -------------------------
NEUTRAL_KEYWORDS = {"average","okay","fine","decent","so-so","not bad","mediocre"}
POSITIVE_WORDS = {"good","great","fantastic","amazing","loved","best"}
NEGATIVE_WORDS = {"bad","awful","terrible","horrible","worst","waste"}

def adjust_prediction(review, label, probs, label_encoder):
    review_lc = review.lower()

    # Neutral keyword override
    if any(kw in review_lc for kw in NEUTRAL_KEYWORDS):
        if probs and max(probs.values()) < 0.75:
            return "neutral"

    # Mixed sentiment override
    if any(pw in review_lc for pw in POSITIVE_WORDS) and any(nw in review_lc for nw in NEGATIVE_WORDS):
        return "negative"   # negatives dominate in reviews

    return label

# -------------------------
# Prediction
# -------------------------
def predict_sentiment(text, model, vectorizer, label_encoder):
    cleaned = clean_text(text)
    feats = vectorizer.transform([cleaned])
    pred = model.predict(feats)[0]
    label = label_encoder.inverse_transform([pred])[0]

    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats)[0]
        return label, dict(zip(label_encoder.classes_, probs))
    return label, None

# -------------------------
# Hybrid Prediction (Model + Rules + Fallback)
# -------------------------
def hybrid_predict(review, model, vectorizer, label_encoder):
    label, probs = predict_sentiment(review, model, vectorizer, label_encoder)
    tb_label = textblob_label(review)

    # Apply rule-based adjustments
    label = adjust_prediction(review, label, probs, label_encoder)

    # Fallback: use TextBlob if confidence weak or strong disagreement
    if probs:
        model_conf = max(probs.values())
        if model_conf < 0.7 or (label != tb_label and model_conf < 0.8):
            return tb_label, probs, "TextBlob Override"
    return label, probs, "Model+Rules"

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(page_title="🎬 Sentiment Analyzer", layout="centered")
    st.title("🎬 Movie Review Sentiment Analyzer")

    review = st.text_area("Your review:", height=150)

    if st.button("Analyze Sentiment"):
        if review.strip() == "":
            st.warning("⚠️ Please enter a review first.")
        else:
            try:
                model, vectorizer, label_encoder = load_artifacts()
                label, probs, source = hybrid_predict(review, model, vectorizer, label_encoder)

                st.subheader(f"🧠 Predicted Sentiment: **{label.capitalize()}**")
                st.caption(f"Decision Source → {source}")

                if probs:
                    st.write("Confidence:")
                    st.json({k: f"{v:.2f}" for k,v in probs.items()})
            except Exception as e:
                st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
