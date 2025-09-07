import streamlit as st
import joblib
import re

# -------------------------
# Load artifacts
# -------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("best_sentiment_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer (1).pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return model, vectorizer, label_encoder
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        return None, None, None

# -------------------------
# Rule-based logic
# -------------------------
def rule_based_sentiment(text):
    text_lower = text.lower().strip()

    # Neutral expressions
    neutral_phrases = [
        "neither good nor bad", "not good not bad", "not bad not good",
        "so so", "meh", "okayish", "average", "fine", "decent",
        "nothing special", "mediocre", "forgettable"
    ]
    if any(p in text_lower for p in neutral_phrases):
        return "Neutral"

    # Strong negative
    strong_neg = [
        "terrible", "awful", "horrible", "worst", "pathetic",
        "disgusting", "waste of time", "boring", "dull", "poorly made",
        "laughable", "painful to watch"
    ]
    if any(word in text_lower for word in strong_neg):
        return "Negative"

    # Strong positive
    strong_pos = [
        "excellent", "amazing", "fantastic", "outstanding", "masterpiece",
        "brilliant", "superb", "phenomenal", "spectacular", "top notch",
        "well made", "loved it", "best ever"
    ]
    if any(word in text_lower for word in strong_pos):
        return "Positive"

    # Negation handling
    negation_words = ["not", "n't", "never", "no", "dont", "don't", "cannot", "can't", "hardly"]
    pos_words = ["good", "great", "amazing", "fantastic", "excellent", "wonderful", "awesome"]
    neg_words = ["bad", "terrible", "awful", "horrible", "poor"]

    if any(word in text_lower.split() for word in negation_words):
        if any(p in text_lower for p in pos_words):
            return "Negative"
        if any(n in text_lower for n in neg_words):
            return "Positive"

    # Mixed feelings
    if "good" in text_lower and "bad" in text_lower:
        return "Neutral"
    if "love" in text_lower and "hate" in text_lower:
        return "Neutral"
    if "better than expected" in text_lower:
        return "Positive"
    if "worse than expected" in text_lower:
        return "Negative"

    # Sarcasm-like cues
    sarcasm_patterns = [
        "yeah right", "as if", "sure it was great", "totally amazing", "what a joke"
    ]
    if any(p in text_lower for p in sarcasm_patterns):
        return "Negative"

    return None  # let ML model decide

# -------------------------
# Hybrid prediction
# -------------------------
def predict_sentiment(text, model, vectorizer, label_encoder):
    # Rule-based first
    rule_label = rule_based_sentiment(text)
    if rule_label:
        return rule_label

    # ML model fallback
    X = vectorizer.transform([text])
    pred = model.predict(X)
    return label_encoder.inverse_transform(pred)[0].capitalize()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="centered")

st.title("🎭 Sentiment Analysis App")
st.write("Hybrid model: ML + rule-based fixes for tricky cases.")

model, vectorizer, label_encoder = load_artifacts()

if model and vectorizer and label_encoder:
    user_input = st.text_area("Enter a review/comment:", "")
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            prediction = predict_sentiment(user_input, model, vectorizer, label_encoder)
            st.success(f"Predicted Sentiment: **{prediction}**")
        else:
            st.warning("⚠️ Please enter some text.")
else:
    st.error("Model artifacts not found. Please upload your .pkl files.")
