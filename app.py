import streamlit as st
import joblib
import zipfile
import os

@st.cache_resource
def load_artifacts():
    model_path = "best_sentiment_model.zip"
    vectorizer_path = "tfidf_vectorizer (1).pkl"
    encoder_path = "label_encoder.pkl"

    # Step 1: Unzip model if not already extracted
    extracted_model_path = "best_sentiment_model.pkl"
    if not os.path.exists(extracted_model_path):
        with zipfile.ZipFile(model_path, "r") as zip_ref:
            zip_ref.extractall(".")  # Extract in current dir

    # Step 2: Load artifacts
    model = joblib.load(extracted_model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(encoder_path)

    return model, vectorizer, label_encoder

# Example usage
st.title("Sentiment Analysis App")

model, vectorizer, label_encoder = load_artifacts()

user_input = st.text_area("Enter text to analyze:")
if st.button("Predict"):
    if user_input.strip():
        X = vectorizer.transform([user_input])
        pred = model.predict(X)
        sentiment = label_encoder.inverse_transform(pred)[0]
        st.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text.")
