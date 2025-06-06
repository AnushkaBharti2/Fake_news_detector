import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Fake News Detector")

user_input = st.text_area("Enter news text here:")

if st.button("Predict"):
    if user_input.strip():
        vector_input = vectorizer.transform([user_input])
        prediction = model.predict(vector_input)[0]
        label = "REAL" if prediction == 1 else "FAKE"
        st.success(f"The news is likely: {label}")
    else:
        st.error("Please enter some news text.")
