import joblib

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Take user input
news_text = input("Enter news text: ")

# Clean it (same as in training)
import string
def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

cleaned = clean_text(news_text)

# Vectorize and predict
vectorized = vectorizer.transform([cleaned])
prediction = model.predict(vectorized)[0]

# Output result
if prediction == 1:
    print("✅ This news is likely REAL.")
else:
    print("🚨 This news is likely FAKE.")

