# Fake News Detector

This project is a simple machine learning application that detects whether a news article is real or fake based on the given text.

## Features

- Uses a TF-IDF vectorizer to process text data
- Trains a Logistic Regression model on a fake news dataset
- Provides a Streamlit web interface for user-friendly interaction
- Saves and loads model and vectorizer for predictions

## Dataset

The dataset contains news headlines and text labeled as real (`1`) or fake (`0`). It is stored in `fake_news_dataset.csv`.

## How to Run

1. Clone the repo or download the files.
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Train the model:
    ```bash
    python train_model.py
    ```
4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
5. Enter news text in the app to check if it's real or fake.

## Files

- `load_data.py`: Loads and displays the dataset.
- `train_model.py`: Trains the fake news detection model.
- `predict_news.py`: Script to predict news labels using saved model.
- `app.py`: Streamlit web application.
- `fake_news_dataset.csv`: Dataset file.
- `fake_news_model.pkl`: Saved model.
- `tfidf_vectorizer.pkl`: Saved vectorizer.

## Improvements & Next Steps

- Use a larger and more diverse dataset.
- Experiment with more advanced NLP models.
- Improve web app UI and add more features.
- Deploy the app to cloud platforms like Heroku or Streamlit Sharing.

---

Feel free to customize or ask me to add anything else! Want me to generate the full markdown file content?
