import pandas as pd

# Load the dataset
data = pd.read_csv("fake_news_dataset.csv")

# Print first few rows
print(data.head())

# Print total samples and label distribution
print(f"Total samples: {len(data)}")
print(data['label'].value_counts())
