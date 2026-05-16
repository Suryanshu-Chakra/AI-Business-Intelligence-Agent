import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/Amazon_Unlocked_Mobile.csv")

# Keep important columns only
df = df[['Product Name', 'Brand Name', 'Rating', 'Reviews']]

# Remove missing values
df.dropna(inplace=True)

# Remove duplicate reviews
df.drop_duplicates(subset=['Reviews'], inplace=True)

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters but KEEP spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Join words back properly
    return " ".join(words)

# Apply cleaning
df['Cleaned_Reviews'] = df['Reviews'].apply(clean_text)

# Save cleaned dataset
df.to_csv("data/cleaned_mobile_reviews.csv", index=False)

print("Dataset cleaned successfully!")
print(df.head())