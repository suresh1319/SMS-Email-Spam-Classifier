import pickle
import streamlit as st
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
@st.cache_resource
def download_nltk_data():
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)

    nltk.download("punkt", download_dir=nltk_data_path)
    nltk.download("stopwords", download_dir=nltk_data_path)

download_nltk_data()

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load vectorizer and model with error handling
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

# Streamlit app UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message you want to check")

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)  # FIX: use correct tokenizer
    processed_tokens = [ps.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(processed_tokens)

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.header("🚫 Spam")
        else:
            st.header("✅ Not Spam")
