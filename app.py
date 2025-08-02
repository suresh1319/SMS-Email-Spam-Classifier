import pickle
import streamlit as st
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords

# Download necessary NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('punkt')

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
    tokens = nltk.word_tokenize(text)
    processed_tokens = [ps.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(processed_tokens)

# Prediction logic
if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        # Preprocess input
        transformed_sms = transform_text(input_sms)

        # Vectorize input
        vector_input = tfidf.transform([transformed_sms])

        # Make prediction
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.header("🚫 Spam")
        else:
            st.header("✅ Not Spam")

        # Show prediction confidence if model supports it
        try:
            confidence = model.predict_proba(vector_input)[0][1 if result == 1 else 0]
            st.subheader(f"Prediction Confidence: {confidence:.2%}")
        except AttributeError:
            st.info("Confidence score not available for this model.")
