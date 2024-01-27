
import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
from farasa.stemmer import FarasaStemmer

# Load the TF-IDF vectorizer and the trained model
with open('tfidf_vectorizer.joblib', 'rb') as file:
    tfidf_vectorizer = joblib.load(file)

with open('random_forest_model.joblib', 'rb') as file:
    model = joblib.load(file)

# Arabic text processing
def preprocess_arabic_text(text):
    text = re.sub(r'[^ء-ي\s]', '', text)
    words = word_tokenize(text)
    words = [re.sub(r'[\u064B-\u0652]', '', word) for word in words]
    stop_words = set(nltk.corpus.stopwords.words('arabic'))
    words = [word for word in words if word not in stop_words]
    stemmer = FarasaStemmer(interactive=True)
    words = [stemmer.stem(word) for word in words]
    processed_text = ' '.join(words)
    return processed_text

# Streamlit UI
st.title('Fatwa Author Attribution Chatbot')

user_input = st.text_input('Enter your fatwa:')
if st.button('Predict Author'):
    if user_input:
        # Preprocess user input
        processed_input = preprocess_arabic_text(user_input)

        # Vectorize the input using the loaded TF-IDF vectorizer
        input_tfidf = tfidf_vectorizer.transform([processed_input])

        # Make predictions using the loaded model
        prediction = model.predict(input_tfidf)

        st.success(f'The predicted author for the fatwa is: {prediction[0]}')
    else:
        st.warning('Please enter a fatwa.')
