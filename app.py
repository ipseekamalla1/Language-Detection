import streamlit as st
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore warnings for cleaner output
warnings.simplefilter("ignore")

# Load and preprocess the data
@st.cache_data
def load_data():
    data = pd.read_csv("Language Detection.csv")
    return data

# Function to preprocess text data
def preprocess_text(text):
    text = re.sub(r'[!@#$(),\n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'\[\]', ' ', text)
    text = text.lower()
    return text

# Load data
data = load_data()

# Preprocess and prepare data
X = data["Text"].apply(preprocess_text)
y = data["Language"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Vectorize text data
cv = CountVectorizer()
X = cv.fit_transform(X).toarray()

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Train the model
model = MultinomialNB()
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Streamlit app
st.title("Language Detection App")


# Text input for prediction
user_input = st.text_area("Enter text to detect language:", "")

# Add an "Enter" button
if st.button("Enter"):
    if user_input:
        processed_input = preprocess_text(user_input)
        x_input = cv.transform([processed_input]).toarray()
        lang_pred = model.predict(x_input)
        lang_pred = le.inverse_transform(lang_pred)
        st.write(f"The predicted language is: {lang_pred[0]}")
    else:
        st.write("Please enter some text to predict the language.")