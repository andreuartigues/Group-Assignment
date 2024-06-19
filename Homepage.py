import streamlit as st
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set page configuration
st.set_page_config(
    page_title="LLM",
    page_icon="👩‍⚕️",
    layout="wide",
    initial_sidebar_state='expanded'
)


# Custom CSS
custom_css = """
<style>
    /* Title style */
    .stTitle {
        color: #006064;
    }
    /* Header in the sidebar */
    .sidebar .sidebar-content h1 {
        color: #004d40;
    }
    /* Button color */
    div.stButton > button {
        background-color: #004d40;
        color: white;
    }
    /* Information box */
    .stAlert p {
        color: #004d40;
    }
    /* Markdown color */
    .stMarkdown p {
        color: #004d40;
    }
    /* Disclaimer text */
    .disclaimer {
        color: grey;
        font-size: small;
    }
</style>
"""

df_counts = pd.read_csv('df_counts.csv')
df_locations = pd.read_csv('df_locations.csv')

# Inject the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Title and description
st.title("👩‍⚕️ Medical Trends Through Entity Recognition & Synthetic Datasets")
st.markdown("---")
st.write("Enter your symptoms to explore medical entities and visualize symptom frequencies.")

# Sidebar information
with st.sidebar:
    st.header("About This App")
    st.write("""
    This application leverages large language models (LLM) and synthetic datasets to provide medical insights.
    Enter your symptoms, and the system will recognize medical entities and provide a potential diagnosis.
    """)
    st.markdown("Created by: IE Students")
    st.markdown("Contact: Natural Language Processing Course")

# Text input for symptoms
st.subheader("Describe your symptoms:")
symptoms = st.text_area("", height=150, placeholder="e.g., fever, headache, muscle pain...")




class KeywordExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(self.extract_keywords)

    def extract_keywords(self, text):
        doc = nlp(text)
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        return ' '.join(keywords)

# Function to visualize our words on the map
def visualize_symptoms_on_map(word, df_locations):
    # Filter the locations for the given word
    word_locations = df_locations[df_locations['word'] == word]
    
    # Count occurrences for each hospital
    hospital_counts = word_locations['hospital_name'].value_counts().reset_index()
    hospital_counts.columns = ['hospital_name', 'count']

    # Merge counts with word locations
    word_locations = word_locations.merge(hospital_counts, on='hospital_name', how='left')

    # Create a map centered around the average coordinates in Madrid
    map_center = [40.4168, -3.7038]  # Coordinates for Madrid city center
    word_map = folium.Map(location=map_center, zoom_start=10)

    # Add markers for each hospital location
    for idx, row in word_locations.iterrows():
        latitude = row['longitude']
        longitude = row['latitude']

        print(f"Adding marker: {row['hospital_name']}, Lat: {latitude}, Lon: {longitude}")

        folium.Marker(
            location=[latitude, longitude],
            popup=f"Cases: {row['count']} <br> {row['hospital_name']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(word_map)
        

    return word_map



pipeline = joblib.load('disease_prediction_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')


if st.button("Analyze Symptoms"):
    if symptoms:
        #30-year-old female office worker with a history of migraines and a family history of hypertension.
        symptoms_series = pd.Series([symptoms])
        predicted_label = pipeline.predict(symptoms_series)
        predicted_disease = label_encoder.inverse_transform(predicted_label)
            
        st.markdown("### Recognized Medical Entities")
        st.write(f'Predicted Disease: {predicted_disease[0]}')
        
        st.markdown("### Map")
        word_map=visualize_symptoms_on_map(predicted_disease[0], df_locations)
        folium_static(word_map)
        
    else:
        st.write("No medical entities found in the input.")
else:
    pass

# Footer
st.markdown("---")
st.markdown('<p class="disclaimer">Disclaimer: This application is for educational purposes only and does not provide medical advice.</p>', unsafe_allow_html=True)
