import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib

#Mirar cosas de chache
pipeline = joblib.load('disease_prediction_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')

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

# Button to analyze symptoms
if st.button("Analyze Symptoms"):
    if symptoms:
        predicted_label = pipeline.predict([symptoms])
        predicted_disease = label_encoder.inverse_transform(predicted_label)
            
        st.markdown("### Recognized Medical Entities")
        st.write(f'Predicted Disease: {predicted_disease[0]}')
    else:
        st.write("No medical entities found in the input.")
else:
    st.info("Click the button to analyze your symptoms.")

# Footer
st.markdown("---")
st.markdown('<p class="disclaimer">Disclaimer: This application is for educational purposes only and does not provide medical advice.</p>', unsafe_allow_html=True)
