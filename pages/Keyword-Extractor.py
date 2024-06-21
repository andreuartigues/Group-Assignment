import streamlit as st
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="NLP",
    page_icon="👩‍⚕️",
    layout="wide",
    initial_sidebar_state='expanded'
)

# Inject the custom CSS
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

# Sidebar information
with st.sidebar:
    st.header("About This App")
    st.write("""
    This application leverages large language models (LLM) and synthetic datasets to provide medical insights.
    Enter your symptoms, and the system will recognize medical entities and provide a potential diagnosis.
    """)
    st.markdown("Created by: IE Students")
    st.markdown("Contact: Natural Language Processing Course")

df_counts = pd.read_csv('df_counts.csv')
df_locations = pd.read_csv('df_locations.csv')
# Title and description
st.title("👩‍⚕️ Entity recognition based on keyword")
st.markdown("---")
st.write("Enter your keywords to explore medical entities and visualize where in Madrid are given and with what frequencies.")

# Text input for symptoms
st.subheader("Entity:")
symptoms = st.text_area("", height=50, placeholder="e.g., fever, headache, muscle pain...")


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

if st.button("Analyze Symptoms"):
    if symptoms:
        print(symptoms)
        word_map=visualize_symptoms_on_map(symptoms.lower(), df_locations)
        folium_static(word_map)


