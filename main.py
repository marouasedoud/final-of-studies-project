import streamlit as st
import pandas as pd
from PIL import Image
from page2 import show_data_page
from page3 import show_data_pagee, sendtofile3

def main():
    # Placer le logo dans le coin supérieur gauche de la page principale
    col1, col2, col3 = st.columns([2, 20, 2])

    # Redimensionner et afficher l'image 1 avec une taille spécifique
    image1 = Image.open("so1.png")
    image1_resized = image1.resize((500, 500))  # Redimensionner l'image à la taille souhaitée
    col1.image(image1_resized)

    # Afficher le titre avec une mise en page personnalisée
    col2.markdown("<h1 style='text-align: center; font-size: 40px; color: orange;'>Artificial Intelligence for Virtual Metering of Well Production AI_VM_PROD</h1>", unsafe_allow_html=True)

    # Redimensionner et afficher l'image 2 avec une taille spécifique
    image2 = Image.open("so2.png")
    image2_resized = image2.resize((500, 500))  # Redimensionner l'image à la taille souhaitée
    col3.image(image2_resized)

    # Afficher la page d'accueil
    show_home()

    # Afficher le contenu en fonction de la page sélectionnée
    pages = ["None", "AI for Daily Well Production Accounting", "AI for Well Production Forecasts"]
    page = st.selectbox("Select the AI feature you wish to use", pages)

    if page == "AI for Daily Well Production Accounting":
        show_page1()
    elif page == "AI for Well Production Forecasts":
        show_page2()

def show_home():
      # Afficher le titre avec la couleur orange
    title_style = """<style>h1 { text-align ;font-size: 40px;color: orange;}</style>"""
    st.markdown(title_style, unsafe_allow_html=True)
    st.markdown("<p style='font-style: italic; font-weight: bold; font-size: 20px; color: grey;'>Welcome to AI_VM_PROD application.</p>", unsafe_allow_html=True)

def show_page1():
    st.markdown("<p style=' font-weight: bold; text-align:center; font-size: 30px; color: orange;'>Upload Wells Gauging Data File</p>", unsafe_allow_html=True)
    uploaded_file_jaugeage = st.file_uploader("Wells gauging data file is required to generate AI models.",type=["csv"])

    # Vérifier si les fichiers ont été téléchargés
    if uploaded_file_jaugeage is not None:
        # Lire les fichiers CSV en utilisant Pandas
        df_jaugeage = pd.read_csv(uploaded_file_jaugeage)

        # Rediriger vers la page de visualisation des données
        show_data_page(df_jaugeage)
        sendtofile3(df_jaugeage)
    else:
        st.write("No CSV file uploaded yet.")

def show_page2():
    st.markdown("<p style=' font-weight: bold; text-align:center; font-size: 30px; color: orange;'>Upload daily mesurement data file</p>", unsafe_allow_html=True)
    uploaded_file_daily = st.file_uploader("Daily mesurement data file is required to start forecasting", type=["csv"])

    if uploaded_file_daily is not None:
        # Lire les fichiers CSV en utilisant Pandas
        df_daily = pd.read_csv(uploaded_file_daily)
        show_data_pagee(df_daily)
    else:
        st.write("No CSV file uploaded yet.")

if __name__ == "__main__":
    main()
