import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline

# Function to perform sentiment analysis on dialogues
def perform_sentiment_analysis(dialogues):
    tokens = word_tokenize(dialogues)
    sid = SentimentIntensityAnalyzer()
    sentiments = [sid.polarity_scores(word)['compound'] for word in tokens]
    word_sentiments = list(zip(tokens, sentiments))
    positive_words = [word for word, sentiment in word_sentiments if sentiment > 0]
    negative_words = [word for word, sentiment in word_sentiments if sentiment < 0]
    return positive_words, negative_words

# Function to generate word cloud
def generate_word_cloud(words1, words2, title1, title2):
    text1 = ' '.join(words1)
    text2 = ' '.join(words2)
    wordcloud1 = WordCloud(width=500, height=300, background_color="white", colormap='hsv_r').generate_from_text(text1)
    wordcloud2 = WordCloud(width=500, height=300, background_color="black", colormap='hsv_r').generate_from_text(text2)
    col1, col2 = st.columns(2)
    with col1:
        st.image(wordcloud1.to_array(), caption=title1, use_column_width=True)
    with col2:
        st.image(wordcloud2.to_array(), caption=title2, use_column_width=True)

# Function to count dialogues for a specific character
def count_dialogues_for_character(df, character_name):
    dialogues_count = 0
    for index, row in df.iterrows():
        scene_dialogues = row["Scene_Dialogue"]
        if pd.notnull(scene_dialogues):
            character_dialogues = re.findall(r"\b" + re.escape(character_name) + r"\b", scene_dialogues, flags=re.IGNORECASE)
            dialogues_count += len(character_dialogues)
    return dialogues_count

# Function to count scenes for a specific character
def count_scenes_for_character(df, character_name):
    scenes_count = 0
    for index, row in df.iterrows():
        scene_characters = row["Scene_Characters"]
        if pd.notnull(scene_characters):
            if re.search(r"\b" + re.escape(character_name) + r"\b", scene_characters, flags=re.IGNORECASE):
                scenes_count += 1
    return scenes_count

# Function to analyze character relationships
def analyze_relationships(df):
    relationships = {}
    for index, row in df.iterrows():
        scene_characters = row["Scene_Characters"]
        if pd.notnull(scene_characters):
            characters = [character.strip() for character in scene_characters.split(",")]
            for character1 in characters:
                for character2 in characters:
                    if character1 != character2:
                        key = (character1, character2)
                        relationships[key] = relationships.get(key, 0) + 1
    return relationships

# Function to count scenes for each character
def count_scenes_per_character(df):
    scenes_count = {}
    for index, row in df.iterrows():
        scene_characters = row["Scene_Characters"]
        if pd.notnull(scene_characters):
            characters = [character.strip() for character in scene_characters.split(",")]
            for character in characters:
                scenes_count[character] = scenes_count.get(character, 0) + 1
    return scenes_count

def split_text_into_lines(texts):
    return texts.splitlines()

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Film Pre-production Analysis",
        page_icon="🎬",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    
    st.markdown("""
        <style>
        .main-header {background-color: #004466; padding: 10px; border-radius: 10px; margin-bottom: 20px; color: white; text-align: center;}
        </style>
        <div class="main-header"><h1>Film Pre-Production Analysis 🎥</h1></div>
    """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page_options = ["Home", "Word Cloud", "Character Names", "Character Dialogue Counts", "Character Scene Counts", "Bar Graph on Dialogue Count", "Bar Graph on Scene Count", "Character Interactions", "Character Relationships", "Character Emotion Analysis", "Text Emotion Analysis"]
    page = st.sidebar.radio("Go to", page_options, index=0, help="Select a page to navigate to")

    if page == "Home":
        st.write("""
            🙏🏻Welcome to the Movie Character Analysis App!

            Are you ready to dive deep into the world of your favorite movie characters? Our app makes it easy to explore the depths of characters and their roles within the movie script. Whether you're a filmmaker, screenwriter, or simply a fan looking to analyze character dynamics, this interface is designed to provide you with insightful details without any hassle.

            🤷‍♀️How it Works:

            1. Upload Your CSV File: Begin by uploading your CSV file containing the movie data. This file should include information extracted from the movie script, such as character names, dialogues, and scene details.
            2. Analyze Character Data: Once the file is uploaded, our app will analyze the data and present you with a breakdown of each character. You'll discover the number of dialogues and scenes for each character, allowing you to gauge their significance in the storyline.
            3. Explore Emotional Analysis: Dive into the emotional aspects of the movie by exploring the sentiments expressed by each character. Our app performs sentiment analysis on dialogues, revealing the range of emotions portrayed throughout the script.
            4. Visualize Character Relationships: Gain insights into character relationships and interactions with our interactive visualization. You'll uncover the dynamics between characters and the frequency of their dialogues with one another.
            5. Word Cloud Visualization: Delve deeper into the language used by characters with our word cloud visualizations. Easily identify the most commonly used words and phrases, providing additional context to character personalities and themes.
            6. Interactive Graphs: Visualize dialogue and scene counts with interactive bar graphs, allowing for easy comparison between characters and their involvement in the movie.
            7. Distribution of Emotions: Explore the distribution of emotions portrayed by characters through a colorful pie chart. Understand the dominant emotions conveyed by each character and their impact on the overall storyline.
           
            ❔Why Use Our App?

            • Efficient Analysis: Our app streamlines the process of character analysis, saving you time and effort.
            • Insightful Visualizations: Visual representations make it easy to interpret data and uncover patterns within the movie script.
            • Comprehensive Understanding: By examining dialogue counts, emotional analysis, and character relationships, you'll gain a comprehensive understanding of the movie's characters and themes.
           
            👉🏻Get Started Today!

            Upload your CSV file now and embark on a journey to unravel the mysteries behind your favorite movie characters. Whether you're a filmmaker seeking deeper insights or a fan looking to explore character dynamics, our app has everything you need for an enriching analysis experience.
        """)
   
    elif page == "Word Cloud":
        st.subheader("WordCloud for Positive 😌 and Negative Words 😱")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            all_dialogues = ' '.join(df['Scene_Dialogue'].dropna())
            positive_words, negative_words = perform_sentiment_analysis(all_dialogues)
            generate_word_cloud(positive_words, negative_words, "✅Positive Words", "🤬Negative Words")

    elif page == "Character Names":
        st.subheader("Character Names 😏")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            character_names = set()
            for scene_characters in df["Scene_Characters"]:
                if pd.notnull(scene_characters):
                    names = re.findall(r"\b[A-Z][a-zA-Z\s]+\b", scene_characters)
                    character_names.update(names)
            st.write(f"Total Characters: {len(character_names)}")
            st.write(sorted(character_names))

    elif page == "Character Dialogue Counts":
        st.subheader("Character Dialogue Counts")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            character_name = st.text_input("Enter a character name to count their dialogues")
            if character_name:
                dialogues_count = count_dialogues_for_character(df, character_name)
                st.write(f"Total dialogues for {character_name}: {dialogues_count}")

    elif page == "Character Scene Counts":
        st.subheader("Character Scene Counts")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            character_name = st.text_input("Enter a character name to count their scenes")
            if character_name:
                scenes_count = count_scenes_for_character(df, character_name)
                st.write(f"Total scenes for {character_name}: {scenes_count}")

    elif page == "Bar Graph on Dialogue Count":
        st.subheader("Bar Graph on Dialogue Count")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            character_names = set()
            for scene_characters in df["Scene_Characters"]:
                if pd.notnull(scene_characters):
                    names = re.findall(r"\b[A-Z][a-zA-Z\s]+\b", scene_characters)
                    character_names.update(names)
            dialogues_counts = {character: count_dialogues_for_character(df, character) for character in character_names}
            sorted_dialogues_counts = dict(sorted(dialogues_counts.items(), key=lambda item: item[1], reverse=True))
            fig = px.bar(x=list(sorted_dialogues_counts.keys()), y=list(sorted_dialogues_counts.values()), labels={'x':'Characters', 'y':'Dialogue Count'}, title="Dialogue Counts by Character")
            st.plotly_chart(fig)

    elif page == "Bar Graph on Scene Count":
        st.subheader("Bar Graph on Scene Count")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            scenes_counts = count_scenes_per_character(df)
            sorted_scenes_counts = dict(sorted(scenes_counts.items(), key=lambda item: item[1], reverse=True))
            fig = px.bar(x=list(sorted_scenes_counts.keys()), y=list(sorted_scenes_counts.values()), labels={'x':'Characters', 'y':'Scene Count'}, title="Scene Counts by Character")
            st.plotly_chart(fig)

    elif page == "Character Interactions":
        st.subheader("Character Interactions")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            relationships = analyze_relationships(df)
            st.write("Character Interactions (Pair and Number of Interactions):")
            st.write(relationships)

    elif page == "Character Relationships":
        st.subheader("Character Relationships")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            relationships = analyze_relationships(df)
            fig = px.scatter(x=[pair[0] for pair in relationships.keys()], y=[pair[1] for pair in relationships.keys()], size=list(relationships.values()), labels={'x':'Character 1', 'y':'Character 2', 'size':'Interactions'}, title="Character Relationships")
            st.plotly_chart(fig)

    elif page == "Character Emotion Analysis":
        st.subheader("Character Emotion Analysis")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            character_name = st.text_input("Enter a character name to analyze their emotions")
            if character_name:
                dialogues = df[df["Scene_Characters"].str.contains(character_name, na=False)]["Scene_Dialogue"].dropna().tolist()
                dialogues_combined = ' '.join(dialogues)
                emotion_analyzer = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")
                emotions = emotion_analyzer(dialogues_combined)
                st.write(f"Emotions for {character_name}:")
                st.write(emotions)

    elif page == "Text Emotion Analysis":
        st.subheader("Text Emotion Analysis")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            dialogues = df["Scene_Dialogue"].dropna().tolist()
            dialogues_combined = ' '.join(dialogues)
            emotion_analyzer = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")
            emotions = emotion_analyzer(dialogues_combined)
            st.write("Overall Emotions in the Movie:")
            st.write(emotions)

if __name__ == "__main__":
    main()
