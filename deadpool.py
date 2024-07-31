import pandas as pd
import re
import nltk
nltk.download('vader_lexicon')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
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
    wordcloud1 = WordCloud(width=500, height=300, background_color="white").generate(text1)
    wordcloud2 = WordCloud(width=500, height=300, background_color="black").generate(text2)
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
                        if key not in relationships:
                            relationships[key] = 1
                        else:
                            relationships[key] += 1
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
    st.set_page_config(page_title="Film Pre-production Analysis", page_icon="🎬", layout="centered", initial_sidebar_state="expanded")
    html_temp = """
    <div style="background-color:#004466;padding:10px;border-radius:10px;margin-bottom:20px;">
    <h1 style="color:WHITE;text-align:center;"> Film Pre-Production Analysis  🎥 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    page_options = ["Home", "Word Cloud", "Character Names", "Character Dialogue Counts", "Character Scene Counts", "Bar Graph on Dialogue Count", "Bar Graph on Scene Count", "Character Interactions", "Character Relationships", "Character Emotion Analysis", "Text Emotion Analysis"]
    page = st.sidebar.radio("Go to", page_options, index=0, help="Select a page to navigate to")

    if page == "Home":
        st.write("""
            🙏🏻Welcome to the Movie Character Analysis App!

            Are you ready to dive deep into the world of your favorite movie characters? Our app makes it easy to explore the depths of characters and their roles within the movie script. Whether you're a filmmaker, screenwriter, or simply a fan looking to analyze character dynamics, this interface is designed to provide you with insightful details without any hassle.

            🤷‍♀️How it Works:

            1. Upload Your CSV File: Begin by uploading your CSV file containing the movie data. This file should include information extracted from the movie script, such as character names, dialogues, and scene details.\n
            2. Analyze Character Data: Once the file is uploaded, our app will analyze the data and present you with a breakdown of each character. You'll discover the number of dialogues and scenes for each character, allowing you to gauge their significance in the storyline.\n
            3. Explore Emotional Analysis: Dive into the emotional aspects of the movie by exploring the sentiments expressed by each character. Our app performs sentiment analysis on dialogues, revealing the range of emotions portrayed throughout the script.\n
            4. Visualize Character Relationships: Gain insights into character relationships and interactions with our interactive visualization. You'll uncover the dynamics between characters and the frequency of their dialogues with one another.\n
            5. Word Cloud Visualization: Delve deeper into the language used by characters with our word cloud visualizations. Easily identify the most commonly used words and phrases, providing additional context to character personalities and themes.\n
            6. Interactive Graphs: Visualize dialogue and scene counts with interactive bar graphs, allowing for easy comparison between characters and their involvement in the movie.\n
            7. Distribution of Emotions: Explore the distribution of emotions portrayed by characters through a colorful pie chart. Understand the dominant emotions conveyed by each character and their impact on the overall storyline.
           
            ❔Why Use Our App?

            • Efficient Analysis: Our app streamlines the process of character analysis, saving you time and effort.\n
            • Insightful Visualizations: Visual representations make it easy to interpret data and uncover patterns within the movie script.\n
            • Comprehensive Understanding: By examining dialogue counts, emotional analysis, and character relationships, you'll gain a comprehensive understanding of the movie's characters and themes.
           
            👉🏻Get Started Today!

            Upload your CSV file now and embark on a journey to unravel the mysteries behind your favorite movie characters. Whether you're a filmmaker seeking deeper insights or a fan looking to explore character dynamics, our app has everything you need for an enriching analysis experience.
            """)

    elif page == "Word Cloud":
        st.subheader("WordCloud for Positive 😌 and Negative Words 😱")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Scene_Dialogue' in df.columns:
                all_dialogues = ' '.join(df['Scene_Dialogue'].dropna())
                positive_words, negative_words = perform_sentiment_analysis(all_dialogues)
                generate_word_cloud(positive_words, negative_words, "✅Positive Words", "🤬Negative Words")
            else:
                st.error("CSV file does not contain 'Scene_Dialogue' column")

    elif page == "Character Names":
        st.subheader("Character Names 😏")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Scene_Characters' in df.columns:
                character_names = set()
                for scene_characters in df["Scene_Characters"]:
                    if pd.notnull(scene_characters):
                        names = re.findall(r"\b[A-Z][a-zA-Z\s]+\b", scene_characters)
                        character_names.update(names)
                character_names = list(character_names)
                st.write("Character Names:", character_names)
            else:
                st.error("CSV file does not contain 'Scene_Characters' column")

    elif page == "Character Dialogue Counts":
        st.subheader("Character Dialogue Analysis 🗣")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Scene_Dialogue' in df.columns:
                character_name = st.text_input("Enter the character's name:")
                if character_name:
                    dialogues_count = count_dialogues_for_character(df, character_name)
                    st.write(f"{character_name} has {dialogues_count} dialogues in the movie.")
            else:
                st.error("CSV file does not contain 'Scene_Dialogue' column")

    elif page == "Character Scene Counts":
        st.subheader("Character Scene Count 🎬")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Scene_Characters' in df.columns:
                character_name = st.text_input("Enter the character's name:")
                if character_name:
                    scenes_count = count_scenes_for_character(df, character_name)
                    st.write(f"{character_name} appears in {scenes_count} scenes in the movie.")
            else:
                st.error("CSV file does not contain 'Scene_Characters' column")

    elif page == "Bar Graph on Dialogue Count":
        st.subheader("Character Dialogue Count")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Scene_Dialogue' in df.columns:
                characters = st.multiselect("Select characters", df['Scene_Characters'].unique())
                if characters:
                    dialogue_counts = {character: count_dialogues_for_character(df, character) for character in characters}
                    fig = px.bar(x=list(dialogue_counts.keys()), y=list(dialogue_counts.values()), labels={'x': 'Character', 'y': 'Dialogue Count'}, title="Dialogue Count per Character")
                    st.plotly_chart(fig)
            else:
                st.error("CSV file does not contain 'Scene_Dialogue' column")

    elif page == "Bar Graph on Scene Count":
        st.subheader("Character Scene Count")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Scene_Characters' in df.columns:
                scenes_count = count_scenes_per_character(df)
                fig = px.bar(x=list(scenes_count.keys()), y=list(scenes_count.values()), labels={'x': 'Character', 'y': 'Scene Count'}, title="Scene Count per Character")
                st.plotly_chart(fig)
            else:
                st.error("CSV file does not contain 'Scene_Characters' column")

    elif page == "Character Interactions":
        st.subheader("Character Interactions 🔄")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Scene_Characters' in df.columns:
                relationships = analyze_relationships(df)
                character1 = st.selectbox("Select the first character", df['Scene_Characters'].unique())
                character2 = st.selectbox("Select the second character", df['Scene_Characters'].unique())
                if character1 and character2:
                    interaction_count = relationships.get((character1, character2), 0)
                    st.write(f"{character1} and {character2} interact in {interaction_count} scenes.")
            else:
                st.error("CSV file does not contain 'Scene_Characters' column")

    elif page == "Character Relationships":
        st.subheader("Character Relationships ❤️")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Scene_Characters' in df.columns:
                relationships = analyze_relationships(df)
                characters = st.multiselect("Select characters", df['Scene_Characters'].unique())
                if characters:
                    relationship_counts = {(character1, character2): count for (character1, character2), count in relationships.items() if character1 in characters and character2 in characters}
                    fig = go.Figure()
                    for (character1, character2), count in relationship_counts.items():
                        fig.add_trace(go.Bar(name=f'{character1} and {character2}', x=[f'{character1} and {character2}'], y=[count]))
                    fig.update_layout(barmode='stack', title="Character Relationships", xaxis_title="Characters", yaxis_title="Number of Scenes Together")
                    st.plotly_chart(fig)
            else:
                st.error("CSV file does not contain 'Scene_Characters' column")

    elif page == "Character Emotion Analysis":
        st.subheader("Character Emotion Analysis")
        emotion_analysis = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'Scene_Dialogue' in df.columns:
                character_name = st.text_input("Enter the character's name:")
                if character_name:
                    character_dialogues = df[df["Scene_Dialogue"].str.contains(character_name, na=False, case=False)]["Scene_Dialogue"]
                    all_dialogues = ' '.join(character_dialogues.dropna())
                    emotion_scores = emotion_analysis(all_dialogues)
                    emotion_labels = [score['label'] for score in emotion_scores[0]]
                    emotion_values = [score['score'] for score in emotion_scores[0]]
                    fig = px.pie(values=emotion_values, names=emotion_labels, title=f"Emotion Distribution for {character_name}")
                    st.plotly_chart(fig)
            else:
                st.error("CSV file does not contain 'Scene_Dialogue' column")

    elif page == "Text Emotion Analysis":
        st.subheader("Text Emotion Analysis")
        emotion_analysis = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
        input_text = st.text_area("Enter text for emotion analysis")
        if input_text:
            emotion_scores = emotion_analysis(input_text)
            emotion_labels = [score['label'] for score in emotion_scores[0]]
            emotion_values = [score['score'] for score in emotion_scores[0]]
            fig = px.pie(values=emotion_values, names=emotion_labels, title="Emotion Distribution for Input Text")
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
