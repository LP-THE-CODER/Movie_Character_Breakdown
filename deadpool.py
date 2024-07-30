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
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import tempfile
import os
from transformers import pipeline

# Function to perform sentiment analysis on dialogues
def perform_sentiment_analysis(dialogues):
    # Tokenize the dialogues
    tokens = word_tokenize(dialogues)
   
    # Perform sentiment analysis using Vader
    sid = SentimentIntensityAnalyzer()
    sentiments = [sid.polarity_scores(word)['compound'] for word in tokens]
   
    # Combine words and sentiments
    word_sentiments = list(zip(tokens, sentiments))
   
    # Separate positive and negative words
    positive_words = [word for word, sentiment in word_sentiments if sentiment > 0]
    negative_words = [word for word, sentiment in word_sentiments if sentiment < 0]
   
    return positive_words, negative_words

# Function to generate word cloud
def generate_word_cloud(words1, words2, title1, title2):
    # Join words into a single string
    text1 = ' '.join(words1)
    text2 = ' '.join(words2)
   
    # Create two word clouds
    wordcloud1 = WordCloud(width=500, height=300, background_color="white", colormap='hsv_r')
    wordcloud1.generate_from_text(text1)

    wordcloud2 = WordCloud(width=500, height=300, background_color="black", colormap='hsv_r')
    wordcloud2.generate_from_text(text2)
   
    # Display the word clouds side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(wordcloud1.to_array(), caption=title1, use_column_width=True)
    with col2:
        st.image(wordcloud2.to_array(), caption=title2, use_column_width=True)

# Function to count dialogues for a specific character
def count_dialogues_for_character(df, character_name):
    dialogues_count = 0

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        scene_dialogues = row["Scene_Dialogue"]
        if pd.notnull(scene_dialogues):
            # Use regular expression to find character names in dialogues
            character_dialogues = re.findall(r"\b" + re.escape(character_name) + r"\b", scene_dialogues, flags=re.IGNORECASE)
            dialogues_count += len(character_dialogues)

    return dialogues_count

# Function to count scenes for a specific character
def count_scenes_for_character(df, character_name):
    scenes_count = 0

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        scene_characters = row["Scene_Characters"]
        if pd.notnull(scene_characters):
            # Use regular expression to find character names in scene characters
            if re.search(r"\b" + re.escape(character_name) + r"\b", scene_characters, flags=re.IGNORECASE):
                scenes_count += 1

    return scenes_count

# Function to analyze character relationships
def analyze_relationships(df):
    relationships = {}

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        scene_characters = row["Scene_Characters"]
        if pd.notnull(scene_characters):
            # Split the scene_characters into a list of characters
            characters = [character.strip() for character in scene_characters.split(",")]
            # Update relationships dictionary
            for character1 in characters:
                for character2 in characters:
                    if character1 != character2:
                        key = (character1, character2)
                        # Increment the interaction count between character1 and character2
                        if key not in relationships:
                            relationships[key] = 1
                        else:
                            relationships[key] += 1

    return relationships

# Function to count scenes for each character
def count_scenes_per_character(df):
    scenes_count = {}

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        scene_characters = row["Scene_Characters"]
        if pd.notnull(scene_characters):
            # Split the scene_characters into a list of characters
            characters = [character.strip() for character in scene_characters.split(",")]
            # Update the scenes count for each character
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
            character_names = list(character_names)
            st.write("Character Names:", character_names)

    elif page == "Character Dialogue Counts":
        st.subheader("Character Dialogue Analysis 🗣")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Dictionary to store the dialogue counts for each character
            character_dialogue_counts = {}
            # Iterate through the DataFrame and count the dialogues for each character
            for index, row in df.iterrows():
                characters = row['Scene_Characters']
                if pd.notnull(characters):
                    characters = characters.split(", ")
                    for character in characters:
                        character_name = character.strip("[]")
                        if character_name in character_dialogue_counts:
                            character_dialogue_counts[character_name] += 1
                        else:
                            character_dialogue_counts[character_name] = 1
            # Display the dialogue counts for each character
            st.write("Number of dialogues for each character:")
            for character, count in character_dialogue_counts.items():
                st.write(f"{character}: {count} dialogues")

                        
    elif page == "Character Scene Counts":
        st.subheader("Character Scene Analysis 🎞")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Calculate the number of scenes for each character
            character_scenes_count = count_scenes_per_character(df)
            # Sum up the counts for each character
            total_scenes_count = {}
            for character, count in character_scenes_count.items():
                character_name = character.strip("[]")  # Remove brackets from character name
                total_scenes_count[character_name] = total_scenes_count.get(character_name, 0) + count
            # Display the total number of scenes for each character
            st.write("Total Scenes Count for Each Character:")
            for character, count in total_scenes_count.items():
                st.write(f"{character}: {count} scenes")
            

    elif page == "Bar Graph on Dialogue Count":
        st.subheader("Dialogue Counts for Each Character")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            character_dialogue_counts = {}
            # Iterate through the DataFrame and count the dialogues for each character
            for index, row in df.iterrows():
                characters = row['Scene_Characters']
                if pd.notnull(characters):
                    characters = characters.split(", ")
                    for character in characters:
                        character_name = character.strip("[]")
                        if character_name in character_dialogue_counts:
                            character_dialogue_counts[character_name] += 1
                        else:
                            character_dialogue_counts[character_name] = 1
            # Convert the dialogue counts dictionary to a DataFrame
            dialogue_counts_df = pd.DataFrame(list(character_dialogue_counts.items()), columns=["Character", "Dialogue Count"])
            # Create a bar graph using Plotly Express
            fig = px.bar(dialogue_counts_df, x="Character", y="Dialogue Count", title="Number of Dialogues for Each Character",
                         color_discrete_sequence=["blue"])  # Set the color to orange
            # Add text annotations for each bar
            fig.update_traces(texttemplate='%{y}', textposition='outside')
            # Show the plot
            st.plotly_chart(fig)


    elif page == "Bar Graph on Scene Count":
        st.subheader("Scene Counts for Each Character")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Calculate the number of scenes for each character
            character_scenes_count = count_scenes_per_character(df)
            # Sum up the counts for each character
            total_scenes_count = {}
            for character, count in character_scenes_count.items():
                character_name = character.strip("[]")  # Remove brackets from character name
                total_scenes_count[character_name] = total_scenes_count.get(character_name, 0) + count
            # Convert the total scenes count dictionary to a DataFrame
            scenes_counts_df = pd.DataFrame(list(total_scenes_count.items()), columns=["Character", "Scenes Count"])
            # Create a bar graph using Plotly Express
            fig = px.bar(scenes_counts_df, x="Character", y="Scenes Count", title="Total Scenes Count for Each Character",
                         color_discrete_sequence=["black"])  # Set the color to blue
            # Add text annotations for each bar
            fig.update_traces(texttemplate='%{y}', textposition='outside')
            # Show the plot
            st.plotly_chart(fig)

    elif page == "Character Interactions":
        st.subheader("Character Interactions/Relationships Analysis 🤝🏻")
        uploaded_file = st.file_uploader("Upload a CSV file containing movie data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Analyze relationships for all characters
            character_relationships = analyze_relationships(df)
            # Display character interactions
            st.write("Character Relationships:")
            for characters, count in character_relationships.items():
                character1, character2 = characters
                # Remove brackets from character names if present
                character1 = character1.strip("[]")
                character2 = character2.strip("[]")
                st.write(f"{character1} - {character2}: {count} interactions")
    

    elif page == "Character Relationships":
        st.title("Character Relationships Graph")
        st.write("Below is the graph showing character relationships:")
        # Path to your PNG file
        image_path = "deadpool_relationships.png"
        # Display the image
        st.image(image_path, use_column_width=True)

    
    elif page == "Character Emotion Analysis":
        st.title("Character Emotion Graph")
        st.write("Below is the graph showing character emotion:")
        # Path to your PNG file
        image_path = "deadpool_relationships_emotions.png"
        # Display the image
        st.image(image_path, use_column_width=True)
 

    elif page == "Text Emotion Analysis":
        st.title("Text Emotion Analysis")
        # Get user input text
        if st.button("Analyze Emotions"):
            # Split text into lines
            texts = """ The story begins with Wade Wilson (Ryan Reynolds), a former Special Forces operative turned mercenary, living a rough but relatively content life in New York City. He spends his days taking small-time jobs, protecting teenage girls from stalkers, and generally keeping a low profile. Wade frequents Sister Margaret's School for Wayward Girls, a local bar where mercenaries hang out and take on jobs. The bar is run by his best friend, Weasel (T.J. Miller).

            One night at the bar, Wade meets Vanessa Carlysle (Morena Baccarin), a prostitute with a sharp wit. The two quickly hit it off, beginning a whirlwind romance filled with humor, adventure, and love. Vanessa and Wade's relationship deepens, and just as Wade is about to propose, he is diagnosed with terminal cancer. Devastated, Wade struggles with the idea of leaving Vanessa behind.

            In desperation, Wade is approached by a mysterious recruiter (Jed Rees) who offers him a cure for his cancer through a secret experimental program. The program, run by Ajax (Ed Skrein) and Angel Dust (Gina Carano), promises not only to cure Wade but also to give him extraordinary abilities. Although skeptical, Wade decides to undergo the procedure to save his life and stay with Vanessa.

            The procedure is brutal and torturous. Ajax, whose real name is Francis, and Angel Dust subject Wade to intense physical and psychological abuse, attempting to awaken any dormant mutant genes within him. The treatment involves injecting a serum and then exposing him to extreme stress to trigger a mutation. Ajax reveals that they are not heroes but are creating super slaves to be sold as living weapons. Wade's mutation eventually kicks in, giving him superhuman healing abilities but leaving him horribly disfigured with a scarred face and body.

            Fueled by anger and a desire for revenge, Wade adopts the alter ego "Deadpool." He escapes the facility by causing an explosion and vows to hunt down Ajax to force him to fix his disfigurement. Believing he is too hideous for Vanessa to love him, Wade keeps his survival a secret from her, despite still watching over her from afar.

            Wade dons a red suit and mask to hide his appearance and embarks on a mission to find Ajax. He goes on a rampage through the criminal underworld, torturing and interrogating anyone who might know Ajax's whereabouts. During this time, he encounters Colossus (voiced by Stefan Kapičić) and Negasonic Teenage Warhead (Brianna Hildebrand), members of the X-Men. Colossus tries to recruit Deadpool to join the X-Men and use his powers for good, but Deadpool rejects the offer, preferring his own methods of justice.

            Deadpool finally tracks Ajax to a convoy and attacks it, leading to a brutal and chaotic highway battle. He kills many of Ajax's henchmen before confronting Ajax himself. Just as Deadpool is about to capture Ajax, Colossus and Negasonic Teenage Warhead intervene, allowing Ajax to escape. Colossus handcuffs Deadpool and tries to convince him to change his ways, but Deadpool cuts off his own hand to escape, knowing it will regenerate later.

            Ajax learns of Deadpool's connection to Vanessa and kidnaps her to lure Deadpool into a trap. Ajax and Angel Dust take Vanessa to a decommissioned helicarrier where they plan to kill her. Deadpool, with the help of Weasel, finds out about the plan and decides to rescue Vanessa. He reluctantly teams up with Colossus and Negasonic Teenage Warhead, who agree to help him in exchange for Deadpool considering joining the X-Men.

            The final showdown takes place at the scrapyard where the helicarrier is located. Deadpool, Colossus, and Negasonic Teenage Warhead fight their way through Ajax's goons. Colossus battles Angel Dust, while Negasonic uses her explosive powers to assist Deadpool. Deadpool confronts Ajax and, after a fierce battle, subdues him. Ajax reveals that there is no cure for Wade's disfigurement, enraging Deadpool further. Despite Colossus's plea to spare Ajax's life and be a true hero, Deadpool kills Ajax, much to Colossus's dismay.

            With Ajax dead, Deadpool frees Vanessa and explains why he disappeared. Vanessa, though initially shocked by Wade's appearance, still loves him. The two reconcile and share a kiss, with Wade feeling hopeful about their future together.

            The movie ends with Deadpool acknowledging his place as an unconventional anti-hero, breaking the fourth wall to speak directly to the audience. He humorously addresses his journey, the love he has for Vanessa, and his acceptance of his new identity. In a post-credits scene, Deadpool teases the audience with the possibility of a sequel and the introduction of a new character, Cable."""
            pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", return_all_scores=True)
            texts = split_text_into_lines(texts)
            # Initialize emotion lists
            anger = []
            anticipation = []
            disgust = []
            fear = []
            joy = []
            love = []
            optimism = []
            pessimism = []
            sadness = []
            surprise = []
            trust = []
            # Loop through each line of text and analyze emotions
            for text in texts:
                emotions = pipe(text)
                for feel in emotions[0]:
                    if feel['label'] == 'anger':
                        anger.append(feel['score'])
                    elif feel['label'] == 'anticipation':
                        anticipation.append(feel['score'])
                    elif feel['label'] == 'disgust':
                        disgust.append(feel['score'])
                    elif feel['label'] == 'fear':
                        fear.append(feel['score'])
                    elif feel['label'] == 'joy':
                        joy.append(feel['score'])
                    elif feel['label'] == 'love':
                        love.append(feel['score'])
                    elif feel['label'] == 'optimism':
                        optimism.append(feel['score'])
                    elif feel['label'] == 'pessimism':
                        pessimism.append(feel['score'])
                    elif feel['label'] == 'sadness':
                        sadness.append(feel['score'])
                    elif feel['label'] == 'surprise':
                        surprise.append(feel['score'])
                    elif feel['label'] == 'trust':
                        trust.append(feel['score'])
            # Create Plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=anger, mode='lines', name='anger'))
            fig.add_trace(go.Scatter(y=anticipation, mode='lines', name='anticipation'))
            fig.add_trace(go.Scatter(y=disgust, mode='lines', name='disgust'))
            fig.add_trace(go.Scatter(y=fear, mode='lines', name='fear'))
            fig.add_trace(go.Scatter(y=joy, mode='lines', name='joy'))
            fig.add_trace(go.Scatter(y=love, mode='lines', name='love'))
            fig.add_trace(go.Scatter(y=optimism, mode='lines', name='optimism'))
            fig.add_trace(go.Scatter(y=pessimism, mode='lines', name='pessimism'))
            fig.add_trace(go.Scatter(y=sadness, mode='lines', name='sadness'))
            fig.add_trace(go.Scatter(y=surprise, mode='lines', name='surprise'))
            fig.add_trace(go.Scatter(y=trust, mode='lines', name='trust'))
            # Display Plotly figure
            st.plotly_chart(fig)


# Run the Streamlit app
if __name__ == "__main__":
    main()
