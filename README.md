
---

# Film Pre-Production Analysis App

## Overview

Welcome to the **Film Pre-Production Analysis App**! 🎬 Dive deep into your favorite movie scripts with our powerful tool designed for filmmakers, screenwriters, and movie enthusiasts. This app provides interactive visualizations and detailed insights into character interactions, dialogue counts, sentiment analysis, and emotional breakdowns.

## 📹 Watch the Demo

[![Watch the Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

*Click the video to see the app in action!*

## Features

- **Home**: Introduction and how-to guide.
- **Word Cloud**: Visualize positive and negative words from dialogues.
- **Character Names**: Extract and display unique character names.
- **Character Dialogue Counts**: Count and display dialogues for each character.
- **Character Scene Counts**: Analyze and count scenes involving each character.
- **Bar Graph on Dialogue Count**: Interactive bar graph for dialogue counts.
- **Bar Graph on Scene Count**: Interactive bar graph for scene counts.
- **Character Interactions**: Analyze and display character interactions.
- **Character Relationships**: Graphical representation of character relationships.
- **Character Emotion Analysis**: Analyze and visualize character emotions.
- **Text Emotion Analysis**: Analyze and visualize emotions from a given text.

## Installation

To run this app locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/LP-THE-CODER/Movie_Character_Breakdown.git
   cd Movie_Character_Breakdown
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App:**
   ```bash
   streamlit run deadpool.py
   ```

## Usage

1. **Upload CSV File:**
   - Navigate to the desired feature page (e.g., Word Cloud, Character Names).
   - Upload a CSV file containing your movie data. Ensure it includes columns like `Scene_Dialogue` and `Scene_Characters`.

2. **Explore Features:**
   - **Word Cloud**: Analyze and visualize positive and negative words.
   - **Character Names**: View unique character names.
   - **Character Dialogue Counts**: See dialogue counts for each character.
   - **Character Scene Counts**: View scene counts for each character.
   - **Bar Graph on Dialogue Count**: Compare dialogue counts across characters.
   - **Bar Graph on Scene Count**: Compare scene counts across characters.
   - **Character Interactions**: Analyze character interactions.
   - **Character Relationships**: View relationships in a graphical format.
   - **Character Emotion Analysis**: Analyze character emotions.
   - **Text Emotion Analysis**: Analyze emotions from a given text.

## Dependencies

- `pandas`
- `re`
- `nltk`
- `wordcloud`
- `matplotlib`
- `numpy`
- `streamlit`
- `plotly`
- `networkx`
- `transformers`

## License

This project is licensed under the MIT License. See the [LICENSE](License) file for details.

## Contributing

We welcome contributions to this project! Please fork the repository and submit a pull request with your changes.



---
