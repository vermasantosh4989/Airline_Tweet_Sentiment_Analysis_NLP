
import streamlit as st
import joblib
import pandas as pd
import re
import nltk

from nltk import word_tokenize, pos_tag
from text_preprocessing import clean_text


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Airline Sentiment Studio",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM CSS DESIGN
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F4F7FC;
    color: #172B4D;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: #F4F7FC;
}

[data-testid="stSidebar"] {
    background: #10213D;
}

[data-testid="stSidebar"] * {
    color: #EAF1FF;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #172B4D !important;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #10213D, #174A83, #2476C7);
    padding: 38px 42px;
    border-radius: 22px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 12px 30px rgba(16, 42, 80, 0.16);
}

.hero h1 {
    color: white !important;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #D9E8FF !important;
    font-size: 1rem;
    margin-top: 12px;
}

.section-title {
    color: #173B68;
    font-size: 1.35rem;
    font-weight: 750;
    margin: 24px 0 14px 0;
}

.info-card {
    background: white;
    border: 1px solid #E1EAF5;
    border-radius: 17px;
    padding: 22px;
    box-shadow: 0 5px 18px rgba(30, 65, 110, 0.06);
    min-height: 115px;
}

.info-label {
    color: #71829B;
    font-size: 0.83rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.info-value {
    color: #173B68;
    font-size: 1.55rem;
    font-weight: 800;
    margin-top: 10px;
}

.result-card {
    background: white;
    border: 1px solid #E1EAF5;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 8px 25px rgba(30, 65, 110, 0.08);
    text-align: center;
}

.result-label {
    color: #71829B;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.result-value {
    font-size: 2.1rem;
    font-weight: 800;
    margin-top: 12px;
}

div[data-testid="stProgress"] > div > div {
    background-color: #3182CE;
}

div.stButton > button {
    background: linear-gradient(90deg, #2563EB, #0EA5E9);
    color: white;
    border: none;
    border-radius: 11px;
    padding: 0.72rem 1.2rem;
    font-weight: 700;
    min-height: 45px;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #1D4ED8, #0284C7);
    color: white;
    border: none;
    transform: translateY(-1px);
}

div[data-testid="stTextArea"] textarea {
    background: white !important;
    color: #172B4D !important;
    border: 1px solid #D5E1F0 !important;
    border-radius: 12px !important;
}

div[data-testid="stTextArea"] label {
    color: #29486E !important;
    font-weight: 600;
}

div[data-testid="stExpander"] {
    background: white;
    border: 1px solid #E1EAF5;
    border-radius: 12px;
}

hr {
    border-color: #DCE6F2;
}

.footer {
    text-align: center;
    color: #8292A8;
    font-size: 0.8rem;
    padding-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("sentiment_pipeline.pkl")

    return model


model = load_model()


# =========================================================
# 4. DOWNLOAD NLTK RESOURCES
# =========================================================

@st.cache_resource
def download_nltk_resources():

    nltk.download("punkt", quiet=True)

    nltk.download("punkt_tab", quiet=True)

    nltk.download(
        "averaged_perceptron_tagger_eng",
        quiet=True
    )


download_nltk_resources()


# =========================================================
# 5. SENTIMENT LABEL FUNCTIONS
# =========================================================

def get_sentiment_name(label):

    # Mapping must match the labels used during model training
    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    if isinstance(label, str):

        return label.strip().capitalize()

    return label_map.get(int(label), str(label))


def get_sentiment_emoji(sentiment):

    emoji_map = {
        "Negative": "😞",
        "Neutral": "😐",
        "Positive": "😊"
    }

    return emoji_map.get(sentiment, "💬")


def get_sentiment_color(sentiment):

    color_map = {
        "Negative": "#DC5252",
        "Neutral": "#D69B26",
        "Positive": "#15966B"
    }

    return color_map.get(sentiment, "#2563EB")


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ✈️ Airline Studio")

    st.caption("CUSTOMER EXPERIENCE ANALYTICS")

    st.divider()

    st.markdown("### 📝 Try an Example")

    examples = {
        "Negative example":
        "My flight was delayed for five hours and the customer service was terrible.",

        "Neutral example":
        "My flight from Delhi to Mumbai is scheduled for tomorrow.",

        "Positive example":
        "Thank you for the excellent service. The staff was very helpful!"
    }

    selected_example = st.selectbox(
        "Choose a sample tweet",
        list(examples.keys())
    )

    if st.button("Use This Example", use_container_width=True):

        st.session_state["tweet_input"] = examples[selected_example]

    st.divider()

    st.markdown("### 📊 Project Details")

    st.write("**Domain:** Airline Customer Feedback")

    st.write("**Technology:** NLP and Machine Learning")

    st.write("**Model:** Logistic Regression")

    st.write("**Features:** TF-IDF / Text Vectorization")

    st.write("**Classes:** Negative, Neutral, Positive")

    st.divider()

    st.caption("NLP Airline Sentiment Analysis Project")


# =========================================================
# 7. MAIN HEADER
# =========================================================

hero_html = """<div class="hero">
<h1>✈️ Airline Tweet Sentiment Studio</h1>
<p>
Understand customer opinions, discover feedback patterns,
and analyse airline experiences using Natural Language Processing.
</p>
</div>"""

st.markdown(hero_html, unsafe_allow_html=True)


# =========================================================
# 8. CUSTOMER TWEET INPUT
# =========================================================

st.markdown(
    '<div class="section-title">💬 Analyse Customer Feedback</div>',
    unsafe_allow_html=True
)

tweet = st.text_area(
    "Enter an airline customer tweet",
    placeholder="Example: The flight was comfortable and the staff were very helpful...",
    height=135,
    key="tweet_input"
)

analyse_button = st.button(
    "🔍 Analyse Sentiment",
    use_container_width=True
)


# =========================================================
# 9. SENTIMENT PREDICTION
# =========================================================

if analyse_button:

    if not tweet.strip():

        st.warning("Please enter a tweet before analysing.")

    else:

        cleaned_tweet = clean_text(tweet)

        if not cleaned_tweet.strip():

            st.warning(
                "The entered text does not contain enough usable text."
            )

        else:

            # Predict sentiment
            prediction = model.predict([cleaned_tweet])[0]

            sentiment = get_sentiment_name(prediction)

            emoji = get_sentiment_emoji(sentiment)

            color = get_sentiment_color(sentiment)

            st.divider()

            st.markdown(
                '<div class="section-title">📊 Sentiment Analysis Result</div>',
                unsafe_allow_html=True
            )

            # Display sentiment result card
            result_html = f"""<div class="result-card">
<div class="result-label">PREDICTED CUSTOMER SENTIMENT</div>
<div class="result-value" style="color:{color};">
{emoji} {sentiment}
</div>
<p style="color:#71829B; margin-top:12px;">
The model has analysed the text and identified its overall sentiment.
</p>
</div>"""

            st.markdown(
                result_html,
                unsafe_allow_html=True
            )


            # =================================================
            # 10. SENTIMENT CONFIDENCE
            # =================================================

            if hasattr(model, "predict_proba"):

                # Calculate prediction probabilities
                probabilities = model.predict_proba([cleaned_tweet])[0]

                # Convert model classes into readable names
                probability_labels = [
                    get_sentiment_name(label)
                    for label in model.classes_
                ]

                # Create probability DataFrame
                probability_df = pd.DataFrame({
                    "Sentiment": probability_labels,
                    "Probability": probabilities
                })

                # Sort from highest to lowest probability
                probability_df = probability_df.sort_values(
                    by="Probability",
                    ascending=False
                )

                st.markdown(
                    '<div class="section-title">📈 Sentiment Confidence</div>',
                    unsafe_allow_html=True
                )

                st.markdown("**Prediction Probability**")

                # Display each sentiment using native Streamlit progress bars
                for _, row in probability_df.iterrows():

                    label = row["Sentiment"]

                    probability = float(row["Probability"])

                    percentage = probability * 100

                    # Display sentiment name and percentage
                    col1, col2 = st.columns([4, 1])

                    with col1:

                        st.write(f"**{label}**")

                    with col2:

                        st.write(f"**{percentage:.2f}%**")

                    # Display probability bar
                    st.progress(probability)


                # Display probability table
                st.markdown("**Probability Details**")

                display_df = probability_df.copy()

                display_df["Probability"] = (
                    display_df["Probability"] * 100
                )

                display_df = display_df.rename(
                    columns={
                        "Probability": "Probability (%)"
                    }
                )

                st.dataframe(
                    display_df.style.format({
                        "Probability (%)": "{:.2f}%"
                    }),
                    use_container_width=True,
                    hide_index=True
                )


            # =================================================
            # 11. TWEET INSIGHTS
            # =================================================

            mentions = re.findall(r"@\w+", tweet)

            hashtags = re.findall(r"#\w+", tweet)

            word_count = len(tweet.split())

            st.markdown(
                '<div class="section-title">🔎 Tweet Insights</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            # Word count card
            with col1:

                st.markdown(
                    f"""<div class="info-card">
<div class="info-label">Word Count</div>
<div class="info-value">{word_count}</div>
</div>""",
                    unsafe_allow_html=True
                )

            # Mentions card
            with col2:

                st.markdown(
                    f"""<div class="info-card">
<div class="info-label">Mentions</div>
<div class="info-value">{len(mentions)}</div>
</div>""",
                    unsafe_allow_html=True
                )

            # Hashtags card
            with col3:

                st.markdown(
                    f"""<div class="info-card">
<div class="info-label">Hashtags</div>
<div class="info-value">{len(hashtags)}</div>
</div>""",
                    unsafe_allow_html=True
                )

            if mentions:

                st.write("**Mentions:**", ", ".join(mentions))

            if hashtags:

                st.write("**Hashtags:**", ", ".join(hashtags))

            st.markdown("**Original Tweet**")

            st.info(tweet)


            # =================================================
            # 12. POS TAGGING ANALYSIS
            # =================================================

            st.markdown(
                '<div class="section-title">🔤 POS Tagging Analysis</div>',
                unsafe_allow_html=True
            )

            with st.expander("View Parts of Speech Analysis"):

                # Tokenize original tweet
                tokens = word_tokenize(tweet)

                # Apply POS tagging
                tagged_words = pos_tag(tokens)

                # POS tag meanings
                pos_meanings = {

                    "NN": "Noun (Singular)",
                    "NNS": "Noun (Plural)",
                    "NNP": "Proper Noun",
                    "NNPS": "Proper Noun (Plural)",

                    "VB": "Verb (Base Form)",
                    "VBD": "Verb (Past Tense)",
                    "VBG": "Verb (Gerund)",
                    "VBN": "Verb (Past Participle)",
                    "VBP": "Verb (Present)",
                    "VBZ": "Verb (Present Singular)",

                    "JJ": "Adjective",
                    "JJR": "Comparative Adjective",
                    "JJS": "Superlative Adjective",

                    "RB": "Adverb",
                    "RBR": "Comparative Adverb",
                    "RBS": "Superlative Adverb",

                    "PRP": "Personal Pronoun",
                    "PRP$": "Possessive Pronoun",

                    "IN": "Preposition",
                    "DT": "Determiner",
                    "CC": "Coordinating Conjunction",
                    "CD": "Number",
                    "UH": "Interjection",
                    "MD": "Modal Verb",
                    "TO": "To",
                    "RP": "Particle",

                    "WDT": "Wh-Determiner",
                    "WP": "Wh-Pronoun",
                    "WRB": "Wh-Adverb"

                }

                # Create POS tagging DataFrame
                tagging_df = pd.DataFrame(
                    tagged_words,
                    columns=["Word", "POS Tag"]
                )

                # Add tag meanings
                tagging_df["Meaning"] = tagging_df["POS Tag"].apply(
                    lambda tag: pos_meanings.get(tag, "Other")
                )

                st.write("**Word-by-word POS Tagging:**")

                st.dataframe(
                    tagging_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Extract nouns
                nouns = [
                    word
                    for word, tag in tagged_words
                    if tag in ["NN", "NNS", "NNP", "NNPS"]
                ]

                st.write("**Nouns identified in the tweet:**")

                if nouns:

                    st.success(", ".join(nouns))

                else:

                    st.info("No nouns were identified in this tweet.")

                # Display token and noun counts
                col4, col5 = st.columns(2)

                with col4:

                    st.metric(
                        "Total Tokens",
                        len(tokens)
                    )

                with col5:

                    st.metric(
                        "Total Nouns",
                        len(nouns)
                    )


# =========================================================
# 13. FOOTER
# =========================================================

st.markdown(
    """<div class="footer">
Built with Streamlit • Natural Language Processing • Machine Learning
<br>
Airline Tweet Sentiment Analysis
</div>""",
    unsafe_allow_html=True
)
