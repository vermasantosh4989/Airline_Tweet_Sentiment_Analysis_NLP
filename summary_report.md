# Airline Tweet Sentiment Analysis: Summary Report

## 1. Business Problem and Dataset

The main objective of this project was to analyse airline customer feedback on social media and automatically identify whether customers expressed positive, negative, or neutral opinions. Airlines receive a large number of tweets about flight delays, cancellations, customer service, staff behaviour, and travel experiences. Manually analysing these tweets can be time-consuming. Sentiment analysis can help airlines understand customer satisfaction, identify common complaints, and improve their services.

For this project, I used the **Airline Tweets Sentiment Dataset**, which contains customer tweets and sentiment labels. The dataset includes important columns such as `airline`, `airline_sentiment`, `negativereason`, and `text`.

## 2. Data Cleaning and Feature Engineering

The first step involved cleaning and preprocessing the tweets using Python, Pandas, and Regular Expressions. I converted text to lowercase, removed HTML tags, URLs, unnecessary special characters, and extra whitespace. I also handled missing values and duplicate records where required.

During preprocessing, I preserved important negative words such as "not" and "no" because removing them could change the meaning of a sentence.

I explored different text representation techniques, including Bag of Words (BoW), TF-IDF, and N-Grams. I also trained Word2Vec models using CBOW and Skip-gram to understand semantic relationships between words. PCA was used to visualise the Word2Vec embeddings.

## 3. Classification and Model Performance

For sentiment classification, I experimented with multiple approaches, including BoW, TF-IDF, TF-IDF with N-Grams, and averaged Word2Vec embeddings combined with Logistic Regression.

The best-performing model was **[Enter Best Model]**, using **[Enter Best Feature Method]**.

Its performance was:

- Accuracy: **[Enter Accuracy]%**
- Macro Precision: **[Enter Macro Precision]**
- Macro Recall: **[Enter Macro Recall]**
- Macro F1-score: **[Enter Macro F1]**

The majority-class baseline achieved **[Enter Baseline Accuracy]% accuracy**. Comparing the baseline with the trained models demonstrated why accuracy alone is insufficient for an imbalanced dataset. Macro F1-score provided a more balanced evaluation across sentiment classes.

## 4. Sentiment Approaches and POS Tagging

I compared three sentiment analysis approaches: a custom heuristic method, TextBlob, and the trained machine learning model. The heuristic approach used predefined positive and negative words, while TextBlob provided ready-made sentiment polarity analysis. The machine learning model learned patterns from the airline dataset.

**[Enter the best-performing approach and its actual macro F1-score.]**

I also used NLTK Part-of-Speech (POS) tagging to identify nouns in customer complaints. The frequent nouns helped reveal common complaint topics, such as **[Enter your actual top complaint-related nouns]**.

## 5. Streamlit Application and Future Improvements

Finally, I developed an interactive Streamlit application named Airline Tweet Sentiment Studio. Users can enter airline tweets and receive predicted sentiment, probability scores, tweet insights, and POS tagging results.

In the future, I would improve the project by collecting more training data, supporting Hinglish and multilingual tweets, improving sarcasm detection, and experimenting with advanced Transformer models such as DistilBERT.

## Conclusion

This project provided practical experience in the complete Natural Language Processing workflow, from text preprocessing and feature extraction to model evaluation, POS tagging, and application deployment. It demonstrated how machine learning can help transform unstructured customer feedback into useful business insights.