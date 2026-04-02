# Social Speech Classification on Twitter / X

This project focuses on the classification of Twitter / X data using Natural Language Processing (NLP) techniques. The goal is to process and analyze a dataset of tweets to distinguish between different types of speech, focusing on identifying offensive or hate speech through binary classification.

## 1. Repository Structure

* **social_speech_classification.ipynb**: A Jupyter Notebook containing the full pipeline for data loading, exploratory analysis, text preprocessing, and visualization.
* **twitter.csv**: The dataset containing 31,962 unique tweet entries with labels.

## 2. Dataset Description

The `twitter.csv` file consists of the following columns:

* **id**: Unique identifier for each tweet.
* **label**: Binary label (0 for positive/neutral tweets or 1 for offensive/hate speech).
* **tweet**: The raw text content of the Twitter post.

## 3. Requirements

To run the analysis, the following Python libraries are required:

* pandas
* nltk
* scikit-learn
* seaborn
* matplotlib
* wordcloud
* re (Regular Expressions)

You can install the necessary dependencies using:

```bash
pip install pandas nltk scikit-learn seaborn matplotlib wordcloud
```

## 4. Methodology

The notebook has the following structured workflow:

1. Data Observation

Initial exploration of the dataset using pandas to understand the distribution of labels and general data structure.
Text Preprocessing
* Lowercasing: Converting text to lowercase for uniformity.
* Cleaning: Removing URLs, user mentions (@user), and hashtags (#).
* Punctuation: Removing punctuation and special characters.
* Tokenization: Breaking down tweets into individual words.
* Stopword Removal: Filtering out common words (e.g., "the", "is", "and") that do not add significant meaning.
* Lemmatization: Reducing words to their base or root form using the NLTK WordNetLemmatizer.

2. Exploratory Data Analysis (EDA)

Visualizing the data using Seaborn and generating WordClouds to identify frequent terms in different speech categories.

3. Feature Extraction

Implementation of TfidfVectorizer to convert text data into numerical format (TF-IDF features) for model training.
