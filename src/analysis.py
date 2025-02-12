import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.util import ngrams
from textstat import flesch_reading_ease

nltk.download('punkt')
nltk.download('stopwords')

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Preprocesses text by converting to lowercase, removing punctuation, and stopwords."""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(tokens)

def plot_word_frequencies(word_counts, top_n=20):
    """Plots the top N most common words."""
    most_common_words = word_counts.most_common(top_n)
    words, counts = zip(*most_common_words)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(words), y=list(counts), palette="viridis")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} Most Common Words")
    plt.xticks(rotation=45)
    plt.show()

def plot_tfidf_top_words(tfidf_matrix, feature_names, top_n=20):
    """Plots the top N words based on TF-IDF scores."""
    mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = np.argsort(mean_tfidf)[-top_n:]
    top_words = [feature_names[i] for i in top_indices]
    top_scores = [mean_tfidf[i] for i in top_indices]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_words, y=top_scores, palette="magma")
    plt.xlabel("Words")
    plt.ylabel("Average TF-IDF Score")
    plt.title(f"Top {top_n} Words by TF-IDF Score")
    plt.xticks(rotation=45)
    plt.show()

def plot_bow_distribution(bow_matrix):
    """Plots the distribution of word counts in the Bag-of-Words representation."""
    word_counts = np.array(bow_matrix.sum(axis=0)).flatten()
    plt.figure(figsize=(12, 6))
    sns.histplot(word_counts, bins=30, kde=True, color="blue", log_scale=True)
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.title("Distribution of Word Counts in BoW")
    plt.show()

def compute_nlp_statistics(df):
    """Computes and visualizes basic NLP statistics such as word frequency, TF-IDF, and Bag-of-Words."""
    df['processed_abstract'] = df['abstract'].apply(preprocess_text)
    
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(df['processed_abstract'])
    bow_feature_names = vectorizer.get_feature_names_out()
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_abstract'])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    word_counts = Counter(" ".join(df['processed_abstract']).split())
    
    # Generate visualizations
    plot_word_frequencies(word_counts)
    plot_tfidf_top_words(tfidf_matrix, tfidf_feature_names)
    plot_bow_distribution(bow_matrix)
    
    return {
        "bow_matrix": bow_matrix,
        "bow_feature_names": bow_feature_names,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_feature_names": tfidf_feature_names,
        "word_counts": word_counts
    }

def plot_nlp_statistics_per_venue(df):
    """Generates separate 4x3 grids of plots for BoW and TF-IDF statistics for the top 12 venues using Seaborn."""
    top_venues = df['venue'].value_counts().nlargest(12).index
    
    fig_bow, axes_bow = plt.subplots(4, 3, figsize=(32, 42))
    fig_tfidf, axes_tfidf = plt.subplots(4, 3, figsize=(32, 42))
    
    for idx, venue in enumerate(top_venues):
        row, col = divmod(idx, 3)
        venue_df = df[df['venue'] == venue]
        venue_df['processed_abstract'] = venue_df['abstract'].apply(preprocess_text)
        
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(venue_df['processed_abstract'])
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(venue_df['processed_abstract'])
        word_counts = Counter(" ".join(venue_df['processed_abstract']).split())
        
        # Get top 10 words for BoW
        most_common_words = word_counts.most_common(10)
        words, counts = zip(*most_common_words) if most_common_words else ([], [])
        
        # Get top 10 words for TF-IDF
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = np.argsort(mean_tfidf)[-10:]
        tfidf_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in top_indices]
        tfidf_scores = [mean_tfidf[i] for i in top_indices]
        
        # BoW Bar Plot
        sns.barplot(x=list(words), y=list(counts), ax=axes_bow[row, col], palette="Blues")
        axes_bow[row, col].set_title(f"BoW - {venue}")
        axes_bow[row, col].set_xticklabels(words, rotation=45)
        
        # TF-IDF Bar Plot
        sns.barplot(x=tfidf_words, y=tfidf_scores, ax=axes_tfidf[row, col], palette="Reds")
        axes_tfidf[row, col].set_title(f"TF-IDF - {venue}")
        axes_tfidf[row, col].set_xticklabels(tfidf_words, rotation=45)

    plt.show()
    
def univariate_analysis(df, column):
    if pd.api.types.is_numeric_dtype(df[column]):
        print(f"Summary Statistics for {column}:")
        print(df[column].describe())
        
        # Histogram
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df[column], kde=True, ax=ax[0])
        ax[0].set_title(f"Histogram of {column}")
        sns.boxplot(x=df[column], ax=ax[1])
        ax[1].set_title(f"Boxplot of {column}")
        plt.show()
        
    elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == 'object':
        print(f"Value Counts for {column}:")
        print(df[column].value_counts())
        
        # Visualization: Bar Chart
        plt.figure(figsize=(8, 5))
        sns.countplot(x=df[column], order=df[column].value_counts().index)
        plt.title(f"Bar Chart of {column}")
        plt.xticks(rotation=45)
        plt.show()
    return

def plot_correlation_matrix(df, figsize=(10, 8), cmap='coolwarm', annot=True, title='Correlation Matrix'):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap)
    plt.title(title)
    plt.show()
    return

def generate_ngrams(texts, n=2, top_n=10):
    """Generates the top N most frequent n-grams."""
    all_ngrams = []
    for text in texts:
        tokens = preprocess_text(text)
        all_ngrams.extend([" ".join(ngram) for ngram in ngrams(tokens, n)])
    
    ngram_counts = Counter(all_ngrams)
    return ngram_counts.most_common(top_n)

def plot_ngrams(df, n=2, top_n=10):
    """Plots the top N bigrams and trigrams smartly."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    bigrams = generate_ngrams(df['abstract'], n=2, top_n=top_n)
    trigrams = generate_ngrams(df['abstract'], n=3, top_n=top_n)
    
    if bigrams:
        words, counts = zip(*bigrams)
        sns.barplot(x=list(counts), y=list(words), ax=axes[0], palette="Blues")
        axes[0].set_title("Top Bigrams")
        axes[0].set_xlabel("Frequency")
    
    if trigrams:
        words, counts = zip(*trigrams)
        sns.barplot(x=list(counts), y=list(words), ax=axes[1], palette="Reds")
        axes[1].set_title("Top Trigrams")
        axes[1].set_xlabel("Frequency")
    
    plt.tight_layout()
    plt.show()

def abstract_length_distribution(texts):
    """Plots the distribution of abstract lengths."""
    lengths = [len(text.split()) for text in texts]
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=20, kde=True, color="purple")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.title("Abstract Length Distribution")
    plt.show()
    
def lexical_diversity(texts):
    """Computes the lexical diversity ratio (unique words / total words)."""
    all_words = [word for text in texts for word in preprocess_text(text)]
    return len(set(all_words)) / len(all_words) if len(all_words) > 0 else 0

def readability_scores(texts):
    """Computes readability scores (Flesch Reading Ease)."""
    return [flesch_reading_ease(text) for text in texts]

def plot_lexical_diversity(df):
    """Plots lexical diversity as a function of venues."""
    top_venues = df['venue'].value_counts().nlargest(12).index
    
    venue_stats = {
        "venue": [],
        "lexical_diversity": []
    }
    
    for venue in top_venues:
        venue_df = df[df['venue'] == venue]
        venue_stats["venue"].append(venue)
        venue_stats["lexical_diversity"].append(lexical_diversity(venue_df['abstract']))
    
    stats_df = pd.DataFrame(venue_stats)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=stats_df["lexical_diversity"], y=stats_df["venue"], palette="Blues")
    plt.xlabel("Lexical Diversity")
    plt.ylabel("Venue")
    plt.title("Lexical Diversity per Venue")
    plt.show()

def plot_readability_scores(df):
    """Plots readability scores as a function of venues."""
    top_venues = df['venue'].value_counts().nlargest(12).index
    
    venue_stats = {
        "venue": [],
        "readability": []
    }
    
    for venue in top_venues:
        venue_df = df[df['venue'] == venue]
        venue_stats["venue"].append(venue)
        venue_stats["readability"].append(np.mean(readability_scores(venue_df['abstract'])))
    
    stats_df = pd.DataFrame(venue_stats)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=stats_df["readability"], y=stats_df["venue"], palette="Reds")
    plt.xlabel("Readability Score")
    plt.ylabel("Venue")
    plt.title("Readability Score per Venue")
    plt.show()

