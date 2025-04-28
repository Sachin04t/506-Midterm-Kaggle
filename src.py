import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import nltk
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_df = pd.read_csv('train.csv')

# Examine basic information
print(f"Train data shape: {train_df.shape}")
print(f"Number of reviews with missing scores: {train_df['Score'].isna().sum()}")

# Split into training (with scores) and prediction set (missing scores)
train_with_scores = train_df[~train_df['Score'].isna()].copy()
prediction_set = train_df[train_df['Score'].isna()].copy()

# Check distribution of scores in the training set
print(train_with_scores['Score'].value_counts().sort_index())

# Basic statistics
print(train_with_scores.describe())

# Check missing values
print(train_with_scores.isna().sum())

# Look at a few examples of reviews
print(train_with_scores[['reviewText', 'summary', 'Score']].sample(3))

# Check data types
print(train_with_scores.dtypes)

# Examine correlation between features
numeric_cols = train_with_scores.select_dtypes(include=[np.number]).columns
correlation = train_with_scores[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Check distribution of review lengths
train_with_scores['review_length'] = train_with_scores['reviewText'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
plt.figure(figsize=(10, 6))
sns.histplot(train_with_scores['review_length'], bins=50)
plt.title('Distribution of Review Lengths')
plt.xlabel('Number of Words')
plt.xlim(0, 500)  # Limiting to 500 words for better visualization
plt.show()

# Genre distribution
if 'genres' in train_with_scores.columns:
    # Create a list of all genres
    all_genres = []
    for genres in train_with_scores['genres'].dropna():
        all_genres.extend([g.strip() for g in str(genres).split(',')])
    
    # Count frequency of each genre
    from collections import Counter
    genre_counts = Counter(all_genres)
    
    # Plot top 20 genres
    plt.figure(figsize=(12, 8))
    top_genres = pd.DataFrame(genre_counts.most_common(20), columns=['Genre', 'Count'])
    sns.barplot(x='Count', y='Genre', data=top_genres)
    plt.title('Top 20 Genres')
    plt.show()

# Examine relationship between score and helpful votes
plt.figure(figsize=(10, 6))
sns.boxplot(x='Score', y='VotedHelpful', data=train_with_scores)
plt.title('Relationship Between Score and Helpful Votes')
plt.show()