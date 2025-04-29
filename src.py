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

# Drop ID columns if any (or store them separately if needed for final output)
id_col = 'ReviewID' if 'ReviewID' in train_with_scores.columns else None

# Fill missing text fields
train_with_scores['reviewText'] = train_with_scores['reviewText'].fillna('')
train_with_scores['summary'] = train_with_scores['summary'].fillna('')

# Combine reviewText and summary for richer text features
train_with_scores['full_text'] = train_with_scores['summary'] + ' ' + train_with_scores['reviewText']
prediction_set['reviewText'] = prediction_set['reviewText'].fillna('')
prediction_set['summary'] = prediction_set['summary'].fillna('')
prediction_set['full_text'] = prediction_set['summary'] + ' ' + prediction_set['reviewText']

# TF-IDF vectorization
tfidf = TfidfVectorizer(
    max_features=1000,  # limit features for speed
    stop_words='english'
)

X_text = tfidf.fit_transform(train_with_scores['full_text'])
X_pred_text = tfidf.transform(prediction_set['full_text'])

# Add numeric features (e.g., review length)
train_with_scores['review_length'] = train_with_scores['full_text'].apply(lambda x: len(x.split()))
prediction_set['review_length'] = prediction_set['full_text'].apply(lambda x: len(x.split()))

from scipy.sparse import hstack

# Final training matrices
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_num = scaler.fit_transform(train_with_scores[['review_length']])
X_pred_num = scaler.transform(prediction_set[['review_length']])

# Combine text and numeric features
X_train = hstack([X_text, X_num])
X_pred = hstack([X_pred_text, X_pred_num])

# Target variable
y_train = train_with_scores['Score']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import loguniform

# Initialize a basic logistic regression
lr = LogisticRegression(max_iter=1000, random_state=42)

# (Optional) Small hyperparameter tuning for Logistic Regression
param_distributions = {
    'C': loguniform(0.01, 10),  # Random search over C from 0.01 to 10
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # liblinear supports both l1 and l2
}

random_search = RandomizedSearchCV(
    estimator=lr,
    param_distributions=param_distributions,
    n_iter=15,            # number of random hyperparameters
    scoring='f1_macro',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit random search
random_search.fit(X_train, y_train)

print(f"Best Logistic Regression Params: {random_search.best_params_}")
print(f"Best Logistic Regression Macro-F1: {random_search.best_score_:.4f}")

# Best model
best_model = random_search.best_estimator_

# Fit best model on full training set
best_model.fit(X_train, y_train)

# Predict on test set
test_preds = best_model.predict(X_pred)

test_df = pd.read_csv('test.csv')

# Create submission
submission = test_df[['id']].copy()
submission['Score'] = test_preds

submission.to_csv('submission.csv', index=False)

print("✅ Submission file 'submission.csv' created!")