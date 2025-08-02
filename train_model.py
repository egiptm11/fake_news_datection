import os
import re
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from collections import Counter
from wordcloud import WordCloud

# ML Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                            precision_recall_curve, PrecisionRecallDisplay,
                            ConfusionMatrixDisplay, accuracy_score)
from sklearn.preprocessing import LabelEncoder

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = DATA_DIR / "plots"

# Create directories
for directory in [DATA_DIR, MODEL_DIR, PLOT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Enhanced Indonesian stopwords
STOPWORDS = {
    'yang', 'dan', 'di', 'ada', 'adalah', 'sama', 'itu', 'ini', 'tidak', 'dengan',
    'juga', 'atau', 'akan', 'telah', 'untuk', 'pada', 'dari', 'dalam', 'ke', 'kepada'
}

def clean_text(text):
    """Advanced text cleaning"""
    if not isinstance(text, str) or pd.isna(text):
        return "[empty]"
    
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', ' ', text)
    text = re.sub(r'\[.*?\]|\(.*?\)|\<.*?\>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', text).strip() or "[empty]"

def save_plot(fig, filename):
    """Save plot with proper resource handling"""
    try:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        with open(filename, 'wb') as f:
            f.write(buf.read())
        # Set permissions
        os.chmod(filename, 0o644)
        return True
    except Exception as e:
        print(f"Error saving plot {filename}: {str(e)}")
        return False
    finally:
        plt.close(fig)
        buf.close()

def generate_wordcloud(text, filename, color_scheme):
    """Generate and save wordcloud"""
    try:
        wc = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap=color_scheme,
            stopwords=STOPWORDS,
            max_words=200
        ).generate(text)
        
        fig = plt.figure(figsize=(15,8))
        plt.imshow(wc)
        plt.axis("off")
        plt.tight_layout()
        return save_plot(fig, filename)
    except Exception as e:
        print(f"Error generating wordcloud: {str(e)}")
        return False

def train_and_evaluate():
    """Main training pipeline"""
    try:
        # Load data
        df = pd.read_csv(DATA_DIR / 'dataset.csv')
        df['cleaned_text'] = df['berita'].apply(clean_text)

        # Verify classes
        if len(df['kategori'].unique()) < 2:
            raise ValueError("Need both 'valid' and 'hoax' classes")

        # Prepare features
        le = LabelEncoder()
        y = le.fit_transform(df['kategori'])
        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words=list(STOPWORDS))
        X = tfidf.fit_transform(df['cleaned_text'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train models
        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), early_stopping=True)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl")

        # Save preprocessing
        joblib.dump(tfidf, MODEL_DIR / 'tfidf_vectorizer.pkl')
        joblib.dump(le, MODEL_DIR / 'label_encoder.pkl')

        # Generate visualizations
        for label in ['hoax', 'valid']:
            subset = df[df['kategori'] == label]
            text = ' '.join(subset['cleaned_text'])
            if text.strip():
                generate_wordcloud(text, PLOT_DIR / f'{label}_wordcloud.png', 
                                 'Reds' if label == 'hoax' else 'Greens')

        return True
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    train_and_evaluate()