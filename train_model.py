import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                            precision_recall_curve, PrecisionRecallDisplay,
                            ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
from collections import Counter

# Configuration
os.makedirs('data/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Enhanced Indonesian stopwords
STOPWORDS = {
    'yang', 'dan', 'di', 'ada', 'adalah', 'sama', 'itu', 'ini', 'tidak', 'dengan',
    'juga', 'atau', 'akan', 'telah', 'untuk', 'pada', 'dari', 'dalam', 'ke', 'kepada',
    'katanya', 'konon', 'disebut', 'dituduh', 'dituding', 'dikabarkan', 'dilaporkan'
}

def clean_text(text):
    """Advanced text cleaning with hoax pattern handling"""
    if not isinstance(text, str) or pd.isna(text):
        return "[empty]"
    
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', ' ', text)
    text = re.sub(r'\[.*?\]|\(.*?\)|\<.*?\>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "[empty]"

def generate_visualizations(df, models, X_test, y_test):
    """Generate all required visualizations with safety checks"""
    
    # 1. Verify we have data for both classes
    class_counts = df['kategori'].value_counts()
    if len(class_counts) < 2:
        print(f"Warning: Only {len(class_counts)} class(es) found. Need both 'valid' and 'hoax'.")
        return
    
    # 2. Class Distribution
    plt.figure(figsize=(8,6))
    ax = class_counts.plot(kind='bar', color=['#4CAF50', '#F44336'])
    plt.title('Class Distribution', pad=20)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height()/total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() + 0.02 * total
        ax.annotate(percentage, (x, y), ha='center', fontsize=10)
    
    plt.savefig('data/plots/class_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Word Clouds (with content checks)
    for label in class_counts.index:
        color = 'Greens' if label == 'valid' else 'Reds'
        subset = df[df['kategori'] == label]
        text = ' '.join(subset['cleaned_text'])
        
        # Skip if no meaningful text
        if len(text.strip()) == 0 or text == "[empty]":
            print(f"Warning: No text available for {label} word cloud")
            continue
            
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap=color,
            stopwords=STOPWORDS,
            max_words=200
        ).generate(text)
        
        plt.figure(figsize=(15,8))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title(f'Most Frequent Words - {label.capitalize()} News', pad=20, fontsize=16)
        plt.savefig(f'data/plots/{label}_wordcloud.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    # 4. Top Words Analysis
    for label in class_counts.index:
        subset = df[df['kategori'] == label]
        text = ' '.join(subset['cleaned_text'])
        
        if len(text.strip()) == 0:
            continue
            
        words = [word for word in text.split() if word not in STOPWORDS and word != "[empty]"]
        if not words:
            continue
            
        word_counts = Counter(words).most_common(20)
        words, counts = zip(*word_counts)
        
        plt.figure(figsize=(10,6))
        plt.barh(words, counts, color='#F44336' if label == 'hoax' else '#4CAF50')
        plt.title(f'Top 20 Words - {label.capitalize()} News', pad=20)
        plt.xlabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'data/plots/top_{label}_words.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    # 5. Model Evaluation Visualizations
    for name, model in models.items():
        try:
            # Confusion Matrix
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(6,6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=class_counts.index)
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.savefig(f'data/plots/cm_{name.lower().replace(" ", "_")}.png', 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            # Precision-Recall Curve
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:,1]
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                plt.figure(figsize=(8,6))
                disp = PrecisionRecallDisplay(precision=precision, recall=recall)
                disp.plot()
                plt.title(f'Precision-Recall Curve - {name}')
                plt.savefig(f'data/plots/pr_{name.lower().replace(" ", "_")}.png', 
                           bbox_inches='tight', dpi=300)
                plt.close()
        except Exception as e:
            print(f"Error generating visuals for {name}: {str(e)}")

def train_and_evaluate():
    print("Loading and preprocessing data...")
    try:
        df = pd.read_csv('data/dataset.csv')
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Data Validation
    print("\n=== Data Validation ===")
    print(f"Total samples: {len(df)}")
    print("Columns:", df.columns.tolist())
    
    required_cols = ['berita', 'kategori']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Dataset must contain {required_cols}")
        return
    
    print("\nClass distribution:")
    print(df['kategori'].value_counts())
    
    print("\nMissing values:")
    print(df.isna().sum())
    
    # Preprocessing
    df['cleaned_text'] = df['berita'].apply(clean_text)
    
    print("\nAfter cleaning:")
    empty_texts = df['cleaned_text'].str.strip().eq('') | df['cleaned_text'].eq('[empty]')
    print(f"Empty texts: {empty_texts.sum()}")
    
    # Verify we have both classes
    if len(df['kategori'].unique()) < 2:
        print("Error: Need both 'valid' and 'hoax' classes")
        return
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['kategori'])  # 0=valid, 1=hoax
    
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        stop_words=list(STOPWORDS))
    X = tfidf.fit_transform(df['cleaned_text'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), early_stopping=True)
    }
    
    # Train models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(X_train, y_train)
            joblib.dump(model, f'models/{name.lower().replace(" ", "_")}.pkl')
            print(f"{name} trained successfully!")
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    # Save preprocessing artifacts
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(df, models, X_test, y_test)
    
    print("\n✅ Training completed!")
    print(f"Models saved to: {os.path.abspath('models')}")
    print(f"Visualizations saved to: {os.path.abspath('data/plots')}")

if __name__ == "__main__":
    train_and_evaluate()