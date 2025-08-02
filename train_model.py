import os
import re
import time
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from collections import Counter
from wordcloud import WordCloud

# Machine Learning Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, PrecisionRecallDisplay,
    ConfusionMatrixDisplay, accuracy_score
)
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = DATA_DIR / "plots"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, PLOT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Enhanced Indonesian stopwords
STOPWORDS = {
    'yang', 'dan', 'di', 'ada', 'adalah', 'sama', 'itu', 'ini', 'tidak', 'dengan',
    'juga', 'atau', 'akan', 'telah', 'untuk', 'pala', 'dari', 'dalam', 'ke', 'kepada',
    'katanya', 'konon', 'disebut', 'dituduh', 'dituding', 'dikabarkan', 'dilaporkan'
}

def clean_text(text):
    """Advanced text cleaning with hoax pattern handling"""
    if not isinstance(text, str) or pd.isna(text):
        return "[empty]"
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', ' ', text)
    # Remove special patterns common in fake news
    text = re.sub(r'\[.*?\]|\(.*?\)|\<.*?\>', ' ', text)
    # Remove punctuation and lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "[empty]"

def save_figure(fig, filename, dpi=300):
    """Save matplotlib figure with proper resource handling"""
    try:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        with open(filename, 'wb') as f:
            f.write(buf.read())
        logger.info(f"Saved figure: {filename}")
    except Exception as e:
        logger.error(f"Error saving figure {filename}: {str(e)}")
    finally:
        buf.close()
        plt.close(fig)

def generate_visualizations(df, models, X_test, y_test):
    """Generate all required visualizations with enhanced error handling"""
    
    # Verify we have data for both classes
    class_counts = df['kategori'].value_counts()
    if len(class_counts) < 2:
        logger.warning(f"Only {len(class_counts)} class(es) found. Need both 'valid' and 'hoax'.")
        return
    
    # 1. Class Distribution Plot
    try:
        fig, ax = plt.subplots(figsize=(8,6))
        class_counts.plot(kind='bar', color=['#4CAF50', '#F44336'], ax=ax)
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
        
        save_figure(fig, PLOT_DIR / 'class_distribution.png')
    except Exception as e:
        logger.error(f"Error generating class distribution: {str(e)}")

    # 2. Word Clouds
    for label in class_counts.index:
        try:
            color = 'Greens' if label == 'valid' else 'Reds'
            subset = df[df['kategori'] == label]
            text = ' '.join(subset['cleaned_text'])
            
            if len(text.strip()) == 0 or text == "[empty]":
                logger.warning(f"No text available for {label} word cloud")
                continue
                
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color='white',
                colormap=color,
                stopwords=STOPWORDS,
                max_words=200
            ).generate(text)
            
            fig = plt.figure(figsize=(15,8))
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.title(f'Most Frequent Words - {label.capitalize()} News', pad=20, fontsize=16)
            save_figure(fig, PLOT_DIR / f'{label}_wordcloud.png')
        except Exception as e:
            logger.error(f"Error generating {label} wordcloud: {str(e)}")

    # 3. Top Words Analysis
    for label in class_counts.index:
        try:
            subset = df[df['kategori'] == label]
            text = ' '.join(subset['cleaned_text'])
            
            if len(text.strip()) == 0:
                continue
                
            words = [word for word in text.split() if word not in STOPWORDS and word != "[empty]"]
            if not words:
                continue
                
            word_counts = Counter(words).most_common(20)
            words, counts = zip(*word_counts)
            
            fig = plt.figure(figsize=(10,6))
            plt.barh(words, counts, color='#F44336' if label == 'hoax' else '#4CAF50')
            plt.title(f'Top 20 Words - {label.capitalize()} News', pad=20)
            plt.xlabel('Frequency')
            plt.tight_layout()
            save_figure(fig, PLOT_DIR / f'top_{label}_words.png')
        except Exception as e:
            logger.error(f"Error generating top words for {label}: {str(e)}")

    # 4. Model Evaluation Visualizations
    for name, model in models.items():
        try:
            # Confusion Matrix
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(6,6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=class_counts.index)
            disp.plot(cmap='Blues', ax=ax)
            plt.title(f'Confusion Matrix - {name}')
            save_figure(fig, PLOT_DIR / f'cm_{name.lower().replace(" ", "_")}.png')
            
            # Precision-Recall Curve
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:,1]
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                
                fig, ax = plt.subplots(figsize=(8,6))
                disp = PrecisionRecallDisplay(precision=precision, recall=recall)
                disp.plot(ax=ax)
                plt.title(f'Precision-Recall Curve - {name}')
                save_figure(fig, PLOT_DIR / f'pr_{name.lower().replace(" ", "_")}.png')
        except Exception as e:
            logger.error(f"Error generating visuals for {name}: {str(e)}")

def train_and_evaluate():
    """Main training pipeline with enhanced error handling"""
    # Create training lock file
    lock_file = BASE_DIR / "training.lock"
    try:
        with open(lock_file, "w") as f:
            f.write(str(time.time()))
        
        logger.info("Starting model training process...")
        
        # 1. Data Loading
        logger.info("Loading dataset...")
        try:
            df = pd.read_csv(DATA_DIR / 'dataset.csv')
            logger.info(f"Loaded {len(df)} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise

        # 2. Data Validation
        logger.info("\n=== Data Validation ===")
        required_cols = ['berita', 'kategori']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Dataset must contain columns: {required_cols}")

        logger.info("Class distribution:\n" + str(df['kategori'].value_counts()))
        logger.info("Missing values:\n" + str(df.isna().sum()))

        if len(df['kategori'].unique()) < 2:
            raise ValueError("Need both 'valid' and 'hoax' classes in dataset")

        # 3. Preprocessing
        logger.info("Cleaning text...")
        df['cleaned_text'] = df['berita'].apply(clean_text)
        empty_texts = df['cleaned_text'].str.strip().eq('') | df['cleaned_text'].eq('[empty]')
        logger.info(f"Empty texts after cleaning: {empty_texts.sum()}")

        # 4. Feature Engineering
        logger.info("Creating TF-IDF features...")
        le = LabelEncoder()
        y = le.fit_transform(df['kategori'])  # 0=valid, 1=hoax
        
        tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2),
            stop_words=list(STOPWORDS))
        X = tfidf.fit_transform(df['cleaned_text'])
        
        # 5. Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # 6. Model Training
        models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42),
            'Naive Bayes': MultinomialNB(),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100,),
                early_stopping=True,
                random_state=42)
        }
        
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            try:
                model.fit(X_train, y_train)
                
                # Save model
                model_path = MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"Saved model to {model_path}")
                
                # Evaluate
                train_acc = accuracy_score(y_train, model.predict(X_train))
                test_acc = accuracy_score(y_test, model.predict(X_test))
                logger.info(f"{name} Accuracy - Train: {train_acc:.2f}, Test: {test_acc:.2f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue

        # 7. Save Preprocessing Artifacts
        joblib.dump(tfidf, MODEL_DIR / 'tfidf_vectorizer.pkl')
        joblib.dump(le, MODEL_DIR / 'label_encoder.pkl')
        logger.info("Saved preprocessing artifacts")

        # 8. Generate Visualizations
        logger.info("Generating visualizations...")
        generate_visualizations(df, models, X_test, y_test)

        logger.info("\n✅ Training completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        return False
    finally:
        # Create completion marker
        if lock_file.exists():
            if train_and_evaluate():
                with open(BASE_DIR / "training_complete.flag", "w") as f:
                    f.write("1")
            lock_file.unlink(missing_ok=True)

if __name__ == "__main__":
    train_and_evaluate()
