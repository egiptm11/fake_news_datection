import os
import re
import time
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from pathlib import Path
from PIL import Image
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay

# Config
st.set_page_config(layout="wide", page_title="Fake News Detector")

# Path setup
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "data" / "plots"

def load_models():
    """Load models with caching"""
    try:
        return {
            'models': {
                'Logistic Regression': joblib.load(MODEL_DIR / 'logistic_regression.pkl'),
                'Naive Bayes': joblib.load(MODEL_DIR / 'naive_bayes.pkl'),
                'Neural Network': joblib.load(MODEL_DIR / 'neural_network.pkl')
            },
            'vectorizer': joblib.load(MODEL_DIR / 'tfidf_vectorizer.pkl'),
            'encoder': joblib.load(MODEL_DIR / 'label_encoder.pkl')
        }
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None

def show_wordcloud(tab, image_name, caption):
    """Display wordcloud with error handling"""
    try:
        image_path = PLOT_DIR / image_name
        if not image_path.exists():
            tab.warning(f"Image not found: {image_path.name}")
            return
            
        # Use cache busting
        tab.image(str(image_path) + f"?{time.time()}",
                caption=caption,
                use_container_width=True)
    except Exception as e:
        tab.error(f"Failed to load image: {str(e)}")

def main():
    st.title("🔍 Fake News Detection System")
    
    # Load models
    assets = load_models()
    if not assets:
        st.error("Please train models first")
        return

    # Sidebar
    model_option = st.sidebar.selectbox(
        "Select Model:",
        ('Logistic Regression', 'Naive Bayes', 'Neural Network')
    )

    # Prediction
    st.subheader("🔎 Analyze News")
    news_input = st.text_area("Enter news text:", height=200)
    
    if st.button("Analyze"):
        if news_input and len(news_input) > 20:
            cleaned = re.sub(r'[^\w\s]', ' ', str(news_input).lower())
            vectorized = assets['vectorizer'].transform([cleaned])
            pred = assets['models'][model_option].predict(vectorized)
            result = assets['encoder'].inverse_transform(pred)[0]
            st.success(f"Result: {result.upper()}")

    # Visualizations
    st.markdown("---")
    with st.expander("📊 Word Clouds"):
        tab1, tab2 = st.tabs(["Fake News", "Real News"])
        with tab1:
            show_wordcloud(tab1, "hoax_wordcloud.png", "Fake News Word Cloud")
        with tab2:
            show_wordcloud(tab2, "valid_wordcloud.png", "Real News Word Cloud")

if __name__ == "__main__":
    main()