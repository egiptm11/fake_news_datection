import os
import re
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from PIL import Image
import subprocess
import time
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay

# Config
st.set_page_config(layout="wide", page_title="Fake News Detector", page_icon="🔍")

# Training status tracker
TRAINING_LOCK_FILE = "training.lock"
TRAINING_COMPLETE_FILE = "training_complete.flag"

def check_training_status():
    if os.path.exists(TRAINING_LOCK_FILE):
        return "in_progress"
    elif os.path.exists(TRAINING_COMPLETE_FILE):
        return "complete"
    return "not_started"

def start_training():
    with open(TRAINING_LOCK_FILE, "w") as f:
        f.write(str(time.time()))
    
    try:
        subprocess.Popen(["python", "train_model.py"])
        return True
    except Exception as e:
        st.error(f"Failed to start training: {str(e)}")
        if os.path.exists(TRAINING_LOCK_FILE):
            os.remove(TRAINING_LOCK_FILE)
        return False

def show_training_ui():
    st.warning("Models not trained or need updating!")
    
    if st.button("🚀 Train Models Now"):
        if start_training():
            st.info("Model training in progress...")
            st.session_state.training_started = True
    
    if st.session_state.get("training_started"):
        with st.empty():
            while check_training_status() == "in_progress":
                st.write("⏳ Training models...")
                time.sleep(5)
            
            if check_training_status() == "complete":
                st.success("✅ Training complete! Models ready to use")
                os.remove(TRAINING_COMPLETE_FILE)
                st.rerun()
            else:
                st.error("❌ Training failed")

def models_exist():
    required_files = [
        'models/logistic_regression.pkl',
        'models/naive_bayes.pkl',
        'models/neural_network.pkl',
        'models/tfidf_vectorizer.pkl',
        'models/label_encoder.pkl'
    ]
    return all(os.path.exists(f) for f in required_files)

@st.cache_resource(ttl=3600)
def load_models():
    try:
        return {
            'models': {
                'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
                'Naive Bayes': joblib.load('models/naive_bayes.pkl'),
                'Neural Network': joblib.load('models/neural_network.pkl')
            },
            'vectorizer': joblib.load('models/tfidf_vectorizer.pkl'),
            'encoder': joblib.load('models/label_encoder.pkl')
        }
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None

def clean_text(text):
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    text = re.sub(r'\d+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict(text, model, vectorizer, encoder):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)
    proba = model.predict_proba(vectorized)[0]
    return encoder.inverse_transform(pred)[0], max(proba)

def show_header():
    st.title("🔍 Fake News Detection System")
    st.markdown("""
    <style>
    .stMarkdown > div > div {
        padding-bottom: 1rem;
        border-bottom: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

def show_sidebar():
    with st.sidebar:
        st.header("⚙️ Model Settings")
        model_option = st.selectbox(
            "Select Model:",
            ('Logistic Regression', 'Naive Bayes', 'Neural Network'),
            help="Select machine learning model for detection"
        )
        
        if st.button("🔄 Reload Models"):
            st.cache_resource.clear()
            st.success("Model cache cleared, reloading!")
            time.sleep(1)
            st.rerun()
        
        st.markdown("---")
        
        if st.button("🎯 Train New Models", help="Run model retraining"):
            if start_training():
                st.session_state.training_triggered = True
        
        if st.session_state.get("training_triggered"):
            st.info("Training started... check terminal for progress")
        
        st.markdown("---")
        st.info("""
        **Model Guide:**
        - **Logistic Regression**: Fast and interpretable
        - **Naive Bayes**: Optimal for text classification
        - **Neural Network**: High accuracy but needs more data
        """)

        return model_option

def show_model_metrics():
    with st.expander("📈 Model Evaluation Metrics", expanded=False):
        tabs = st.tabs(["Logistic Regression", "Naive Bayes", "Neural Network"])
        
        for i, model_name in enumerate(['logistic_regression', 'naive_bayes', 'neural_network']):
            with tabs[i]:
                col1, col2 = st.columns(2)
                
                with col1:
                    try:
                        st.image(f"data/plots/cm_{model_name}.png",
                               caption=f"Confusion Matrix",
                               use_container_width=True)
                    except:
                        st.warning("Confusion matrix not available")
                
                with col2:
                    try:
                        st.image(f"data/plots/pr_{model_name}.png",
                               caption=f"Precision-Recall Curve",
                               use_container_width=True)
                    except:
                        st.warning("PR curve not available")

def show_dataset_analysis():
    with st.expander("📊 Dataset Analysis", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Class Distribution", "Word Clouds", "Top Words"])
        
        with tab1:
            cols = st.columns([2, 3])
            with cols[0]:
                try:
                    st.image("data/plots/class_distribution.png",
                           use_container_width=True)
                except:
                    st.warning("Distribution chart not available")
            with cols[1]:
                try:
                    df = pd.read_csv('data/dataset.csv')
                    fig = px.pie(
                        names=df['kategori'].value_counts().index,
                        values=df['kategori'].value_counts().values,
                        title='News Category Distribution',
                        hole=0.3
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.warning("Pie chart not available")
        
        with tab2:
            tabs = st.tabs(["Fake News", "Real News"])
            with tabs[0]:
                try:
                    st.image("data/plots/hoax_wordcloud.png",
                           caption="Fake News",
                           use_container_width=True)
                except:
                    st.warning("Fake news wordcloud not available")
            with tabs[1]:
                try:
                    st.image("data/plots/valid_wordcloud.png",
                           caption="Real News",
                           use_container_width=True)
                except:
                    st.warning("Real news wordcloud not available")
        
        with tab3:
            cols = st.columns(2)
            with cols[0]:
                try:
                    st.image("data/plots/top_hoax_words.png",
                           caption="Fake News Top Words",
                           use_container_width=True)
                except:
                    st.warning("Fake news words not available")
            with cols[1]:
                try:
                    st.image("data/plots/top_valid_words.png",
                           caption="Real News Top Words",
                           use_container_width=True)
                except:
                    st.warning("Real news words not available")

def main():
    # Initialize session state
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False
    if 'training_triggered' not in st.session_state:
        st.session_state.training_triggered = False
    
    show_header()
    
    # Check model availability
    if not models_exist():
        show_training_ui()
        return
    
    # Load models
    assets = load_models()
    if assets is None:
        show_training_ui()
        return
    
    model_option = show_sidebar()
    
    # Prediction Section
    st.subheader("🔎 Analyze News")
    news_input = st.text_area(
        "Enter news text to analyze:", 
        height=200,
        placeholder="Example: 'COVID vaccines contain microchips...'",
        help="Enter at least 20 characters for optimal results"
    )
    
    col1, col2 = st.columns([3,1])
    with col2:
        predict_btn = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    if predict_btn:
        if not news_input or len(news_input.strip()) < 20:
            st.warning("Please enter longer text (minimum 20 characters)")
        else:
            with st.spinner("Analyzing text..."):
                try:
                    result, confidence = predict(
                        news_input,
                        assets['models'][model_option],
                        assets['vectorizer'],
                        assets['encoder']
                    )
                    
                    if result == 'hoax':
                        st.error(f"**Result:** FAKE ❌ (Confidence: {confidence:.2%})")
                    else:
                        st.success(f"**Result:** REAL ✅ (Confidence: {confidence:.2%})")
                    
                    with st.expander("🔍 Analysis Details"):
                        st.write(f"**Model used:** {model_option}")
                        st.write(f"**Analyzed text:**")
                        st.code(news_input[:500] + ("..." if len(news_input) > 500 else ""))
                        
                        st.write("**Other Model Comparisons:**")
                        comp_cols = st.columns(3)
                        for idx, (name, model) in enumerate(assets['models'].items()):
                            if name != model_option:
                                with comp_cols[idx-1]:
                                    res, conf = predict(news_input, model, assets['vectorizer'], assets['encoder'])
                                    st.metric(
                                        label=name,
                                        value=res.upper(),
                                        help=f"Confidence: {conf:.2%}"
                                    )
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    # Visualization Section
    st.markdown("---")
    show_model_metrics()
    st.markdown("---")
    show_dataset_analysis()

if __name__ == "__main__":
    main()