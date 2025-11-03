import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import nltk
from nltk.tokenize import sent_tokenize
import textstat
import re  # For fallback

# Cloud/local compat (cache dirs for downloads, CPU device)
os.environ['TRANSFORMERS_CACHE'] = './cache/'
nltk.data.path.append('./nltk_data')
try:
    nltk.download('punkt', quiet=False)
except Exception as e:
    pass  # Handled in safe_sent_tokenize

# Repo-relative paths
DATA_DIR = "./data"
MODELS_DIR = "./models"

@st.cache_resource
def load_artifacts():
    """Load with pickle for robustness."""
    try:
        st.info("Loading artifacts... (first run ~2 min for models).")
        
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # CPU for local/cloud
        
        vectorizer_path = os.path.join(DATA_DIR, "vectorizer.pkl")
        if not os.path.exists(vectorizer_path):
            st.error(f"Missing {vectorizer_path}. Run notebook to generate.")
            return None, None, None, None, None
        vectorizer = joblib.load(vectorizer_path)
        
        clf_path = os.path.join(MODELS_DIR, "quality_model.pkl")
        if os.path.exists(clf_path):
            clf = joblib.load(clf_path)
        else:
            st.warning("No quality_model.pkl—using rules only.")
            clf = None
        
        # Pickle load (replaces CSV)
        ref_path = os.path.join(DATA_DIR, "reference_data.pkl")
        if not os.path.exists(ref_path):
            st.error(f"Missing {ref_path}. Run notebook to save pickle.")
            return None, None, None, None, None
        ref_dict = joblib.load(ref_path)
        reference_embeddings = ref_dict['embeddings']
        reference_urls = ref_dict['urls']
        
        st.success(f"Loaded! Dataset: {len(reference_urls)} pages, Embeddings shape: {reference_embeddings.shape}")
        return model, vectorizer, clf, reference_embeddings, reference_urls
    except Exception as e:
        st.error(f"Load failed: {str(e)}")
        st.info("Tips: Run seo_pipeline.ipynb to generate files. Check requirements.txt has torch.")
        return None, None, None, None, None

# Functions
def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = ' '.join(text.split())
    return text

def safe_sent_tokenize(text):
    try:
        return sent_tokenize(text)
    except:
        return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)

def get_top_keywords(tfidf_row, vectorizer, top_k=5):
    try:
        feature_names = vectorizer.get_feature_names_out()
        top_indices = tfidf_row.toarray().argsort()[0][-top_k:][::-1]
        return '|'.join([feature_names[i] for i in top_indices])
    except Exception as e:
        return "N/A (short text)"

def assign_quality(word_count, flesch_reading_ease):
    if word_count > 1500 and 50 <= flesch_reading_ease <= 70:
        return 'High'
    elif word_count < 500 or flesch_reading_ease < 30:
        return 'Low'
    return 'Medium'

def analyze_url(url, model, vectorizer, clf, reference_embeddings, reference_urls):
    result = {'url': url}
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        result['title'] = soup.title.string.strip() if soup.title else "No title"
        content_areas = soup.find_all(['p', 'article', 'main', 'div'], class_=['content', 'post-body', 'entry-content'])
        if not content_areas:
            content_areas = soup.find_all(['p'])[:20]
        body_text = ' '.join([area.get_text().strip() for area in content_areas if area.get_text().strip()])
        word_count = len(body_text.split())
        result['word_count'] = word_count
        
        clean_text = preprocess_text(body_text)
        result['sentence_count'] = len(safe_sent_tokenize(clean_text))
        flesch_reading_ease = textstat.flesch_reading_ease(clean_text) if clean_text else 0
        result['flesch_reading_ease'] = round(flesch_reading_ease, 2)
        
        tfidf_single = vectorizer.transform([clean_text])
        result['top_keywords'] = get_top_keywords(tfidf_single[0], vectorizer)
        
        emb_single = model.encode([clean_text], show_progress_bar=False, batch_size=1, device='cpu')
        emb_array = np.array(emb_single[0])
        
        result['quality_label'] = assign_quality(word_count, flesch_reading_ease)
        result['is_thin'] = word_count < 500
        
        if clf:
            pred_features = np.array([[word_count, result['sentence_count'], flesch_reading_ease]])
            result['model_quality'] = clf.predict(pred_features)[0]
        else:
            result['model_quality'] = result['quality_label']
            st.warning("Using rules—load model for ML pred.")
        
        if len(reference_embeddings) > 0:
            similarities = cosine_similarity([emb_array], reference_embeddings)[0]
            similar_to = [{'matched_url': reference_urls[i], 'similarity': round(float(sim), 3)} for i, sim in enumerate(similarities) if sim > 0.80]
            result['similar_to'] = similar_to
        else:
            result['similar_to'] = []
        
        return result
    except Exception as e:
        result['error'] = str(e)
        return result

# UI
st.set_page_config(page_title="SEO Analyzer", page_icon="🔍", layout="wide")
st.title("🔍 SEO Content Quality & Duplicate Detector")
st.markdown("Enter URL for SEO analysis: quality, thin content, keywords, duplicates vs. 81-page dataset.")

# Load
model, vectorizer, clf, reference_embeddings, reference_urls = load_artifacts()

col1, col2 = st.columns([3, 1])
with col1:
    url_input = st.text_input("Enter URL:", placeholder="https://example.com", help="http/https only")
with col2:
    if st.button("🔍 Analyze", type="primary"):
        if url_input.startswith(('http://', 'https://')) and model:
            with st.spinner("Analyzing... (scraping + ML)"):
                result = analyze_url(url_input, model, vectorizer, clf, reference_embeddings, reference_urls)
                if 'error' in result:
                    st.error(f"Analysis failed: {result['error']}")
                else:
                    st.success("Analysis complete!")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Word Count", result['word_count'])
                    col_b.metric("Readability", result['flesch_reading_ease'], "60-70 ideal")
                    col_c.metric("Quality", result['model_quality'])
                    
                    st.subheader("Thin Content?")
                    st.info("Yes – Expand for better SEO!" if result['is_thin'] else "No – Good length.")
                    
                    st.subheader("Top Keywords (TF-IDF)")
                    st.write(result['top_keywords'])
                    
                    st.subheader("Duplicate Risks")
                    if result['similar_to']:
                        for dup in result['similar_to']:
                            st.warning(f"⚠️ Similar to: {dup['matched_url']} (Similarity: {dup['similarity']})")
                    else:
                        st.success("✅ No duplicates (>0.80 similarity).")
                    
                    st.subheader("Full Results")
                    st.json(result)
        else:
            st.warning("Enter valid URL. Artifacts loaded? Run notebook if not.")

st.markdown("---")
st.caption("Dataset: 81 pages | Model F1: 0.81 | Built with Streamlit for SEO Assignment.")
