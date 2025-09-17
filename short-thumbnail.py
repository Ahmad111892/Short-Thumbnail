import streamlit as st
from PIL import Image
import numpy as np
import cv2
import colorsys
from sklearn.cluster import KMeans
import torch
import os
import requests
from io import BytesIO
import hashlib
import time
import google.generativeai as genai
import math

# --- AI Model Loading (Cached, Robust, and Centralized) ---

@st.cache_resource
def load_ai_models():
    """Loads all core AI models into memory once with comprehensive error handling."""
    models = {}
    st.write("Cache miss: Loading core AI models...")

    # 1. YOLOv8 for Object Detection
    try:
        from ultralytics import YOLO
        # This will auto-download yolov8n.pt if not present
        models['yolo'] = YOLO('yolov8n.pt')
    except Exception as e:
        st.error(f"Could not load YOLOv8 model. Object detection will be disabled. Error: {e}")
        models['yolo'] = None

    # 2. EasyOCR for Text Detection
    try:
        import easyocr
        # Using a broader language set for global reach
        models['ocr'] = easyocr.Reader(['en', 'ur'], gpu=False, verbose=False)
    except Exception as e:
        st.error(f"Could not load EasyOCR model. Text analysis will be limited. Error: {e}")
        models['ocr'] = None
        
    # 3. Saliency Model for User Attention
    models['saliency'] = cv2.saliency.StaticSaliencyFineGrained_create()
    
    st.success("✅ Core AI models loaded successfully!")
    return models

@st.cache_resource
def initialize_gemini(api_key):
    """Initializes the Google Gemini Pro Vision model."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro-vision')
        st.success("✅ Gemini AI Connected!")
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini AI: {e}")
        return None

# --- Advanced CSS Styling ---
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; font-size: 3.5rem; font-weight: 900;
        margin-bottom: 2rem; text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    }
    .ultra-card {
        background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px); padding: 25px; border-radius: 20px; color: white;
        text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin: 10px 0;
        transition: transform 0.3s ease-in-out;
    }
    .ultra-card:hover { transform: translateY(-5px); box-shadow: 0 15px 40px rgba(0,0,0,0.2); }
    .score-excellent { background: linear-gradient(135deg, #00c851, #007e33) !important; animation: pulse-green 2s infinite; }
    .score-good { background: linear-gradient(135deg, #2196F3, #0D47A1) !important; }
    .score-average { background: linear-gradient(135deg, #ffbb33, #ff8800) !important; }
    .score-poor { background: linear-gradient(135deg, #ff4444, #cc0000) !important; }
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(0, 200, 81, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0, 200, 81, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 200, 81, 0); }
    }
    .ai-insight {
        background: rgba(255,255,255,0.05); backdrop-filter: blur(5px);
        padding: 20px; border-radius: 15px; border-left: 5px solid #667eea;
        margin: 15px 0; font-weight: 400; box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .upload-area {
        border: 2px dashed #667eea; border-radius: 15px; padding: 30px; text-align: center;
        background: rgba(102, 126, 234, 0.05); margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Main Application ---

def main():
    load_custom_css()
    st.markdown('<h1 class="main-header">🚀 Ultimate AI Thumbnail Analyzer 🎯</h1>', unsafe_allow_html=True)

    # Initialize Session State
    if 'analysis_results' not in st.session_state: st.session_state['analysis_results'] = None
    if 'gemini_model' not in st.session_state: st.session_state['gemini_model'] = None
    if 'gemini_review' not in st.session_state: st.session_state['gemini_review'] = None

    setup_sidebar()
    
    tab_list = ["🎯 AI Analysis", "🧠 Gemini Strategic Review", "🏆 Competitor Insights", "🎨 Design Studio", "📈 Analytics Dashboard"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

    with tab1: ai_analysis_tab()
    with tab2: gemini_review_tab()
    with tab3: competitor_insights_tab()
    with tab4: design_studio_tab()
    with tab5: analytics_dashboard_tab()

def setup_sidebar():
    with st.sidebar:
        st.header("🔑 AI Configuration")
        gemini_api_key = st.text_input("Google Gemini API Key", type="password", help="Required for Gemini Strategic Review")
        
        if gemini_api_key and not st.session_state.get('gemini_model'):
            st.session_state['gemini_model'] = initialize_gemini(gemini_api_key)

        st.markdown("---")
        st.header("⚙️ Analysis Settings")
        st.selectbox("Analysis Depth", ["Standard", "Deep (Slower)"], key="analysis_depth")
        st.toggle("Emotion Detection", value=True, key="enable_emotion")
        st.toggle("Object Detection", value=True, key="enable_object")

def ai_analysis_tab():
    core_models = load_ai_models()
    
    st.markdown("### Upload Thumbnail for a Hyper-Detailed Analysis")
    col1, col2 = st.columns([1, 1.2]) # Adjusted column ratio
    
    with col1:
        with st.container(border=True):
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload Thumbnail File", type=['png', 'jpg', 'jpeg'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            thumbnail_url = st.text_input("Or Paste Image URL")
            
            image, image_hash = None, None
            
            try:
                if uploaded_file:
                    image = Image.open(uploaded_file).convert('RGB')
                elif thumbnail_url:
                    response = requests.get(thumbnail_url)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                
                if image: image_hash = hashlib.md5(image.tobytes()).hexdigest()
            except Exception as e:
                st.error(f"Error loading image: {e}"); return

            if image and image_hash != st.session_state.get('last_image_hash'):
                st.session_state['last_image_hash'] = image_hash
                st.session_state['gemini_review'] = None # Reset Gemini review for new image
                progress_bar = st.progress(0, "Initializing Analysis...")
                st.session_state['analysis_results'] = perform_ultimate_analysis(image, core_models, progress_bar)
                progress_bar.empty()

            if st.session_state.get('analysis_results'):
                st.image(image, caption="Analyzed Thumbnail", use_container_width=True)

    with col2:
        if st.session_state.get('analysis_results'):
            display_analysis_results()
        else:
            st.info("Upload a thumbnail or paste a URL to begin analysis.")
            st.image("https://i.imgur.com/gYv6T5d.png", caption="Example analysis dashboard")

def display_analysis_results():
    results = st.session_state['analysis_results']
    final_scores = results['final_scores']
    overall_score = final_scores['overall_score']
    
    st.markdown(f"""
    <div class="ultra-card {get_score_class(overall_score)}">
        <h2 style="font-size:1.5rem;">Ultimate Performance Score</h2>
        <h1 style="font-size:4rem; margin-top:0; margin-bottom:1rem;">{overall_score}</h1>
        <p style="font-size:1.1rem; font-weight:600;">{get_score_description(overall_score)}</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 Core Metrics Breakdown")
    metrics_to_show = {
        "🧠 Click Psychology": results['psychology_analysis']['psychology_score'],
        "👤 Emotional Impact": results['face_analysis']['emotional_impact_score'],
        "🎨 Color Score": results['color_analysis']['color_score'],
        "🏗️ Composition Score": results['composition_analysis']['composition_score'],
        "✨ Clarity & Quality": results['visual_analysis']['quality_score'],
        "📝 Text Score": results['text_analysis']['text_score'],
    }
    cols = st.columns(2)
    for i, (name, score) in enumerate(metrics_to_show.items()):
        with cols[i % 2]:
            st.markdown(f"**{name}**")
            st.progress(int(score), text=f"{int(score)}/100")

# --- Ultimate Analysis Pipeline ---

def perform_ultimate_analysis(image, models, progress_bar):
    img_array = np.array(image)
    results = {'image': image}

    def update_progress(val, text): progress_bar.progress(val, text=text)

    update_progress(10, "Analyzing Colors & Psychology...")
    results['color_analysis'] = analyze_colors_advanced(img_array)

    update_progress(25, "Analyzing Faces & Emotions...")
    results['face_analysis'] = analyze_faces_advanced(img_array) if st.session_state.enable_emotion else {'emotional_impact_score': 10, 'face_count': 0, 'dominant_emotion': 'N/A'}

    update_progress(40, "Analyzing Text...")
    results['text_analysis'] = analyze_text_advanced(img_array, models)

    update_progress(55, "Analyzing Composition & Aesthetics...")
    results['composition_analysis'] = analyze_composition_advanced(img_array)

    update_progress(70, "Analyzing Technical Quality...")
    results['visual_analysis'] = analyze_quality_advanced(img_array)

    update_progress(80, "Analyzing Objects & Clarity...")
    results['object_analysis'] = analyze_objects_advanced(img_array, models) if st.session_state.enable_object else {'clarity_score': 50, 'key_objects': []}
    
    update_progress(90, "Analyzing Click Psychology...")
    results['psychology_analysis'] = analyze_click_psychology(results)

    update_progress(100, "Calculating Final Scores...")
    results['final_scores'] = calculate_final_scores(results)
    
    return results

# --- Granular Analysis Functions ---

def analyze_faces_advanced(img_array):
    try:
        from deepface import DeepFace
        analysis = DeepFace.analyze(img_path=img_array, actions=['emotion'], enforce_detection=False, detector_backend='ssd')
        if analysis and isinstance(analysis, list):
            face = analysis[0]
            emotion = face['dominant_emotion']
            confidence = face['emotion'][emotion]
            impact = confidence
            if emotion in ['happy', 'surprise']: impact += 25
            if emotion in ['fear', 'sad']: impact -= 10
            return {'emotional_impact_score': min(100, impact), 'dominant_emotion': emotion.capitalize(), 'face_count': len(analysis)}
    except Exception: pass
    return {'emotional_impact_score': 10, 'dominant_emotion': 'N/A', 'face_count': 0}

def analyze_text_advanced(img_array, models):
    if not models.get('ocr'): return {'text_score': 30, 'word_count': 0, 'full_text': ""}
    results = models['ocr'].readtext(img_array)
    if not results: return {'text_score': 50, 'word_count': 0, 'full_text': ""}
    
    total_area = img_array.shape[0] * img_array.shape[1]
    text_area, word_count = 0, 0
    for res in results:
        points = res[0]
        word_count += len(res[1].split())
        w = max(p[0] for p in points) - min(p[0] for p in points)
        h = max(p[1] for p in points) - min(p[1] for p in points)
        text_area += w * h
    
    size_score = min(100, (text_area / total_area) * 750)
    word_penalty = max(0, word_count - 5) * 15
    text_score = max(0, size_score - word_penalty)
    return {'text_score': int(text_score), 'word_count': word_count, 'full_text': " ".join([r[1] for r in results])}

def analyze_objects_advanced(img_array, models):
    if not models.get('yolo'): return {'clarity_score': 50, 'key_objects': []}
    preds = models['yolo'].predict(img_array, verbose=False)
    obj_count = len(preds[0].boxes)
    clarity_score = max(0, 100 - (obj_count * 8)) # Less penalty per object
    key_objects = [preds[0].names[int(c)] for c in preds[0].boxes.cls[:3]]
    return {'clarity_score': int(clarity_score), 'key_objects': key_objects}

def analyze_composition_advanced(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    balance = (100 - abs(np.mean(gray[:, :gray.shape[1]//2]) - np.mean(gray[:, gray.shape[1]//2:])))
    thirds_interest = 0
    h, w = gray.shape
    for x_ratio in [1/3, 2/3]:
        for y_ratio in [1/3, 2/3]:
            x, y = int(w * x_ratio), int(h * y_ratio)
            thirds_interest += np.var(gray[y-15:y+15, x-15:x+15])
    thirds_score = min(100, thirds_interest / 1500)
    return {'composition_score': int(balance * 0.5 + thirds_score * 0.5)}

def analyze_colors_advanced(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    contrast = np.std(gray)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    saturation = np.mean(hsv[:, :, 1])
    color_score = min(100, (contrast / 2.55) * 0.6 + (saturation / 2.55) * 0.4)
    return {'color_score': int(color_score)}

def analyze_quality_advanced(img_array):
    clarity = cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    quality_score = min(100, clarity / 25) # Adjusted scaling
    return {'quality_score': int(quality_score)}

def analyze_click_psychology(results):
    # Urgency from colors (reds/oranges)
    urgency_score = results['color_analysis']['color_score'] * 0.5 if results['color_analysis']['color_score'] > 60 else 30
    # Curiosity from objects and text
    curiosity_score = results['object_analysis']['clarity_score'] * 0.4 + (100 - results['text_analysis']['text_score']) * 0.6
    psychology_score = urgency_score * 0.5 + curiosity_score * 0.5
    return {'psychology_score': int(psychology_score)}

def calculate_final_scores(results):
    weights = {'emotion': 0.30, 'text': 0.15, 'clarity': 0.20, 'quality': 0.10, 'color': 0.15, 'composition': 0.10}
    overall_score = (
        results['face_analysis']['emotional_impact_score'] * weights['emotion'] +
        results['text_analysis']['text_score'] * weights['text'] +
        results['object_analysis']['clarity_score'] * weights['clarity'] +
        results['visual_analysis']['quality_score'] * weights['quality'] +
        results['color_analysis']['color_score'] * weights['color'] +
        results['composition_analysis']['composition_score'] * weights['composition']
    )
    mobile_score = (results['text_analysis']['text_score'] * 0.6 + results['visual_analysis']['quality_score'] * 0.4)
    return {'overall_score': int(min(100, overall_score)), 'mobile_optimization_score': int(min(100, mobile_score))}

# --- Other Tabs ---

def gemini_review_tab():
    st.header("🧠 Gemini Strategic Review")
    if not st.session_state.get('gemini_model'):
        st.warning("Please enter your Google Gemini API Key in the sidebar to enable this feature.")
        return
    if not st.session_state.get('analysis_results'):
        st.info("Analyze a thumbnail on the first tab to generate a Gemini review.")
        return

    if st.button("✨ Generate Gemini Strategic Review", use_container_width=True, key="gemini_button"):
        with st.spinner("Gemini AI is crafting your strategic review..."):
            results = st.session_state['analysis_results']
            data_summary = f"""
            Quantitative Analysis Summary:
            - Overall Score: {results['final_scores']['overall_score']}/100
            - Dominant Emotion: {results['face_analysis']['dominant_emotion']} (Impact: {results['face_analysis']['emotional_impact_score']}/100)
            - Text Found: "{results['text_analysis']['full_text']}" (Word Count: {results['text_analysis']['word_count']})
            - Key Objects: {results['object_analysis']['key_objects']}
            - Clarity Score: {results['visual_analysis']['quality_score']}/100
            - Color Score: {results['color_analysis']['color_score']}/100
            """
            prompt = ["You are a world-class YouTube thumbnail strategist. Analyze the provided thumbnail image and its data summary. Provide a concise, expert review in markdown. Your review MUST include: 1. **Executive Summary:** A one-sentence powerful summary. 2. **Target Audience Analysis:** Who is this thumbnail for? 3. **Key Strengths:** 2-3 bullet points. 4. **Major Weaknesses:** 2-3 bullet points. 5. **Actionable Recommendations:** 3 specific, actionable tips to improve CTR.", data_summary, results['image']]
            
            try:
                response = st.session_state['gemini_model'].generate_content(prompt)
                st.session_state['gemini_review'] = response.text
            except Exception as e:
                st.error(f"Gemini API Error: {e}")

    if st.session_state.get('gemini_review'):
        st.markdown(st.session_state['gemini_review'])

def design_studio_tab():
    st.header("🎨 Design Studio & Best Practices"); st.info("General design principles for creating high-CTR thumbnails.")
def competitor_insights_tab():
    st.header("🏆 Competitor Insights"); st.warning("This feature is under development.")
def analytics_dashboard_tab():
    st.header("📈 Analytics Dashboard"); st.warning("This feature is under development.")
def ab_testing_tab():
    st.header("🔬 A/B Testing Ideas"); st.info("Analyze a thumbnail to get A/B testing ideas.")

# --- UI Helper Functions ---
def get_score_class(score):
    if score >= 85: return 'score-excellent'
    if score >= 70: return 'score-good'
    if score >= 50: return 'score-average'
    return 'score-poor'

def get_score_description(score):
    if score >= 85: return "Excellent! Optimized for high performance."
    if score >= 70: return "Good. Strong potential but can be improved."
    if score >= 50: return "Average. Meets basic criteria."
    return "Poor. Needs significant improvements."

if __name__ == "__main__":
    main()
