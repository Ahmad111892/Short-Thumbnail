# =================================================================================
# ULTIMATE AI THUMBNAIL ANALYZER V3.1 - STABLE EDITION
#
# This version has removed the 'rembg' library to ensure stable deployment
# on Streamlit Cloud, while retaining all other advanced analysis features.
# =================================================================================

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import colorsys
from sklearn.cluster import KMeans
import easyocr
import requests
from io import BytesIO
import hashlib
import time
import json
import google.generativeai as genai
import math
from collections import Counter

# --- AI Model Loading (Cached, Robust, and Centralized) ---

@st.cache_resource
def load_ai_models():
    """Loads all core AI models into memory once."""
    models = {}
    st.write("Cache miss: Loading core AI models...")

    # 1. YOLOv8 for Object Detection
    try:
        from ultralytics import YOLO
        models['yolo'] = YOLO('yolov8n.pt')
    except Exception as e:
        st.error(f"YOLOv8 Error: {e}"); models['yolo'] = None

    # 2. EasyOCR for Text Detection
    try:
        models['ocr'] = easyocr.Reader(['en', 'ur'], gpu=False, verbose=False)
    except Exception as e:
        st.error(f"EasyOCR Error: {e}"); models['ocr'] = None
        
    # 3. MediaPipe for Face Detection
    try:
        import mediapipe as mp
        models['face_detector'] = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    except Exception as e:
        st.error(f"MediaPipe Error: {e}"); models['face_detector'] = None
        
    # 4. Dlib for Facial Landmarks
    try:
        import dlib
        # Ensure 'shape_predictor_68_face_landmarks.dat' is in the root directory
        if os.path.exists('shape_predictor_68_face_landmarks.dat'):
            models['dlib_predictor'] = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        else:
            st.error("Dlib Error: `shape_predictor_68_face_landmarks.dat` not found. Please upload it.")
            models['dlib_predictor'] = None
    except Exception as e:
        st.error(f"Dlib Error: {e}")
        models['dlib_predictor'] = None

    # 5. Sentence Transformer for Semantic Coherence
    try:
        from sentence_transformers import SentenceTransformer
        models['semantic_model'] = SentenceTransformer('clip-ViT-B-32')
    except Exception as e:
        st.warning(f"SentenceTransformer Warning: Semantic Coherence will be disabled. Error: {e}")
        models['semantic_model'] = None
    
    st.success("✅ Core AI models loaded successfully!")
    return models

@st.cache_resource
def initialize_gemini(api_key):
    """Initializes the Google Gemini Pro Vision model."""
    if not api_key: return None
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
    st.set_page_config(page_title="🚀 Ultimate AI Thumbnail Analyzer", page_icon=":dart:", layout="wide")
    load_custom_css()
    st.markdown('<h1 class="main-header">🚀 Ultimate AI Thumbnail Analyzer 🎯</h1>', unsafe_allow_html=True)

    if 'analysis_results' not in st.session_state: st.session_state['analysis_results'] = None
    
    setup_sidebar()
    
    tab_list = ["🎯 AI Analysis", "🧠 Gemini Strategic Review", "🎨 Design Studio", "🏆 Competitor Insights"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_list)

    with tab1: ai_analysis_tab()
    with tab2: gemini_review_tab()
    with tab3: st.header("🏆 Competitor Insights"); st.warning("This feature is under development.")
    with tab4: design_studio_tab()

def setup_sidebar():
    with st.sidebar:
        st.header("🔑 AI Configuration")
        gemini_api_key = st.text_input("Google Gemini API Key", type="password", help="Required for Gemini Strategic Review")
        if gemini_api_key and 'gemini_model' not in st.session_state:
            st.session_state['gemini_model'] = initialize_gemini(gemini_api_key)
        st.markdown("---")
        st.header("⚙️ Analysis Settings")
        st.toggle("Emotion Detection (Slow)", value=True, key="enable_emotion")
        st.toggle("Object Detection", value=True, key="enable_object")
        st.toggle("Semantic Coherence", value=True, key="enable_semantic", help="Memory intensive, might be slow.")
        st.text_input("Uniqueness Check Keyword", key="uniqueness_keyword", placeholder="e.g., 'cricket highlights'")

def ai_analysis_tab():
    core_models = load_ai_models()
    st.markdown("### Upload Thumbnail for Ultimate Analysis")
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload Thumbnail File", type=['png', 'jpg', 'jpeg'])
            thumbnail_url = st.text_input("Or Paste Image URL")
            image, image_hash = None, None
            try:
                if uploaded_file: image = Image.open(uploaded_file).convert('RGB')
                elif thumbnail_url:
                    response = requests.get(thumbnail_url, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                if image: image_hash = hashlib.md5(image.tobytes()).hexdigest()
            except Exception as e:
                st.error(f"Error loading image: {e}"); return

            if image and image_hash != st.session_state.get('last_image_hash'):
                st.session_state['last_image_hash'] = image_hash
                if 'gemini_review' in st.session_state: del st.session_state['gemini_review']
                progress_bar = st.progress(0, "Initializing Ultimate Analysis...")
                st.session_state['analysis_results'] = perform_ultimate_analysis(image, core_models, progress_bar)
                progress_bar.empty()

            if st.session_state.get('analysis_results'):
                st.image(image, caption="Analyzed Thumbnail", width='stretch')

    with col2:
        if st.session_state.get('analysis_results'):
            display_analysis_results()
        else:
            st.info("Upload a thumbnail or paste a URL to begin analysis.")

def display_analysis_results():
    results = st.session_state['analysis_results']
    final_scores = results['final_scores']
    overall_score = final_scores['overall_score']
    
    st.markdown(f'<div class="ultra-card {get_score_class(overall_score)}"><h2 style="font-size:1.5rem;">Ultimate Performance Score</h2><h1 style="font-size:4rem; margin:0;">{overall_score}</h1><p style="font-size:1.1rem; font-weight:600;">{get_score_description(overall_score)}</p></div>', unsafe_allow_html=True)
    st.subheader("📊 Core Metrics Breakdown")
    # Display new metrics from secret features
    metrics_to_show = {
        "🧠 Click Psychology": results['psychology']['psychology_score'],
        "👤 Emotional Impact": results['face']['emotional_impact'],
        "👀 Gaze Score": results['face']['gaze_score'],
        "💎 Uniqueness Score": results['uniqueness']['uniqueness_score'],
        "🎨 Color Score": results['color']['color_score'],
        "🏗️ Composition Score": results['composition']['composition_score'],
        "✨ Clarity & Quality": results['quality']['quality_score'],
        "📝 Text Score": results['text']['text_score'],
        "🤝 Semantic Coherence": results['coherence']['coherence_score'],
    }
    cols = st.columns(2)
    for i, (name, score) in enumerate(metrics_to_show.items()):
        with cols[i % 2]:
            st.markdown(f"**{name}**")
            st.progress(int(score), text=f"{int(score)}/100")

# --- Ultimate Analysis Pipeline ---
def perform_ultimate_analysis(image, models, progress_bar):
    img_array = np.array(image)
    results = {'image': image} # Store for later use
    def update_progress(val, text): progress_bar.progress(val, text=text)

    update_progress(10, "Analyzing Colors...")
    results['color'] = analyze_colors_advanced(img_array)
    update_progress(20, "Analyzing Faces, Emotion & Gaze...")
    results['face'] = analyze_faces_ultimate(img_array, models) if st.session_state.enable_emotion else {'emotional_impact': 10, 'face_count': 0, 'dominant_emotion': 'N/A', 'gaze_score': 50}
    update_progress(35, "Analyzing Text & Font...")
    results['text'] = analyze_text_advanced(img_array, models)
    update_progress(50, "Analyzing Composition...")
    results['composition'] = analyze_composition_advanced(img_array)
    update_progress(60, "Analyzing Technical Quality...")
    results['quality'] = analyze_quality_advanced(img_array)
    update_progress(70, "Analyzing Objects & Clarity...")
    results['object'] = analyze_objects_advanced(img_array, models) if st.session_state.enable_object else {'clarity_score': 50, 'key_objects': []}
    update_progress(80, "Analyzing Visual Flow & Negative Space...")
    results['design'] = analyze_design_flow(results, img_array)
    update_progress(90, "Analyzing Uniqueness & Coherence...")
    results['uniqueness'] = analyze_uniqueness(image, st.session_state.uniqueness_keyword)
    results['coherence'] = analyze_semantic_coherence(results, models) if st.session_state.enable_semantic else {'coherence_score': 50}
    update_progress(95, "Analyzing Click Psychology...")
    results['psychology'] = analyze_click_psychology(results)
    update_progress(100, "Calculating Final Scores...")
    results['final_scores'] = calculate_final_scores(results)
    
    return results

# --- Granular and "Secret" Analysis Functions ---

def analyze_faces_ultimate(img_array, models):
    """Ultimate Two-Stage Face Analysis: MediaPipe for detection, Dlib for Gaze, DeepFace for emotion."""
    face_detector = models.get('face_detector')
    dlib_predictor = models.get('dlib_predictor')
    if not face_detector or not dlib_predictor: return {'emotional_impact': 10, 'dominant_emotion': 'N/A', 'face_count': 0, 'gaze_score': 50}
    
    rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    mp_results = face_detector.process(rgb_image)
    if not mp_results.detections: return {'emotional_impact': 10, 'dominant_emotion': 'N/A', 'face_count': 0, 'gaze_score': 50}
        
    face_count = len(mp_results.detections)
    emotions, gaze_scores = [], []
    
    try:
        from deepface import DeepFace
        import dlib
        for detection in mp_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img_array.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            face_rect = dlib.rectangle(x, y, x + w, y + h)
            shape = dlib_predictor(rgb_image, face_rect)
            
            left_eye_y = (shape.part(37).y + shape.part(38).y + shape.part(40).y + shape.part(41).y) / 4
            right_eye_y = (shape.part(43).y + shape.part(44).y + shape.part(46).y + shape.part(47).y) / 4
            gaze_score = max(0, 100 - abs(left_eye_y - right_eye_y) * 10)
            gaze_scores.append(gaze_score)

            face_crop = img_array[y:y+h, x:x+w]
            emotion_analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, silent=True)
            if emotion_analysis: emotions.append(emotion_analysis[0]['dominant_emotion'])

    except Exception: pass

    dominant_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "Neutral"
    impact = 70
    if dominant_emotion in ['happy', 'surprise']: impact += 25
    if dominant_emotion in ['fear', 'sad', 'disgust']: impact -= 15

    return {
        'emotional_impact': min(100, impact), 'dominant_emotion': dominant_emotion.capitalize(),
        'face_count': face_count, 'gaze_score': np.mean(gaze_scores) if gaze_scores else 50
    }

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
    clarity_score = max(0, 100 - (obj_count * 8))
    key_objects = [preds[0].names[int(c)] for c in preds[0].boxes.cls[:3]]
    return {'clarity_score': int(clarity_score), 'key_objects': key_objects}

def analyze_composition_advanced(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    balance_score = (100 - abs(np.mean(gray[:, :gray.shape[1]//2]) - np.mean(gray[:, gray.shape[1]//2:])))
    h, w = gray.shape
    thirds_interest = 0
    for x_ratio in [1/3, 2/3]:
        for y_ratio in [1/3, 2/3]:
            x, y = int(w * x_ratio), int(h * y_ratio)
            region = gray[max(0, y-15):min(h, y+15), max(0, x-15):min(w, x+15)]
            if region.size > 0: thirds_interest += np.var(region)
    thirds_score = min(100, thirds_interest / 1500)
    return {'composition_score': int(balance_score * 0.5 + thirds_score * 0.5)}

def analyze_colors_advanced(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    contrast = np.std(gray)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    saturation = np.mean(hsv[:, :, 1])
    color_score = min(100, (contrast / 2.55) * 0.6 + (saturation / 2.55) * 0.4)
    return {'color_score': int(color_score)}

def analyze_quality_advanced(img_array):
    clarity = cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    quality_score = min(100, clarity / 25)
    return {'quality_score': int(quality_score)}

def analyze_click_psychology(results):
    urgency_score = results['color']['color_score'] * 0.5 if results['color']['color_score'] > 60 else 30
    curiosity_score = results['object']['clarity_score'] * 0.4 + (100 - results['text']['text_score']) * 0.6
    psychology_score = urgency_score * 0.5 + curiosity_score * 0.5
    return {'psychology_score': int(psychology_score)}

def analyze_design_flow(results, img_array):
    # This is a placeholder for a more complex algorithm.
    # A real implementation would analyze coordinates of faces, text, and objects.
    negative_space_score = results['object']['clarity_score'] # Use clarity as a proxy
    visual_flow_score = results['composition']['composition_score'] # Use composition as a proxy
    return {'negative_space_score': negative_space_score, 'visual_flow_score': visual_flow_score}

def analyze_uniqueness(image, keyword):
    if not keyword: return {'uniqueness_score': 50}
    try:
        from pytube import Search
        import imagehash
        
        s = Search(keyword)
        competitor_hashes = []
        for v in s.results[:10]:
            try:
                response = requests.get(v.thumbnail_url, timeout=5)
                if response.status_code == 200:
                    competitor_image = Image.open(BytesIO(response.content))
                    competitor_hashes.append(imagehash.phash(competitor_image))
            except Exception: continue
            
        if not competitor_hashes: return {'uniqueness_score': 80} # High score if no competitors found

        own_hash = imagehash.phash(image)
        min_diff = min(abs(own_hash - ch) for ch in competitor_hashes)
        
        # Higher difference means more unique
        uniqueness_score = min(100, (min_diff / 64) * 200)
        return {'uniqueness_score': int(uniqueness_score)}
    except Exception:
        return {'uniqueness_score': 50}

def analyze_semantic_coherence(results, models):
    model = models.get('semantic_model')
    text = results['text'].get('full_text')
    image = results['image']
    if not model or not text: return {'coherence_score': 50}
    
    try:
        text_embedding = model.encode([text])
        image_embedding = model.encode(image)
        
        from sentence_transformers.util import cos_sim
        similarity = cos_sim(text_embedding, image_embedding).item()
        return {'coherence_score': int((similarity + 1) / 2 * 100)}
    except Exception:
        return {'coherence_score': 50}

def calculate_final_scores(results):
    """Calculates the final weighted score from all advanced metrics."""
    weights = {
        'emotion': 0.20, 'text': 0.15, 'clarity': 0.10, 'quality': 0.10,
        'color': 0.10, 'composition': 0.10, 'gaze': 0.10,
        'uniqueness': 0.05, 'coherence': 0.05, 'psychology': 0.05
    }
    
    score = (
        results['face']['emotional_impact'] * weights['emotion'] +
        results['face']['gaze_score'] * weights['gaze'] +
        results['text']['text_score'] * weights['text'] +
        results['object']['clarity_score'] * weights['clarity'] +
        results['quality']['quality_score'] * weights['quality'] +
        results['color']['color_score'] * weights['color'] +
        results['composition']['composition_score'] * weights['composition'] +
        results['uniqueness']['uniqueness_score'] * weights['uniqueness'] +
        results['coherence']['coherence_score'] * weights['coherence'] +
        results['psychology']['psychology_score'] * weights['psychology']
    )
    return {'overall_score': int(min(100, score))}

# --- Other Tabs & UI Helpers ---
def design_studio_tab():
    st.header("🎨 Design Studio")
    st.info("Analyze a thumbnail on the first tab to see personalized design tips here.")
def gemini_review_tab(): st.header("🧠 Gemini Strategic Review"); st.warning("Enter Gemini API key in sidebar.")
def competitor_insights_tab(): st.header("🏆 Competitor Insights"); st.warning("This feature is under development.")
def get_score_class(score):
    if score >= 85: return 'score-excellent'
    if score >= 70: return 'score-good'
    if score >= 50: return 'score-average'
    return 'score-poor'
def get_score_description(score):
    if score >= 85: return "Excellent! Optimized for high performance."
    if score >= 70: return "Good. Strong potential."
    if score >= 50: return "Average. Meets basic criteria."
    return "Poor. Needs significant improvements."

if __name__ == "__main__":
    main()
