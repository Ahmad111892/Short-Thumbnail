import streamlit as st
from PIL import Image
import numpy as np
import cv2
import colorsys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
import easyocr
import os

# --- Model Loading (Cached for Performance) ---

@st.cache_resource
def load_easyocr_model():
    """This function loads the EasyOCR model and is cached to prevent reloading on every run."""
    # The model is downloaded automatically on the first run
    reader = easyocr.Reader(['en']) # 'en' for English language
    return reader

@st.cache_resource
def load_haar_cascade():
    """Loads the Haar Cascade file and is cached."""
    # Check if the file exists before loading
    if os.path.exists('haarcascade_frontalface_default.xml'):
        return cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    else:
        st.error("Fatal Error: `haarcascade_frontalface_default.xml` not found. Please upload it to your GitHub repository.")
        return None

# --- Main Application ---

def main():
    st.set_page_config(
        page_title="🚀 AI Thumbnail Analyzer Pro",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Modern CSS Styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5rem; font-weight: 900;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 20px; border-radius: 15px;
        color: white; text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .score-high { background: linear-gradient(135deg, #00c851, #007e33) !important; }
    .score-medium { background: linear-gradient(135deg, #ffbb33, #ff8800) !important; }
    .score-low { background: linear-gradient(135deg, #ff4444, #cc0000) !important; }
    .ai-insight {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px; border-radius: 10px;
        color: white; margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🚀 AI THUMBNAIL ANALYZER PRO 🎯</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Billion Dollar Level YouTube Thumbnail Intelligence</p>', unsafe_allow_html=True)
    
    setup_api_sidebar()
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 AI Analysis", 
        "📊 Performance Prediction", 
        "🏆 Competitor Analysis", 
        "🎨 Design Optimization"
    ])
    
    with tab1:
        ai_analysis_tab()
    with tab2:
        performance_prediction_tab()
    with tab3:
        competitor_analysis_tab()
    with tab4:
        design_optimization_tab()

def setup_api_sidebar():
    with st.sidebar:
        st.header("🔑 API Configuration")
        st.text_input("YouTube Data API Key", type="password")
        st.text_input("Google Gemini AI API Key", type="password")
        st.markdown("---")
        st.header("⚙️ Analysis Settings")
        st.selectbox("Analysis Depth", ["Quick Scan", "Deep Analysis"], index=1)

def ai_analysis_tab():
    st.header("🎯 AI-Powered Thumbnail Analysis")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📤 Upload Your Thumbnail",
            type=['png', 'jpg', 'jpeg', 'webp']
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Analyze ONLY if it's a new file and save to session state
            if st.session_state.get('last_uploaded_name') != uploaded_file.name:
                with st.spinner('🤖 AI is analyzing your thumbnail... Please wait.'):
                    st.session_state['analysis_results'] = perform_ai_analysis(image)
                    st.session_state['last_uploaded_name'] = uploaded_file.name
            
            st.image(image, caption="Uploaded Thumbnail", use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Resolution", f"{image.width}x{image.height}")
            c2.metric("File Size", f"{len(uploaded_file.getvalue()) / 1024:.1f} KB")

    with col2:
        if st.session_state.get('analysis_results'):
            results = st.session_state['analysis_results']
            overall_score = results['overall_score']
            st.markdown(f"""
            <div class="metric-card {get_score_class(overall_score)}">
                <h2>Overall Thumbnail Score</h2>
                <h1>{overall_score}/100</h1>
                <p>{get_score_description(overall_score)}</p>
            </div>
            """, unsafe_allow_html=True)
            display_detailed_metrics(results)
        else:
            st.info("👈 Upload a thumbnail to begin the AI analysis.")

# --- Core Analysis Functions ---

def perform_ai_analysis(image):
    img_array = np.array(image)
    
    color_analysis = analyze_colors(img_array)
    face_analysis = detect_faces(img_array)
    text_analysis = detect_text_regions(img_array)
    composition_analysis = calculate_composition(img_array)
    visual_appeal = calculate_visual_appeal(img_array)
    
    click_potential = predict_click_potential(
        color_analysis, face_analysis, text_analysis, 
        composition_analysis, visual_appeal
    )
    
    overall_score = calculate_overall_score(
        color_analysis, face_analysis, text_analysis,
        composition_analysis, visual_appeal, click_potential
    )
    
    return {
        'overall_score': overall_score,
        'color_analysis': color_analysis,
        'face_analysis': face_analysis,
        'text_analysis': text_analysis,
        'composition_analysis': composition_analysis,
        'visual_appeal': visual_appeal,
        'click_potential': click_potential,
        'recommendations': generate_ai_recommendations(overall_score, color_analysis, face_analysis, text_analysis)
    }

def detect_faces(img_array):
    """REAL face detection using OpenCV's Haar Cascade Classifier."""
    face_cascade = load_haar_cascade()
    if face_cascade is None:
        return {'faces_count': 0, 'face_sizes': [], 'face_positions': [], 'face_quality': 0, 'face_score': 0}

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    faces_found = len(faces)
    face_sizes, face_positions = [], []
    img_height, img_width = img_array.shape[:2]
    
    for (x, y, w, h) in faces:
        face_sizes.append((w * h) / (img_width * img_height))
        face_positions.append(((x + w / 2) / img_width, (y + h / 2) / img_height))
            
    face_quality = assess_face_quality(faces_found, face_sizes)
    return {
        'faces_count': faces_found, 'face_sizes': face_sizes, 
        'face_positions': face_positions, 'face_quality': face_quality,
        'face_score': min(100, face_quality * 100)
    }

def detect_text_regions(img_array):
    """REAL text detection and analysis using EasyOCR."""
    try:
        reader = load_easyocr_model()
        results = reader.readtext(img_array)
        
        if not results:
            return {'text_regions_count': 0, 'text_coverage': 0, 'readability_score': 0.5, 'text_score': 50}

        text_regions, total_text_area = [], 0
        for (bbox, text, prob) in results:
            top_left, _, bottom_right, _ = bbox
            x, y = int(top_left[0]), int(top_left[1])
            w, h = int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])
            area = abs(w * h)
            total_text_area += area
            text_regions.append({'position': (x, y, w, h), 'area': area})

        readability_score = calculate_text_readability(text_regions, img_array.shape)
        
        word_count = sum(len(res[1].split()) for res in results)
        if word_count > 6: readability_score *= 0.8 # Penalty for too many words
        
        return {
            'text_regions_count': len(results),
            'text_coverage': total_text_area / (img_array.shape[0] * img_array.shape[1]),
            'readability_score': readability_score, 'text_score': min(100, readability_score * 100)
        }
    except Exception as e:
        st.warning(f"Text detection (EasyOCR) failed. It may be downloading models. Please wait and try again. Error: {e}")
        return {'text_regions_count': 0, 'text_coverage': 0, 'readability_score': 0.5, 'text_score': 50}

def analyze_colors(img_array):
    pixels = img_array.reshape(-1, 3)
    data = pixels
    if len(data) > 20000:
        indices = np.random.choice(len(data), 20000, replace=False)
        data = data[indices]
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    kmeans.fit(data)
    dominant_colors = kmeans.cluster_centers_.astype(int).tolist()

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    harmony = calculate_color_harmony(dominant_colors)
    contrast = np.std(gray)
    saturation = np.mean(hsv[:, :, 1])
    
    return {
        'dominant_colors': dominant_colors, 'harmony_score': harmony,
        'contrast': contrast, 'saturation': saturation,
        'color_score': min(100, (harmony * 0.3 + min(contrast / 50, 1) * 0.4 + min(saturation / 200, 1) * 0.3) * 100)
    }

# --- Scoring and Helper Functions ---
def calculate_composition(img_array):
    height, width, _ = img_array.shape
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    third_w, third_h = width // 3, height // 3
    interest_points = [(third_w, third_h), (2 * third_w, third_h), (third_w, 2 * third_h), (2 * third_w, 2 * third_h)]
    rule_of_thirds_score = 0
    for x, y in interest_points:
        region = gray[max(0, y - 20):min(height, y + 20), max(0, x - 20):min(width, x + 20)]
        if region.size > 0: rule_of_thirds_score += np.std(region)
    rule_of_thirds_score = min(1.0, rule_of_thirds_score / (4 * 70))

    left_half_mean = np.mean(gray[:, :width//2])
    right_half_mean = np.mean(gray[:, width//2:])
    balance_score = 1 - abs(left_half_mean - right_half_mean) / 255

    return { 'composition_score': min(100, (rule_of_thirds_score * 0.6 + balance_score * 0.4) * 100) }

def calculate_visual_appeal(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 50, 150)
    clarity = 1 - (np.sum(edges > 0) / edges.size) / 0.2
    vibrancy = np.mean(hsv[:, :, 1]) / 255
    return {'appeal_score': min(100, (clarity * 0.5 + vibrancy * 0.5) * 100)}

def predict_click_potential(color, face, text, composition, appeal):
    weights = {'faces': 0.30, 'colors': 0.25, 'text': 0.15, 'composition': 0.15, 'appeal': 0.15}
    potential = (face['face_score'] * weights['faces'] + color['color_score'] * weights['colors'] +
                 text['text_score'] * weights['text'] + composition['composition_score'] * weights['composition'] +
                 appeal['appeal_score'] * weights['appeal'])
    if face['faces_count'] > 0: potential += 10
    if color['contrast'] > 50: potential += 5
    return min(100, potential)

def calculate_overall_score(color, face, text, composition, appeal, potential):
    scores = [color['color_score'], face['face_score'], text['text_score'], 
              composition['composition_score'], appeal['appeal_score'], potential]
    weights = [0.15, 0.20, 0.15, 0.15, 0.15, 0.20]
    return int(min(100, sum(s * w for s, w in zip(scores, weights))))

def generate_ai_recommendations(score, color, face, text):
    recs = []
    if score < 60: recs.append("🚨 **Major improvements needed.** Focus on basics: clear subject, high contrast, and readable text.")
    elif score < 80: recs.append("⚠️ **Good thumbnail, but can be better.** Try increasing color vibrancy or improving composition.")
    else: recs.append("✅ **Excellent thumbnail!** Looks professional and is likely to perform well.")

    if face['faces_count'] == 0: recs.append("👤 **Consider adding a human face.** Thumbnails with faces often get higher click-through rates.")
    if color['contrast'] < 40: recs.append("🎨 **Increase contrast.** Make your main subject 'pop' from the background.")
    if text['text_score'] < 60: recs.append("📝 **Improve text readability.** Use fewer words, a larger font, and high-contrast colors for your text.")
    return recs

def display_detailed_metrics(results):
    st.subheader("📊 Detailed Score Breakdown")
    metrics = [
        ("Color Impact", results['color_analysis']['color_score']),
        ("Face Presence", results['face_analysis']['face_score']),
        ("Text Readability", results['text_analysis']['text_score']),
        ("Composition", results['composition_analysis']['composition_score']),
        ("Visual Appeal", results['visual_appeal']['appeal_score']),
        ("Click Potential", results['click_potential'])
    ]
    for name, score in metrics:
        st.markdown(f"""
        <div class="metric-card {get_score_class(score)}" style="margin: 5px 0; padding: 10px;">
            <h4>{name}</h4><h3>{score:.0f}/100</h3>
        </div>""", unsafe_allow_html=True)
    st.subheader("💡 AI Recommendations")
    for rec in results['recommendations']:
        st.markdown(f'<div class="ai-insight">{rec}</div>', unsafe_allow_html=True)

# --- Other Tabs and Helper Functions ---
def performance_prediction_tab():
    st.header("📈 Performance Prediction")
    if not st.session_state.get('analysis_results'):
        st.info("Analyze a thumbnail first to see predictions.")
        return
    score = st.session_state['analysis_results']['overall_score']
    predicted_ctr = 2.0 + (score / 100) ** 3 * 15
    st.metric("Predicted Click-Through Rate (CTR)", f"{predicted_ctr:.2f}%")
    st.bar_chart({'Your Thumbnail': predicted_ctr, 'Average (3%)': 3.0, 'Good (7%)': 7.0, 'Excellent (12%)': 12.0})

def competitor_analysis_tab():
    st.header("🏆 Competitor Analysis")
    st.info("This is a demo feature. A full version would use the YouTube API to analyze competitor thumbnails.")
    
def design_optimization_tab():
    st.header("🎨 Design Optimization Studio")
    st.subheader("Quick Wins to Boost Your Score")
    st.markdown("""
    - **Rule of Thirds:** Place your main subject off-center. 
    - **Color Psychology:** Use Red/Orange for excitement, Blue for trust.
    - **Font Choice:** Use a bold, clean, sans-serif font (like Impact).
    - **Add an Outline:** A white or black outline around subjects or text makes them 'pop'.
    - **Less is More:** Don't clutter the thumbnail. One clear subject is best.
    """)

def get_score_class(score):
    if score >= 80: return 'score-high'
    if score >= 60: return 'score-medium'
    return 'score-low'

def get_score_description(score):
    if score >= 90: return 'Exceptional! A top-tier thumbnail.'
    if score >= 80: return 'Excellent! High performance expected.'
    if score >= 60: return 'Good. Solid with room for improvement.'
    return 'Needs Improvement. Re-evaluate key elements.'

def assess_face_quality(count, sizes):
    if count == 0: return 0.3
    quality = 0.5
    if count == 1: quality += 0.3
    for size in sizes:
        if 0.1 < size < 0.4: quality += 0.1 # Bonus for optimally sized face
    return min(1.0, quality)

def calculate_color_harmony(colors):
    if not colors or len(colors) < 2: return 0.5
    harmony_score, comparisons = 0, 0
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            h1, _, _ = colorsys.rgb_to_hsv(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255)
            h2, _, _ = colorsys.rgb_to_hsv(colors[j][0]/255, colors[j][1]/255, colors[j][2]/255)
            hue_diff = abs(h1 - h2)
            if hue_diff > 0.5: hue_diff = 1 - hue_diff
            if 0.4 < hue_diff < 0.6: harmony_score += 1.0
            else: harmony_score += 0.3
            comparisons += 1
    return harmony_score / comparisons if comparisons > 0 else 0.5

def calculate_text_readability(text_regions, img_shape):
    if not text_regions: return 0.5
    total_area = img_shape[0] * img_shape[1]
    if sum(r['area'] for r in text_regions) / total_area > 0.3: return 0.4
    
    readability = sum(min(1, r['area'] / (total_area * 0.1)) for r in text_regions)
    return readability / len(text_regions)

# --- App Execution ---
if __name__ == "__main__":
    # Initialize session state variables
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None
    if 'last_uploaded_name' not in st.session_state:
        st.session_state['last_uploaded_name'] = None
    main()
