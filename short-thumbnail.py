# ==============================================================================
# AI THUMBNAIL ANALYZER PRO - CORRECTED CODE
#
# Requirements:
# Install all necessary libraries by running:
# pip install streamlit numpy opencv-python-headless Pillow scikit-learn matplotlib seaborn
# ==============================================================================

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import colorsys
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans

def main():
    st.set_page_config(
        page_title="🚀 AI Thumbnail Analyzer Pro",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Ultra Modern CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .score-high { background: linear-gradient(135deg, #00c851, #007e33) !important; }
    .score-medium { background: linear-gradient(135deg, #ffbb33, #ff8800) !important; }
    .score-low { background: linear-gradient(135deg, #ff4444, #cc0000) !important; }
    .analysis-box {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        margin: 15px 0;
    }
    .competitor-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
    }
    .ai-insight {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🚀 AI THUMBNAIL ANALYZER PRO 🎯</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Billion Dollar Level YouTube Thumbnail Intelligence</p>', unsafe_allow_html=True)
    
    # API Configuration Sidebar
    setup_api_sidebar()
    
    # Main Layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 AI Analysis", 
        "📊 Performance Prediction", 
        "🏆 Competitor Analysis", 
        "🎨 Design Optimization", 
        "📈 Advanced Metrics"
    ])
    
    with tab1:
        ai_analysis_tab()
    
    with tab2:
        performance_prediction_tab()
    
    with tab3:
        competitor_analysis_tab()
    
    with tab4:
        design_optimization_tab()
    
    with tab5:
        advanced_metrics_tab()

def setup_api_sidebar():
    with st.sidebar:
        st.header("🔑 API Configuration")
        # API key inputs can remain as they are, they correctly use session_state
        youtube_api = st.text_input("YouTube Data API Key", type="password", help="Get from Google Cloud Console")
        if youtube_api:
            st.session_state['youtube_api'] = youtube_api
            st.success("✅ YouTube API Connected")
        
        gemini_api = st.text_input("Google Gemini AI API Key", type="password", help="For advanced AI analysis")
        if gemini_api:
            st.session_state['gemini_api'] = gemini_api
            st.success("✅ Gemini AI Connected")
        
        st.markdown("---")
        st.header("⚙️ Analysis Settings")
        st.selectbox("Analysis Depth", ["Quick Scan", "Deep Analysis", "Pro Max Analysis"], index=2)

def ai_analysis_tab():
    st.header("🎯 AI-Powered Thumbnail Analysis")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📤 Upload Your Thumbnail",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload high-quality thumbnail for AI analysis"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Perform analysis ONLY if it's a new file and save to session state
            if st.session_state.get('last_uploaded_name') != uploaded_file.name:
                with st.spinner('🤖 AI is analyzing your thumbnail... Please wait.'):
                    st.session_state['analysis_results'] = perform_ai_analysis(image)
                    st.session_state['last_uploaded_name'] = uploaded_file.name
            
            # Display image and stats
            st.image(image, caption="Uploaded Thumbnail", use_column_width=True)
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            width, height = image.size
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Resolution", f"{width}x{height}")
            c2.metric("File Size", f"{file_size:.1f} MB")
            c3.metric("Aspect Ratio", f"{width/height:.2f}")

    with col2:
        # Read analysis results from session state
        if st.session_state.get('analysis_results'):
            st.subheader("🤖 AI Analysis Results")
            results = st.session_state['analysis_results']
            
            overall_score = results['overall_score']
            score_class = get_score_class(overall_score)
            
            st.markdown(f"""
            <div class="metric-card {score_class}">
                <h2>Overall Thumbnail Score</h2>
                <h1>{overall_score}/100</h1>
                <p>{get_score_description(overall_score)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            display_detailed_metrics(results)
        else:
            st.info("👈 Upload a thumbnail to begin the AI analysis.")

def perform_ai_analysis(image):
    img_array = np.array(image)
    
    color_analysis = analyze_colors(img_array)
    face_analysis = detect_faces(img_array)
    text_analysis = detect_text_regions(img_array)
    composition_analysis = analyze_composition(img_array)
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
    """
    WARNING: This is a SIMULATED face detection function.
    It does not use a real AI model. It approximates face presence based on image properties.
    For real face detection, use cv2.CascadeClassifier or a deep learning model like MTCNN or MediaPipe.
    """
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        center_region = gray[height//4:3*height//4, width//4:3*width//4]
        
        faces_found = 0
        face_sizes = []
        face_positions = []
        
        # Simple simulation based on high contrast/detail in the center (often where faces are)
        if np.std(center_region) > 45 and center_region.mean() > 50:
            faces_found = 1
            face_sizes = [0.3]  # Assume a relative size of 30% of the image
            face_positions = [(0.5, 0.4)]  # Assume a central position
            
        face_quality = assess_face_quality(faces_found, face_sizes, face_positions)
        
        return {
            'faces_count': faces_found,
            'face_sizes': face_sizes,
            'face_positions': face_positions,
            'face_quality': face_quality,
            'face_score': min(100, face_quality * 100)
        }
    except Exception as e:
        st.warning(f"Face detection error: {e}")
        return {'faces_count': 0, 'face_sizes': [], 'face_positions': [], 'face_quality': 0, 'face_score': 30}

def detect_text_regions(img_array):
    """
    WARNING: This is a SIMULATED text detection function.
    It uses edge detection to find text-like regions, but it does not perform Optical Character Recognition (OCR).
    For real text detection and reading, use a library like EasyOCR or pytesseract.
    """
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            if 0.2 < aspect_ratio < 10 and area > 500 and (w > 20 or h > 20):
                text_regions.append({'position': (x, y, w, h), 'area': area})
        
        readability_score = calculate_text_readability(text_regions, img_array.shape)
        return {
            'text_regions_count': len(text_regions),
            'text_coverage': sum(r['area'] for r in text_regions) / (img_array.shape[0] * img_array.shape[1]),
            'readability_score': readability_score,
            'text_score': min(100, readability_score * 100)
        }
    except Exception as e:
        st.warning(f"Text detection error: {e}")
        return {'text_regions_count': 0, 'text_coverage': 0, 'readability_score': 0.5, 'text_score': 50}

def get_dominant_colors(pixels, k=5):
    data = pixels.reshape((-1, 3))
    # Sample data for performance
    if len(data) > 20000:
        indices = np.random.choice(len(data), 20000, replace=False)
        data = data[indices]
    
    try:
        # Use n_init='auto' to avoid future warnings
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data)
        return kmeans.cluster_centers_.astype(int).tolist()
    except Exception as e:
        st.warning(f"Color analysis (KMeans) failed: {e}. Using fallback.")
        return [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]]

# --- The rest of the functions from your original code ---
# (No major changes needed in the functions below, so they are included as is for completeness)

def analyze_colors(img_array):
    rgb_img = img_array
    pixels = rgb_img.reshape(-1, 3)
    dominant_colors = get_dominant_colors(pixels, k=5)
    harmony_score = calculate_color_harmony(dominant_colors)
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    saturation = np.mean(hsv[:, :, 1])
    return {
        'dominant_colors': dominant_colors,
        'harmony_score': harmony_score,
        'brightness': brightness,
        'contrast': contrast,
        'saturation': saturation,
        'color_score': min(100, (harmony_score * 0.3 + min(contrast / 50, 1) * 0.4 + min(saturation / 200, 1) * 0.3) * 100)
    }

def analyze_composition(img_array):
    height, width, _ = img_array.shape
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Rule of thirds
    third_w, third_h = width // 3, height // 3
    interest_points = [(third_w, third_h), (2 * third_w, third_h), (third_w, 2 * third_h), (2 * third_w, 2 * third_h)]
    rule_of_thirds_score = 0
    for x, y in interest_points:
        region = gray[max(0, y - 20):min(height, y + 20), max(0, x - 20):min(width, x + 20)]
        if region.size > 0:
            rule_of_thirds_score += np.std(region)
    rule_of_thirds_score = min(1.0, rule_of_thirds_score / (4 * 70))  # Normalize

    # Balance
    left_half = np.mean(gray[:, :width//2])
    right_half = np.mean(gray[:, width//2:])
    balance_score = 1 - abs(left_half - right_half) / 255

    overall_composition = (rule_of_thirds_score * 0.6 + balance_score * 0.4)
    return {
        'rule_of_thirds_score': rule_of_thirds_score,
        'balance_score': balance_score,
        'composition_score': min(100, overall_composition * 100)
    }

def calculate_visual_appeal(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 50, 150)
    
    clarity = 1 - (np.sum(edges > 0) / edges.size) / 0.2 # Lower edge density -> higher clarity
    vibrancy = np.mean(hsv[:, :, 1]) / 255
    
    appeal = (clarity * 0.5 + vibrancy * 0.5)
    return {'appeal_score': min(100, appeal * 100)}

def predict_click_potential(color_analysis, face_analysis, text_analysis, composition_analysis, visual_appeal):
    weights = {'faces': 0.30, 'colors': 0.25, 'text': 0.15, 'composition': 0.15, 'appeal': 0.15}
    click_potential = (
        face_analysis['face_score'] * weights['faces'] +
        color_analysis['color_score'] * weights['colors'] +
        text_analysis['text_score'] * weights['text'] +
        composition_analysis['composition_score'] * weights['composition'] +
        visual_appeal['appeal_score'] * weights['appeal']
    )
    if face_analysis['faces_count'] > 0: click_potential += 10
    if color_analysis['contrast'] > 50: click_potential += 5
    return min(100, click_potential)

def calculate_overall_score(color_analysis, face_analysis, text_analysis, composition_analysis, visual_appeal, click_potential):
    scores = [
        color_analysis['color_score'], face_analysis['face_score'], text_analysis['text_score'],
        composition_analysis['composition_score'], visual_appeal['appeal_score'], click_potential
    ]
    weights = [0.15, 0.20, 0.15, 0.15, 0.15, 0.20] # More weight on faces and click potential
    overall_score = sum(score * weight for score, weight in zip(scores, weights))
    return int(min(100, overall_score))

def generate_ai_recommendations(overall_score, color_analysis, face_analysis, text_analysis):
    recs = []
    if overall_score < 60: recs.append("🚨 **Major improvements needed.** Focus on the basics: clear subject, high contrast, and readable text.")
    elif overall_score < 80: recs.append("⚠️ **Good thumbnail, but can be better.** Try increasing color vibrancy or improving composition.")
    else: recs.append("✅ **Excellent thumbnail!** Looks professional and is likely to perform well.")

    if face_analysis['faces_count'] == 0: recs.append("👤 **Consider adding a human face.** Thumbnails with faces often see a significant CTR boost.")
    if color_analysis['contrast'] < 40: recs.append("🎨 **Increase contrast.** Make sure your main subject stands out clearly from the background, especially on mobile screens.")
    if text_analysis['text_score'] < 60: recs.append("📝 **Improve text readability.** Use fewer words, a larger font, and high-contrast colors for your text.")
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
        score_class = get_score_class(score)
        st.markdown(f"""
        <div class="metric-card {score_class}" style="margin: 5px 0; padding: 10px;">
            <h4>{name}</h4>
            <h3>{score:.0f}/100</h3>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("💡 AI Recommendations")
    for rec in results['recommendations']:
        st.markdown(f'<div class="ai-insight">{rec}</div>', unsafe_allow_html=True)

def performance_prediction_tab():
    st.header("📈 Performance Prediction")
    if not st.session_state.get('analysis_results'):
        st.info("Analyze a thumbnail first to see predictions.")
        return

    results = st.session_state['analysis_results']
    predicted_ctr = predict_ctr(results)
    st.metric("Predicted Click-Through Rate (CTR)", f"{predicted_ctr:.2f}%", help="Based on our model, this is the expected CTR against average videos in this category.")
    
    ctr_data = {
        'Your Thumbnail': predicted_ctr,
        'Average (3%)': 3.0,
        'Good (7%)': 7.0,
        'Excellent (12%)': 12.0
    }
    st.bar_chart(ctr_data)

def competitor_analysis_tab():
    st.header("🏆 Competitor Analysis")
    st.info("This feature is a demo. In a real app, this would use the YouTube API to fetch and analyze competitor thumbnails.")
    category = st.selectbox("Select Your Video Category", ["Gaming", "Tech Reviews", "Cooking", "Fitness", "Education"])
    if st.button("Analyze Top Competitors in " + category):
        st.subheader(f"🔥 Common Patterns in Top {category} Thumbnails")
        patterns = {
            "Gaming": "Bright, saturated colors, expressive faces (shock/excitement), bold text, and often include arrows or circles.",
            "Tech Reviews": "Clean, high-quality product images, often on a simple background. Minimal text, usually the product name.",
            "Cooking": "Vibrant, delicious-looking food close-ups. Often shows the final dish. Warm and inviting colors are common.",
            "Fitness": "Action shots or clear before/after images. High-energy and motivational feel. Text is often bold and direct.",
            "Education": "Clear diagrams, intriguing questions as text, and often a person pointing to the subject matter. High clarity is key."
        }
        st.markdown(f'<div class="competitor-card">{patterns[category]}</div>', unsafe_allow_html=True)

def design_optimization_tab():
    st.header("🎨 Design Optimization Studio")
    if not st.session_state.get('analysis_results'):
        st.info("Analyze a thumbnail first to get design tips.")
        return
    st.subheader("Quick Wins to Boost Your Score")
    tips = [
        "**Rule of Thirds:** Place your main subject or face off-center, at one of the four intersection points.",
        "**Color Psychology:** Use Red/Orange for excitement, Blue for trust, Green for growth/calm.",
        "**Font Choice:** Use a bold, clean, sans-serif font (like Montserrat, Impact, or Arial Black).",
        "**Add an Outline:** A black or white outline around your subject or text can make it 'pop' from the background.",
        "**Less is More:** Don't clutter the thumbnail. Focus on one clear subject, a few words of text, and a clean background."
    ]
    for tip in tips:
        st.markdown(f"- {tip}")

def advanced_metrics_tab():
    st.header("📊 Advanced Analytics Dashboard")
    if not st.session_state.get('analysis_results'):
        st.info("Analyze a thumbnail first to see advanced metrics.")
        return
    
    st.subheader("🔥 Visual Attention Heatmap (Simulation)")
    st.info("This simulated heatmap shows where a viewer's eyes are most likely to be drawn. Red indicates high attention areas.")
    
    # Create a simulated heatmap
    heatmap_data = np.random.rand(10, 15)
    if st.session_state['analysis_results']['face_analysis']['faces_count'] > 0:
        # If face is detected, concentrate heat in the center
        heatmap_data[3:7, 5:10] += np.random.rand(4, 5) * 2
    
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, cmap='YlOrRd', cbar=False, xticklabels=False, yticklabels=False, ax=ax)
    st.pyplot(fig)


# --- Helper Functions ---
def get_score_class(score):
    if score >= 80: return 'score-high'
    if score >= 60: return 'score-medium'
    return 'score-low'

def get_score_description(score):
    if score >= 90: return 'Exceptional! A top-tier thumbnail.'
    if score >= 80: return 'Excellent! High performance expected.'
    if score >= 70: return 'Good! Solid and above average.'
    if score >= 60: return 'Fair. Has potential but needs tweaks.'
    return 'Needs Improvement. Re-evaluate key elements.'

def predict_ctr(results):
    score = results['overall_score']
    # A simple non-linear mapping from score to CTR
    base_ctr = 2.0
    bonus = (score / 100) ** 3 * 15 
    return base_ctr + bonus

def calculate_color_harmony(colors):
    if not colors or len(colors) < 2: return 0.5
    harmony_score = 0
    comparisons = 0
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            r1, g1, b1 = colors[i]
            r2, g2, b2 = colors[j]
            h1, _, _ = colorsys.rgb_to_hsv(r1 / 255, g1 / 255, b1 / 255)
            h2, _, _ = colorsys.rgb_to_hsv(r2 / 255, g2 / 255, b2 / 255)
            hue_diff = abs(h1 - h2)
            if hue_diff > 0.5: hue_diff = 1 - hue_diff
            if 0.4 < hue_diff < 0.6: harmony_score += 1.0 # Complementary
            elif hue_diff < 0.1: harmony_score += 0.7 # Analogous
            else: harmony_score += 0.3
            comparisons += 1
    return harmony_score / comparisons if comparisons > 0 else 0.5

def assess_face_quality(count, sizes, positions):
    if count == 0: return 0.3
    quality = 0.5
    if count == 1: quality += 0.3
    elif count == 2: quality += 0.1
    else: quality -= 0.2
    
    for size in sizes:
        if 0.2 < size < 0.5: quality += 0.1 # Good size
    return min(1.0, quality)

def calculate_text_readability(text_regions, img_shape):
    if not text_regions: return 0.5 # Neutral score
    height, width, _ = img_shape
    total_area = height * width
    readability = 0
    
    # Penalize if too much text
    total_text_area = sum(r['area'] for r in text_regions)
    if total_text_area / total_area > 0.25:
        return 0.4 # Too cluttered
        
    for region in text_regions:
        x, y, w, h = region['position']
        # Larger text is better
        size_score = min(1, (w * h) / (total_area * 0.1))
        # Central positions are better, avoid edges
        pos_score = 1.0 - (abs(x + w / 2 - width / 2) / (width / 2)) * 0.5
        readability += size_score * 0.7 + pos_score * 0.3
        
    return readability / len(text_regions)


# --- Run the App ---
if __name__ == "__main__":
    # Initialize session state variables if they don't exist
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None
    if 'last_uploaded_name' not in st.session_state:
        st.session_state['last_uploaded_name'] = None
        
    main()
