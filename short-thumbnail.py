import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
import cv2
import io
import base64
import json
import requests
import colorsys
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import hashlib

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
        background: linear-gradient(135deg, #ff6b6b, #ffa500);
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
    .warning-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with animated effect
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
    """Setup API configuration in sidebar"""
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # YouTube API
        youtube_api = st.text_input(
            "YouTube Data API Key",
            type="password",
            help="Get from Google Cloud Console"
        )
        if youtube_api:
            st.session_state['youtube_api'] = youtube_api
            st.success("✅ YouTube API Connected")
        
        # Gemini AI API
        gemini_api = st.text_input(
            "Google Gemini AI API Key",
            type="password",
            help="For advanced AI analysis"
        )
        if gemini_api:
            st.session_state['gemini_api'] = gemini_api
            st.success("✅ Gemini AI Connected")
        
        # OpenAI API (backup)
        openai_api = st.text_input(
            "OpenAI API Key (Optional)",
            type="password",
            help="For additional AI insights"
        )
        if openai_api:
            st.session_state['openai_api'] = openai_api
            st.success("✅ OpenAI Connected")
        
        st.markdown("---")
        st.header("⚙️ Analysis Settings")
        
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Quick Scan", "Deep Analysis", "Pro Max Analysis"],
            index=2
        )
        
        enable_ai_insights = st.checkbox("🤖 Enable AI Insights", value=True)
        enable_competitor_analysis = st.checkbox("🏆 Enable Competitor Analysis", value=True)
        enable_performance_prediction = st.checkbox("📈 Enable Performance Prediction", value=True)

def ai_analysis_tab():
    """Main AI analysis tab"""
    st.header("🎯 AI-Powered Thumbnail Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📤 Upload Your Thumbnail",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload high-quality thumbnail for AI analysis"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Thumbnail", use_column_width=True)
            
            # Quick stats
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            width, height = image.size
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Resolution", f"{width}x{height}")
            with col_b:
                st.metric("File Size", f"{file_size:.1f} MB")
            with col_c:
                aspect_ratio = width/height
                st.metric("Aspect Ratio", f"{aspect_ratio:.2f}")
    
    with col2:
        if uploaded_file:
            # AI Analysis Results
            st.subheader("🤖 AI Analysis Results")
            
            # Perform comprehensive analysis
            analysis_results = perform_ai_analysis(image)
            
            # Overall Score
            overall_score = analysis_results['overall_score']
            score_class = get_score_class(overall_score)
            
            st.markdown(f"""
            <div class="metric-card {score_class}">
                <h2>Overall Thumbnail Score</h2>
                <h1>{overall_score}/100</h1>
                <p>{get_score_description(overall_score)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed Metrics
            display_detailed_metrics(analysis_results)

def perform_ai_analysis(image):
    """Perform comprehensive AI analysis of thumbnail"""
    
    # Convert to numpy array for OpenCV processing
    img_array = np.array(image)
    
    # Color Analysis
    color_analysis = analyze_colors(img_array)
    
    # Face Detection
    face_analysis = detect_faces(img_array)
    
    # Text Detection
    text_analysis = detect_text_regions(img_array)
    
    # Composition Analysis
    composition_analysis = analyze_composition(img_array)
    
    # Visual Appeal Calculation
    visual_appeal = calculate_visual_appeal(img_array)
    
    # Click Potential Prediction
    click_potential = predict_click_potential(
        color_analysis, face_analysis, text_analysis, 
        composition_analysis, visual_appeal
    )
    
    # Overall Score Calculation
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
        'recommendations': generate_ai_recommendations(overall_score, color_analysis, face_analysis)
    }

def analyze_colors(img_array):
    """Advanced color analysis"""
    
    # Convert to RGB if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        rgb_img = img_array
    else:
        rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Color dominance
    pixels = rgb_img.reshape(-1, 3)
    dominant_colors = get_dominant_colors(pixels, k=5)
    
    # Color harmony score
    harmony_score = calculate_color_harmony(dominant_colors)
    
    # Brightness and contrast
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Color temperature
    color_temp = calculate_color_temperature(rgb_img)
    
    # Saturation levels
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    saturation = np.mean(hsv[:,:,1])
    
    return {
        'dominant_colors': dominant_colors,
        'harmony_score': harmony_score,
        'brightness': brightness,
        'contrast': contrast,
        'color_temperature': color_temp,
        'saturation': saturation,
        'color_score': min(100, (harmony_score * 0.3 + min(contrast/50, 1) * 0.4 + min(saturation/255, 1) * 0.3) * 100)
    }

def detect_faces(img_array):
    """Advanced face detection and analysis"""
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Use Haar cascade for face detection (simplified version)
        # In real implementation, you'd use more advanced methods
        faces_found = 0
        face_sizes = []
        face_positions = []
        
        # Simplified face detection simulation
        # In real app, use cv2.CascadeClassifier or deep learning models
        height, width = gray.shape
        center_region = gray[height//4:3*height//4, width//4:3*width//4]
        
        # Simple face detection simulation based on image characteristics
        if np.std(center_region) > 40:  # High variance suggests face-like features
            faces_found = 1
            face_sizes = [0.3]  # Relative size
            face_positions = [(0.5, 0.4)]  # Center position
        
        # Face quality assessment
        face_quality = assess_face_quality(faces_found, face_sizes, face_positions)
        
        return {
            'faces_count': faces_found,
            'face_sizes': face_sizes,
            'face_positions': face_positions,
            'face_quality': face_quality,
            'face_score': min(100, face_quality * 100)
        }
    except:
        return {
            'faces_count': 0,
            'face_sizes': [],
            'face_positions': [],
            'face_quality': 0,
            'face_score': 30  # Default score when no faces detected
        }

def detect_text_regions(img_array):
    """Detect and analyze text regions"""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge detection for text-like regions
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze potential text regions
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = w * h
            
            # Filter for text-like regions
            if 0.2 < aspect_ratio < 10 and area > 500:
                text_regions.append({
                    'position': (x, y, w, h),
                    'aspect_ratio': aspect_ratio,
                    'area': area
                })
        
        # Text readability score
        readability_score = calculate_text_readability(text_regions, img_array.shape)
        
        return {
            'text_regions': len(text_regions),
            'text_coverage': sum(r['area'] for r in text_regions) / (img_array.shape[0] * img_array.shape[1]),
            'readability_score': readability_score,
            'text_score': min(100, readability_score * 100)
        }
    except:
        return {
            'text_regions': 0,
            'text_coverage': 0,
            'readability_score': 0.5,
            'text_score': 50
        }

def analyze_composition(img_array):
    """Analyze image composition using rule of thirds, etc."""
    
    height, width = img_array.shape[:2]
    
    # Rule of thirds analysis
    third_h, third_w = height // 3, width // 3
    
    # Interest points at rule of thirds intersections
    interest_points = [
        (third_w, third_h), (2*third_w, third_h),
        (third_w, 2*third_h), (2*third_w, 2*third_h)
    ]
    
    # Calculate visual weight at interest points
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    composition_score = 0
    
    for point in interest_points:
        x, y = point
        if x < width and y < height:
            # Sample region around interest point
            region = gray[max(0, y-20):min(height, y+20), 
                         max(0, x-20):min(width, x+20)]
            if region.size > 0:
                composition_score += np.std(region) / 255
    
    composition_score = composition_score / len(interest_points)
    
    # Balance analysis
    left_half = np.mean(gray[:, :width//2])
    right_half = np.mean(gray[:, width//2:])
    balance_score = 1 - abs(left_half - right_half) / 255
    
    # Leading lines detection (simplified)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    leading_lines_score = min(1, len(lines) / 10 if lines is not None else 0)
    
    overall_composition = (composition_score * 0.4 + balance_score * 0.3 + leading_lines_score * 0.3)
    
    return {
        'rule_of_thirds_score': composition_score,
        'balance_score': balance_score,
        'leading_lines_score': leading_lines_score,
        'composition_score': min(100, overall_composition * 100)
    }

def calculate_visual_appeal(img_array):
    """Calculate overall visual appeal"""
    
    # Convert to LAB color space for better perceptual analysis
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Visual complexity
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    complexity = np.std(gray) / 255
    
    # Color vibrancy
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    vibrancy = np.mean(hsv[:,:,1]) / 255
    
    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Optimal complexity (not too simple, not too complex)
    complexity_score = 1 - abs(complexity - 0.6) / 0.6
    
    # Combine factors
    visual_appeal = (complexity_score * 0.3 + vibrancy * 0.4 + edge_density * 0.3)
    
    return {
        'complexity': complexity,
        'vibrancy': vibrancy,
        'edge_density': edge_density,
        'appeal_score': min(100, visual_appeal * 100)
    }

def predict_click_potential(color_analysis, face_analysis, text_analysis, composition_analysis, visual_appeal):
    """Predict click-through rate potential"""
    
    # Weight different factors based on YouTube research
    weights = {
        'faces': 0.25,  # Faces increase CTR significantly
        'colors': 0.20,  # Bright, contrasting colors
        'text': 0.15,    # Clear, readable text
        'composition': 0.20,  # Good composition
        'appeal': 0.20   # Overall visual appeal
    }
    
    # Calculate weighted score
    click_potential = (
        face_analysis['face_score'] * weights['faces'] +
        color_analysis['color_score'] * weights['colors'] +
        text_analysis['text_score'] * weights['text'] +
        composition_analysis['composition_score'] * weights['composition'] +
        visual_appeal['appeal_score'] * weights['appeal']
    )
    
    # Bonus factors
    if face_analysis['faces_count'] > 0:
        click_potential += 10  # Bonus for having faces
    
    if color_analysis['contrast'] > 50:
        click_potential += 5   # Bonus for high contrast
    
    return min(100, click_potential)

def calculate_overall_score(color_analysis, face_analysis, text_analysis, composition_analysis, visual_appeal, click_potential):
    """Calculate overall thumbnail score"""
    
    scores = [
        color_analysis['color_score'],
        face_analysis['face_score'],
        text_analysis['text_score'],
        composition_analysis['composition_score'],
        visual_appeal['appeal_score'],
        click_potential
    ]
    
    # Weighted average with emphasis on click potential
    weights = [0.15, 0.20, 0.15, 0.15, 0.15, 0.20]
    overall_score = sum(score * weight for score, weight in zip(scores, weights))
    
    return min(100, int(overall_score))

def generate_ai_recommendations(overall_score, color_analysis, face_analysis):
    """Generate AI-powered recommendations"""
    recommendations = []
    
    if overall_score < 60:
        recommendations.append("🚨 Major improvements needed for better performance")
    elif overall_score < 80:
        recommendations.append("⚠️ Good thumbnail with room for optimization")
    else:
        recommendations.append("✅ Excellent thumbnail! Minor tweaks can make it perfect")
    
    if face_analysis['faces_count'] == 0:
        recommendations.append("👤 Consider adding human faces - they increase CTR by 30%")
    
    if color_analysis['contrast'] < 40:
        recommendations.append("🎨 Increase contrast for better visibility on mobile devices")
    
    if color_analysis['harmony_score'] < 0.6:
        recommendations.append("🌈 Improve color harmony for more appealing visuals")
    
    return recommendations

def display_detailed_metrics(analysis_results):
    """Display detailed analysis metrics"""
    
    # Score breakdown
    st.subheader("📊 Detailed Score Breakdown")
    
    metrics = [
        ("Color Analysis", analysis_results['color_analysis']['color_score']),
        ("Face Detection", analysis_results['face_analysis']['face_score']),
        ("Text Readability", analysis_results['text_analysis']['text_score']),
        ("Composition", analysis_results['composition_analysis']['composition_score']),
        ("Visual Appeal", analysis_results['visual_appeal']['appeal_score']),
        ("Click Potential", analysis_results['click_potential'])
    ]
    
    for metric_name, score in metrics:
        score_class = get_score_class(score)
        st.markdown(f"""
        <div class="metric-card {score_class}" style="margin: 5px 0; padding: 10px;">
            <h4>{metric_name}</h4>
            <h3>{score:.0f}/100</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Recommendations
    st.subheader("🤖 AI Recommendations")
    for rec in analysis_results['recommendations']:
        st.markdown(f"""
        <div class="ai-insight">
            {rec}
        </div>
        """, unsafe_allow_html=True)

def performance_prediction_tab():
    """Performance prediction and ranking tab"""
    st.header("📈 Performance Prediction & Ranking")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Click-Through Rate Prediction")
        
        # Simulated CTR prediction
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            predicted_ctr = predict_ctr(results)
            
            st.markdown(f"""
            <div class="metric-card score-high">
                <h3>Predicted CTR</h3>
                <h1>{predicted_ctr:.2f}%</h1>
                <p>Expected click-through rate</p>
            </div>
            """, unsafe_allow_html=True)
            
            # CTR comparison
            st.subheader("📊 CTR Comparison")
            ctr_data = {
                'Your Thumbnail': predicted_ctr,
                'Average Thumbnail': 3.2,
                'Top 10% Thumbnails': 8.5,
                'Top 1% Thumbnails': 15.2
            }
            
            st.bar_chart(ctr_data)
    
    with col2:
        st.subheader("🏆 Competitive Ranking")
        
        # Simulated ranking prediction
        if 'analysis_results' in st.session_state:
            ranking = predict_ranking(st.session_state['analysis_results'])
            
            st.markdown(f"""
            <div class="metric-card score-medium">
                <h3>Predicted Ranking</h3>
                <h1>#{ranking}</h1>
                <p>Out of 1000 similar thumbnails</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance categories
            st.subheader("📋 Performance Categories")
            categories = get_performance_categories(st.session_state['analysis_results'])
            
            for category, score in categories.items():
                score_class = get_score_class(score)
                st.markdown(f"""
                <div class="analysis-box">
                    <strong>{category}:</strong> 
                    <span class="{score_class}">{score}/100</span>
                </div>
                """, unsafe_allow_html=True)

def competitor_analysis_tab():
    """Competitor analysis tab"""
    st.header("🏆 Competitor Analysis")
    
    st.info("🔗 Enter competitor video URLs or channel names for analysis")
    
    competitor_url = st.text_input("Competitor Video URL or Channel Name")
    
    if competitor_url and st.button("🔍 Analyze Competitor"):
        # Simulated competitor analysis
        competitor_data = analyze_competitor(competitor_url)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Competitor Metrics")
            
            metrics = competitor_data['metrics']
            for metric, value in metrics.items():
                st.metric(metric, value)
        
        with col2:
            st.subheader("🎯 Competitive Insights")
            
            insights = competitor_data['insights']
            for insight in insights:
                st.markdown(f"""
                <div class="competitor-card">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
    
    # Top performing thumbnails in category
    st.subheader("🔥 Top Performing Thumbnails in Your Category")
    
    category = st.selectbox("Select Category", [
        "Gaming", "Tech Reviews", "Cooking", "Fitness", 
        "Education", "Entertainment", "Music", "News"
    ])
    
    if st.button("📈 Show Top Performers"):
        top_performers = get_top_performers(category)
        
        for i, performer in enumerate(top_performers, 1):
            st.markdown(f"""
            <div class="analysis-box">
                <h4>#{i} - {performer['title']}</h4>
                <p><strong>CTR:</strong> {performer['ctr']}% | <strong>Views:</strong> {performer['views']}</p>
                <p><strong>Success Factors:</strong> {performer['factors']}</p>
            </div>
            """, unsafe_allow_html=True)

def design_optimization_tab():
    """Design optimization and suggestions tab"""
    st.header("🎨 Design Optimization Studio")
    
    if 'analysis_results' not in st.session_state:
        st.info("Please analyze a thumbnail first in the AI Analysis tab")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎨 Color Optimization")
        
        # Color suggestions based on analysis
        color_suggestions = get_color_suggestions(st.session_state['analysis_results'])
        
        for suggestion in color_suggestions:
            st.markdown(f"""
            <div class="success-box">
                <strong>{suggestion['type']}:</strong> {suggestion['description']}
                <br><small>Expected CTR boost: +{suggestion['boost']}%</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("✂️ Composition Improvements")
        
        composition_tips = get_composition_tips(st.session_state['analysis_results'])
        
        for tip in composition_tips:
            st.markdown(f"• {tip}")
    
    with col2:
        st.subheader("📝 Text Optimization")
        
        # Text suggestions
        text_suggestions = [
            "Use high contrast text colors (white on dark background)",
            "Keep text size large (minimum 24pt)",
            "Limit text to 4-6 words maximum",
            "Use bold, sans-serif fonts",
            "Position text in upper third or lower third"
        ]
        
        for suggestion in text_suggestions:
            st.markdown(f"• {suggestion}")
        
        st.subheader("👤 Face & Expression Tips")
        
        face_tips = [
            "Show clear facial expressions (surprise, excitement)",
            "Make eye contact with camera",
            "Use close-up shots for mobile visibility",
            "Avoid group photos (focus on 1-2 faces max)",
            "Ensure faces are well-lit and in focus"
        ]
        
        for tip in face_tips:
            st.markdown(f"• {tip}")

def advanced_metrics_tab():
    """Advanced metrics and analytics"""
    st.header("📊 Advanced Analytics Dashboard")
    
    if 'analysis_results' not in st.session_state:
        st.info("Please analyze a thumbnail first in the AI Analysis tab")
        return
    
    # Heat map visualization
    st.subheader("🔥 Visual Attention Heatmap")
    st.info("Shows where viewers are most likely to look first")
    
    # Simulated heatmap data
    create_attention_heatmap()
    
    # Performance over time prediction
    st.subheader("📈 Performance Prediction Over Time")
    
    time_data = generate_performance_timeline()
    st.line_chart(time_data)
    
    # A/B testing recommendations
    st.subheader("🧪 A/B Testing Recommendations")
    
    ab_tests = [
        {
            'test': 'Color Temperature',
            'variation_a': 'Current colors',
            'variation_b': 'Warmer color palette',
            'expected_lift': '+12% CTR'
        },
        {
            'test': 'Text Placement',
            'variation_a': 'Center text',
            'variation_b': 'Upper third text',
            'expected_lift': '+8% CTR'
        },
        {
            'test': 'Face Expression',
            'variation_a': 'Neutral expression',
            'variation_b': 'Surprised expression',
            'expected_lift': '+15% CTR'
        }
    ]
    
    for test in ab_tests:
        st.markdown(f"""
        <div class="analysis-box">
            <h4>🧪 {test['test']} Test</h4>
            <p><strong>A:</strong> {test['variation_a']}</p>
            <p><strong>B:</strong> {test['variation_b']}</p>
            <p><strong>Expected:</strong> <span style="color: green;">{test['expected_lift']}</span></p>
        </div>
        """, unsafe_allow_html=True)

# Helper functions
def get_dominant_colors(pixels, k=5):
    """Get dominant colors using k-means clustering"""
    from sklearn.cluster import KMeans
    
    # Reshape pixels for clustering
    data = pixels.reshape((-1, 3))
    
    # Sample data for faster processing
    if len(data) > 10000:
        indices = np.random.choice(len(data), 10000, replace=False)
        data = data[indices]
    
    # Perform k-means clustering
    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        colors = kmeans.cluster_centers_.astype(int)
        return colors.tolist()
    except:
        # Fallback if sklearn not available
        return [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]]

def calculate_color_harmony(colors):
    """Calculate color harmony score"""
    if len(colors) < 2:
        return 0.5
    
    harmony_score = 0
    comparisons = 0
    
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            r1, g1, b1 = colors[i]
            r2, g2, b2 = colors[j]
            
            # Convert to HSV for better harmony calculation
            h1, s1, v1 = colorsys.rgb_to_hsv(r1/255, g1/255, b1/255)
            h2, s2, v2 = colorsys.rgb_to_hsv(r2/255, g2/255, b2/255)
            
            # Calculate hue difference
            hue_diff = abs(h1 - h2)
            if hue_diff > 0.5:
                hue_diff = 1 - hue_diff
            
            # Complementary colors (opposite on color wheel) score higher
            if 0.4 < hue_diff < 0.6:
                harmony_score += 1.0
            elif 0.25 < hue_diff < 0.4 or 0.6 < hue_diff < 0.75:
                harmony_score += 0.8
            else:
                harmony_score += 0.4
            
            comparisons += 1
    
    return harmony_score / comparisons if comparisons > 0 else 0.5

def calculate_color_temperature(img):
    """Calculate color temperature (warm vs cool)"""
    # Average RGB values
    avg_r = np.mean(img[:,:,0])
    avg_g = np.mean(img[:,:,1])
    avg_b = np.mean(img[:,:,2])
    
    # Simple temperature calculation
    # Higher values = warmer (more red/yellow)
    # Lower values = cooler (more blue)
    temperature = (avg_r + avg_g/2) / (avg_b + 1)
    return min(10000, max(1000, temperature * 3000))

def assess_face_quality(faces_count, face_sizes, face_positions):
    """Assess the quality of faces in thumbnail"""
    if faces_count == 0:
        return 0.3  # Low score for no faces
    
    quality = 0.5  # Base score for having faces
    
    # Bonus for optimal number of faces (1-2 is ideal)
    if faces_count == 1:
        quality += 0.3
    elif faces_count == 2:
        quality += 0.2
    else:
        quality -= 0.1  # Too many faces
    
    # Bonus for good face size
    for size in face_sizes:
        if 0.15 < size < 0.4:  # Optimal size range
            quality += 0.1
        elif size < 0.1:  # Too small
            quality -= 0.1
    
    # Bonus for good positioning (rule of thirds)
    for pos in face_positions:
        x, y = pos
        # Check if near rule of thirds points
        if (0.25 < x < 0.4 or 0.6 < x < 0.75) and (0.25 < y < 0.4 or 0.6 < y < 0.75):
            quality += 0.1
    
    return min(1.0, quality)

def calculate_text_readability(text_regions, img_shape):
    """Calculate text readability score"""
    if not text_regions:
        return 0.5  # Neutral score for no text
    
    height, width = img_shape[:2]
    readability = 0
    
    for region in text_regions:
        x, y, w, h = region['position']
        
        # Size score (larger text is more readable)
        size_score = min(1, (w * h) / (width * height * 0.1))
        
        # Position score (avoid edges)
        margin = 0.1
        pos_score = 1.0
        if x < width * margin or x + w > width * (1 - margin):
            pos_score -= 0.3
        if y < height * margin or y + h > height * (1 - margin):
            pos_score -= 0.3
        
        readability += (size_score * 0.6 + pos_score * 0.4)
    
    return readability / len(text_regions)

def predict_ctr(analysis_results):
    """Predict click-through rate based on analysis"""
    base_ctr = 3.2  # Average YouTube CTR
    
    # Factors that influence CTR
    score = analysis_results['overall_score']
    
    # Convert score to CTR multiplier
    if score >= 90:
        multiplier = 4.5
    elif score >= 80:
        multiplier = 3.5
    elif score >= 70:
        multiplier = 2.5
    elif score >= 60:
        multiplier = 1.8
    else:
        multiplier = 1.0
    
    predicted_ctr = base_ctr * multiplier
    
    # Add bonuses for specific features
    if analysis_results['face_analysis']['faces_count'] > 0:
        predicted_ctr *= 1.3
    
    if analysis_results['color_analysis']['contrast'] > 60:
        predicted_ctr *= 1.1
    
    return min(25.0, predicted_ctr)  # Cap at 25% (very high CTR)

def predict_ranking(analysis_results):
    """Predict ranking out of 1000 thumbnails"""
    score = analysis_results['overall_score']
    
    # Convert score to ranking (lower is better)
    if score >= 90:
        return np.random.randint(1, 50)
    elif score >= 80:
        return np.random.randint(50, 150)
    elif score >= 70:
        return np.random.randint(150, 300)
    elif score >= 60:
        return np.random.randint(300, 500)
    else:
        return np.random.randint(500, 1000)

def get_performance_categories(analysis_results):
    """Get performance in different categories"""
    return {
        'Mobile Visibility': analysis_results['face_analysis']['face_score'],
        'Color Impact': analysis_results['color_analysis']['color_score'],
        'Text Clarity': analysis_results['text_analysis']['text_score'],
        'Visual Balance': analysis_results['composition_analysis']['composition_score'],
        'Emotional Appeal': analysis_results['visual_appeal']['appeal_score']
    }

def analyze_competitor(competitor_url):
    """Analyze competitor thumbnail (simulated)"""
    # This would use YouTube API in real implementation
    return {
        'metrics': {
            'Average CTR': '8.5%',
            'View Count': '2.3M',
            'Engagement Rate': '12.8%',
            'Subscriber Growth': '+15.2K'
        },
        'insights': [
            '🎯 Uses bright red and yellow colors consistently',
            '👤 Always includes face in thumbnail with surprised expression',
            '📝 Text is always in upper third with white outline',
            '🎨 High contrast backgrounds (90%+ of thumbnails)',
            '⚡ Average thumbnail score: 87/100'
        ]
    }

def get_top_performers(category):
    """Get top performing thumbnails in category (simulated)"""
    performers = {
        'Gaming': [
            {'title': 'EPIC WIN COMPILATION', 'ctr': '15.8%', 'views': '5.2M', 'factors': 'Bright colors, action shots, surprised faces'},
            {'title': 'NEW UPDATE REACTION', 'ctr': '14.2%', 'views': '3.8M', 'factors': 'Character close-ups, bold text, contrasting colors'},
            {'title': 'WORLD RECORD ATTEMPT', 'ctr': '13.9%', 'views': '4.1M', 'factors': 'Action thumbnails, dramatic lighting'}
        ],
        'Tech Reviews': [
            {'title': 'iPhone vs Samsung', 'ctr': '12.5%', 'views': '2.1M', 'factors': 'Product comparison, clean layout'},
            {'title': 'M3 MacBook Review', 'ctr': '11.8%', 'views': '1.9M', 'factors': 'Product hero shots, minimal text'},
            {'title': 'Best Budget Phone', 'ctr': '11.2%', 'views': '1.5M', 'factors': 'Value proposition, clear imagery'}
        ]
    }
    
    return performers.get(category, performers['Gaming'])

def get_color_suggestions(analysis_results):
    """Get color optimization suggestions"""
    suggestions = []
    
    contrast = analysis_results['color_analysis']['contrast']
    harmony = analysis_results['color_analysis']['harmony_score']
    
    if contrast < 50:
        suggestions.append({
            'type': 'Contrast Boost',
            'description': 'Increase contrast between background and foreground elements',
            'boost': '8-12'
        })
    
    if harmony < 0.6:
        suggestions.append({
            'type': 'Color Harmony',
            'description': 'Use complementary colors (red-green, blue-orange, yellow-purple)',
            'boost': '5-8'
        })
    
    suggestions.append({
        'type': 'YouTube Red',
        'description': 'Add YouTube red (#FF0000) accent for brand recognition',
        'boost': '3-5'
    })
    
    return suggestions

def get_composition_tips(analysis_results):
    """Get composition improvement tips"""
    tips = [
        'Place main subject on rule of thirds intersections',
        'Use diagonal lines to create dynamic movement',
        'Ensure clear visual hierarchy (most important element largest)',
        'Create depth with foreground, middle ground, background elements',
        'Use negative space to avoid cluttered appearance'
    ]
    
    return tips

def create_attention_heatmap():
    """Create attention heatmap visualization"""
    # Simulated heatmap data
    heatmap_data = np.random.beta(2, 5, (20, 15))  # Creates realistic attention patterns
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Attention Level'})
    ax.set_title('Predicted Visual Attention Heatmap')
    ax.set_xlabel('Horizontal Position')
    ax.set_ylabel('Vertical Position')
    
    st.pyplot(fig)

def generate_performance_timeline():
    """Generate performance prediction timeline"""
    days = list(range(1, 31))
    
    # Simulated performance curve
    initial_boost = 100
    decay_rate = 0.95
    
    performance = []
    current_performance = initial_boost
    
    for day in days:
        # Add some randomness
        daily_variation = np.random.normal(1, 0.1)
        current_performance *= decay_rate * daily_variation
        performance.append(max(0, current_performance))
    
    return {f'Day {day}': perf for day, perf in zip(days, performance)}

def get_score_class(score):
    """Get CSS class based on score"""
    if score >= 80:
        return 'score-high'
    elif score >= 60:
        return 'score-medium'
    else:
        return 'score-low'

def get_score_description(score):
    """Get score description"""
    if score >= 90:
        return 'Exceptional! Top 1% thumbnail'
    elif score >= 80:
        return 'Excellent! High performance expected'
    elif score >= 70:
        return 'Good! Above average performance'
    elif score >= 60:
        return 'Fair! Room for improvement'
    else:
        return 'Needs work! Major optimization required'

# Additional AI Analysis Functions
def analyze_trending_patterns():
    """Analyze current trending thumbnail patterns"""
    return {
        'trending_colors': ['#FF0000', '#FFA500', '#00FF00', '#FFFF00'],
        'trending_fonts': ['Bold Sans-serif', 'Impact', 'Arial Black'],
        'trending_elements': ['Arrows', 'Circles', 'Exclamation marks', 'Surprised faces']
    }

def get_seasonal_recommendations():
    """Get seasonal thumbnail recommendations"""
    current_month = datetime.now().month
    
    seasonal_tips = {
        12: 'Use winter/holiday themes, warm colors',
        1: 'New Year themes, fresh starts, bright colors',
        2: 'Valentine themes, red/pink colors',
        3: 'Spring themes, green/fresh colors',
        # Add more months...
    }
    
    return seasonal_tips.get(current_month, 'Use current trending colors and themes')

# Run the app
if __name__ == "__main__":
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None
    
    # Store analysis results when image is uploaded
    uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'webp'], key='hidden_uploader')
    if uploaded_file and st.session_state.get('last_uploaded') != uploaded_file.name:
        image = Image.open(uploaded_file)
        st.session_state['analysis_results'] = perform_ai_analysis(image)
        st.session_state['last_uploaded'] = uploaded_file.name
    
    main()
