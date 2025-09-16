import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import base64

def main():
    st.set_page_config(
        page_title="YouTube Shorts Thumbnail Checker",
        page_icon="📱",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #FF0000;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .preview-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
    }
    .thumbnail-frame {
        background: #000;
        padding: 10px;
        border-radius: 10px;
        display: inline-block;
        margin: 10px;
    }
    .specs-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF0000;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📱 YouTube Shorts Thumbnail Checker</h1>', unsafe_allow_html=True)
    
    # Sidebar for specifications
    with st.sidebar:
        st.header("📋 YouTube Shorts Specs")
        st.markdown("""
        <div class="specs-box">
        <h3>Recommended Dimensions:</h3>
        <ul>
        <li><strong>Resolution:</strong> 1080x1920 (9:16)</li>
        <li><strong>Format:</strong> JPG, PNG, GIF, BMP</li>
        <li><strong>File Size:</strong> Under 2MB</li>
        <li><strong>Safe Area:</strong> Keep text away from edges</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("🎨 Preview Options")
        show_safe_area = st.checkbox("Show Safe Area", value=True)
        show_title_overlay = st.checkbox("Show Title Overlay", value=True)
        preview_style = st.selectbox("Preview Style", ["Mobile View", "Desktop View", "Both"])
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Your Thumbnail")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload your thumbnail image to preview how it will look on YouTube Shorts"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            st.info(f"📁 File: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # Load and analyze image
            image = Image.open(uploaded_file)
            width, height = image.size
            aspect_ratio = width / height
            
            st.markdown("### 📊 Image Analysis")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Width", f"{width}px")
            with col_b:
                st.metric("Height", f"{height}px")
            with col_c:
                st.metric("Aspect Ratio", f"{aspect_ratio:.2f}")
            
            # Check if dimensions are optimal
            if abs(aspect_ratio - 0.5625) < 0.1:  # 9:16 = 0.5625
                st.success("✅ Perfect aspect ratio for YouTube Shorts!")
            elif aspect_ratio < 0.5625:
                st.warning("⚠️ Too tall - consider cropping or adding padding")
            else:
                st.error("❌ Too wide - needs to be taller for Shorts format")
    
    with col2:
        if uploaded_file is not None:
            st.header("👀 Thumbnail Preview")
            
            # Create YouTube Shorts preview
            preview_image = create_shorts_preview(
                image, 
                show_safe_area, 
                show_title_overlay,
                uploaded_file.name
            )
            
            if preview_style in ["Mobile View", "Both"]:
                st.markdown("#### 📱 Mobile Preview")
                st.image(preview_image, width=300, caption="How it looks on mobile")
            
            if preview_style in ["Desktop View", "Both"]:
                st.markdown("#### 💻 Desktop Preview")
                st.image(preview_image, width=200, caption="How it looks on desktop")
            
            # Download button for optimized version
            if st.button("📥 Download Optimized Version"):
                optimized_image = optimize_for_shorts(image)
                img_buffer = io.BytesIO()
                optimized_image.save(img_buffer, format='PNG')
                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                href = f'<a href="data:image/png;base64,{img_str}" download="optimized_thumbnail.png">Download Optimized Thumbnail</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        else:
            st.info("👆 Upload an image to see the preview")
    
    # Tips section
    st.markdown("---")
    st.header("💡 Pro Tips for YouTube Shorts Thumbnails")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **🎯 Design Tips:**
        - Use bold, contrasting colors
        - Keep text large and readable
        - Focus on faces and emotions
        - Avoid cluttered designs
        - Test readability on small screens
        """)
    
    with tips_col2:
        st.markdown("""
        **📐 Technical Tips:**
        - Use 1080x1920 resolution
        - Keep file size under 2MB
        - Save as PNG for quality
        - Leave 10% margin from edges
        - Test on different devices
        """)

def create_shorts_preview(image, show_safe_area=True, show_title_overlay=True, filename=""):
    """Create a YouTube Shorts-style preview of the thumbnail"""
    
    # Resize to shorts format (9:16)
    target_width = 300
    target_height = int(target_width * 16/9)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image maintaining aspect ratio
    img_aspect = image.size[0] / image.size[1]
    target_aspect = target_width / target_height
    
    if img_aspect > target_aspect:
        # Image is wider, fit to height
        new_height = target_height
        new_width = int(new_height * img_aspect)
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Center crop
        left = (new_width - target_width) // 2
        preview = resized.crop((left, 0, left + target_width, target_height))
    else:
        # Image is taller, fit to width
        new_width = target_width
        new_height = int(new_width / img_aspect)
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Center crop
        top = (new_height - target_height) // 2
        preview = resized.crop((0, top, target_width, top + target_height))
    
    # Add overlays
    draw = ImageDraw.Draw(preview)
    
    # Safe area overlay
    if show_safe_area:
        margin = 15
        draw.rectangle([margin, margin, target_width-margin, target_height-margin], 
                      outline='red', width=2)
        draw.text((margin+5, margin+5), "Safe Area", fill='red')
    
    # Title overlay simulation
    if show_title_overlay:
        # Simulate YouTube UI elements
        draw.rectangle([0, target_height-60, target_width, target_height], 
                      fill=(0, 0, 0, 128))
        draw.text((10, target_height-50), "Sample Title Here", fill='white')
        draw.text((10, target_height-30), "👍 1.2K  💬 45  ↗️ Share", fill='white')
    
    return preview

def optimize_for_shorts(image):
    """Optimize image for YouTube Shorts format"""
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Target dimensions for YouTube Shorts
    target_width = 1080
    target_height = 1920
    
    # Calculate scaling
    img_aspect = image.size[0] / image.size[1]
    target_aspect = target_width / target_height
    
    if img_aspect > target_aspect:
        # Image is wider than target, fit to height
        new_height = target_height
        new_width = int(new_height * img_aspect)
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop
        left = (new_width - target_width) // 2
        optimized = resized.crop((left, 0, left + target_width, target_height))
    else:
        # Image is taller than target, fit to width  
        new_width = target_width
        new_height = int(new_width / img_aspect)
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop
        top = (new_height - target_height) // 2
        optimized = resized.crop((0, top, target_width, top + target_height))
    
    return optimized

if __name__ == "__main__":
    main()
