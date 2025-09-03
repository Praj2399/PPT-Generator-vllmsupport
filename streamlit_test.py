#!/usr/bin/env python3
"""
Streamlit Frontend for PPT Generator API Testing
Enhanced with bullet count control and detailed metrics
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import plotly.express as px

# ==================== Configuration ====================
API_BASE_URL = "http://localhost:8000/api"  # Update if your API runs on different port

# Page configuration
st.set_page_config(
    page_title="PPT Generator Test Suite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .slide-container {
        border: 2px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        background: #f9f9f9;
    }
    .bullet-item {
        background: white;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        border-left: 3px solid #4CAF50;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
    }
    .timing-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    div[data-testid="stSidebar"] {
        min-width: 350px;
        max-width: 350px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Session State Management ====================
if 'generated_slides' not in st.session_state:
    st.session_state.generated_slides = None
if 'generation_params' not in st.session_state:
    st.session_state.generation_params = {}
if 'performance_history' not in st.session_state:
    st.session_state.performance_history = []
if 'cache_status' not in st.session_state:
    st.session_state.cache_status = None
if 'last_operation_metrics' not in st.session_state:
    st.session_state.last_operation_metrics = {}
if 'detailed_metrics' not in st.session_state:
    st.session_state.detailed_metrics = []

# ==================== Helper Functions ====================
def make_api_call(endpoint: str, method: str = "GET", data: Dict = None, files: List = None) -> Dict:
    """Make API call and measure detailed time"""
    start_time = time.time()
    
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, data=data)
        else:
            return {"success": False, "error": "Invalid method"}
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            result['elapsed_time'] = elapsed_time
            return result
        else:
            return {
                "success": False,
                "error": f"API Error: {response.status_code} - {response.text}",
                "elapsed_time": elapsed_time
            }
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time
        }

def get_cache_status():
    """Get current cache status from API"""
    return make_api_call("cache-status", "GET")

def format_time(seconds: float) -> str:
    """Format time in seconds to readable string"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"

def get_bullet_color(char_count: int, tone: str) -> str:
    """Get color indicator based on bullet length and tone"""
    if tone == "concise":
        if char_count < 25:
            return "🔵"  # Too short
        elif char_count > 60:
            return "🔴"  # Too long
        else:
            return "🟢"  # Good
    else:  # comprehensive
        if char_count < 60:
            return "🔵"  # Too short
        elif char_count > 120:
            return "🔴"  # Too long
        else:
            return "🟢"  # Good

def display_detailed_metrics(metrics: Dict):
    """Display detailed performance and retrieval metrics"""
    with st.expander("📊 Detailed Metrics", expanded=True):
        # Performance Timing
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("⏱️ Total Time", format_time(metrics.get('elapsed_time', 0)))
            
        with col2:
            st.metric("📄 Slides", metrics.get('total_slides', 0))
            
        with col3:
            st.metric("📝 Context Length", f"{metrics.get('context_length', 0):,} chars")
            
        with col4:
            st.metric("🎯 Method", metrics.get('approach', 'N/A').upper())
        
        # Additional metrics if available
        if 'chunks_retrieved' in metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🔍 Chunks Retrieved", metrics.get('chunks_retrieved', 0))
            
            with col2:
                st.metric("📚 Total Chunks", metrics.get('total_chunks', 0))
            
            with col3:
                retrieval_method = metrics.get('retrieval_method', 'N/A')
                st.metric("🔄 Retrieval", retrieval_method)
            
            with col4:
                cache_used = "✅ Yes" if metrics.get('cache_used') else "❌ No"
                st.metric("💾 Cache Used", cache_used)
        
        # Timing breakdown if available
        if 'timing_breakdown' in metrics:
            st.subheader("⏱️ Timing Breakdown")
            timing = metrics['timing_breakdown']
            
            df_timing = pd.DataFrame([
                {"Phase": "Document Processing", "Time": format_time(timing.get('document_processing', 0))},
                {"Phase": "Embedding Creation", "Time": format_time(timing.get('embedding_creation', 0))},
                {"Phase": "Vectorstore Build", "Time": format_time(timing.get('vectorstore_build', 0))},
                {"Phase": "Context Retrieval", "Time": format_time(timing.get('retrieval', 0))},
                {"Phase": "LLM Generation", "Time": format_time(timing.get('generation', 0))},
                {"Phase": "Validation", "Time": format_time(timing.get('validation', 0))},
            ])
            
            st.dataframe(df_timing, hide_index=True, use_container_width=True)

# ==================== Sidebar Controls ====================
with st.sidebar:
    st.header("🎯 PPT Generator Controls")
    
    # Input Parameters Section
    st.subheader("📝 Generation Parameters")
    
    # Topic Input
    topic = st.text_input(
        "Presentation Topic",
        value=st.session_state.generation_params.get('topic', 'Artificial Intelligence in Healthcare'),
        key="sidebar_topic",
        help="Main topic for your presentation"
    )
    
    # Number of Slides
    num_slides = st.slider(
        "Number of Slides",
        min_value=1,
        max_value=20,
        value=st.session_state.generation_params.get('num_slides', 6),
        key="sidebar_num_slides"
    )
    
    # Bullet Count Control - NEW!
    st.markdown("### 📍 Bullet Points")
    bullet_count = st.slider(
        "Bullets per Slide",
        min_value=2,
        max_value=10,
        value=st.session_state.generation_params.get('bullet_count', 4),
        key="sidebar_bullet_count",
        help="Number of bullet points for each slide"
    )
    
    # Show bullet count impact
    if bullet_count > 6:
        st.info(f"📊 Will extract 20% more context for {bullet_count} bullets")
    
    # Tone Selection
    tone_options = ["concise", "comprehensive"]
    current_tone = st.session_state.generation_params.get('tone', 'concise')
    tone_index = tone_options.index(current_tone) if current_tone in tone_options else 0
    
    tone = st.selectbox(
        "Tone",
        options=tone_options,
        index=tone_index,
        key="sidebar_tone",
        help="Concise: 25-60 chars | Comprehensive: 60-120 chars"
    )
    
    # File Upload
    st.subheader("📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload Documents (Optional)",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        key="sidebar_files",
        help="Upload documents to extract content from"
    )
    
    # Display file info
    if uploaded_files:
        for file in uploaded_files:
            file_size = len(file.read()) / 1024  # KB
            file.seek(0)  # Reset pointer
            st.caption(f"📄 {file.name} ({file_size:.1f} KB)")
    
    # Generate Button
    if st.button("🚀 Generate Presentation", type="primary", key="generate_btn", use_container_width=True):
        with st.spinner("Generating presentation..."):
            # Record generation start time
            gen_start = time.time()
            
            # Prepare files for upload
            files_data = []
            if uploaded_files:
                for file in uploaded_files:
                    files_data.append(('files', (file.name, file.read(), file.type)))
            
            # Make API call
            data = {
                "topic": topic,
                "num_slides": str(num_slides),
                "tone": tone,
                "bullet_count": str(bullet_count)  # NEW!
            }
            
            result = make_api_call("generate-content", "POST", data=data, files=files_data)
            
            if result.get('success'):
                st.session_state.generated_slides = result['slides']
                st.session_state.generation_params = {
                    'topic': topic,
                    'num_slides': num_slides,
                    'tone': tone,
                    'bullet_count': bullet_count
                }
                st.session_state.last_operation_metrics = result
                
                # Record performance with detailed metrics
                st.session_state.performance_history.append({
                    'operation': 'Generate',
                    'time': format_time(result['elapsed_time']),
                    'cache_used': False,
                    'context_length': result.get('context_length', 0),
                    'chunks': result.get('chunks_retrieved', 'N/A'),
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                st.success(f"✅ Generated in {format_time(result['elapsed_time'])}")
                st.rerun()
            else:
                st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    st.divider()
    
    # Cache & Metrics Status
    st.subheader("📦 System Status")
    
    cache_status = get_cache_status()
    
    if cache_status.get('has_vectorstore'):
        st.success("✅ Vectorstore Active")
        with st.expander("Details", expanded=False):
            st.text(f"Topic: {cache_status.get('topic', 'N/A')}")
            st.text(f"Created: {cache_status.get('created_at', 'N/A')[:19]}")
            params = cache_status.get('original_params', {})
            if params:
                st.text(f"Slides: {params.get('num_slides', 'N/A')}")
                st.text(f"Tone: {params.get('tone', 'N/A')}")
                st.text(f"Bullets: {params.get('bullet_count', 4)}")
            
            # Display chunk info if available
            if 'chunks_count' in cache_status:
                st.text(f"Chunks: {cache_status['chunks_count']}")
    else:
        st.warning("❌ No Cache")
    
    if st.button("🗑️ Clear Cache", key="clear_cache_btn", use_container_width=True):
        result = make_api_call("clear-cache", "POST")
        if result.get('message'):
            st.success("Cache cleared!")
            st.rerun()
    
    st.divider()
    
    # Performance History
    st.subheader("📊 Performance")
    if st.session_state.performance_history:
        # Show last 5 operations with more details
        recent = st.session_state.performance_history[-5:]
        df = pd.DataFrame(recent)
        
        # Display selected columns
        display_cols = ['operation', 'time', 'cache_used']
        if 'chunks' in df.columns:
            display_cols.append('chunks')
        
        st.dataframe(df[display_cols], hide_index=True, use_container_width=True)
        
        # Calculate and show improvements
        gen_times = [h for h in st.session_state.performance_history if 'Generate' in h['operation']]
        regen_times = [h for h in st.session_state.performance_history if 'Regen' in h['operation']]
        
        if gen_times and regen_times:
            def parse_time(time_str):
                if 'ms' in time_str:
                    return float(time_str.replace('ms', '')) / 1000
                else:
                    return float(time_str.replace('s', ''))
            
            avg_gen = sum(parse_time(g['time']) for g in gen_times) / len(gen_times)
            avg_regen = sum(parse_time(r['time']) for r in regen_times) / len(regen_times)
            
            if avg_regen > 0:
                speedup = avg_gen / avg_regen
                st.metric("⚡ Speed Boost", f"{speedup:.1f}x faster")
    else:
        st.caption("No operations yet")

# ==================== Main Content Area ====================
st.title("🎯 PPT Generator Test Suite - Advanced Metrics")

# Display last operation metrics if available
if st.session_state.last_operation_metrics:
    display_detailed_metrics(st.session_state.last_operation_metrics)

# Check if slides are generated
if st.session_state.generated_slides:
    
    # Top Section: Regenerate All Controls
    with st.container():
        st.markdown("### ♻️ Regenerate All Slides")
        
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            all_slides_prompt = st.text_input(
                "Instructions for All Slides",
                placeholder="E.g., 'Make all slides more technical', 'Add more statistics'",
                key="all_slides_prompt"
            )
        
        with col2:
            # Bullet count for regenerate all
            regen_all_bullets = st.number_input(
                "Bullets/Slide",
                min_value=2,
                max_value=10,
                value=st.session_state.generation_params.get('bullet_count', 4),
                key="regen_all_bullets"
            )
        
        with col3:
            st.write("")  # Spacer
            regen_all_btn = st.button(
                "♻️ Regenerate All",
                type="primary",
                key="regen_all_btn"
            )
        
        with col4:
            st.write("")  # Spacer
            st.metric("Total Slides", len(st.session_state.generated_slides))
    
    # Handle Regenerate All
    if regen_all_btn:
        with st.spinner("Regenerating all slides..."):
            regen_start = time.time()
            
            data = {
                "topic": topic,
                "num_slides": str(num_slides),
                "tone": tone,
                "bullet_count": str(regen_all_bullets),
                "current_slides_content": json.dumps(st.session_state.generated_slides)
            }
            
            if all_slides_prompt:
                data["user_prompt"] = all_slides_prompt
            
            result = make_api_call("regenerate-all-slides", "POST", data=data)
            
            if result.get('success'):
                st.session_state.generated_slides = result['slides']
                st.session_state.generation_params['bullet_count'] = regen_all_bullets
                st.session_state.last_operation_metrics = result
                
                # Record detailed performance
                st.session_state.performance_history.append({
                    'operation': 'Regen All',
                    'time': format_time(result['elapsed_time']),
                    'cache_used': result.get('cache_used', False),
                    'context_length': result.get('context_length', 0),
                    'chunks': result.get('chunks_retrieved', 'N/A'),
                    'bullets': regen_all_bullets,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                st.success(f"✅ All slides regenerated in {format_time(result['elapsed_time'])}")
                if result.get('cache_used'):
                    st.info("⚡ Used cached vectorstore - 10x faster!")
                st.rerun()
            else:
                st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    st.divider()
    
    # Display All Slides with Individual Controls
    st.markdown("### 📑 Generated Slides")
    
    for idx, slide in enumerate(st.session_state.generated_slides):
        with st.container():
            # Slide Header with metrics
            col_title, col_metric1, col_metric2 = st.columns([4, 1, 1])
            
            with col_title:
                st.subheader(f"Slide {slide['slide']}: {slide['title']}")
            
            with col_metric1:
                actual_bullets = len(slide.get('bullets', []))
                expected_bullets = st.session_state.generation_params.get('bullet_count', 4)
                bullet_status = "✅" if actual_bullets == expected_bullets else "⚠️"
                st.metric("Bullets", f"{actual_bullets} {bullet_status}")
            
            with col_metric2:
                avg_length = sum(len(b) for b in slide.get('bullets', [])) / max(actual_bullets, 1)
                st.metric("Avg Length", f"{avg_length:.0f} chars")
            
            # Individual slide controls
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                slide_prompt = st.text_input(
                    f"Custom instructions for Slide {slide['slide']}",
                    placeholder="E.g., 'Add statistics', 'Make it simpler'",
                    key=f"prompt_slide_{idx}",
                    label_visibility="collapsed"
                )
            
            with col2:
                slide_bullets = st.number_input(
                    "Bullets",
                    min_value=2,
                    max_value=10,
                    value=actual_bullets,
                    key=f"bullets_slide_{idx}"
                )
            
            with col3:
                regen_btn = st.button(
                    "🔄 Regenerate",
                    key=f"regen_btn_{idx}"
                )
            
            # Handle individual slide regeneration
            if regen_btn:
                with st.spinner(f"Regenerating slide {slide['slide']} with {slide_bullets} bullets..."):
                    regen_start = time.time()
                    
                    data = {
                        "slide_number": str(slide['slide']),
                        "slide_title": slide['title'],
                        "tone": tone,
                        "bullet_count": str(slide_bullets),
                        "current_slide": json.dumps(slide)
                    }
                    
                    if slide_prompt:
                        data["user_prompt"] = slide_prompt
                    
                    result = make_api_call("regenerate-slide", "POST", data=data)
                    
                    if result.get('success'):
                        st.session_state.generated_slides[idx] = result['slide']
                        st.session_state.last_operation_metrics = result
                        
                        # Record detailed performance
                        st.session_state.performance_history.append({
                            'operation': f'Regen Slide {slide["slide"]}',
                            'time': format_time(result['elapsed_time']),
                            'cache_used': result.get('cache_used', False),
                            'context_length': result.get('context_length', 0),
                            'chunks': result.get('chunks_retrieved', 'N/A'),
                            'bullets': slide_bullets,
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                        
                        st.success(f"✅ Slide {slide['slide']} regenerated in {format_time(result['elapsed_time'])}")
                        if result.get('cache_used'):
                            st.info(f"⚡ Cache hit! Retrieved {result.get('chunks_retrieved', 'N/A')} chunks")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
            # Display Bullets with analysis
            if 'bullets' in slide:
                for bullet_idx, bullet in enumerate(slide['bullets'], 1):
                    char_count = len(bullet)
                    color = get_bullet_color(char_count, st.session_state.generation_params.get('tone', 'concise'))
                    
                    # Create bullet with character count
                    bullet_col1, bullet_col2 = st.columns([6, 1])
                    with bullet_col1:
                        st.markdown(f"{color} **{bullet_idx}.** {bullet}")
                    with bullet_col2:
                        st.caption(f"{char_count} chars")
            
            st.markdown("---")
    
    # Advanced Analytics Section
    with st.expander("📈 Advanced Analytics & Export"):
        tab1, tab2, tab3 = st.tabs(["Statistics", "Performance Chart", "Export"])
        
        with tab1:
            # Calculate statistics
            total_bullets = sum(len(slide.get('bullets', [])) for slide in st.session_state.generated_slides)
            total_chars = sum(
                len(bullet) for slide in st.session_state.generated_slides 
                for bullet in slide.get('bullets', [])
            )
            avg_bullet_length = total_chars / max(total_bullets, 1)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Bullets", total_bullets)
            with col2:
                st.metric("Total Characters", f"{total_chars:,}")
            with col3:
                st.metric("Avg Bullet Length", f"{avg_bullet_length:.0f} chars")
            with col4:
                bullets_per_slide = total_bullets / len(st.session_state.generated_slides)
                st.metric("Avg Bullets/Slide", f"{bullets_per_slide:.1f}")
        
        with tab2:
            # Performance visualization
            if st.session_state.performance_history:
                df = pd.DataFrame(st.session_state.performance_history[-10:])
                
                # Parse times for chart
                df['time_seconds'] = df['time'].apply(
                    lambda x: float(x.replace('s', '')) if 's' in x else float(x.replace('ms', ''))/1000
                )
                
                # Create performance chart
                fig = go.Figure()
                
                # Add bars colored by cache usage
                colors = ['green' if cache else 'blue' for cache in df.get('cache_used', [False]*len(df))]
                
                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df['time_seconds'],
                    text=df['time'],
                    textposition='auto',
                    marker_color=colors,
                    customdata=df[['operation', 'timestamp']],
                    hovertemplate='%{customdata[0]}<br>Time: %{y:.2f}s<br>At: %{customdata[1]}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Operation Performance History",
                    xaxis_title="Operation Index",
                    yaxis_title="Time (seconds)",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                # Export slides as JSON
                json_str = json.dumps(st.session_state.generated_slides, indent=2)
                st.download_button(
                    label="📥 Download Slides (JSON)",
                    data=json_str,
                    file_name=f"slides_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json"
                )
            
            with col2:
                # Export metrics as CSV
                if st.session_state.performance_history:
                    df_export = pd.DataFrame(st.session_state.performance_history)
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Metrics (CSV)",
                        data=csv,
                        file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_csv"
                    )

else:
    # No slides generated yet
    st.info("👈 Use the sidebar controls to generate your presentation")
    
    # Show instructions
    with st.expander("📖 How to Use - Enhanced Features", expanded=True):
        st.markdown("""
        ### 🆕 New Features
        
        **Bullet Count Control:**
        - Set 2-10 bullets per slide in sidebar
        - Adjust individually for each slide during regeneration
        - System extracts 20% more context for 7+ bullets
        
        **Detailed Metrics:**
        - View chunks retrieved for each operation
        - See context length in characters
        - Track cache usage and speedup
        - Monitor timing for different phases
        
        **Performance Tracking:**
        - Real-time performance charts
        - Cache hit/miss statistics
        - Speed improvement calculations
        - Export metrics as CSV
        
        ### 📊 Metrics Explained
        
        | Metric | Meaning |
        |--------|---------|
        | **Chunks Retrieved** | Number of document segments used |
        | **Context Length** | Total characters of context sent to LLM |
        | **Cache Used** | Whether vectorstore was reused (10x faster) |
        | **Total Chunks** | Total document chunks in vectorstore |
        | **Retrieval Method** | Hybrid (semantic + keyword) or MMR |
        
        ### ⚡ Expected Performance
        
        - **Initial Generation**: 5-10s (creates vectorstore)
        - **Single Slide Regen**: 0.5-1s (uses cache)
        - **All Slides Regen**: 1-2s (uses cache)
        - **More Bullets**: Slightly longer due to extra context
        """)

# Footer with system info
st.divider()

# Display system configuration
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"🔗 API: {API_BASE_URL}")

with col2:
    cache_status = get_cache_status()
    cache_indicator = "✅" if cache_status.get('has_vectorstore') else "❌"
    st.caption(f"💾 Cache: {cache_indicator}")

with col3:
    st.caption("⚡ 10x faster with cache")

st.caption("PPT Generator Test Suite v2.0 - Enhanced with Bullet Control & Advanced Metrics")