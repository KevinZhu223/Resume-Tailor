#!/usr/bin/env python3
"""
Streamlit Web Interface for Resume Tailoring Tool
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import our main classes (assumes they're in resume_tailor.py)
from resume_tailor import ResumeTailoringPipeline, JobDescriptionAnalyzer, DocumentParser

# Page configuration
st.set_page_config(
    page_title="AI Resume Tailor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .suggestion-item {
        background-color: #e8f4fd;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        border-left: 3px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎯 AI Resume Tailor</h1>', unsafe_allow_html=True)
    st.markdown("**Transform your resume to match any job description using AI**")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys"
        )
        
        # Model selection
        model = st.selectbox(
            "AI Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            help="Choose the AI model for rewriting"
        )
        
        # Mode selection
        st.markdown("### 🎯 Operation Mode")
        mode = st.selectbox(
            "Choose operation mode",
            ["tailor", "polish", "gap"],
            format_func=lambda x: {
                "tailor": "Tailor Mode - Match job requirements",
                "polish": "Polish Mode - Improve readability",
                "gap": "Gap Analysis - Identify missing skills"
            }[x],
            help="""
            Tailor Mode: Customize resume for job requirements
            Polish Mode: Improve readability and professionalism
            Gap Analysis: Identify missing skills and opportunities
            """
        )
        
        # Output format
        st.markdown("### 📄 Output Format")
        output_format = st.selectbox(
            "Choose output format",
            ["standard", "detailed"],
            format_func=lambda x: {
                "standard": "Standard - Clean resume output",
                "detailed": "Detailed - Full analysis with recommendations"
            }[x],
            help="""
            Standard: Clean, professional resume output
            Detailed: Complete analysis with skills and recommendations
            """
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            verify_truth = st.checkbox(
                "Enable strict truth verification",
                True,
                help="Verify that rewritten content maintains truthfulness"
            )
            bullets_only = st.checkbox(
                "Output only bullet points",
                False,
                help="Show only the rewritten bullet points with context"
            )
            clear_cache = st.checkbox(
                "Clear cache before running",
                False,
                help="Clear previous results cache for fresh analysis"
            )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📋 Job Description</h2>', unsafe_allow_html=True)
        
        # Job description input options
        job_input_method = st.radio(
            "How would you like to provide the job description?",
            ["Paste text", "Upload file", "Enter URL"]
        )
        
        job_description = ""
        
        if job_input_method == "Paste text":
            job_description = st.text_area(
                "Paste the job description here",
                height=300,
                placeholder="Copy and paste the complete job description..."
            )
        
        elif job_input_method == "Upload file":
            uploaded_job_file = st.file_uploader(
                "Upload job description file",
                type=['txt', 'pdf', 'docx'],
                help="Upload a file containing the job description"
            )
            
            if uploaded_job_file:
                # Save the uploaded file to a temporary file
                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, uploaded_job_file.name)
                
                with open(temp_file_path, 'wb') as f:
                    f.write(uploaded_job_file.getvalue())
                
                try:
                    parser = DocumentParser()
                    job_description = parser.parse_document(temp_file_path)
                    st.success("✅ Job description loaded successfully!")
                    
                    # Show preview
                    with st.expander("📖 Job Description Preview"):
                        st.text_area("Job Description Content", job_description[:500] + "...", height=150, disabled=True)
                
                except Exception as e:
                    st.error(f"❌ Error parsing job file: {e}")
                    st.info("💡 Make sure your file is in PDF, Word, or text format")
                finally:
                    # Clean up temporary files
                    try:
                        os.remove(temp_file_path)
                        os.rmdir(temp_dir)
                    except:
                        pass
        
        elif job_input_method == "Enter URL":
            job_url = st.text_input(
                "Enter job posting URL",
                placeholder="https://..."
            )
            if job_url:
                st.info("🔄 URL scraping not implemented in this demo. Please use text or file input.")
    
    with col2:
        st.markdown('<h2 class="section-header">📄 Resume</h2>', unsafe_allow_html=True)
        
        uploaded_resume = st.file_uploader(
            "Upload your resume",
            type=['txt', 'pdf', 'docx'],
            help="Upload your current resume in PDF, Word, or text format"
        )
        
        resume_text = ""
        if uploaded_resume:
                # Save the uploaded file to a temporary file
                temp_dir = tempfile.mkdtemp()
                temp_file_path = os.path.join(temp_dir, uploaded_resume.name)
                
                with open(temp_file_path, 'wb') as f:
                    f.write(uploaded_resume.getvalue())
                
                try:
                    parser = DocumentParser()
                    resume_text = parser.parse_document(temp_file_path)
                    st.success("✅ Resume loaded successfully!")
                    
                    # Show preview
                    with st.expander("📖 Resume Preview"):
                        st.text_area("Resume Content Preview", resume_text[:1000] + "...", height=200, disabled=True)
                
                except Exception as e:
                    st.error(f"❌ Error parsing resume: {e}")
                    st.info("💡 Make sure your resume is in PDF, Word, or text format")
                finally:
                    # Clean up temporary files
                    try:
                        os.remove(temp_file_path)
                        os.rmdir(temp_dir)
                    except:
                        pass
    
    # Process button
    if st.button("🚀 Tailor My Resume", type="primary", use_container_width=True):
        if not api_key:
            st.error("❌ Please enter your OpenAI API key in the sidebar")
            return
        
        if not job_description:
            st.error("❌ Please provide a job description")
            return
        
        if not resume_text:
            st.error("❌ Please upload your resume")
            return
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with st.spinner("🤖 AI is tailoring your resume..."):
                # Initialize pipeline
                status_text.text("Initializing AI pipeline...")
                progress_bar.progress(10)
                
                pipeline = ResumeTailoringPipeline(api_key)
                pipeline.ai_rewriter.model = model
                
                # Save job description to temp file for processing
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as job_file:
                    job_file.write(job_description)
                    job_file_path = job_file.name
                
                status_text.text("Analyzing job description...")
                progress_bar.progress(30)
                
                # Process the resume
                status_text.text("Tailoring resume content...")
                progress_bar.progress(60)
                
                # Create temporary files
                temp_dir = tempfile.mkdtemp()
                resume_path = os.path.join(temp_dir, "resume.txt")
                job_path = os.path.join(temp_dir, "job.txt")
                output_path = os.path.join(temp_dir, "output.txt")
                
                # Save resume and job description to files
                with open(resume_path, 'w', encoding='utf-8') as f:
                    f.write(resume_text)
                with open(job_path, 'w', encoding='utf-8') as f:
                    f.write(job_description)
                
                try:
                    # Process the resume
                    results = pipeline.tailor_resume(
                        job_description=job_description,
                        resume_file=resume_path,
                        output_file=output_path,
                        mode=mode,
                        dry_run=False,
                        debug=True,
                        bullets_only=bullets_only,
                        verify_truth=verify_truth,
                        output_format=output_format
                    )
                    
                    # Read the output file if it exists
                    if os.path.exists(output_path):
                        with open(output_path, 'r', encoding='utf-8') as f:
                            results.output_text = f.read()
                    
                except Exception as e:
                    st.error(f"❌ Error processing resume: {str(e)}")
                    st.info("💡 Please check your inputs and try again")
                    return
                finally:
                    # Clean up temporary files
                    try:
                        for file in [resume_path, job_path, output_path]:
                            if os.path.exists(file):
                                os.remove(file)
                        os.rmdir(temp_dir)
                    except:
                        pass
                
                status_text.text("Generating insights...")
                progress_bar.progress(90)
                
                # Clean up temp files
                os.unlink(job_file_path)
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                # Display results
                display_results(results)
                
        except Exception as e:
            st.error(f"❌ Error processing resume: {e}")
            st.info("💡 Make sure your OpenAI API key is valid and has sufficient credits")

def display_results(results):
    """Display the tailoring results in a nice format"""
    
    st.markdown('<h2 class="section-header">📊 Results</h2>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Match Score Before",
            f"{results.debug_info['initial_match_score']:.1f}%",
            help="How well your original resume matched the job"
        )
    
    with col2:
        improvement = results.debug_info['final_match_score'] - results.debug_info['initial_match_score']
        st.metric(
            "Match Score After",
            f"{results.debug_info['final_match_score']:.1f}%",
            f"+{improvement:.1f}%",
            help="How well your tailored resume matches the job"
        )
    
    with col3:
        st.metric(
            "Bullets Processed",
            results.debug_info['total_bullets_processed'],
            help="Number of bullet points that were processed"
        )
    
    with col4:
        total_suggestions = len(results.notes) if results.notes else 0
        st.metric(
            "AI Suggestions",
            total_suggestions,
            help="Number of improvement suggestions generated"
        )
    
    # Visualization of improvement
    if results.debug_info['final_match_score'] > results.debug_info['initial_match_score']:
        fig = create_improvement_chart(
            results.debug_info['initial_match_score'],
            results.debug_info['final_match_score']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Technical Skills
    st.markdown('<h3 class="section-header">💻 Technical Skills</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Current Skills")
        for category, skills in results.technical_skills.items():
            st.markdown(f"**{category}**")
            st.markdown(", ".join(skills))
    
    with col2:
        st.subheader("⭐ Recommended Skills")
        if results.recommended_skills:
            for skill in sorted(results.recommended_skills):
                st.markdown(f"• {skill}")
        else:
            st.info("No additional skills recommended")
    
    # Notes and Suggestions
    if results.notes:
        st.markdown('<h3 class="section-header">💡 Analysis & Suggestions</h3>', unsafe_allow_html=True)
        
        # Group notes by type
        missing_skills = []
        partial_matches = []
        suggestions = []
        
        for note in results.notes:
            if note.startswith("Missing Skills"):
                missing_skills.append(note)
            elif note.startswith("Partially Matched"):
                partial_matches.append(note)
            else:
                suggestions.append(note)
        
        # Display suggestions
        if suggestions:
            st.subheader("🚀 Improvement Suggestions")
            for suggestion in suggestions:
                st.markdown(f'<div class="suggestion-item">{suggestion}</div>', 
                           unsafe_allow_html=True)
        
        # Display skill matches in columns
        if missing_skills or partial_matches:
            col1, col2 = st.columns(2)
            
            with col1:
                if missing_skills:
                    st.subheader("❌ Missing Skills")
                    for skill in missing_skills:
                        st.markdown(f"• {skill}")
            
            with col2:
                if partial_matches:
                    st.subheader("🔄 Partial Matches")
                    for match in partial_matches:
                        st.markdown(f"• {match}")
    
    # Resume Output
    st.markdown('<h3 class="section-header">📝 Tailored Resume</h3>', unsafe_allow_html=True)
    
    # Output tabs based on format
    if results.debug_info['mode'] == 'gap':
        tab1, tab2 = st.tabs(["📊 Gap Analysis", "📄 Resume"])
        
        with tab1:
            st.markdown("### 🎯 Skill Gap Analysis")
            st.markdown(results.output_text)
        
        with tab2:
            st.text_area("Resume Content", results.polished_resume, height=400)
    else:
        st.text_area("Resume Content", results.polished_resume, height=400)
    
    # Download options
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="💾 Download Resume",
            data=results.polished_resume,
            file_name="tailored_resume.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        if results.debug_info['mode'] == 'gap':
            st.download_button(
                label="📊 Download Analysis",
                data=results.output_text,
                file_name="gap_analysis.txt",
                mime="text/plain",
                use_container_width=True
            )

def create_improvement_chart(initial_score, final_score):
    """Create a chart showing the improvement in match score"""
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=['Before', 'After'],
        y=[initial_score, final_score],
        marker_color=['#ff7f0e', '#1f77b4'],
        text=[f'{initial_score:.1f}%', f'{final_score:.1f}%'],
        textposition='auto',
    ))
    
    # Add improvement arrow
    improvement = final_score - initial_score
    fig.add_annotation(
        x=0.5,
        y=max(initial_score, final_score) + 5,
        text=f"↗️ +{improvement:.1f}% improvement",
        showarrow=False,
        font=dict(size=14, color='green'),
        bgcolor='rgba(144, 238, 144, 0.8)',
        bordercolor='green',
        borderwidth=1
    )
    
    fig.update_layout(
        title="Resume-Job Match Score Improvement",
        yaxis_title="Match Score (%)",
        showlegend=False,
        height=300,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

# Additional utility functions
def show_help():
    """Show help information"""
    st.markdown("""
    ## 🤔 How to Use This Tool
    
    1. **Get an OpenAI API Key**: Visit [OpenAI's website](https://platform.openai.com/api-keys) to get your API key
    2. **Provide Job Description**: Copy-paste or upload the job description you're applying for
    3. **Upload Resume**: Upload your current resume (PDF, Word, or text format)
    4. **Click "Tailor My Resume"**: Let AI analyze and improve your resume
    5. **Review Results**: Check the improvements and download your tailored resume
    
    ## 🎯 What This Tool Does
    
    - **Analyzes** job descriptions to extract key requirements and skills
    - **Rewrites** your resume bullet points to better match the job
    - **Suggests** additional content you might want to include
    - **Scores** how well your resume matches before and after
    
    ## 💡 Tips for Best Results
    
    - Use complete, detailed job descriptions
    - Ensure your resume is well-formatted and complete
    - Review AI suggestions carefully - they're starting points, not final content
    - Always be truthful about your experience and skills
    """)

# Sidebar navigation
with st.sidebar:
    st.markdown("---")
    
    if st.button("❓ Help & Instructions"):
        show_help()
    
    st.markdown("---")
    st.markdown("### 🔧 About This Tool")
    st.markdown("""
    This tool uses AI to help you tailor your resume to specific job descriptions. 
    It identifies key requirements and rewrites your content to better match what employers are looking for.
    """)
    
    st.markdown("### ⚠️ Important Notes")
    st.markdown("""
    - Always review AI-generated content
    - Maintain truthfulness in your resume
    - Use this as a starting point for improvements
    - Your API usage will be charged by OpenAI
    """)

if __name__ == "__main__":
    main()