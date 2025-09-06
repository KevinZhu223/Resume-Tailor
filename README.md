# Resume-Tailor

An intelligent resume tailoring system that helps customize your resume for specific job descriptions while maintaining truthfulness and professionalism.

## 🎯 Features

### Multiple Operation Modes

1. **Polish Mode**
   - Improves readability and professionalism
   - Uses strong action verbs
   - Enhances clarity and impact
   - Preserves all technical details

2. **Tailor Mode**
   - Aligns resume with job requirements
   - Integrates relevant skills naturally
   - Maintains truthfulness
   - Enhances technical skills section

3. **Gap Analysis Mode**
   - Identifies missing skills from job requirements
   - Shows partially matched skills
   - Provides improvement suggestions
   - Helps with skill development planning

### Output Formats

1. **Standard Format**
   - Clean, professional resume output
   - Maintains original structure
   - Perfect for direct submission

2. **Detailed Format**
   - Complete analysis with sections:
     - Polished Resume
     - Technical Skills
     - Recommended Skills
     - Notes and Suggestions
     - Debug Information

### Key Features

- ✅ Accurate job requirement parsing
- ✅ Bullet point preservation
- ✅ Natural skill integration
- ✅ Duplicate prevention
- ✅ Truth verification
- ✅ Technical skill categorization
- ✅ Comprehensive gap analysis
- ✅ Multiple output formats

## 🚀 Getting Started

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix/Linux/Mac
   .\venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Set API Key**
   ```bash
   export OPENAI_API_KEY=your-api-key  # Unix/Linux/Mac
   set OPENAI_API_KEY=your-api-key     # Windows
   ```

3. **Run the Script**
   ```bash
   # Polish Mode
   python resume_tailor.py tailor --job-file job.txt --resume-file resume.pdf --mode polish

   # Tailor Mode
   python resume_tailor.py tailor --job-file job.txt --resume-file resume.pdf --mode tailor

   # Gap Analysis
   python resume_tailor.py tailor --job-file job.txt --resume-file resume.pdf --mode gap
   ```

## 💻 Web Interface

A Streamlit-based web interface is available for easier interaction:
```bash
streamlit run streamlit_app.py
```

## 📋 Usage Tips

1. **Choose the Right Mode**
   - Use Polish for general improvements
   - Use Tailor for job applications
   - Use Gap Analysis for skill development

2. **Output Formats**
   - Standard: Clean resume output
   - Detailed: Full analysis with recommendations

3. **Best Practices**
   - Keep job descriptions in text format
   - Use PDF/DOCX for resumes
   - Review and verify outputs
   - Use gap analysis for preparation

## 🛠️ Technical Details

- Python 3.8+
- OpenAI GPT-3.5
- spaCy for NLP
- NLTK for text processing
- Streamlit for web interface

## 📝 Recent Improvements

- Enhanced job requirement parsing
- Improved bullet point preservation
- Added multiple operation modes
- Implemented detailed gap analysis
- Added truth verification system
- Enhanced technical skills handling
- Improved output formatting
- Added web interface

## 🔜 Future Plans

- Enhanced skill matching algorithms
- More output format options
- Additional customization options
- Improved web interface
- Integration with job boards
