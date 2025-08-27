from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path
import json
import hashlib
import os

# Document processing
import PyPDF2
from docx import Document
import pdfplumber

# NLP and analysis
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from openai import OpenAI

# Web interface (optional)
import streamlit as st

# CLI interface
import click

from dotenv import load_dotenv

load_dotenv()

# Configuration
MAX_API_CALLS = 5  # Configurable limit for API calls
CACHE_FILE = "resume_cache.json"  # Cache file location

#Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResumeSection:
    """Represents a section of a resume"""
    title: str
    content: List[str]
    section_type: str
    
@dataclass
class JobAnalysis:
    """Results of job description analysis"""
    keywords: List[str]
    required_skills: List[str]
    preferred_skills: List[str]
    technologies: List[str]
    responsibilities: List[str]

class BulletPointCache:
    """Manages caching of rewritten bullet points"""
    
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, str]:
        """Load cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Could not save cache: {e}")
    
    def _generate_key(self, bullet_text: str, job_description: str) -> str:
        """Generate cache key from bullet text and job description"""
        combined = f"{bullet_text}|||{job_description}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, bullet_text: str, job_description: str) -> Optional[str]:
        """Get cached rewritten bullet point"""
        key = self._generate_key(bullet_text, job_description)
        return self.cache.get(key)
    
    def set(self, bullet_text: str, job_description: str, rewritten: str):
        """Cache rewritten bullet point"""
        key = self._generate_key(bullet_text, job_description)
        self.cache[key] = rewritten
        self._save_cache()
    
    def clear(self):
        """Clear cache"""
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

class DocumentParser:
    """Handles parsing of various document formats"""
    
    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """Extract text from PDF using pdfplumber for better formatting"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:  # Only add if not None
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error parsing PDF with pdfplumber: {e}")
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:  # Only add if not None
                            text += page_text + "\n"
            except Exception as e2:
                logger.error(f"Error parsing PDF with PyPDF2: {e2}")
                return ""
        
        return text.strip()
    
    def _sections_to_text(self, sections: List[ResumeSection]) -> str:
        """Convert sections back to text format"""
        text = ""
        for section in sections:
            text += f"{section.title}\n"
            for item in section.content:
                text += f"{item}\n"
            text += "\n"
        return text.strip()
    
    def _generate_bullets_only_output(self, bullet_changes: List[Dict], job_analysis: JobAnalysis) -> str:
        """Generate output file containing only the rewritten bullet points with context"""
        output = []
        output.append("=" * 80)
        output.append("TAILORED RESUME BULLET POINTS")
        output.append("=" * 80)
        output.append("")
        
        # Add job analysis summary
        output.append("JOB REQUIREMENTS SUMMARY:")
        output.append("-" * 40)
        if job_analysis.required_skills:
            output.append(f"• Required Skills: {', '.join(job_analysis.required_skills[:8])}")
        if job_analysis.technologies:
            output.append(f"• Technologies: {', '.join(job_analysis.technologies[:8])}")
        if job_analysis.keywords:
            output.append(f"• Key Keywords: {', '.join(job_analysis.keywords[:10])}")
        output.append("")
        output.append("=" * 80)
        output.append("")
        
        # Group changes by section
        sections = {}
        for change in bullet_changes:
            section = change['section']
            if section not in sections:
                sections[section] = []
            sections[section].append(change)
        
        # Generate output for each section
        for section_title, changes in sections.items():
            output.append(f"SECTION: {section_title}")
            output.append("-" * 60)
            
            # Add context for the section
            if changes:
                context = changes[0]['context']
                output.append(f"Context: {context}")
                output.append("")
            
            # Add each bullet point change
            for i, change in enumerate(changes, 1):
                output.append(f"BULLET POINT {i}:")
                output.append("")
                output.append("ORIGINAL:")
                output.append(f"  {change['original']}")
                output.append("")
                output.append("REWRITTEN:")
                output.append(f"  ● {change['rewritten']}")
                output.append("")
                output.append("-" * 40)
                output.append("")
            
            output.append("=" * 80)
            output.append("")
        
        # Add summary
        total_bullets = len(bullet_changes)
        output.append(f"SUMMARY:")
        output.append(f"• Total bullet points rewritten: {total_bullets}")
        output.append(f"• Sections modified: {len(sections)}")
        output.append("")
        output.append("INSTRUCTIONS:")
        output.append("• Copy the REWRITTEN bullet points to replace the corresponding")
        output.append("  original bullets in your resume")
        output.append("• The bullet points are organized by section for easy reference")
        output.append("• Maintain the same formatting and order in your original resume")
        
        return "\n".join(output)

    @staticmethod
    def parse_docx(file_path: str) -> str:
        """Extract text from Word doc"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error parsing DOCX: {e}")
            return ""
    
    @staticmethod
    def parse_text(file_path: str) -> str:
        """Read plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return ""
    
    def parse_document(self, file_path: str) -> str:
        """Parse document based on file extension"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension == '.pdf':
            return self.parse_pdf(str(file_path))
        elif extension in ['.docx', '.doc']:
            return self.parse_docx(str(file_path))
        elif extension == '.txt':
            return self.parse_text(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
class JobDescriptionAnalyzer:
    """Analyzes job descriptions to get information"""

    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.stop_words = set(stopwords.words('english'))

        # Common technical skills and keywords
        self.tech_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift', 'R'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask'],
            'data': ['sql', 'mysql', 'postgresql', 'mongodb', 'pandas', 'numpy', 'scikit-learn'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'tools': ['git', 'jenkins', 'jira', 'confluence', 'slack', 'teams']
        }
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """Extract important keywords using TF-IDF"""
        if not text or not text.strip():
            return []
            
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        if not cleaned_text.strip():
            return []

        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1,2)
        )

        try:
            tfidf_matrix = vectorizer.fit_transform([cleaned_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
            return [kw[0] for kw in keyword_scores[:top_n]]
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
        
    def extract_skills(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract required and preferred skills"""
        if not text:
            return [], []
            
        text_lower = text.lower()

        # Find sections mentioning requirements
        required_patterns = [
            r'required[\s\w]*?skills?[:\-\s]*(.*?)(?=preferred|nice|plus|\n\n|$)',
            r'must have[:\-\s]*(.*?)(?=preferred|nice|plus|\n\n|$)',
            r'requirements?[:\-\s]*(.*?)(?=preferred|nice|plus|\n\n|$)'
        ]
        
        preferred_patterns = [
            r'preferred[\s\w]*?skills?[:\-\s]*(.*?)(?=\n\n|$)',
            r'nice to have[:\-\s]*(.*?)(?=\n\n|$)',
            r'plus[:\-\s]*(.*?)(?=\n\n|$)',
            r'bonus[:\-\s]*(.*?)(?=\n\n|$)'
        ]
        
        required_skills = []
        preferred_skills = []

        # Extract required skills
        for pattern in required_patterns:
            matches = re.findall(pattern, text_lower, re.DOTALL | re.IGNORECASE)
            for match in matches:
                skills = self._extract_skills_from_text(match)
                required_skills.extend(skills)
        
        # Extract preferred skills
        for pattern in preferred_patterns:
            matches = re.findall(pattern, text_lower, re.DOTALL | re.IGNORECASE)
            for match in matches:
                skills = self._extract_skills_from_text(match)
                preferred_skills.extend(skills)
        
        return list(set(required_skills)), list(set(preferred_skills))
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from a block of text"""
        if not text:
            return []
            
        skills = []

        # Check for technical keywords
        for category, keywords in self.tech_keywords.items():
            for keyword in keywords:
                if keyword in text.lower():
                    skills.append(keyword)
        
        # Extract other potential skills (noun phrases)
        if self.nlp:
            try:
                doc = self.nlp(text)
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) <= 3 and chunk.text.lower() not in self.stop_words:
                        skills.append(chunk.text.strip())
            except Exception as e:
                logger.warning(f"Error processing text with spaCy: {e}")
        
        return skills
    
    def analyze_job_description(self, job_text: str) -> JobAnalysis:
        """Comprehensive analysis of job description"""
        if not job_text or not job_text.strip():
            return JobAnalysis(
                keywords=[],
                required_skills=[],
                preferred_skills=[],
                technologies=[],
                responsibilities=[]
            )
            
        keywords = self.extract_keywords(job_text)
        required_skills, preferred_skills = self.extract_skills(job_text)
        
        # Extract technologies (intersection of keywords and tech terms)
        all_tech = []
        for tech_list in self.tech_keywords.values():
            all_tech.extend(tech_list)
        
        technologies = [kw for kw in keywords if any(tech in kw.lower() for tech in all_tech)]
        
        # Extract responsibility keywords
        responsibilities = [kw for kw in keywords if any(word in kw for word in 
                         ['develop', 'design', 'implement', 'manage', 'lead', 'analyze', 'create'])]
        
        return JobAnalysis(
            keywords=keywords,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            technologies=technologies,
            responsibilities=responsibilities
        )
    
class ResumeParser:
    """Parses resume content into structured sections"""
    
    def __init__(self):
        self.section_headers = {
            'experience': ['experience', 'work experience', 'employment', 'professional experience', 'research experience'],
            'education': ['education', 'academic background', 'qualifications'],
            'skills': ['skills', 'technical skills', 'competencies', 'technologies'],
            'projects': ['projects', 'personal projects', 'relevant projects', 'project experience'],
            'achievements': ['achievements', 'accomplishments', 'awards'],
            'research': ['research', 'research experience'],
            'involvement': ['involvement', 'campus involvement', 'activities', 'leadership']
        }
    
    def parse_resume(self, resume_text: str) -> List[ResumeSection]:
        """Parse resume into sections"""
        if not resume_text or not resume_text.strip():
            return []
            
        sections = []
        lines = resume_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            section_type = self._identify_section(line)
            if section_type:
                # Save previous section
                if current_section:
                    sections.append(ResumeSection(
                        title=current_section,
                        content=current_content,
                        section_type=self._get_section_type(current_section)
                    ))
                
                current_section = line
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section:
            sections.append(ResumeSection(
                title=current_section,
                content=current_content,
                section_type=self._get_section_type(current_section)
            ))
        
        return sections
    
    def _identify_section(self, line: str) -> Optional[str]:
        """Identify if a line is a section header"""
        if not line:
            return None
            
        line_lower = line.lower().strip()
        
        # Check for common section patterns
        if any(header in line_lower for headers in self.section_headers.values() for header in headers):
            return line
        
        # Check for lines that are all caps, have colons, or match common patterns
        if (line.isupper() or 
            ':' in line or 
            (len(line.split()) <= 4 and 
             any(keyword in line_lower for keyword in ['experience', 'education', 'skills', 'projects', 'research', 'involvement']))):
            return line
        
        return None
    
    def _get_section_type(self, section_title: str) -> str:
        """Determine the type of section"""
        if not section_title:
            return 'other'
            
        title_lower = section_title.lower()
        
        for section_type, headers in self.section_headers.items():
            if any(header in title_lower for header in headers):
                return section_type
        
        return 'other'
    

class AIResumeRewriter:
    """Uses AI to rewrite resume content to match job descriptions"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", max_api_calls: int = MAX_API_CALLS):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_api_calls = max_api_calls
        self.api_calls_made = 0
        self.cache = BulletPointCache()
    
    def rewrite_bullet_points_batch(self, bullet_points: List[str], job_analysis: JobAnalysis, 
                                  context: str = "", job_description: str = "", 
                                  dry_run: bool = False) -> List[str]:
        """Rewrite multiple bullet points in a single API call"""
        
        if dry_run:
            logger.info(f"DRY RUN: Would rewrite {len(bullet_points)} bullet points:")
            for i, bullet in enumerate(bullet_points, 1):
                logger.info(f"  {i}. {bullet}")
            return bullet_points  # Return original bullets in dry run
        
        # Check cache first
        cached_bullets = []
        uncached_bullets = []
        uncached_indices = []
        
        for i, bullet in enumerate(bullet_points):
            cached = self.cache.get(bullet, job_description)
            if cached:
                cached_bullets.append((i, cached))
            else:
                uncached_bullets.append(bullet)
                uncached_indices.append(i)
        
        logger.info(f"Found {len(cached_bullets)} cached bullets, {len(uncached_bullets)} need rewriting")
        
        # If all bullets are cached, return them
        if not uncached_bullets:
            result = [''] * len(bullet_points)
            for idx, cached_bullet in cached_bullets:
                result[idx] = cached_bullet
            return result
        
        # Check API call limit
        if self.api_calls_made >= self.max_api_calls:
            logger.warning(f"API call limit ({self.max_api_calls}) reached. Skipping rewrite step.")
            return bullet_points
        
        # Prepare batch prompt with clear instructions
        bullet_list = "\n".join([f"{i+1}. {bullet}" for i, bullet in enumerate(uncached_bullets)])
        
        prompt = f"""
        Rewrite the following resume bullet points to better match the job requirements while maintaining truthfulness.
        Return ONLY the rewritten bullets in the same numbered format (1., 2., 3., etc.), preserving the exact order.
        Do not include any headers, explanations, or additional text.
        
        Original bullet points:
        {bullet_list}
        
        Job requirements context:
        - Key skills needed: {', '.join(job_analysis.required_skills[:10])}
        - Technologies mentioned: {', '.join(job_analysis.technologies[:8])}
        - Important keywords: {', '.join(job_analysis.keywords[:15])}
        
        Additional context: {context}
        
        Guidelines:
        1. Keep the core truth and achievements of the original
        2. Use keywords from the job description where appropriate
        3. Quantify results when possible
        4. Use strong action verbs
        5. Make it concise and impactful
        6. Ensure it sounds natural and not keyword-stuffed
        7. Return EXACTLY {len(uncached_bullets)} rewritten bullet points in numbered format
        8. Do not include bullet symbols (●, •, -) in your response - just the numbered text
        
        Rewritten bullet points:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert resume writer who helps tailor resumes to job descriptions while maintaining truthfulness. Always return the exact number of bullet points requested in numbered format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150 * len(uncached_bullets),  # Scale tokens with number of bullets
                temperature=0.7
            )
            
            self.api_calls_made += 1
            logger.info(f"API calls made: {self.api_calls_made}/{self.max_api_calls}")
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            rewritten_bullets = self._parse_numbered_response(response_text, len(uncached_bullets))
            
            # Cache the results
            for original, rewritten in zip(uncached_bullets, rewritten_bullets):
                self.cache.set(original, job_description, rewritten)
            
            # Combine cached and new results
            result = [''] * len(bullet_points)
            
            # Add cached bullets
            for idx, cached_bullet in cached_bullets:
                result[idx] = cached_bullet
            
            # Add newly rewritten bullets
            for result_idx, rewritten in zip(uncached_indices, rewritten_bullets):
                result[result_idx] = rewritten
            
            return result
            
        except Exception as e:
            logger.error(f"Error rewriting bullet points: {e}")
            return bullet_points  # Return originals if AI fails
    
    def _parse_numbered_response(self, response_text: str, expected_count: int) -> List[str]:
        """Parse numbered response from AI and extract bullet points"""
        if not response_text:
            return [f"[Rewrite failed for bullet {i+1}]" for i in range(expected_count)]
            
        lines = response_text.strip().split('\n')
        bullets = []
        
        # First try to find numbered bullets
        for line in lines:
            line = line.strip()
            # Look for numbered lines (1. 2. etc.)
            if re.match(r'^\d+\.\s+', line):
                bullet = re.sub(r'^\d+\.\s+', '', line).strip()
                if bullet:  # Only add non-empty bullets
                    bullets.append(bullet)
        
        # If we didn't get enough numbered bullets, try other patterns
        if len(bullets) < expected_count:
            logger.warning(f"Only found {len(bullets)} numbered bullets, looking for other patterns")
            bullets = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Skip headers or meta text
                if line.lower().startswith(('rewritten', 'bullet point', 'original')):
                    continue
                    
                # Look for bullet-like content (remove various prefixes)
                clean_line = re.sub(r'^[\d\.\-\*\•●]\s*', '', line).strip()
                
                # Must be substantial content (not just numbers or short phrases)
                if len(clean_line) > 15 and not clean_line.isdigit():
                    bullets.append(clean_line)
                    
                if len(bullets) >= expected_count:
                    break
        
        # Ensure we have the right number of bullets
        if len(bullets) < expected_count:
            logger.warning(f"Expected {expected_count} bullets, only got {len(bullets)}")
            # Use original bullets as fallback for missing ones
            while len(bullets) < expected_count:
                bullets.append(f"[Rewrite failed for bullet {len(bullets) + 1}]")
        
        return bullets[:expected_count]
    
    def suggest_new_bullet_points(self, job_analysis: JobAnalysis, 
                                resume_sections: List[ResumeSection],
                                dry_run: bool = False) -> List[str]:
        """Suggest new bullet points based on job requirements and existing resume"""
        
        if dry_run:
            logger.info("DRY RUN: Would generate new bullet point suggestions")
            return []
        
        # Check API call limit
        if self.api_calls_made >= self.max_api_calls:
            logger.warning(f"API call limit ({self.max_api_calls}) reached. Skipping suggestions.")
            return []
        
        existing_content = "\n".join([
            f"{section.title}: {' '.join(section.content)}" 
            for section in resume_sections
        ])
        
        prompt = f"""
        Based on the job requirements and the candidate's existing resume, suggest new bullet points 
        that could strengthen their application. These should be realistic additions based on their background.
        
        Job requirements:
        - Required skills: {', '.join(job_analysis.required_skills[:10])}
        - Preferred skills: {', '.join(job_analysis.preferred_skills[:8])}
        - Key responsibilities: {', '.join(job_analysis.responsibilities[:8])}
        
        Existing resume content:
        {existing_content[:1000]}...
        
        Suggest 3-5 bullet points that could be added to strengthen the resume for this job.
        Focus on skills/experiences the candidate likely has but may not have highlighted.
        
        Format as a numbered list.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert career counselor who helps identify missing elements in resumes."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            self.api_calls_made += 1
            logger.info(f"API calls made: {self.api_calls_made}/{self.max_api_calls}")
            
            suggestions = response.choices[0].message.content.strip()
            # Parse numbered list
            return [line.strip() for line in suggestions.split('\n') if line.strip() and any(c.isdigit() for c in line[:3])]
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []
    
class ResumeTailoringPipeline:
    """Main pipeline that orchestrates the resume tailoring process"""
    
    def __init__(self, openai_api_key: str, max_api_calls: int = MAX_API_CALLS):
        self.doc_parser = DocumentParser()
        self.job_analyzer = JobDescriptionAnalyzer()
        self.resume_parser = ResumeParser()
        self.ai_rewriter = AIResumeRewriter(openai_api_key, max_api_calls=max_api_calls)
    
    def detect_bullets(self, lines: List[str], section_type: str) -> Tuple[List[str], List[str]]:
        """
        Detect bullet points in resume sections, handling various formats and multi-line bullets.
        
        Args:
            lines: List of text lines from a resume section
            section_type: Type of section (experience, projects, etc.)
        
        Returns:
            Tuple of (bullet_points, non_bullet_content)
        """
        # Define bullet pattern regex
        BULLET_REGEX = re.compile(r"^\s*(?:[-*•●▪‣])\s+")
        
        # Action verbs that indicate bullet points in certain sections
        ACTION_VERBS = {
            'developed', 'built', 'promoted', 'attended', 'engaged', 'led', 
            'created', 'engineered', 'implemented', 'analyzed', 'collaborated', 
            'researched', 'managed', 'designed', 'initiated', 'organized', 
            'planned', 'prepared', 'produced', 'programmed', 'qualified', 
            'recruited', 'reduced', 'represented', 'resolved', 'restructured', 
            'revised', 'reviewed', 'scheduled', 'selected', 'served', 'set', 
            'shaped', 'spread', 'streamlined', 'strengthened', 'summarized', 
            'supervised', 'supported', 'trained', 'transformed', 'utilized', 
            'volunteered', 'won', 'wrote', 'awarded', 'cleaned', 'visualized', 
            'practiced'
        }
        
        bullet_points = []
        non_bullet_content = []
        current_bullet = None
        
        # Only apply advanced detection to relevant sections
        relevant_sections = {'experience', 'projects', 'research', 'involvement'}
        use_action_verbs = section_type in relevant_sections
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
                
            # Check if this line starts a new bullet
            is_bullet = False
            
            # Check for bullet symbols
            if BULLET_REGEX.match(line_stripped):
                is_bullet = True
            # Check for action verbs in relevant sections
            elif use_action_verbs:
                first_word = line_stripped.split()[0].lower()
                if first_word in ACTION_VERBS:
                    is_bullet = True
            
            if is_bullet:
                # Save previous bullet if exists
                if current_bullet is not None:
                    bullet_points.append(current_bullet)
                
                # Start new bullet (remove bullet symbol if present)
                if BULLET_REGEX.match(line_stripped):
                    current_bullet = BULLET_REGEX.sub('', line_stripped, count=1).strip()
                else:
                    current_bullet = line_stripped
            else:
                # This is not a bullet starter
                if current_bullet is not None:
                    # Continue the current bullet
                    current_bullet += ' ' + line_stripped
                else:
                    # This is non-bullet content
                    non_bullet_content.append(line_stripped)
        
        # Add the last bullet if exists
        if current_bullet is not None:
            bullet_points.append(current_bullet)
        
        return bullet_points, non_bullet_content
    
    def tailor_resume(self, job_description: str, resume_file: str, 
                     output_file: str = None, dry_run: bool = False, debug: bool = False,
                     bullets_only: bool = False) -> Dict:
        """Complete pipeline to tailor resume to job description"""
        
        logger.info("Starting resume tailoring process...")
        
        # Step 1: Parse documents
        logger.info("Parsing resume...")
        resume_text = self.doc_parser.parse_document(resume_file)
        
        if not resume_text or not resume_text.strip():
            raise ValueError(f"Could not extract text from resume file: {resume_file}")
        
        resume_sections = self.resume_parser.parse_resume(resume_text)
        
        # Debug: Show parsed sections
        if debug:
            logger.info("=== PARSED SECTIONS DEBUG ===")
            for i, section in enumerate(resume_sections):
                logger.info(f"Section {i}: {section.title} (type: {section.section_type})")
                for j, content in enumerate(section.content):
                    logger.info(f"  Content {j}: '{content[:100]}...'")
            logger.info("=== END SECTIONS DEBUG ===")
        
        # Step 2: Analyze job description
        logger.info("Analyzing job description...")
        job_analysis = self.job_analyzer.analyze_job_description(job_description)
        
        # Step 3: Calculate initial match score
        initial_score = self._calculate_match_score(resume_text, job_analysis)
        
        # Step 4: Rewrite relevant sections in batches
        logger.info("Processing resume content...")
        tailored_sections = []
        total_bullets_processed = 0
        bullet_changes = []  # Track bullet changes for bullets-only output
        
        for section in resume_sections:
            if section.section_type in ['experience', 'projects', 'research']:
                # Extract bullet points - more flexible detection
                bullet_points, non_bullet_content = self.detect_bullets(section.content, section.section_type)
                
                if debug:
                    logger.info(f"=== {section.section_type.upper()} SECTION DEBUG ===")
                    logger.info(f"Total content items: {len(section.content)}")
                    logger.info(f"Detected bullet points: {len(bullet_points)}")
                    for bp in bullet_points:
                        logger.info(f"  Bullet: '{bp[:60]}...'")
                    logger.info(f"Non-bullet content: {len(non_bullet_content)}")
                    logger.info("=== END SECTION DEBUG ===")
                
                if bullet_points:
                    logger.info(f"Processing {len(bullet_points)} bullets from {section.section_type} section")
                    
                    # Rewrite all bullet points in this section at once
                    rewritten_bullets = self.ai_rewriter.rewrite_bullet_points_batch(
                        bullet_points, 
                        job_analysis, 
                        f"This is from the {section.section_type} section",
                        job_description,
                        dry_run
                    )
                    
                    # Track changes for bullets-only output
                    if bullets_only and not dry_run:
                        # Find the section context (company, role, etc.)
                        context_lines = []
                        for item in section.content:
                            if not (item.strip().startswith(('•', '-', '*', '●')) or 
                                   (len(item.strip()) > 20 and 
                                    any(item.strip().lower().startswith(verb) for verb in 
                                        ['developed', 'built', 'created', 'implemented', 'designed', 
                                         'analyzed', 'managed', 'led', 'collaborated', 'engineered',
                                         'promoted', 'leading', 'awarded', 'cleaned', 'visualized',
                                         'attend', 'practice', 'engaged']))):
                                context_lines.append(item.strip())
                        
                        section_context = section.title
                        if context_lines:
                            section_context += f" - {' | '.join(context_lines[:2])}"
                        
                        for orig, rewritten in zip(bullet_points, rewritten_bullets):
                            bullet_changes.append({
                                'section': section.title,
                                'section_type': section.section_type,
                                'context': section_context,
                                'original': orig.strip(),
                                'rewritten': rewritten.strip()
                            })
                    
                    # Combine non-bullet content with rewritten bullets
                    new_content = []
                    bullet_idx = 0
                    for item in section.content:
                        item_stripped = item.strip()
                        # Use same detection logic as above
                        if (item_stripped.startswith('•') or 
                            item_stripped.startswith('-') or 
                            item_stripped.startswith('*') or
                            item_stripped.startswith('●') or
                            (len(item_stripped) > 20 and 
                             any(item_stripped.lower().startswith(verb) for verb in 
                                 ['developed', 'built', 'created', 'implemented', 'designed', 
                                  'analyzed', 'managed', 'led', 'collaborated', 'engineered',
                                  'promoted', 'leading', 'awarded', 'cleaned', 'visualized',
                                  'attend', 'practice', 'engaged']))):
                            if bullet_idx < len(rewritten_bullets):
                                # Preserve original bullet formatting
                                if item_stripped.startswith(('•', '-', '*', '●')):
                                    prefix = item_stripped[0] + ' '
                                    new_content.append(prefix + rewritten_bullets[bullet_idx])
                                else:
                                    new_content.append('● ' + rewritten_bullets[bullet_idx])
                                bullet_idx += 1
                            else:
                                new_content.append(item)
                        else:
                            new_content.append(item)
                    
                    total_bullets_processed += len(bullet_points)
                    
                    tailored_sections.append(ResumeSection(
                        title=section.title,
                        content=new_content,
                        section_type=section.section_type
                    ))
                else:
                    tailored_sections.append(section)
            else:
                tailored_sections.append(section)
        
        # Step 5: Generate suggestions
        logger.info("Generating improvement suggestions...")
        suggestions = self.ai_rewriter.suggest_new_bullet_points(
            job_analysis, resume_sections, dry_run
        )
        
        # Step 6: Create output
        tailored_resume_text = self._sections_to_text(tailored_sections)
        final_score = self._calculate_match_score(tailored_resume_text, job_analysis)
        
        # Save output if specified and not in dry run
        if output_file and not dry_run:
            if bullets_only and bullet_changes:
                # Generate bullets-only output
                bullets_output = self.doc_parser._generate_bullets_only_output(bullet_changes, job_analysis)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(bullets_output)
            else:
                # Generate full resume output
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(tailored_resume_text)
        
        return {
            'original_resume': resume_text,
            'tailored_resume': tailored_resume_text,
            'job_analysis': job_analysis,
            'initial_match_score': initial_score,
            'final_match_score': final_score,
            'improvement_suggestions': suggestions,
            'total_bullets_processed': total_bullets_processed,
            'api_calls_made': self.ai_rewriter.api_calls_made,
            'dry_run': dry_run,
            'bullet_changes': bullet_changes if bullets_only else [],
            'bullets_only': bullets_only
        }
    
    def _calculate_match_score(self, resume_text: str, job_analysis: JobAnalysis) -> float:
        """Calculate how well resume matches job requirements"""
        if not resume_text or not resume_text.strip():
            return 0.0
            
        resume_lower = resume_text.lower()
        
        # Count keyword matches
        keyword_matches = sum(1 for kw in job_analysis.keywords if kw.lower() in resume_lower)
        skill_matches = sum(1 for skill in job_analysis.required_skills if skill.lower() in resume_lower)
        tech_matches = sum(1 for tech in job_analysis.technologies if tech.lower() in resume_lower)
        
        total_requirements = len(job_analysis.keywords) + len(job_analysis.required_skills) + len(job_analysis.technologies)
        total_matches = keyword_matches + skill_matches + tech_matches
        
        return (total_matches / max(total_requirements, 1)) * 100
    
    def _sections_to_text(self, sections: List[ResumeSection]) -> str:
        """Convert sections back to text format"""
        text = ""
        for section in sections:
            text += f"{section.title}\n"
            for item in section.content:
                text += f"{item}\n"
            text += "\n"
        return text.strip()

# CLI Interface
@click.command()
@click.option('--job-file', '-j', required=True, help='Path to job description file')
@click.option('--resume-file', '-r', required=True, help='Path to resume file')
@click.option('--output-file', '-o', help='Path to output tailored resume')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
@click.option('--max-calls', default=MAX_API_CALLS, help=f'Maximum API calls allowed (default: {MAX_API_CALLS})')
@click.option('--dry-run', is_flag=True, help='Run without making API calls (preview mode)')
@click.option('--clear-cache', is_flag=True, help='Clear the bullet point cache before running')
@click.option('--debug', is_flag=True, help='Enable debug output to see bullet detection')
@click.option('--bullets-only', is_flag=True, help='Output only the rewritten bullet points with context (not the full resume)')
def main(job_file, resume_file, output_file, api_key, max_calls, dry_run, clear_cache, debug, bullets_only):
    """Tailor resume to match job description using AI"""
    
    if not dry_run and not api_key:
        click.echo("Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key")
        return
    
    if clear_cache:
        cache = BulletPointCache()
        cache.clear()
        click.echo("✅ Cache cleared")
        if not job_file:  # If only clearing cache
            return
    
    if not output_file:
        output_file = f"tailored_{Path(resume_file).stem}.txt"
    
    try:
        # Read job description
        with open(job_file, 'r', encoding='utf-8') as f:
            job_description = f.read()
        
        if not job_description.strip():
            click.echo("Error: Job description file is empty")
            return
        
        # Initialize pipeline
        if dry_run:
            # Use dummy API key for dry run
            pipeline = ResumeTailoringPipeline("dummy-key", max_api_calls=max_calls)
        else:
            pipeline = ResumeTailoringPipeline(api_key, max_api_calls=max_calls)
        
        # Process resume
        results = pipeline.tailor_resume(job_description, resume_file, output_file, dry_run, debug, bullets_only)
        
        # Display results
        if dry_run:
            click.echo(f"\n🔍 DRY RUN COMPLETED")
            click.echo(f"📄 Would process: {resume_file}")
            click.echo(f"📊 Current match score: {results['initial_match_score']:.1f}%")
            click.echo(f"🔧 Bullets that would be rewritten: {results['total_bullets_processed']}")
            click.echo(f"📞 API calls that would be made: ~{results['api_calls_made'] if results['api_calls_made'] > 0 else 'up to 2'}")
            if bullets_only:
                click.echo(f"📋 Would generate bullets-only output to: {output_file}")
            else:
                click.echo(f"💾 Would generate full resume output to: {output_file}")
        else:
            click.echo(f"\n✅ Resume tailoring completed!")
            if bullets_only:
                click.echo(f"📋 Rewritten bullet points saved to: {output_file}")
                click.echo(f"📊 {len(results['bullet_changes'])} bullet points were rewritten")
            else:
                click.echo(f"📄 Tailored resume saved to: {output_file}")
            click.echo(f"📊 Match score improved: {results['initial_match_score']:.1f}% → {results['final_match_score']:.1f}%")
            click.echo(f"🔧 Bullets processed: {results['total_bullets_processed']}")
            click.echo(f"📞 API calls made: {results['api_calls_made']}/{max_calls}")
        
        if results['improvement_suggestions']:
            click.echo(f"\n💡 Suggestions for further improvement:")
            for i, suggestion in enumerate(results['improvement_suggestions'], 1):
                click.echo(f"   {i}. {suggestion}")
        
        # Show cache info
        if not dry_run:
            cache = BulletPointCache()
            cache_size = len(cache.cache)
            click.echo(f"💾 Cache contains {cache_size} rewritten bullet points")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")

# Additional CLI commands for cache management
@click.group()
def cli():
    """Resume tailoring tool with caching and batch processing"""
    pass

@cli.command()
def clear_cache():
    """Clear the bullet point cache"""
    cache = BulletPointCache()
    cache.clear()
    click.echo("✅ Cache cleared")

@cli.command()
def cache_info():
    """Show cache information"""
    cache = BulletPointCache()
    cache_size = len(cache.cache)
    click.echo(f"Cache file: {cache.cache_file}")
    click.echo(f"Cached items: {cache_size}")
    
    if cache_size > 0:
        # Show some sample cache keys (first 5)
        sample_keys = list(cache.cache.keys())[:5]
        click.echo("\nSample cache entries:")
        for key in sample_keys:
            click.echo(f"  {key[:16]}... -> {cache.cache[key][:50]}...")

# Add the main command to the CLI group
cli.add_command(main, name='tailor')

if __name__ == "__main__":
    cli()