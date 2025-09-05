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
        """Generate cache key from bullet text and job description hash"""
        # Create a hash of the job description to ensure cache invalidation when job changes
        job_hash = hashlib.md5(job_description.encode()).hexdigest()[:8]  # Use first 8 chars for brevity
        combined = f"{bullet_text}|||{job_hash}"
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
        
        # Group changes by section, but clean up section titles
        sections = {}
        for change in bullet_changes:
            section_title = change['section']
            
            # Clean up malformed section titles (remove bullet symbols)
            section_title = re.sub(r'^[•\-*●◆▪–]\s*', '', section_title).strip()
            
            if section_title not in sections:
                sections[section_title] = []
            sections[section_title].append(change)
        
        # Generate output for each section
        for section_title, changes in sections.items():
            output.append(f"SECTION: {section_title}")
            output.append("-" * 60)
            
            # Add context for the section (cleaned up)
            if changes:
                context = changes[0]['context']
                # Clean up context - remove redundant section titles
                context_parts = context.split(' - ', 1)
                if len(context_parts) > 1:
                    context = context_parts[1]  # Use the part after the dash
                else:
                    context = context_parts[0]
                
                if context and context != section_title:
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
                output.append(f"  • {change['rewritten']}")
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
        """Extract required and preferred skills with precise targeting"""
        if not text:
            return [], []
            
        text_lower = text.lower()

        # Split text into sections
        sections = re.split(r'\n\s*\n', text_lower)
        
        # Find required and preferred sections
        required_section = ""
        preferred_section = ""
        responsibilities_section = ""
        
        for section in sections:
            if any(marker in section.lower() for marker in ['required', 'requirements', 'qualifications', 'must have']):
                required_section = section
            elif any(marker in section.lower() for marker in ['preferred', 'nice to have', 'plus', 'bonus']):
                preferred_section = section
            elif any(marker in section.lower() for marker in ['responsibilities', 'duties', 'what you will do']):
                responsibilities_section = section

        # Enhanced patterns for finding requirements
        required_patterns = [
            # Direct requirement statements
            r'required[\s\w]*?(?:qualifications?|capabilities?|skills?)[:\-\s]*(.*?)(?=(?:preferred|nice|plus|capabilities|skills|\n\n|$))',
            r'must have[:\-\s]*(.*?)(?=(?:preferred|nice|plus|\n\n|$))',
            r'requirements?[:\-\s]*(.*?)(?=(?:preferred|nice|plus|\n\n|$))',
            
            # Knowledge and proficiency statements
            r'(?:baseline|foundational) knowledge[:\-\s]*(.*?)(?=(?:preferred|nice|plus|\n\n|$))',
            r'proficiency (?:in|with)[:\-\s]*(.*?)(?=(?:preferred|nice|plus|\n\n|$))',
            r'understanding of[:\-\s]*(.*?)(?=(?:preferred|nice|plus|\n\n|$))',
            
            # Technical requirements
            r'technical (?:skills?|requirements?)[:\-\s]*(.*?)(?=(?:preferred|nice|plus|\n\n|$))',
            r'programming (?:skills?|languages?)[:\-\s]*(.*?)(?=(?:preferred|nice|plus|\n\n|$))',
            r'development (?:skills?|tools?)[:\-\s]*(.*?)(?=(?:preferred|nice|plus|\n\n|$))',
            
            # Bullet point lists
            r'[•\-\*]\s*(.*?)(?=(?:[•\-\*]|\n\n|$))',
            
            # Specific technical mentions
            r'(?:experience|knowledge|proficiency) (?:in|with) (python|java|javascript|c\+\+|c#|react).*?(?=\n)',
            r'(?:cloud|ai|mobile|database|security) (?:development|experience|knowledge).*?(?=\n)',
            r'(?:agile|ci/cd|devops) (?:methodologies|practices|experience).*?(?=\n)'
        ]
        
        preferred_patterns = [
            # Direct preferred statements
            r'preferred[\s\w]*?(?:qualifications?|capabilities?|skills?)[:\-\s]*(.*?)(?=\n\n|$)',
            r'nice to have[:\-\s]*(.*?)(?=\n\n|$)',
            r'plus[:\-\s]*(.*?)(?=\n\n|$)',
            r'bonus[:\-\s]*(.*?)(?=\n\n|$)',
            
            # Advanced skills
            r'strong[\s\w]*?(?:skills?|abilities?)[:\-\s]*(.*?)(?=\n\n|$)',
            r'exceptional[\s\w]*?(?:ability|skills?)[:\-\s]*(.*?)(?=\n\n|$)',
            r'advanced[\s\w]*?(?:knowledge|experience)[:\-\s]*(.*?)(?=\n\n|$)',
            
            # Additional technical skills
            r'additional technical skills[:\-\s]*(.*?)(?=\n\n|$)',
            r'other technologies[:\-\s]*(.*?)(?=\n\n|$)',
            
            # Bullet points in preferred section
            r'[•\-\*]\s*(.*?)(?=(?:[•\-\*]|\n\n|$))'
        ]
        
        required_skills = []
        preferred_skills = []

        # Process required section
        if required_section:
            for pattern in required_patterns:
                matches = re.findall(pattern, required_section, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = ' '.join(match)
                    skills = self._extract_meaningful_skills(match)
                    required_skills.extend(skills)
        
        # Process preferred section
        if preferred_section:
            for pattern in preferred_patterns:
                matches = re.findall(pattern, preferred_section, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = ' '.join(match)
                    skills = self._extract_meaningful_skills(match)
                    preferred_skills.extend(skills)
        
        # Look for skills in responsibilities that aren't already found
        if responsibilities_section:
            resp_skills = self._extract_meaningful_skills(responsibilities_section)
            for skill in resp_skills:
                if skill not in required_skills and skill not in preferred_skills:
                    required_skills.append(skill)
        
        # Process the entire text for any missed critical skills
        critical_skills = {
            'python', 'java', 'javascript', 'c++', 'c#', 'react',
            'cloud', 'artificial intelligence', 'mobile',
            'data structures', 'algorithms',
            'agile', 'ci/cd', 'security'
        }
        
        for skill in critical_skills:
            if skill.lower() in text_lower:
                if skill not in required_skills and skill not in preferred_skills:
                    required_skills.append(skill)
        
        return list(set(required_skills)), list(set(preferred_skills))
    
    def _extract_meaningful_skills(self, text: str) -> List[str]:
        """Extract only meaningful skills, filtering out filler words"""
        if not text:
            return []
        
        # Define meaningful technical skills and qualifications with categories
        skill_categories = {
            'languages': [
            'python', 'java', 'javascript', 'react', 'c++', 'c#', 'node.js', 'html', 'css',
                'sql', 'r', 'swift', 'go', 'rust', 'ruby', 'php', 'kotlin', 'scala'
            ],
            'core_tech': [
            'cloud', 'artificial intelligence', 'mobile', 'databases', 'data structures', 
            'algorithms', 'big data', 'data warehousing', 'ci/cd', 'application resiliency', 
            'security', 'machine learning', 'data analysis', 'software development',
                'web development', 'mobile development', 'devops', 'cloud computing'
            ],
            'tools_frameworks': [
            'developmental toolsets', 'relational databases', 'git', 'github', 'docker',
                'kubernetes', 'aws', 'azure', 'gcp', 'jenkins', 'jira', 'confluence'
            ],
            'professional': [
                'business analysis', 'development', 'maintenance', 'software improvement',
                'agile methodologies', 'problem-solving', 'collaboration', 'computer science',
                'engineering', 'communication', 'interpersonal skills'
            ]
        }
        
        found_skills = []
        text_lower = text.lower()
        
        # Process each category with specific handling
        for category, skills in skill_categories.items():
            # For languages and tools, look for exact matches
            if category in ['languages', 'tools_frameworks']:
                for skill in skills:
                    # Handle special cases like C++ and C#
                    if skill in ['c++', 'c#']:
                        if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                            found_skills.append(skill)
                    else:
                        if re.search(rf'\b{skill}\b', text_lower):
                            found_skills.append(skill)
        
            # For core tech and professional skills, allow for variations
            else:
                for skill in skills:
                    # Look for variations and combinations
                    variations = [skill]
                    if ' ' in skill:
                        # Add variations without spaces
                        variations.append(skill.replace(' ', ''))
                    if category == 'core_tech':
                        # Add common prefixes/suffixes for tech terms
                        variations.extend([
                            f"{skill} development",
                            f"{skill} engineering",
                            f"{skill} architecture",
                            f"{skill} design"
                        ])
                    
                    for var in variations:
                        if re.search(rf'\b{re.escape(var)}\b', text_lower):
                            found_skills.append(skill)
                            break
        
        # Extract skills from specific contexts
        context_patterns = {
            'programming': [
                r'programming languages?.*?(python|react|javascript|java|c\+\+|c#)',
            r'(python|react|javascript|java|c\+\+|c#).*?programming',
                r'(?:knowledge|experience|proficiency) (?:in|with).*?(python|react|javascript|java|c\+\+|c#)',
                r'(?:develop|code|build).*?(?:using|with).*?(python|react|javascript|java|c\+\+|c#)'
            ],
            'databases': [
                r'(?:database|data).*?(sql|mysql|postgresql|oracle|mongodb)',
                r'(relational|nosql).*?database',
                r'data (?:warehousing|storage|management)'
            ],
            'cloud': [
                r'cloud.*?(aws|azure|gcp)',
                r'(aws|azure|gcp).*?cloud',
                r'cloud (?:computing|infrastructure|architecture|services)'
            ],
            'methodologies': [
                r'(agile|scrum|kanban|waterfall)',
                r'(ci/cd|continuous integration|continuous delivery)',
                r'(devops|devsecops|mlops)',
                r'application (?:resiliency|security|performance)'
            ]
        }
        
        for patterns in context_patterns.values():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if isinstance(matches[0], tuple) if matches else False:
                    # Handle tuple results from regex groups
                    found_skills.extend([m for t in matches for m in t if m])
                else:
                    found_skills.extend(matches)
        
        # Filter out common filler words and vague terms
        filler_words = {
            'new', 'skills', 'ideas', 'overview', 'innovative', 'program',
            'ability', 'basic', 'understanding', 'knowledge', 'experience'
        }
        
        filtered_skills = [
            skill for skill in found_skills
            if skill.lower() not in filler_words and len(skill) > 2
        ]
        
        return list(set(filtered_skills))
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from a block of text with improved detection"""
        if not text:
            return []
            
        skills = []
        text_lower = text.lower()

        # Check for technical keywords
        for category, keywords in self.tech_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    skills.append(keyword)
        
        # Enhanced skill patterns
        skill_patterns = [
            # Programming languages
            r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|go|rust|swift|r|php|kotlin|scala|perl)\b',
            # Web technologies
            r'\b(html|css|react|angular|vue|node\.?js|django|flask|spring|express|jquery|bootstrap)\b',
            # Databases
            r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch|oracle|sqlite|dynamodb)\b',
            # Cloud platforms
            r'\b(aws|azure|gcp|google cloud|amazon web services|microsoft azure|kubernetes|docker|terraform)\b',
            # Data science
            r'\b(pandas|numpy|scikit-learn|tensorflow|pytorch|matplotlib|seaborn|plotly|jupyter)\b',
            # Tools and frameworks
            r'\b(git|github|gitlab|jenkins|jira|confluence|slack|teams|agile|scrum|ci/cd)\b',
            # Methodologies
            r'\b(machine learning|artificial intelligence|data analysis|software development|web development|mobile development|devops|cloud computing)\b'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower)
            skills.extend(matches)
        
        # Extract other potential skills (noun phrases)
        if self.nlp:
            try:
                doc = self.nlp(text)
                for chunk in doc.noun_chunks:
                    chunk_text = chunk.text.strip().lower()
                    if (len(chunk.text.split()) <= 3 and 
                        chunk_text not in self.stop_words and
                        len(chunk_text) > 2 and
                        not chunk_text.isdigit()):
                        skills.append(chunk.text.strip())
            except Exception as e:
                logger.warning(f"Error processing text with spaCy: {e}")
        
        return skills
    
    def analyze_job_description(self, job_text: str) -> JobAnalysis:
        """Comprehensive analysis of job description with improved extraction"""
        if not job_text or not job_text.strip():
            return JobAnalysis(
                keywords=[],
                required_skills=[],
                preferred_skills=[],
                technologies=[],
                responsibilities=[]
            )
            
        # Normalize text for better parsing
        normalized_text = self._normalize_job_text(job_text)
        
        keywords = self.extract_keywords(normalized_text)
        required_skills, preferred_skills = self.extract_skills(normalized_text)
        
        # Enhanced technology extraction
        technologies = self._extract_technologies_comprehensive(normalized_text)
        
        # Enhanced responsibility extraction
        responsibilities = self._extract_responsibilities_comprehensive(normalized_text)
        
        # Extract qualifications and soft skills
        qualifications = self._extract_qualifications(normalized_text)
        required_skills.extend(qualifications)
        
        return JobAnalysis(
            keywords=keywords,
            required_skills=list(set(required_skills)),  # Remove duplicates
            preferred_skills=list(set(preferred_skills)),
            technologies=list(set(technologies)),
            responsibilities=list(set(responsibilities))
        )
    
    def _normalize_job_text(self, text: str) -> str:
        """Normalize job description text for better parsing"""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Convert to lowercase for consistent matching
        return text.lower()
    
    def _extract_technologies_comprehensive(self, text: str) -> List[str]:
        """Extract technologies with comprehensive patterns"""
        technologies = []
        
        # Direct technology mentions
        tech_patterns = [
            # Programming languages
            r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|go|rust|swift|r|php|kotlin|scala|perl|react)\b',
            # Web technologies
            r'\b(html|css|react|angular|vue|node\.?js|django|flask|spring|express|jquery|bootstrap)\b',
            # Databases
            r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch|oracle|sqlite|dynamodb|relational databases)\b',
            # Cloud platforms
            r'\b(aws|azure|gcp|google cloud|amazon web services|microsoft azure|kubernetes|docker|terraform|cloud technologies)\b',
            # Data science
            r'\b(pandas|numpy|scikit-learn|tensorflow|pytorch|matplotlib|seaborn|plotly|jupyter)\b',
            # Tools and frameworks
            r'\b(git|github|gitlab|jenkins|jira|confluence|slack|teams|agile|scrum|ci/cd)\b',
            # Methodologies and concepts
            r'\b(machine learning|artificial intelligence|data analysis|software development|web development|mobile development|devops|cloud computing|big data|data warehousing)\b',
            # Specific technical disciplines mentioned in job
            r'\b(cloud|artificial intelligence|mobile|data structures|algorithms|business analysis|development|maintenance|software improvement|application resiliency|security)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technologies.extend(matches)
        
        # Extract from specific sections
        sections = [
            r'baseline knowledge[^.]*?(cloud|artificial intelligence|mobile)',
            r'programming languages[^.]*?(python|react|javascript|java|c\+\+|c#)',
            r'agile methodologies[^.]*?(ci/cd|application resiliency|security)',
            r'big data[^.]*?(data warehousing)',
            r'cloud technologies',
            r'relational databases'
        ]
        
        for section_pattern in sections:
            matches = re.findall(section_pattern, text, re.IGNORECASE)
            technologies.extend(matches)
        
        return list(set(technologies))
    
    def _extract_responsibilities_comprehensive(self, text: str) -> List[str]:
        """Extract responsibilities with comprehensive patterns"""
        responsibilities = []
        
        # Responsibility patterns
        resp_patterns = [
            r'\b(own projects|collaborate|develop skills|create innovative solutions|work on agile teams)\b',
            r'\b(end-to-end|teams and stakeholders|development progress|global team|technologists)\b',
            r'\b(ongoing training|mentorship|senior leaders|innovative solutions|customers|clients|employees)\b',
            r'\b(peers|experienced software engineers|enhance skills|share ideas|innovate)\b',
            r'\b(agile methodologies|ci/cd|application resiliency|security)\b',
            r'\b(business analysis|development|maintenance|software improvement)\b'
        ]
        
        for pattern in resp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            responsibilities.extend(matches)
        
        return list(set(responsibilities))
    
    def _extract_qualifications(self, text: str) -> List[str]:
        """Extract qualifications and soft skills with precise targeting"""
        qualifications = []
        
        # Specific qualification patterns - only capture meaningful terms
        specific_qualifications = [
            # Technical qualifications
            'computer science', 'engineering majors', 'foundational knowledge', 'proficiency',
            'developmental toolsets', 'industry-wide technology trends', 'best practices',
            
            # Soft skills
            'interpersonal skills', 'communication skills', 'problem-solving', 'collaborative environment',
            'exceptional problem-solving', 'strong interpersonal', 'strong communication',
            
            # Experience types
            'exposure to cloud technologies', 'experience with relational databases',
            'well-rounded academic background', 'ability to work effectively',
            'large collaborative teams', 'organizational goals', 'inclusive culture',
            'innovative culture', 'thrive in fast-paced', 'fast-paced collaborative environment'
        ]
        
        # Check for exact matches of meaningful qualifications
        for qual in specific_qualifications:
            if qual in text.lower():
                qualifications.append(qual)
        
        # Extract specific technical disciplines mentioned
        tech_disciplines = re.findall(r'technical discipline.*?(cloud|artificial intelligence|mobile)', text, re.IGNORECASE)
        qualifications.extend(tech_disciplines)
        
        # Extract specific programming languages mentioned
        prog_langs = re.findall(r'programming languages.*?(python|react|javascript|java|c\+\+|c#)', text, re.IGNORECASE)
        qualifications.extend(prog_langs)
        
        return list(set(qualifications))
    
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
        lines = [line.rstrip() for line in resume_text.split('\n')]
        current_section = None
        current_content = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check if line is a major section header
            if self._is_major_section_header(line):
                # Save previous section if it exists and has content
                if current_section and current_content:
                    sections.append(ResumeSection(
                        title=current_section,
                        content=[c for c in current_content if c.strip()],  # Remove empty content
                        section_type=self._get_section_type(current_section)
                    ))
                
                current_section = line
                current_content = []
                i += 1
                continue
            
            # If we're in a section, add content
            if current_section:
                current_content.append(line)
            else:
                # Handle content before first section (like name, contact info)
                if not sections:  # Only for the very beginning
                    if not current_section:
                        current_section = "HEADER"
                    current_content.append(line)
            
            i += 1
        
        # Add final section
        if current_section and current_content:
            sections.append(ResumeSection(
                title=current_section,
                content=[c for c in current_content if c.strip()],
                section_type=self._get_section_type(current_section)
            ))
        
        return sections
    
    def _is_major_section_header(self, line: str) -> bool:
        """Improved section header detection"""
        if not line or len(line) < 3:
            return False
            
        line_lower = line.lower().strip()
        
        # Must contain a known section keyword
        contains_section_keyword = any(
            header in line_lower 
            for headers in self.section_headers.values() 
            for header in headers
        )
        
        if not contains_section_keyword:
            return False
        
        # Additional criteria for section headers
        criteria_met = 0
        
        # Is all caps or title case
        if line.isupper() or line.istitle():
            criteria_met += 1
        
        # Is relatively short (section headers are usually concise)
        if len(line.split()) <= 5:
            criteria_met += 1
        
        # Doesn't start with bullet symbols
        if not line.lstrip().startswith(('•', '-', '*', '●', '◆', '▪')):
            criteria_met += 1
        
        # Doesn't contain typical bullet content indicators
        bullet_indicators = ['developed', 'built', 'analyzed', 'managed', 'led', 'created']
        if not any(indicator in line_lower for indicator in bullet_indicators):
            criteria_met += 1
        
        # Must meet at least 2 additional criteria beyond containing keywords
        return criteria_met >= 2
    
    def _get_section_type(self, section_title: str) -> str:
        """Determine the type of section"""
        if not section_title:
            return 'other'
            
        title_lower = section_title.lower()
        
        # Direct keyword matching with priority order
        section_priorities = [
            ('experience', ['experience', 'work experience', 'professional experience']),
            ('projects', ['project', 'projects']),
            ('research', ['research']),
            ('education', ['education']),
            ('skills', ['skill', 'technical skill', 'competencies']),
            ('involvement', ['involvement', 'activities', 'leadership']),
            ('achievements', ['achievement', 'accomplishment', 'award'])
        ]
        
        for section_type, keywords in section_priorities:
            if any(keyword in title_lower for keyword in keywords):
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
        """Rewrite multiple bullet points in a single API call with enhanced job skill integration"""
        
        if dry_run:
            logger.info(f"DRY RUN: Would rewrite {len(bullet_points)} bullet points:")
            for i, bullet in enumerate(bullet_points, 1):
                logger.info(f"  {i}. {bullet}")
            return bullet_points  # Return original bullets in dry run
        
        # Check cache first and filter out bullets that already strongly match
        cached_bullets = []
        uncached_bullets = []
        uncached_indices = []
        strong_match_bullets = []
        strong_match_indices = []
        bullet_contexts = []  # Store context for each bullet
        
        for i, bullet in enumerate(bullet_points):
            cached = self.cache.get(bullet, job_description)
            if cached:
                cached_bullets.append((i, cached))
            elif job_analysis and self._bullet_strongly_matches_job(bullet, job_analysis):
                # Skip rewriting bullets that already strongly match job requirements
                strong_match_bullets.append((i, bullet))
                strong_match_indices.append(i)
            else:
                uncached_bullets.append(bullet)
                uncached_indices.append(i)
                # Extract technical context from the bullet
                tech_context = self._extract_technical_context(bullet, job_analysis)
                bullet_contexts.append(tech_context)
        
        logger.info(f"Found {len(cached_bullets)} cached bullets, {len(strong_match_bullets)} strong matches, {len(uncached_bullets)} need rewriting")
        
        # If all bullets are cached or strong matches, return them
        if not uncached_bullets:
            result = [''] * len(bullet_points)
            for idx, cached_bullet in cached_bullets:
                result[idx] = cached_bullet
            for idx, strong_bullet in strong_match_bullets:
                result[idx] = strong_bullet
            return result
        
        # Check API call limit
        if self.api_calls_made >= self.max_api_calls:
            logger.warning(f"API call limit ({self.max_api_calls}) reached. Skipping rewrite step.")
            return bullet_points
        
        # Group similar bullets for better context
        bullet_groups = self._group_similar_bullets(uncached_bullets, bullet_contexts)
        
        # Prepare batch prompt with enhanced instructions
        bullet_list = "\n".join([
            f"{i+1}. {bullet} [Context: {bullet_contexts[i]}]" 
            for i, bullet in enumerate(uncached_bullets)
        ])
        
        # Extract key skills and technologies by category
        tech_skills = [skill for skill in job_analysis.technologies 
                      if any(tech in skill.lower() for tech in ['python', 'java', 'javascript', 'c++', 'c#', 'sql', 'react'])]
        cloud_skills = [skill for skill in job_analysis.technologies 
                       if any(cloud in skill.lower() for cloud in ['cloud', 'aws', 'azure', 'gcp'])]
        data_skills = [skill for skill in job_analysis.technologies 
                      if any(data in skill.lower() for data in ['data', 'analytics', 'machine learning', 'ai'])]
        dev_skills = [skill for skill in job_analysis.technologies 
                     if any(dev in skill.lower() for dev in ['agile', 'ci/cd', 'devops', 'development'])]
        
        prompt = f"""
        Rewrite the following resume bullet points to better match the job requirements while maintaining truthfulness and technical depth.
        Return ONLY the rewritten bullets in the same numbered format (1., 2., 3., etc.), preserving the exact order.
        Do not include any headers, explanations, or additional text.
        
        Original bullet points with context:
        {bullet_list}
        
        Job requirements by category:
        Technical Skills:
        - Programming: {', '.join(tech_skills)}
        - Cloud & Infrastructure: {', '.join(cloud_skills)}
        - Data & Analytics: {', '.join(data_skills)}
        - Development & Tools: {', '.join(dev_skills)}
        
        Core Requirements:
        - Required Skills: {', '.join(job_analysis.required_skills[:8])}
        - Key Technologies: {', '.join(job_analysis.technologies[:8])}
        
        Section Context: {context}
        
        CRITICAL GUIDELINES:
        1. PRESERVE TECHNICAL TRUTH:
           - Keep all original technologies, tools, languages, and frameworks
           - Maintain all quantitative metrics and achievements
           - Never fabricate or remove technical details
        
        2. ENHANCE WITH JOB SKILLS:
           - Add relevant job skills ONLY where they naturally fit
           - If a bullet mentions automation, highlight CI/CD or DevOps aspects
           - If data/analytics work is mentioned, emphasize relevant tools/methods
           - For development work, highlight agile/collaborative aspects
        
        3. MAINTAIN TECHNICAL DEPTH:
           - Keep specific implementation details
           - Preserve architecture and design decisions
           - Retain performance metrics and improvements
        
        4. IMPROVE CLARITY AND IMPACT:
           - Use strong technical action verbs
           - Highlight problem-solving and results
           - Remove unnecessary words while keeping technical substance
           - Ensure each bullet demonstrates clear value/impact
        
        5. MATCH JOB REQUIREMENTS:
           - Align with required technical skills where relevant
           - Emphasize matching technologies and methodologies
           - Add context about scale/impact that matches job needs
        
        Examples of EXCELLENT rewrites:
        Original: "Developed Python script for data processing"
        Good: "Engineered scalable Python data processing pipeline using cloud technologies and CI/CD practices, improving efficiency by 40%"
        
        Original: "Built web application with React"
        Good: "Architected and implemented full-stack web application using React and modern JavaScript, following agile methodologies and maintaining 98% test coverage"
        
        Examples of BAD rewrites (avoid these):
        - Removing specific technologies or metrics
        - Adding skills that weren't used in the original work
        - Using vague phrases like "innovative solutions" or "cutting-edge technology"
        - Losing technical implementation details
        
        Return EXACTLY {len(uncached_bullets)} rewritten bullets, preserving all technical substance while integrating relevant job skills naturally.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert technical resume writer who maintains technical accuracy while aligning with job requirements. Never fabricate or remove technical details."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150 * len(uncached_bullets),
                temperature=0.7
            )
            
            self.api_calls_made += 1
            logger.info(f"API calls made: {self.api_calls_made}/{self.max_api_calls}")
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            rewritten_bullets = self._parse_numbered_response(response_text, len(uncached_bullets))
            
            # Validate rewritten bullets
            rewritten_bullets = self._validate_rewritten_bullets(
                original_bullets=uncached_bullets,
                rewritten_bullets=rewritten_bullets,
                job_analysis=job_analysis
            )
            
            # Cache the results
            for original, rewritten in zip(uncached_bullets, rewritten_bullets):
                self.cache.set(original, job_description, rewritten)
            
            # Combine cached, strong match, and new results
            result = [''] * len(bullet_points)
            
            # Add cached bullets
            for idx, cached_bullet in cached_bullets:
                result[idx] = cached_bullet
            
            # Add strong match bullets (keep original)
            for idx, strong_bullet in strong_match_bullets:
                result[idx] = strong_bullet
            
            # Add newly rewritten bullets
            for result_idx, rewritten in zip(uncached_indices, rewritten_bullets):
                result[result_idx] = rewritten
            
            return result
            
        except Exception as e:
            logger.error(f"Error rewriting bullet points: {e}")
            return bullet_points  # Return originals if AI fails
    
    def _extract_technical_context(self, bullet: str, job_analysis: JobAnalysis) -> str:
        """Extract relevant technical context from a bullet point"""
        bullet_lower = bullet.lower()
        context_parts = []
        
        # Check for programming languages
        langs = [lang for lang in ['python', 'java', 'javascript', 'c++', 'c#', 'sql', 'react']
                if lang in bullet_lower]
        if langs:
            context_parts.append(f"Uses {', '.join(langs)}")
        
        # Check for cloud/infrastructure
        cloud = [tech for tech in ['aws', 'azure', 'gcp', 'cloud', 'kubernetes', 'docker']
                if tech in bullet_lower]
        if cloud:
            context_parts.append(f"Cloud tech: {', '.join(cloud)}")
        
        # Check for data/analytics
        data = [tech for tech in ['data', 'analytics', 'machine learning', 'ai', 'algorithm']
               if tech in bullet_lower]
        if data:
            context_parts.append(f"Data focus: {', '.join(data)}")
        
        # Check for development practices
        dev = [practice for practice in ['agile', 'ci/cd', 'devops', 'test', 'automation']
              if practice in bullet_lower]
        if dev:
            context_parts.append(f"Dev practices: {', '.join(dev)}")
        
        # Add relevant job requirements that match
        matching_skills = [skill for skill in job_analysis.required_skills
                         if skill.lower() in bullet_lower]
        if matching_skills:
            context_parts.append(f"Matches requirements: {', '.join(matching_skills[:3])}")
        
        return '; '.join(context_parts) if context_parts else "No specific technical context"
    
    def _group_similar_bullets(self, bullets: List[str], contexts: List[str]) -> List[List[int]]:
        """Group similar bullets based on their technical context"""
        groups = []
        used = set()
        
        for i, (bullet, context) in enumerate(zip(bullets, contexts)):
            if i in used:
                continue
                
            current_group = [i]
            used.add(i)
            
            # Find similar bullets
            for j, (other_bullet, other_context) in enumerate(zip(bullets, contexts)):
                if j in used:
                    continue
                    
                # Check for similar technical context
                if any(tech in other_context for tech in context.split('; ')):
                    current_group.append(j)
                    used.add(j)
            
            groups.append(current_group)
        
        return groups
    
    def _validate_rewritten_bullets(self, original_bullets: List[str], 
                                  rewritten_bullets: List[str],
                                  job_analysis: JobAnalysis) -> List[str]:
        """Validate and fix rewritten bullets to ensure quality"""
        validated_bullets = []
        
        for orig, rewritten in zip(original_bullets, rewritten_bullets):
            # 1. Check if technical details are preserved
            orig_lower = orig.lower()
            rewritten_lower = rewritten.lower()
            
            # Extract technical terms from original
            tech_terms = re.findall(r'\b(?:python|java|javascript|c\+\+|c#|sql|react|aws|azure|gcp|docker|kubernetes)\b',
                                  orig_lower)
            
            # Ensure all technical terms are in rewritten version
            missing_terms = [term for term in tech_terms if term not in rewritten_lower]
            if missing_terms:
                # Fix: Add missing terms back
                rewritten = f"{rewritten} using {', '.join(missing_terms)}"
            
            # 2. Check for metrics preservation
            metrics = re.findall(r'\d+%|\d+x|\d+\s*(?:users|requests|transactions|improvements?|seconds?|minutes?|hours?|days?)',
                               orig_lower)
            missing_metrics = [m for m in metrics if m not in rewritten_lower]
            if missing_metrics:
                # Fix: Add metrics back
                rewritten = f"{rewritten} ({', '.join(missing_metrics)})"
            
            # 3. Check for inappropriate generalizations
            bad_phrases = ['innovative solutions', 'cutting-edge', 'state-of-the-art', 'next-generation']
            if any(phrase in rewritten_lower for phrase in bad_phrases):
                # Fix: Use original if rewritten version has become too vague
                rewritten = orig
            
            # 4. Ensure proper technical context
            if not any(skill.lower() in rewritten_lower for skill in job_analysis.technologies):
                # Try to add relevant context naturally
                matching_skills = [skill for skill in job_analysis.technologies 
                                 if any(term in orig_lower for term in skill.lower().split())]
                if matching_skills:
                    rewritten = f"{rewritten}, leveraging {matching_skills[0]}"
            
            validated_bullets.append(rewritten)
        
        return validated_bullets
        
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
            
            # Combine cached, strong match, and new results
            result = [''] * len(bullet_points)
            
            # Add cached bullets
            for idx, cached_bullet in cached_bullets:
                result[idx] = cached_bullet
            
            # Add strong match bullets (keep original)
            for idx, strong_bullet in strong_match_bullets:
                result[idx] = strong_bullet
            
            # Add newly rewritten bullets
            for result_idx, rewritten in zip(uncached_indices, rewritten_bullets):
                result[result_idx] = rewritten
            
            return result
            
        except Exception as e:
            logger.error(f"Error rewriting bullet points: {e}")
            return bullet_points  # Return originals if AI fails
    
    def _parse_suggestions(self, suggestions_text: str) -> List[str]:
        """Parse numbered suggestions from AI response"""
        if not suggestions_text:
            return []
        
        suggestions = []
        lines = suggestions_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for numbered format: "1. suggestion text"
            match = re.match(r'^(\d+)[\.\)]\s*(.+)', line)
            if match:
                suggestion_text = match.group(2).strip()
                if len(suggestion_text) > 10:  # Must be substantial
                    suggestions.append(suggestion_text)
        
        return suggestions
    
    def _parse_numbered_response(self, response_text: str, expected_count: int) -> List[str]:
        """Improved response parsing with better error handling"""
        if not response_text:
            logger.error("Empty response from AI")
            return [f"[Error: Empty response for bullet {i+1}]" for i in range(expected_count)]
        
        lines = response_text.strip().split('\n')
        bullets = []
        
        # First pass: Look for numbered bullets (1. 2. 3.)
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Match numbered format: "1. content" or "1) content"
            match = re.match(r'^(\d+)[\.\)]\s*(.+)', line)
            if match and match.group(2).strip():
                bullets.append(match.group(2).strip())
        
        # Second pass: If not enough numbered bullets, look for other patterns
        if len(bullets) < expected_count:
            logger.warning(f"Only found {len(bullets)} numbered bullets, trying other patterns")
            bullets = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip headers and meta information
                skip_patterns = [
                    r'^(rewritten|bullet|original|guidelines?|format)',
                    r'^[•\-*●]\s*bullet',  # "• Bullet point 1:"
                    r'^\d+\s*bullet',      # "1 Bullet point:"
                    r'^bullet\s*point',    # "Bullet point 1:"
                ]
                
                if any(re.match(pattern, line.lower()) for pattern in skip_patterns):
                    continue
                
                # Look for substantial content lines
                # Remove various prefixes and check if substantial
                clean_line = re.sub(r'^[\d\.\)\-\*•●]\s*', '', line)
                
                if len(clean_line) > 15 and not clean_line.lower().startswith(('here', 'the following')):
                    bullets.append(clean_line)
                    
                if len(bullets) >= expected_count:
                    break
        
        # Ensure we have the right number
        while len(bullets) < expected_count:
            logger.error(f"Missing bullet {len(bullets) + 1}, using placeholder")
            bullets.append(f"[Error: Could not parse bullet {len(bullets) + 1}]")
        
        # Trim to expected count
        return bullets[:expected_count]
    
    def polish_bullet_points_batch(self, bullet_points: List[str], context: str = "", 
                                  dry_run: bool = False) -> List[str]:
        """Polish bullet points for readability and professionalism without job tailoring"""
        
        if dry_run:
            logger.info(f"DRY RUN: Would polish {len(bullet_points)} bullet points:")
            for i, bullet in enumerate(bullet_points, 1):
                logger.info(f"  {i}. {bullet}")
            return bullet_points  # Return original bullets in dry run
        
        # Check API call limit
        if self.api_calls_made >= self.max_api_calls:
            logger.warning(f"API call limit ({self.max_api_calls}) reached. Skipping polish step.")
            return bullet_points
        
        # Prepare batch prompt for polishing
        bullet_list = "\n".join([f"{i+1}. {bullet}" for i, bullet in enumerate(bullet_points)])
        
        prompt = f"""
        Polish the following resume bullet points for better readability and professionalism.
        Return ONLY the polished bullets in the same numbered format (1., 2., 3., etc.), preserving the exact order.
        Do not include any headers, explanations, or additional text.
        
        Original bullet points:
        {bullet_list}
        
        Context: {context}
        
        POLISHING GUIDELINES:
        1. IMPROVE READABILITY: Make bullets clearer and more professional
        2. STRONG ACTION VERBS: Use powerful, specific action verbs
        3. HIGHLIGHT TECHNOLOGIES: Emphasize technical skills and tools naturally
        4. QUANTIFY IMPACT: Add or improve numbers, percentages, and metrics where possible
        5. BE CONCISE: Remove unnecessary words while keeping essential information
        6. MAINTAIN TRUTH: Keep all original facts and achievements
        7. PROFESSIONAL TONE: Ensure professional, confident language
        8. AVOID FLUFF: No generic phrases or buzzwords
        9. Return EXACTLY {len(bullet_points)} polished bullet points in numbered format
        10. Do not include bullet symbols (●, •, -) in your response - just the numbered text
        
        Examples of GOOD polishing:
        - "Worked with Python to analyze data" → "Developed Python scripts to analyze and visualize datasets, improving efficiency by 20%"
        - "Used R for statistics" → "Applied R programming for statistical analysis and data modeling"
        - "Built a website" → "Developed responsive web application using HTML, CSS, and JavaScript"
        
        Examples of BAD polishing (avoid these):
        - "Worked with Python" → "Leveraged cutting-edge Python technologies to drive innovative solutions"
        - "Used R" → "Utilized advanced R programming methodologies to foster data-driven insights"
        
        Polished bullet points:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert resume writer who improves readability and professionalism. Always return the exact number of bullet points requested in numbered format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150 * len(bullet_points),
                temperature=0.7
            )
            
            self.api_calls_made += 1
            logger.info(f"API calls made: {self.api_calls_made}/{self.max_api_calls}")
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            polished_bullets = self._parse_numbered_response(response_text, len(bullet_points))
            
            return polished_bullets
            
        except Exception as e:
            logger.error(f"Error polishing bullet points: {e}")
            return bullet_points  # Return originals if AI fails
    
    def _bullet_strongly_matches_job(self, bullet: str, job_analysis: JobAnalysis) -> bool:
        """Check if a bullet already strongly matches job requirements"""
        if not bullet or not job_analysis:
            return False
        
        bullet_lower = bullet.lower()
        
        # Count matches with job requirements
        tech_matches = sum(1 for tech in job_analysis.technologies if tech.lower() in bullet_lower)
        skill_matches = sum(1 for skill in job_analysis.required_skills if skill.lower() in bullet_lower)
        keyword_matches = sum(1 for kw in job_analysis.keywords if kw.lower() in bullet_lower)
        
        # Strong match criteria:
        # 1. Contains 2+ technologies from job requirements
        # 2. Contains 2+ skills from job requirements  
        # 3. Contains 3+ keywords from job requirements
        # 4. Contains specific technical terms that are highly relevant
        
        strong_tech_terms = ['python', 'java', 'javascript', 'react', 'r', 'sql', 'machine learning', 
                           'data analysis', 'algorithms', 'data structures', 'cloud', 'agile', 'ci/cd']
        
        has_strong_tech = any(term in bullet_lower for term in strong_tech_terms)
        
        total_matches = tech_matches + skill_matches + keyword_matches
        
        # Consider it a strong match if:
        # - Has 2+ technology matches, OR
        # - Has 2+ skill matches, OR  
        # - Has 3+ total keyword matches, OR
        # - Has strong technical terms AND at least 1 other match
        return (tech_matches >= 2 or 
                skill_matches >= 2 or 
                total_matches >= 3 or 
                (has_strong_tech and total_matches >= 1))
    
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
        Based on the job requirements and the candidate's existing resume, suggest 3-5 new bullet points that could strengthen their application.
        
        Job requirements:
        - Required skills: {', '.join(job_analysis.required_skills[:10])}
        - Preferred skills: {', '.join(job_analysis.preferred_skills[:8])}
        - Key technologies: {', '.join(job_analysis.technologies[:8])}
        
        Existing resume (abbreviated):
        {existing_content[:800]}...
        
        Return ONLY a numbered list of suggestions (1., 2., 3., etc.) with no additional text or explanations.
        Focus on realistic skills/experiences the candidate likely has but may not have emphasized.
        
        Example format:
        1. First suggestion here
        2. Second suggestion here
        3. Third suggestion here
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional resume advisor. Return only numbered suggestions, no explanations or headers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            
            self.api_calls_made += 1
            logger.info(f"API calls made: {self.api_calls_made}/{self.max_api_calls}")
            
            suggestions_text = response.choices[0].message.content.strip()
            return self._parse_suggestions(suggestions_text)
            
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
    
    def _normalize_bullet_for_dedup(self, bullet: str) -> str:
        """Normalize bullet point text for deduplication comparison"""
        # Remove bullet symbols and whitespace
        text = re.sub(r'^[•\-*●◆▪–]\s*', '', bullet.strip())
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except in numbers (e.g., preserve "10.5%")
        text = re.sub(r'(?<!\d)[.,;:!?](?!\d)', '', text)
        
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common filler words
        filler_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.split()
        words = [w for w in words if w not in filler_words]
        
        # Sort words to catch reordered phrases
        words.sort()
        
        return ' '.join(words)
    
    def _bullets_are_similar(self, bullet1: str, bullet2: str) -> bool:
        """Check if two normalized bullet points are similar enough to be considered duplicates"""
        # If exactly the same after normalization, definitely duplicates
        if bullet1 == bullet2:
            return True
        
        # Split into words for more detailed comparison
        words1 = set(bullet1.split())
        words2 = set(bullet2.split())
        
        # Calculate word overlap
        common_words = words1 & words2
        total_words = words1 | words2
        
        # Consider similar if they share more than 80% of their unique words
        similarity = len(common_words) / len(total_words) if total_words else 0
        
        # Extract any numbers/metrics
        numbers1 = set(re.findall(r'\d+(?:\.\d+)?%?', bullet1))
        numbers2 = set(re.findall(r'\d+(?:\.\d+)?%?', bullet2))
        
        # If both have numbers but they're different, not duplicates
        if numbers1 and numbers2 and not numbers1 & numbers2:
            return False
        
        # Extract technical terms
        tech_terms = {
            'python', 'java', 'javascript', 'react', 'c++', 'c#', 'sql',
            'aws', 'azure', 'cloud', 'docker', 'kubernetes', 'ci/cd',
            'agile', 'machine learning', 'data analysis'
        }
        
        terms1 = {word for word in words1 if word in tech_terms}
        terms2 = {word for word in words2 if word in tech_terms}
        
        # If they mention different technologies, not duplicates
        if terms1 and terms2 and not terms1 & terms2:
            return False
        
        return similarity > 0.8
    
    def detect_bullets(self, lines: List[str], section_type: str) -> Tuple[List[str], List[str]]:
        """
        Detect bullet points in resume sections with enhanced preservation and multi-line handling.
        
        Args:
            lines: List of text lines from a resume section
            section_type: Type of section (experience, projects, etc.)
        
        Returns:
            Tuple of (bullet_points, non_bullet_content)
        """
        if not lines:
            return [], []
        
        # Bullet patterns with enhanced detection
        BULLET_SYMBOLS = ['•', '-', '*', '●', '◆', '▪', '–', '∙', '○', '■', '⚫']
        BULLET_REGEX = re.compile(rf"^\s*[{''.join(re.escape(sym) for sym in BULLET_SYMBOLS)}]\s+")
        
        # Comprehensive action verbs for bullet detection
        ACTION_VERBS = {
            # Technical actions
            'developed', 'built', 'implemented', 'designed', 'architected',
            'programmed', 'coded', 'engineered', 'created', 'constructed',
            'deployed', 'maintained', 'optimized', 'debugged', 'tested',
            'integrated', 'automated', 'refactored', 'configured', 'customized',
            
            # Analysis and research
            'analyzed', 'researched', 'investigated', 'evaluated', 'assessed',
            'studied', 'examined', 'explored', 'monitored', 'measured',
            'calculated', 'quantified', 'modeled', 'simulated', 'validated',
            
            # Leadership and management
            'managed', 'led', 'coordinated', 'supervised', 'directed',
            'oversaw', 'guided', 'mentored', 'trained', 'facilitated',
            'organized', 'planned', 'executed', 'administered', 'controlled',
            
            # Achievement and improvement
            'improved', 'enhanced', 'increased', 'reduced', 'streamlined',
            'accelerated', 'strengthened', 'upgraded', 'transformed', 'revolutionized',
            'achieved', 'accomplished', 'delivered', 'generated', 'produced',
            
            # Collaboration and communication
            'collaborated', 'partnered', 'coordinated', 'communicated', 'presented',
            'reported', 'documented', 'wrote', 'authored', 'published',
            'consulted', 'advised', 'educated', 'trained', 'mentored',
            
            # Project-specific
            'launched', 'initiated', 'spearheaded', 'pioneered', 'established',
            'founded', 'instituted', 'introduced', 'piloted', 'prototyped',
            
            # Research and academic
            'conducted', 'performed', 'experimented', 'investigated', 'surveyed',
            'collected', 'compiled', 'synthesized', 'formulated', 'derived',
            
            # Data and analysis
            'visualized', 'processed', 'cleaned', 'mined', 'extracted',
            'transformed', 'aggregated', 'consolidated', 'unified', 'normalized',
            
            # Recognition and awards
            'awarded', 'recognized', 'selected', 'chosen', 'nominated',
            'honored', 'earned', 'received', 'won', 'achieved',
            
            # Participation and engagement
            'participated', 'engaged', 'contributed', 'volunteered', 'assisted',
            'supported', 'helped', 'aided', 'fostered', 'promoted'
        }
        
        # Important phrases that indicate achievements or responsibilities
        IMPORTANT_PHRASES = {
            # Recognition and awards
            'promoted', 'awarded', 'recognized', 'selected', 'nominated',
            'certified', 'licensed', 'graduated', 'completed', 'published',
            'featured', 'honored', 'chosen', 'earned', 'received',
            
            # Leadership roles
            'team lead', 'project lead', 'leader', 'manager', 'supervisor',
            'coordinator', 'chair', 'president', 'director', 'head',
            
            # Technical achievements
            'patent', 'publication', 'research', 'thesis', 'dissertation',
            'prototype', 'algorithm', 'system', 'application', 'framework',
            
            # Academic achievements
            'dean\'s list', 'honor roll', 'scholarship', 'fellowship', 'grant',
            'academic award', 'research award', 'distinction', 'honors', 'summa cum laude'
        }
        
        bullet_points = []
        non_bullet_content = []
        current_bullet = None
        bullet_continuation_indent = None
        
        # Only apply action verb detection to relevant sections
        relevant_sections = {'experience', 'projects', 'research', 'involvement', 'skills'}
        use_action_verbs = section_type in relevant_sections
        
        def is_likely_bullet_continuation(line: str, prev_line: str, indent_level: int) -> bool:
            """Helper to detect if a line is likely a continuation of a bullet point"""
            if not line or not prev_line:
                return False
                
            # Check if indentation matches expected continuation
            if indent_level is not None and len(line) - len(line.lstrip()) != indent_level:
                return False
            
            # Avoid treating metadata as continuations
            if any(re.search(pattern, line.lower()) for pattern in [
                r'\b\d{4}\b',  # Year
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',  # Month
                r'\b(inc|corp|llc|university|college|company)\b',  # Organization
                r'\b[A-Z]{2,}\b',  # Acronyms
                r'\b(http|www)\b'  # URLs
            ]):
                return False
            
            # Check for continuation indicators
            return (
                not BULLET_REGEX.match(line) and  # Not a new bullet
                not line.strip().endswith(('.', ':', ';')) and  # Not a complete thought
                len(line.strip()) > 10 and  # Substantial content
                not any(line.strip().lower().startswith(word) for word in ACTION_VERBS)  # Not a new action
            )
        
        for i, line in enumerate(lines):
            original_line = line
            line_stripped = line.strip()
            current_indent = len(line) - len(line.lstrip())
            
            if not line_stripped:
                continue
            
            # Detect if this is a new bullet point
            is_new_bullet = False
            bullet_content = line_stripped
            
            # Method 1: Explicit bullet symbols
            if BULLET_REGEX.match(line_stripped):
                is_new_bullet = True
                bullet_content = BULLET_REGEX.sub('', line_stripped).strip()
                bullet_continuation_indent = current_indent + 2  # Standard continuation indent
            
            # Method 2: Action verbs (for relevant sections)
            elif use_action_verbs and line_stripped:
                words = line_stripped.split()
                first_word = words[0].lower() if words else ''
                
                # Check for action verbs and ensure it's a substantial bullet
                if (first_word in ACTION_VERBS and 
                    len(line_stripped) > 20 and
                    not any(char in line_stripped[:10] for char in ['(', ')', '|', '–', '-']) and
                    not line_stripped.endswith((',', ':', ';'))):
                    is_new_bullet = True
                    bullet_content = line_stripped
                    bullet_continuation_indent = current_indent + 2
            
            # Method 3: Important phrases
            elif use_action_verbs and line_stripped:
                # Check for important phrases that indicate achievements
                if any(phrase in line_stripped.lower() for phrase in IMPORTANT_PHRASES):
                    if (len(line_stripped) > 15 and
                    not any(char in line_stripped[:10] for char in ['(', ')', '|', '–', '-'])):
                        is_new_bullet = True
                        bullet_content = line_stripped
                        bullet_continuation_indent = current_indent + 2
            
            # Handle bullet point content
            if is_new_bullet:
                # Save previous bullet if it exists
                if current_bullet is not None:
                    bullet_points.append(current_bullet.strip())
                
                # Start new bullet
                current_bullet = bullet_content
            else:
                # Check if this is a continuation of the current bullet
                prev_line = lines[i-1].strip() if i > 0 else ""
                if (current_bullet is not None and 
                    is_likely_bullet_continuation(line, prev_line, bullet_continuation_indent)):
                    # Add continuation to current bullet
                        current_bullet += ' ' + line_stripped
                else:
                    # This is non-bullet content
                    if current_bullet is not None:
                        # Save current bullet before moving to non-bullet content
                        bullet_points.append(current_bullet.strip())
                        current_bullet = None
                    non_bullet_content.append(original_line)
        
        # Add the final bullet if it exists
        if current_bullet is not None:
            bullet_points.append(current_bullet.strip())
        
        # Post-processing to ensure no bullets are lost
        # 1. Check for orphaned continuations
        for i, content in enumerate(non_bullet_content):
            if (i > 0 and content.strip() and
                is_likely_bullet_continuation(content, non_bullet_content[i-1], None)):
                # Try to merge with previous bullet
                if bullet_points:
                    bullet_points[-1] += ' ' + content.strip()
                    non_bullet_content[i] = ''
        
        # 2. Verify all bullet-like content is captured
        for content in non_bullet_content:
            content_stripped = content.strip()
            if (content_stripped and
                (BULLET_REGEX.match(content_stripped) or
                 any(content_stripped.lower().startswith(verb) for verb in ACTION_VERBS))):
                # This looks like a missed bullet
                if BULLET_REGEX.match(content_stripped):
                    content_stripped = BULLET_REGEX.sub('', content_stripped).strip()
                bullet_points.append(content_stripped)
        
        # Clean up non-bullet content
        non_bullet_content = [line for line in non_bullet_content if line.strip()]
        
        # Ensure no duplicate bullets
        bullet_points = list(dict.fromkeys(bullet_points))
        
        return bullet_points, non_bullet_content
    
    def tailor_resume(self, job_description: str, resume_file: str, 
                     output_file: str = None, dry_run: bool = False, debug: bool = False,
                     bullets_only: bool = False, polish_mode: bool = False) -> Dict:
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
        
        # Step 2: Analyze job description (skip in polish mode)
        if polish_mode:
            logger.info("Polish mode: Skipping job description analysis")
            job_analysis = None
        else:
            logger.info("Analyzing job description...")
            job_analysis = self.job_analyzer.analyze_job_description(job_description)
        
        # Step 3: Calculate initial match score (skip in polish mode)
        if polish_mode:
            initial_score = 0.0
        else:
            initial_score = self._calculate_match_score(resume_text, job_analysis)
        
        # Step 4: Rewrite relevant sections in batches
        logger.info("Processing resume content...")
        tailored_sections = []
        total_bullets_processed = 0
        bullet_changes = []  # Track bullet changes for bullets-only output
        
        for section in resume_sections:
            if section.section_type in ['experience', 'projects', 'research', 'involvement', 'skills']:
                # Extract bullet points - more flexible detection
                bullet_points, non_bullet_content = self.detect_bullets(section.content, section.section_type)
                
                # Special handling for Technical Skills section
                if section.section_type == 'skills' and not polish_mode:
                    section = self._enhance_technical_skills_section(section, job_analysis)
                
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
                    if polish_mode:
                        rewritten_bullets = self.ai_rewriter.polish_bullet_points_batch(
                            bullet_points, 
                            f"This is from the {section.section_type} section",
                            dry_run
                        )
                    else:
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
                    seen_bullets = set()  # Track unique bullets
                    
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
                                rewritten = rewritten_bullets[bullet_idx]
                                
                                # Check for near-duplicate content
                                rewritten_key = self._normalize_bullet_for_dedup(rewritten)
                                is_duplicate = False
                                
                                for seen_bullet in seen_bullets:
                                    if self._bullets_are_similar(rewritten_key, seen_bullet):
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    # Add unique bullet with original formatting
                                    if item_stripped.startswith(('•', '-', '*', '●')):
                                        prefix = item_stripped[0] + ' '
                                        new_content.append(prefix + rewritten)
                                else:
                                        new_content.append('● ' + rewritten)
                                        seen_bullets.add(rewritten_key)
                                # Skip duplicate bullet entirely
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
        # Step 5: Generate improvement suggestions (skip in polish mode)
        if polish_mode:
            logger.info("Polish mode: Skipping improvement suggestions")
            suggestions = []
        else:
            logger.info("Generating improvement suggestions...")
            suggestions = self.ai_rewriter.suggest_new_bullet_points(
                job_analysis, resume_sections, dry_run
            )
        
        # Step 6: Create output
        tailored_resume_text = self._sections_to_text(tailored_sections)
        if polish_mode:
            final_score = 0.0
        else:
            final_score = self._calculate_match_score(tailored_resume_text, job_analysis)
        
        # Save output if specified and not in dry run
        if output_file and not dry_run:
            if bullets_only and bullet_changes:
                # Generate bullets-only output
                bullets_output = self.doc_parser._generate_bullets_only_output(bullet_changes, job_analysis if not polish_mode else None)
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
        """Calculate how well resume matches job requirements with enhanced scoring"""
        if not resume_text or not resume_text.strip():
            return 0.0
            
        resume_lower = resume_text.lower()
        
        # Enhanced matching with partial matches
        keyword_matches = self._count_matches_with_partial(job_analysis.keywords, resume_lower)
        skill_matches = self._count_matches_with_partial(job_analysis.required_skills, resume_lower)
        tech_matches = self._count_matches_with_partial(job_analysis.technologies, resume_lower)
        responsibility_matches = self._count_matches_with_partial(job_analysis.responsibilities, resume_lower)
        
        # Calculate weighted scores (technologies and skills are most important)
        keyword_score = keyword_matches * 1.0
        skill_score = skill_matches * 3.0    # Skills are very important
        tech_score = tech_matches * 3.5      # Technologies are most important
        responsibility_score = responsibility_matches * 2.0
        
        total_weighted_matches = keyword_score + skill_score + tech_score + responsibility_score
        
        # Calculate total possible weighted score
        total_keywords = len(job_analysis.keywords) * 1.0
        total_skills = len(job_analysis.required_skills) * 3.0
        total_tech = len(job_analysis.technologies) * 3.5
        total_responsibilities = len(job_analysis.responsibilities) * 2.0
        
        total_possible = total_keywords + total_skills + total_tech + total_responsibilities
        
        if total_possible == 0:
            return 0.0
        
        # Calculate base percentage
        base_score = (total_weighted_matches / total_possible) * 100
        
        # Add bonuses for high match rates in critical areas
        if len(job_analysis.required_skills) > 0:
            skill_match_rate = skill_matches / len(job_analysis.required_skills)
            if skill_match_rate >= 0.6:  # 60%+ skill match
                base_score += 15
            elif skill_match_rate >= 0.4:  # 40%+ skill match
                base_score += 8
        
        if len(job_analysis.technologies) > 0:
            tech_match_rate = tech_matches / len(job_analysis.technologies)
            if tech_match_rate >= 0.5:  # 50%+ tech match
                base_score += 20
            elif tech_match_rate >= 0.3:  # 30%+ tech match
                base_score += 10
        
        # Bonus for having key technical skills
        key_tech_bonus = self._calculate_key_tech_bonus(resume_lower, job_analysis)
        base_score += key_tech_bonus
        
        return min(base_score, 100.0)  # Cap at 100%
    
    def _count_matches_with_partial(self, items: List[str], resume_lower: str) -> int:
        """Count matches including partial matches for better scoring"""
        matches = 0
        for item in items:
            item_lower = item.lower()
            # Exact match
            if item_lower in resume_lower:
                matches += 1
            else:
                # Partial match for multi-word items
                words = item_lower.split()
                if len(words) > 1:
                    word_matches = sum(1 for word in words if word in resume_lower)
                    if word_matches >= len(words) * 0.6:  # 60% of words match
                        matches += 0.5  # Partial credit
        return matches
    
    def _calculate_key_tech_bonus(self, resume_lower: str, job_analysis: JobAnalysis) -> float:
        """Calculate bonus points for having key technical skills"""
        bonus = 0.0
        
        # Key technical skills that are highly valued
        key_skills = {
            'python': 3.0,
            'java': 3.0,
            'javascript': 2.5,
            'react': 2.5,
            'sql': 2.0,
            'machine learning': 4.0,
            'data analysis': 3.0,
            'algorithms': 2.5,
            'data structures': 2.5,
            'cloud': 3.0,
            'agile': 2.0,
            'ci/cd': 2.5
        }
        
        for skill, points in key_skills.items():
            if skill in resume_lower:
                bonus += points
        
        # Cap the bonus to prevent over-inflation
        return min(bonus, 15.0)
    
    def _enhance_technical_skills_section(self, section: ResumeSection, job_analysis: JobAnalysis) -> ResumeSection:
        """Enhance Technical Skills section with missing but truthful skills from job requirements"""
        if not section.content:
            return section
        
        # Define skill categories and their keywords
        skill_categories = {
            'Programming Languages': {
                'keywords': ['programming', 'languages', 'coding'],
                'skills': {'python', 'java', 'javascript', 'typescript', 'html', 'css', 'sql', 
                          'c++', 'c#', 'r', 'swift', 'kotlin', 'go', 'rust', 'php', 'ruby'}
            },
            'Web Technologies': {
                'keywords': ['web', 'frontend', 'backend', 'full-stack'],
                'skills': {'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
                          'spring', 'asp.net', 'jquery', 'bootstrap', 'rest api', 'graphql'}
            },
            'Cloud & DevOps': {
                'keywords': ['cloud', 'devops', 'infrastructure', 'deployment'],
                'skills': {'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins',
                          'ci/cd', 'git', 'github', 'gitlab', 'bitbucket', 'cloud computing'}
            },
            'Data & Analytics': {
                'keywords': ['data', 'analytics', 'database', 'storage'],
                'skills': {'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                          'data analysis', 'data visualization', 'big data', 'data warehousing'}
            },
            'Machine Learning': {
                'keywords': ['machine learning', 'ai', 'artificial intelligence', 'ml'],
                'skills': {'tensorflow', 'pytorch', 'scikit-learn', 'machine learning',
                          'deep learning', 'nlp', 'computer vision', 'artificial intelligence'}
            },
            'Development Tools': {
                'keywords': ['tools', 'environments', 'ide', 'development'],
                'skills': {'git', 'github', 'jira', 'confluence', 'agile', 'scrum',
                          'vs code', 'intellij', 'eclipse', 'postman', 'docker'}
            },
            'Core Concepts': {
                'keywords': ['concepts', 'fundamentals', 'principles'],
                'skills': {'data structures', 'algorithms', 'oop', 'design patterns',
                          'software architecture', 'system design', 'api design'}
            },
            'Professional Skills': {
                'keywords': ['soft skills', 'professional', 'interpersonal'],
                'skills': {'problem-solving', 'communication', 'collaboration', 'leadership',
                          'project management', 'agile methodologies', 'team work'}
            }
        }
        
        # Extract current skills and their categories
        current_categories = {}
        current_skills = set()
        
        for line in section.content:
            line = line.strip()
            if ':' in line:
                category, skills = line.split(':', 1)
                category = category.strip()
                skills = {s.strip().lower() for s in skills.split(',') if s.strip()}
                current_categories[category] = skills
                current_skills.update(skills)
            else:
                # Handle lines without categories
                skills = {s.strip().lower() for s in line.split(',') if s.strip()}
                current_skills.update(skills)
        
        # Extract job-required skills
        job_skills = set()
        for skill in (job_analysis.technologies + job_analysis.required_skills):
            skill_lower = skill.lower()
            # Clean up skill names
            skill_lower = re.sub(r'\s+', ' ', skill_lower)
            job_skills.add(skill_lower)
        
        # Find missing skills that are truthful to add
        missing_skills = {}
        for category_name, category_info in skill_categories.items():
            category_skills = category_info['skills']
            # Find skills that are in both job requirements and our safe list
            relevant_skills = {s for s in job_skills if s.lower() in category_skills}
            # Remove skills we already have
            missing = {s for s in relevant_skills if s.lower() not in current_skills}
            if missing:
                missing_skills[category_name] = missing
        
        if not missing_skills:
            return section
        
        # Prepare enhanced content
        enhanced_content = []
        categories_added = set()
        
        # First, enhance existing categories
        for line in section.content:
            line = line.strip()
            if ':' in line:
                category, skills = line.split(':', 1)
                category = category.strip()
                
                # Find the matching predefined category
                matching_category = None
                for predef_cat, cat_info in skill_categories.items():
                    if any(kw in category.lower() for kw in cat_info['keywords']):
                        matching_category = predef_cat
                        break
                
                if matching_category and matching_category in missing_skills:
                    # Add missing skills to this category
                    current = {s.strip() for s in skills.split(',') if s.strip()}
                    additional = missing_skills[matching_category]
                    all_skills = sorted(current | additional)
                    enhanced_line = f"{category}: {', '.join(all_skills)}"
                    enhanced_content.append(enhanced_line)
                    categories_added.add(matching_category)
                else:
                    enhanced_content.append(line)
            else:
                enhanced_content.append(line)
        
        # Add new categories for remaining missing skills
        for category, skills in missing_skills.items():
            if category not in categories_added and skills:
                enhanced_content.append(f"{category}: {', '.join(sorted(skills))}")
        
        # Ensure proper spacing
        if enhanced_content and not enhanced_content[-1].strip():
            enhanced_content.pop()
        enhanced_content.append('')
        
        return ResumeSection(
            title=section.title,
            content=enhanced_content,
            section_type=section.section_type
        )
    
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
@click.option('--polish', is_flag=True, help='Polish mode: improve readability and professionalism without job tailoring')
def main(job_file, resume_file, output_file, api_key, max_calls, dry_run, clear_cache, debug, bullets_only, polish):
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
        results = pipeline.tailor_resume(job_description, resume_file, output_file, dry_run, debug, bullets_only, polish)
        
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
            if polish:
                click.echo(f"📊 Polish mode: Bullets improved for readability and professionalism")
            else:
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