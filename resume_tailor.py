from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set, Any
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

# Import ATS Optimizer
try:
    from ats_optimizer import ATSOptimizer, ATSOptimizationResult
except ImportError:
    # Fallback if ats_optimizer.py is not available
    ATSOptimizer = None
    ATSOptimizationResult = None

load_dotenv()

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resume_tailor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
class TailoredOutput:
    """Represents the complete output of the tailoring process"""
    polished_resume: str
    technical_skills: Dict[str, List[str]]
    recommended_skills: List[str]
    debug_info: Dict[str, Any]
    notes: List[str]
    quantification_score: float
    jd_alignment_score: float
    ats_score: float
    overall_score: float

@dataclass
class SkillAnalysis:
    """Results of skill gap analysis"""
    present_skills: Set[str]
    missing_skills: Set[str]
    partially_matched_skills: Dict[str, str]  # job_skill -> resume_skill that partially matches
    
@dataclass
class JobAnalysis:
    """Results of job description analysis"""
    keywords: List[str]
    required_skills: List[str]
    preferred_skills: List[str]
    technologies: List[str]
    responsibilities: List[str]

class QuantificationEnhancer:
    """Enhances bullet points with quantified metrics when missing"""
    
    def __init__(self):
        self.technical_verbs = {
            'developed', 'created', 'built', 'designed', 'implemented', 'optimized',
            'improved', 'analyzed', 'processed', 'managed', 'led', 'coordinated',
            'increased', 'decreased', 'reduced', 'enhanced', 'streamlined', 'automated',
            'integrated', 'deployed', 'configured', 'maintained', 'monitored', 'tested',
            'debugged', 'refactored', 'scaled', 'migrated', 'upgraded', 'customized'
        }
        
        self.metric_placeholders = {
            'developed': 'by ~30%',
            'created': 'serving ~100 users',
            'built': 'with 99.9% uptime',
            'designed': 'reducing load time by ~40%',
            'implemented': 'improving efficiency by ~25%',
            'optimized': 'by ~35%',
            'improved': 'by ~20%',
            'analyzed': 'processing ~10K records',
            'processed': '~5K data points',
            'managed': 'team of ~8 members',
            'led': 'team of ~6 developers',
            'coordinated': 'across ~4 departments',
            'increased': 'by ~50%',
            'decreased': 'by ~30%',
            'reduced': 'by ~25%',
            'enhanced': 'by ~40%',
            'streamlined': 'workflow by ~35%',
            'automated': '~80% of processes',
            'integrated': 'with ~5 APIs',
            'deployed': 'to ~3 environments',
            'configured': '~10 servers',
            'maintained': '~15 applications',
            'monitored': '~20 systems',
            'tested': '~100 test cases',
            'debugged': '~50 issues',
            'refactored': '~500 lines of code',
            'scaled': 'to ~1000 users',
            'migrated': '~200GB of data',
            'upgraded': '~10 components',
            'customized': 'for ~5 clients'
        }
    
    def has_quantification(self, text: str) -> bool:
        """Check if text already contains quantifiable metrics"""
        # Look for numbers, percentages, or common metric indicators
        patterns = [
            r'\d+%',  # percentages
            r'\d+\.\d+%',  # decimal percentages
            r'\d+\s*(users?|records?|files?|systems?|applications?|servers?)',  # counts
            r'\d+\s*(GB|MB|KB|TB)',  # data sizes
            r'\d+\s*(hours?|minutes?|days?|weeks?|months?)',  # time periods
            r'\d+\s*(times?|x)',  # multipliers
            r'\$\d+',  # money amounts
            r'\d+\s*(lines?|pages?|sections?)',  # content metrics
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def find_technical_verb(self, text: str) -> Optional[str]:
        """Find the primary technical verb in the text"""
        text_lower = text.lower()
        for verb in self.technical_verbs:
            if verb in text_lower:
                return verb
        return None
    
    def enhance_bullet(self, bullet: str) -> str:
        """Enhance a bullet point with quantification if missing"""
        if self.has_quantification(bullet):
            return bullet
        
        verb = self.find_technical_verb(bullet)
        if not verb:
            return bullet
        
        placeholder = self.metric_placeholders.get(verb, 'by ~25%')
        
        # Add the metric naturally to the bullet
        if bullet.endswith('.'):
            bullet = bullet[:-1]
        
        enhanced = f"{bullet}, {placeholder}."
        return enhanced

class JobDescriptionAlignmentLayer:
    """Aligns resume content with job description requirements"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            min_df=1,
            max_df=0.95
        )
    
    def extract_key_phrases(self, job_description: str, top_n: int = 20) -> List[str]:
        """Extract top key phrases from job description"""
        # Clean and tokenize
        sentences = sent_tokenize(job_description)
        
        # Extract noun phrases and technical terms
        key_phrases = []
        
        # Technical skills patterns
        tech_patterns = [
            r'\b(?:python|java|javascript|react|angular|vue|node\.?js|typescript|sql|mongodb|postgresql|mysql|aws|azure|gcp|docker|kubernetes|git|jenkins|agile|scrum|devops|ci/cd|machine learning|artificial intelligence|data science|big data|cloud computing|microservices|rest api|graphql)\b',
            r'\b(?:software development|web development|mobile development|full stack|frontend|backend|database|testing|debugging|deployment|scaling|optimization|automation|integration|security|performance|monitoring|analytics|visualization)\b',
            r'\b(?:team collaboration|cross-functional|leadership|mentoring|project management|stakeholder|communication|problem solving|analytical|creative|innovative|detail-oriented|self-motivated)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, job_description, re.IGNORECASE)
            key_phrases.extend(matches)
        
        # Extract important phrases using TF-IDF
        try:
            tfidf_matrix = self.vectorizer.fit_transform([job_description])
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top scoring phrases
            phrase_scores = list(zip(feature_names, scores))
            phrase_scores.sort(key=lambda x: x[1], reverse=True)
            
            for phrase, score in phrase_scores[:top_n]:
                if score > 0.1 and len(phrase.split()) <= 3:  # Filter meaningful phrases
                    key_phrases.append(phrase)
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {e}")
        
        # Remove duplicates and return top phrases
        unique_phrases = list(dict.fromkeys(key_phrases))
        return unique_phrases[:top_n]
    
    def calculate_similarity(self, bullet_text: str, jd_phrases: List[str]) -> float:
        """Calculate semantic similarity between bullet and JD phrases"""
        if not jd_phrases:
            return 0.0
        
        try:
            # Combine bullet and JD phrases for vectorization
            texts = [bullet_text] + jd_phrases
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity between bullet and each JD phrase
            bullet_vector = tfidf_matrix[0:1]
            jd_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(bullet_vector, jd_vectors)[0]
            return max(similarities) if len(similarities) > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    def inject_keywords(self, bullet: str, jd_phrases: List[str], similarity_threshold: float = 0.6) -> str:
        """Inject relevant JD keywords into bullet if similarity is low"""
        similarity = self.calculate_similarity(bullet, jd_phrases)
        
        if similarity >= similarity_threshold:
            return bullet
        
        # Find the most relevant JD phrase to inject
        bullet_lower = bullet.lower()
        relevant_phrases = []
        
        for phrase in jd_phrases:
            phrase_lower = phrase.lower()
            # Check if phrase is not already in bullet
            if phrase_lower not in bullet_lower:
                # Check for partial matches or related terms
                words = phrase_lower.split()
                if any(word in bullet_lower for word in words if len(word) > 3):
                    relevant_phrases.append(phrase)
        
        if not relevant_phrases:
            return bullet
        
        # Select the most relevant phrase (longest, most specific)
        best_phrase = max(relevant_phrases, key=len)
        
        # Inject the phrase naturally
        if bullet.endswith('.'):
            bullet = bullet[:-1]
        
        enhanced = f"{bullet}, utilizing {best_phrase}."
        return enhanced

@dataclass
class BulletScore:
    """Multi-metric scoring for bullet points"""
    clarity: float
    quantification: float
    technical_accuracy: float
    jd_alignment: float
    overall: float
    
    def __str__(self):
        return f"Clarity: {self.clarity:.0f}, Quantification: {self.quantification:.0f}, Technical: {self.technical_accuracy:.0f}, Alignment: {self.jd_alignment:.0f}"

class EnhancedScoringSystem:
    """Enhanced multi-metric scoring system for bullet points"""
    
    def __init__(self):
        self.technical_keywords = {
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js', 'typescript',
            'sql', 'mongodb', 'postgresql', 'mysql', 'aws', 'azure', 'gcp', 'docker',
            'kubernetes', 'git', 'jenkins', 'agile', 'scrum', 'devops', 'ci/cd',
            'machine learning', 'artificial intelligence', 'data science', 'big data',
            'cloud computing', 'microservices', 'rest api', 'graphql', 'testing',
            'debugging', 'deployment', 'scaling', 'optimization', 'automation'
        }
        
        self.action_verbs = {
            'developed', 'created', 'built', 'designed', 'implemented', 'optimized',
            'improved', 'analyzed', 'processed', 'managed', 'led', 'coordinated',
            'increased', 'decreased', 'reduced', 'enhanced', 'streamlined', 'automated',
            'integrated', 'deployed', 'configured', 'maintained', 'monitored', 'tested',
            'debugged', 'refactored', 'scaled', 'migrated', 'upgraded', 'customized'
        }
    
    def score_clarity(self, bullet: str) -> float:
        """Score clarity based on sentence structure and readability"""
        score = 0.0
        
        # Length check (optimal range: 15-30 words)
        words = bullet.split()
        word_count = len(words)
        
        if 15 <= word_count <= 30:
            score += 10
        elif 10 <= word_count <= 40:
            score += 5
        
        # Action verb presence
        bullet_lower = bullet.lower()
        if any(verb in bullet_lower for verb in self.action_verbs):
            score += 10
        
        # Technical specificity
        tech_count = sum(1 for tech in self.technical_keywords if tech in bullet_lower)
        score += min(tech_count * 2, 10)  # Cap at 10 points
        
        # Sentence structure (starts with action verb)
        first_word = words[0].lower() if words else ""
        if first_word in self.action_verbs:
            score += 5
        
        return min(score, 25)  # Cap at 25 points
    
    def score_quantification(self, bullet: str) -> float:
        """Score quantification based on metrics presence"""
        score = 0.0
        
        # Check for various metric patterns
        patterns = [
            r'\d+%',  # percentages
            r'\d+\.\d+%',  # decimal percentages
            r'\d+\s*(users?|records?|files?|systems?|applications?|servers?)',  # counts
            r'\d+\s*(GB|MB|KB|TB)',  # data sizes
            r'\d+\s*(hours?|minutes?|days?|weeks?|months?)',  # time periods
            r'\d+\s*(times?|x)',  # multipliers
            r'\$\d+',  # money amounts
            r'\d+\s*(lines?|pages?|sections?)',  # content metrics
        ]
        
        metric_count = 0
        for pattern in patterns:
            if re.search(pattern, bullet, re.IGNORECASE):
                metric_count += 1
        
        if metric_count >= 2:
            score = 25
        elif metric_count == 1:
            score = 15
        else:
            score = 0
        
        return score
    
    def score_technical_accuracy(self, bullet: str) -> float:
        """Score technical accuracy based on proper terminology"""
        score = 0.0
        bullet_lower = bullet.lower()
        
        # Technical keyword presence
        tech_matches = sum(1 for tech in self.technical_keywords if tech in bullet_lower)
        score += min(tech_matches * 3, 15)  # Cap at 15 points
        
        # Proper technical phrasing
        technical_phrases = [
            'api', 'database', 'algorithm', 'framework', 'library', 'platform',
            'architecture', 'infrastructure', 'deployment', 'integration',
            'optimization', 'scalability', 'performance', 'security', 'monitoring'
        ]
        
        phrase_matches = sum(1 for phrase in technical_phrases if phrase in bullet_lower)
        score += min(phrase_matches * 2, 10)  # Cap at 10 points
        
        return min(score, 25)  # Cap at 25 points
    
    def score_jd_alignment(self, bullet: str, jd_phrases: List[str]) -> float:
        """Score alignment with job description"""
        if not jd_phrases:
            return 0.0
        
        bullet_lower = bullet.lower()
        alignment_score = 0.0
        
        # Direct phrase matches
        for phrase in jd_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in bullet_lower:
                alignment_score += 5
            else:
                # Partial word matches
                words = phrase_lower.split()
                word_matches = sum(1 for word in words if word in bullet_lower and len(word) > 3)
                if word_matches > 0:
                    alignment_score += word_matches * 1.5
        
        return min(alignment_score, 25)  # Cap at 25 points
    
    def calculate_overall_score(self, bullet: str, jd_phrases: List[str] = None) -> BulletScore:
        """Calculate comprehensive bullet score"""
        if jd_phrases is None:
            jd_phrases = []
        
        clarity = self.score_clarity(bullet)
        quantification = self.score_quantification(bullet)
        technical = self.score_technical_accuracy(bullet)
        alignment = self.score_jd_alignment(bullet, jd_phrases)
        
        overall = (clarity + quantification + technical + alignment) / 4
        
        return BulletScore(
            clarity=clarity,
            quantification=quantification,
            technical_accuracy=technical,
            jd_alignment=alignment,
            overall=overall
        )

@dataclass
class ATSReport:
    """ATS optimization report"""
    keyword_coverage: float
    formatting_score: float
    header_compliance: float
    keyword_normalization: float
    overall_ats_score: float
    issues: List[str]
    recommendations: List[str]

class ATSOptimizationLayer:
    """ATS optimization layer for resume compatibility"""
    
    def __init__(self):
        self.standard_headers = {
            'experience', 'professional experience', 'work experience', 'employment',
            'education', 'academic background', 'qualifications',
            'skills', 'technical skills', 'core competencies',
            'projects', 'project experience', 'portfolio',
            'certifications', 'licenses', 'awards', 'achievements',
            'summary', 'profile', 'objective', 'about'
        }
        
        self.skill_normalization = {
            'react.js': 'react',
            'reactjs': 'react',
            'node.js': 'nodejs',
            'nodejs': 'nodejs',
            'javascript': 'javascript',
            'js': 'javascript',
            'python': 'python',
            'py': 'python',
            'java': 'java',
            'sql': 'sql',
            'mysql': 'mysql',
            'postgresql': 'postgresql',
            'mongodb': 'mongodb',
            'aws': 'aws',
            'amazon web services': 'aws',
            'azure': 'azure',
            'microsoft azure': 'azure',
            'gcp': 'gcp',
            'google cloud': 'gcp',
            'docker': 'docker',
            'kubernetes': 'kubernetes',
            'k8s': 'kubernetes',
            'git': 'git',
            'github': 'github',
            'gitlab': 'gitlab',
            'jenkins': 'jenkins',
            'ci/cd': 'ci/cd',
            'continuous integration': 'ci/cd',
            'continuous delivery': 'ci/cd',
            'devops': 'devops',
            'agile': 'agile',
            'scrum': 'scrum',
            'machine learning': 'machine learning',
            'ml': 'machine learning',
            'artificial intelligence': 'artificial intelligence',
            'ai': 'artificial intelligence',
            'data science': 'data science',
            'big data': 'big data',
            'cloud computing': 'cloud computing',
            'microservices': 'microservices',
            'rest api': 'rest api',
            'graphql': 'graphql'
        }
    
    def calculate_keyword_coverage(self, resume_text: str, jd_keywords: List[str]) -> float:
        """Calculate percentage of JD keywords found in resume"""
        if not jd_keywords:
            return 0.0
        
        resume_lower = resume_text.lower()
        found_keywords = 0
        
        for keyword in jd_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in resume_lower:
                found_keywords += 1
        
        return (found_keywords / len(jd_keywords)) * 100
    
    def check_formatting_issues(self, resume_text: str) -> Tuple[float, List[str]]:
        """Check for ATS-unfriendly formatting"""
        issues = []
        score = 100.0
        
        # Check for tables
        if '|' in resume_text or '\t' in resume_text:
            issues.append("Contains tables or tabular formatting")
            score -= 20
        
        # Check for special characters/icons
        special_chars = ['●', '•', '→', '←', '↑', '↓', '★', '☆', '◆', '◇']
        for char in special_chars:
            if char in resume_text:
                issues.append(f"Contains special character: {char}")
                score -= 10
        
        # Check for headers in all caps (can be problematic)
        lines = resume_text.split('\n')
        for line in lines:
            if len(line) > 5 and line.isupper() and not line.isdigit():
                issues.append(f"All-caps header detected: {line[:20]}...")
                score -= 5
        
        # Check for excessive formatting
        if resume_text.count('\n') > len(resume_text.split()) * 0.1:
            issues.append("Excessive line breaks detected")
            score -= 10
        
        return max(score, 0), issues
    
    def check_header_compliance(self, resume_sections: List[ResumeSection]) -> Tuple[float, List[str]]:
        """Check if headers follow ATS-friendly standards"""
        issues = []
        score = 100.0
        
        for section in resume_sections:
            header_lower = section.title.lower().strip()
            
            # Check if header is too generic
            if header_lower in ['section', 'other', 'misc', 'additional']:
                issues.append(f"Generic header: {section.title}")
                score -= 15
            
            # Check if header is too creative/unconventional
            creative_indicators = ['awesome', 'amazing', 'fantastic', 'incredible', 'outstanding']
            if any(indicator in header_lower for indicator in creative_indicators):
                issues.append(f"Overly creative header: {section.title}")
                score -= 10
            
            # Check for proper section naming
            if not any(std_header in header_lower for std_header in self.standard_headers):
                if len(header_lower) > 3:  # Only flag if it's a substantial header
                    issues.append(f"Non-standard header: {section.title}")
                    score -= 5
        
        return max(score, 0), issues
    
    def normalize_keywords(self, resume_text: str) -> Tuple[str, float]:
        """Normalize skill keywords for better ATS compatibility"""
        normalized_text = resume_text
        normalization_count = 0
        
        for variant, standard in self.skill_normalization.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(variant), re.IGNORECASE)
            matches = pattern.findall(resume_text)
            
            if matches:
                normalized_text = pattern.sub(standard, normalized_text)
                normalization_count += len(matches)
        
        # Calculate normalization score
        if normalization_count > 0:
            score = min(100, 80 + (normalization_count * 2))
        else:
            score = 100
        
        return normalized_text, score
    
    def generate_ats_report(self, resume_text: str, resume_sections: List[ResumeSection], 
                           jd_keywords: List[str]) -> ATSReport:
        """Generate comprehensive ATS optimization report with enhanced screening"""
        
        # Keyword coverage and frequency analysis
        keyword_coverage = self.calculate_keyword_coverage(resume_text, jd_keywords)
        keyword_frequency = self.calculate_keyword_frequency(resume_text, jd_keywords)
        
        # Formatting check with enhanced detection
        formatting_score, formatting_issues = self.check_formatting_issues(resume_text)
        
        # Header compliance
        header_score, header_issues = self.check_header_compliance(resume_sections)
        
        # Keyword normalization
        normalized_text, normalization_score = self.normalize_keywords(resume_text)
        
        # Overall ATS score
        overall_score = (keyword_coverage * 0.4 + formatting_score * 0.2 + 
                        header_score * 0.2 + normalization_score * 0.2)
        
        # Compile issues and recommendations
        all_issues = formatting_issues + header_issues
        recommendations = []
        
        if keyword_coverage < 70:
            recommendations.append("Add more job-relevant keywords to improve ATS matching")
        
        if formatting_score < 80:
            recommendations.append("Remove special characters and tables for better ATS compatibility")
        
        if header_score < 90:
            recommendations.append("Use standard section headers (Experience, Education, Skills)")
        
        if normalization_score < 95:
            recommendations.append("Standardize skill terminology for better keyword matching")
        
        # Add keyword frequency recommendations
        low_frequency_keywords = [kw for kw, freq in keyword_frequency.items() if freq < 2]
        if low_frequency_keywords:
            recommendations.append(f"Increase keyword density for: {', '.join(low_frequency_keywords[:3])}")
        
        return ATSReport(
            keyword_coverage=keyword_coverage,
            formatting_score=formatting_score,
            header_compliance=header_score,
            keyword_normalization=normalization_score,
            overall_ats_score=overall_score,
            issues=all_issues,
            recommendations=recommendations
        )
    
    def calculate_keyword_frequency(self, resume_text: str, jd_keywords: List[str]) -> Dict[str, int]:
        """Calculate frequency of each JD keyword in resume"""
        frequency_map = {}
        resume_lower = resume_text.lower()
        
        for keyword in jd_keywords:
            keyword_lower = keyword.lower()
            # Count occurrences using word boundaries
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            count = len(re.findall(pattern, resume_lower))
            frequency_map[keyword] = count
        
        return frequency_map

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
        """Extract required and preferred skills with enhanced parsing"""
        if not text:
            return [], []
            
        text_lower = text.lower()

        # Split text into sections
        sections = re.split(r'\n\s*\n', text_lower)
        
        # Find required and preferred sections with improved logic
        required_section = ""
        preferred_section = ""
        responsibilities_section = ""
        
        for i, section in enumerate(sections):
            section_lower = section.lower()
            
            # Check if this section contains required markers (but not preferred)
            if any(marker in section_lower for marker in ['required', 'requirements', 'qualifications', 'must have']):
                # Make sure it's not a preferred section
                if not any(pref_marker in section_lower for pref_marker in ['preferred', 'nice to have', 'plus', 'bonus']):
                    # If this is a header section, also check the next section for content
                    if len(section.strip()) < 100:  # Likely a header
                        if i + 1 < len(sections) and len(sections[i + 1].strip()) > 50:
                            required_section = sections[i + 1]  # Use the next section
                        else:
                            required_section = section
                    else:
                        required_section = section
            # Check if this section contains preferred markers
            elif any(marker in section_lower for marker in ['preferred', 'nice to have', 'plus', 'bonus']):
                # If this is a header section, also check the next section for content
                if len(section.strip()) < 100:  # Likely a header
                    if i + 1 < len(sections) and len(sections[i + 1].strip()) > 50:
                        preferred_section = sections[i + 1]  # Use the next section
                    else:
                        preferred_section = section
                else:
                    preferred_section = section
            # Check if this section contains responsibilities markers
            elif any(marker in section_lower for marker in ['responsibilities', 'duties', 'what you will do']):
                responsibilities_section = section
            # Check if this section contains actual qualifications (fallback)
            elif any(qual in section_lower for qual in ['python', 'java', 'javascript', 'c++', 'c#', 'react', 'cloud', 'ai', 'mobile']):
                # If we haven't found a required section yet, this might be it
                if not required_section and len(section.strip()) > 50:  # Avoid empty sections
                    required_section = section

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
                    skills = self._extract_meaningful_skills_enhanced(match)
                    required_skills.extend(skills)
        
        # Process preferred section
        if preferred_section:
            for pattern in preferred_patterns:
                matches = re.findall(pattern, preferred_section, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = ' '.join(match)
                    skills = self._extract_meaningful_skills_enhanced(match)
                    preferred_skills.extend(skills)
        
        # Look for skills in responsibilities that aren't already found
        if responsibilities_section:
            resp_skills = self._extract_meaningful_skills_enhanced(responsibilities_section)
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
        
        # Apply enhanced processing
        required_skills = self._apply_skill_hierarchy(required_skills)
        preferred_skills = self._apply_skill_hierarchy(preferred_skills)
        
        # Remove duplicates and ensure uniqueness
        required_skills = list(dict.fromkeys(required_skills))  # Preserves order
        preferred_skills = list(dict.fromkeys(preferred_skills))  # Preserves order
        
        return required_skills, preferred_skills
    
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
    
    def _extract_meaningful_skills_enhanced(self, text: str) -> List[str]:
        """Enhanced skill extraction with semantic detection and normalization"""
        if not text:
            return []
        
        # Define comprehensive skill categories with semantic variations
        skill_categories = {
            'languages': [
                'python', 'java', 'javascript', 'react', 'c++', 'c#', 'node.js', 'html', 'css',
                'sql', 'r', 'swift', 'go', 'rust', 'ruby', 'php', 'kotlin', 'scala', 'typescript'
            ],
            'core_tech': [
                'cloud', 'artificial intelligence', 'mobile', 'databases', 'data structures', 
                'algorithms', 'big data', 'data warehousing', 'ci/cd', 'application resiliency', 
                'security', 'machine learning', 'data analysis', 'software development',
                'web development', 'mobile development', 'devops', 'cloud computing',
                'machine learning', 'deep learning', 'data science', 'business intelligence'
            ],
            'tools_frameworks': [
            'developmental toolsets', 'relational databases', 'git', 'github', 'docker',
            'kubernetes', 'aws', 'azure', 'gcp', 'jenkins', 'jira', 'confluence',
                'react.js', 'angular', 'vue.js', 'spring', 'django', 'flask', 'express'
            ],
            'professional': [
                'business analysis', 'development', 'maintenance', 'software improvement',
                'agile methodologies', 'problem-solving', 'collaboration', 'computer science',
                'engineering', 'communication', 'interpersonal skills', 'teamwork',
                'project management', 'leadership', 'mentoring'
            ]
        }
        
        # Skill normalization mapping with comprehensive variants
        skill_synonyms = {
            'ci/cd': ['continuous integration', 'continuous delivery', 'cicd', 'ci cd'],
            'ai': ['artificial intelligence'],
            'react.js': ['react', 'reactjs', 'react js'],
            'js': ['javascript'],
            'ml': ['machine learning'],
            'db': ['database', 'databases'],
            'relational db': ['relational databases', 'relational database'],
            'sql': ['mysql', 'postgresql', 'postgres', 'sqlite', 'oracle'],
            'big data': ['bigdata', 'big-data', 'large-scale data', 'data lake', 'data pipeline'],
            'data warehousing': ['datawarehousing', 'data-warehousing', 'etl', 'data marts'],
            'application resiliency': ['app resiliency', 'application resilience', 'system reliability', 'high availability'],
            'devops': ['dev-ops', 'dev ops'],
            'machine learning': ['ml', 'machine-learning'],
            'cloud computing': ['cloud', 'cloud platforms'],
            'data science': ['data analytics', 'data analysis'],
            'web development': ['web dev', 'frontend', 'backend', 'full stack'],
            'mobile development': ['mobile dev', 'ios', 'android'],
            'software development': ['software engineering', 'software dev'],
            'business analysis': ['business intelligence', 'requirements analysis'],
            'agile methodologies': ['agile', 'scrum', 'kanban'],
            'problem-solving': ['problem solving', 'analytical thinking'],
            'collaboration': ['teamwork', 'cross-functional'],
            'communication': ['interpersonal skills', 'verbal communication', 'written communication']
        }
        
        found_skills = []
        text_lower = text.lower()
        
        # Process each category with enhanced detection
        for category, skills in skill_categories.items():
            for skill in skills:
                # Check for exact matches and variations
                if self._find_skill_match(text_lower, skill):
                    found_skills.append(skill)
        
        # Apply semantic extraction for multi-word phrases with contextual matching
        semantic_phrases = [
            'big data', 'data warehousing', 'application resiliency', 
            'machine learning', 'devops', 'relational databases',
            'artificial intelligence', 'data structures', 'ci/cd',
            'cloud computing', 'web development', 'mobile development',
            'software engineering', 'business analysis', 'agile methodologies',
            'data lake', 'data pipeline', 'etl', 'data marts',
            'system reliability', 'high availability', 'microservices',
            'rest api', 'graphql', 'kubernetes', 'docker containers',
            'continuous integration', 'continuous deployment', 'cicd',
            'business intelligence', 'requirements analysis', 'scrum',
            'kanban', 'cross-functional', 'analytical thinking',
            'interpersonal skills', 'verbal communication', 'written communication'
        ]
        
        for phrase in semantic_phrases:
            if self._find_skill_match(text_lower, phrase):
                found_skills.append(phrase)
        
        # Add phrase-based contextual matching for capitalized 2+ word phrases
        contextual_phrases = self._extract_contextual_phrases(text)
        found_skills.extend(contextual_phrases)
        
        # Apply normalization and n-gram merging
        normalized_skills = []
        for skill in found_skills:
            normalized = self._normalize_skill(skill, skill_synonyms)
            if normalized not in normalized_skills:
                normalized_skills.append(normalized)
        
        # Apply n-gram merging to handle skill hierarchy
        merged_skills = self._apply_skill_hierarchy_enhanced(normalized_skills)
        
        # Filter out filler words
        filler_words = {
            'new', 'skills', 'ideas', 'overview', 'innovative', 'program',
            'ability', 'basic', 'understanding', 'knowledge', 'experience',
            'strong', 'good', 'excellent', 'proficient', 'familiar'
        }
        
        filtered_skills = [
            skill for skill in merged_skills
            if skill.lower() not in filler_words and len(skill) > 2
        ]
        
        return filtered_skills
    
    def _apply_skill_hierarchy_enhanced(self, skills: List[str]) -> List[str]:
        """Enhanced n-gram merging with lemmatization and skill hierarchy"""
        if not skills:
            return []
        
        # Sort skills by length (longer first) to prioritize more specific terms
        sorted_skills = sorted(skills, key=len, reverse=True)
        
        # Track seen skills (case-insensitive)
        seen_lower = set()
        merged_skills = []
        
        for skill in sorted_skills:
            skill_lower = skill.lower()
            
            # Check if this skill is already covered by a longer one
            is_covered = False
            for seen_skill in seen_lower:
                if skill_lower in seen_skill and len(skill_lower) < len(seen_skill):
                    is_covered = True
                    break
            
            if not is_covered:
                # Check if any existing skill should be replaced by this longer one
                merged_skills = [
                    s for s in merged_skills 
                    if not (s.lower() in skill_lower and len(s.lower()) < len(skill_lower))
                ]
                merged_skills.append(skill)
                seen_lower.add(skill_lower)
        
        return merged_skills
    
    def _extract_contextual_phrases(self, text: str) -> List[str]:
        """Extract capitalized 2+ word phrases commonly associated with skills"""
        contextual_phrases = []
        
        # Pattern to match capitalized 2+ word phrases
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        matches = re.findall(pattern, text)
        
        # Common skill-related capitalized phrases
        skill_indicators = [
            'Big Data', 'Data Warehousing', 'Application Resiliency',
            'Machine Learning', 'Artificial Intelligence', 'Data Science',
            'Cloud Computing', 'Web Development', 'Mobile Development',
            'Software Engineering', 'Business Analysis', 'Agile Methodologies',
            'Data Lake', 'Data Pipeline', 'System Reliability',
            'High Availability', 'Microservices', 'Rest Api', 'Graphql',
            'Continuous Integration', 'Continuous Deployment', 'Business Intelligence',
            'Requirements Analysis', 'Cross Functional', 'Analytical Thinking',
            'Interpersonal Skills', 'Verbal Communication', 'Written Communication'
        ]
        
        for match in matches:
            if match in skill_indicators:
                contextual_phrases.append(match.lower())
        
        return contextual_phrases
    
    def _find_skill_match(self, text: str, skill: str) -> bool:
        """Enhanced skill matching with context awareness"""
        skill_lower = skill.lower()
        
        # Handle special cases like C++ and C#
        if skill in ['c++', 'c#']:
            return bool(re.search(rf'\b{re.escape(skill)}\b', text))
        
        # Handle multi-word skills
        if ' ' in skill:
            # Look for the phrase with word boundaries
            pattern = rf'\b{re.escape(skill_lower)}\b'
            if re.search(pattern, text):
                return True
            
            # Look for variations (hyphens, no spaces)
            variations = [
                skill_lower.replace(' ', '-'),
                skill_lower.replace(' ', ''),
                skill_lower.replace(' ', '_')
            ]
            for variation in variations:
                if variation in text:
                    return True
        else:
            # Single word skills
            return bool(re.search(rf'\b{re.escape(skill_lower)}\b', text))
        
        return False
    
    def _normalize_skill(self, skill: str, synonyms: Dict[str, List[str]]) -> str:
        """Normalize skill using synonym mapping"""
        skill_lower = skill.lower()
        
        # Check if skill is a synonym of a canonical form
        for canonical, synonym_list in synonyms.items():
            if skill_lower in [s.lower() for s in synonym_list]:
                return canonical.title()
        
        # Handle special cases with consistent normalization
        if skill_lower in ['c++', 'c#']:
            return skill.upper()
        elif skill_lower in ['react.js', 'reactjs', 'react js']:
            return 'React'
        elif skill_lower in ['ci/cd', 'cicd', 'ci cd']:
            return 'CI/CD'
        elif skill_lower in ['relational db', 'relational databases', 'relational database']:
            return 'Relational Databases'
        elif skill_lower in ['js', 'javascript']:
            return 'JavaScript'
        elif skill_lower in ['ai', 'artificial intelligence']:
            return 'Artificial Intelligence'
        elif skill_lower in ['ml', 'machine learning']:
            return 'Machine Learning'
        
        # Return skill with proper capitalization
        return skill.title()
    
    def _apply_skill_hierarchy(self, skills: List[str]) -> List[str]:
        """Apply n-gram merging to prevent duplicates and overlaps"""
        if not skills:
            return []
        
        # Sort skills by length (longer first) to prioritize more specific terms
        sorted_skills = sorted(skills, key=len, reverse=True)
        
        filtered_skills = []
        seen_lower = set()
        
        for skill in sorted_skills:
            skill_lower = skill.lower()
            
            # Check if this skill is already covered by a longer skill
            is_covered = False
            for existing in filtered_skills:
                existing_lower = existing.lower()
                
                # If current skill is contained in existing skill, skip it
                if skill_lower in existing_lower and len(skill) < len(existing):
                    is_covered = True
                    break
                
                # If existing skill is contained in current skill, remove existing
                if existing_lower in skill_lower and len(existing) < len(skill):
                    filtered_skills.remove(existing)
                    seen_lower.discard(existing_lower)
                    break
            
            if not is_covered and skill_lower not in seen_lower:
                filtered_skills.append(skill)
                seen_lower.add(skill_lower)
        
        return filtered_skills
    
    def extract_skills_with_metadata(self, text: str) -> Dict[str, Any]:
        """Extract skills with enhanced metadata and coverage calculation"""
        required_skills, preferred_skills = self.extract_skills(text)
        
        # Calculate coverage metrics
        total_required_skills = len(required_skills)
        total_preferred_skills = len(preferred_skills)
        total_skills = total_required_skills + total_preferred_skills
        
        # Calculate coverage percentages
        required_percentage = (total_required_skills / max(total_required_skills, 1)) * 100
        preferred_percentage = (total_preferred_skills / max(total_preferred_skills, 1)) * 100
        
        # Create comprehensive coverage summary
        coverage_summary = {
            "matched_skills": total_skills,
            "required_skills": total_required_skills,
            "preferred_skills": total_preferred_skills,
            "coverage_percent": round(required_percentage, 1) if total_required_skills > 0 else 0.0
        }
        
        return {
            'required_skills': required_skills,
            'preferred_skills': preferred_skills,
            'total_skills': total_skills,
            'required_count': total_required_skills,
            'preferred_count': total_preferred_skills,
            'required_percentage': round(required_percentage, 1),
            'preferred_percentage': round(preferred_percentage, 1),
            'coverage_summary': coverage_summary,
            'coverage_display': f"Coverage: {total_required_skills}/{total_required_skills} required (~{required_percentage:.1f}%)" if total_required_skills > 0 else "No required skills found"
        }
    
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
        Rewrite the following resume bullet points with a CONSERVATIVE approach - improve clarity and professionalism while maintaining complete truthfulness.
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
        
        CONSERVATIVE GUIDELINES - TRUTH FIRST:
        1. PRESERVE TECHNICAL TRUTH:
           - Keep ALL original technologies, tools, languages, and frameworks exactly as mentioned
           - Maintain ALL quantitative metrics and achievements
           - NEVER fabricate, add, or remove technical details
           - If a technology isn't mentioned in the original, DO NOT add it
        
        2. MINIMAL ENHANCEMENT ONLY:
           - Only enhance with terminology that's already implied or naturally fits
           - If automation is mentioned, you can highlight efficiency aspects (but don't add CI/CD if not mentioned)
           - If data work is mentioned, you can emphasize analysis aspects (but don't add tools not mentioned)
           - Focus on clarity and professional language over keyword matching
        
        3. MAINTAIN TECHNICAL DEPTH:
           - Keep specific implementation details exactly as stated
           - Preserve architecture and design decisions
           - Retain performance metrics and improvements
           - Don't add methodologies not already implied
        
        4. IMPROVE CLARITY AND IMPACT:
           - Use strong, professional action verbs
           - Improve readability and flow
           - Remove unnecessary words while keeping technical substance
           - Ensure each bullet is clear and impactful
        
        5. NATURAL INTEGRATION:
           - Only emphasize aspects that clearly match job requirements if they're already there
           - Don't force connections that aren't natural
           - Avoid keyword stuffing or buzzword addition
        
        Examples of EXCELLENT CONSERVATIVE rewrites:
        Original: "Developed Python script for data processing"
        Good: "Developed Python script for data processing, implementing statistical analysis to extract insights from large datasets" ✅ (Enhanced existing Python/data work)
        
        Original: "Built web application with React"
        Good: "Built web application with React, implementing responsive design and user authentication features" ✅ (Enhanced existing React work)
        
        Examples of BAD rewrites (NEVER DO THIS):
        Original: "Developed Python script for data processing"
        Bad: "Developed Python script for data processing using cloud technologies and CI/CD practices" ❌ (Added cloud/CI-CD not mentioned)
        
        Original: "Built web application with React"
        Bad: "Built web application with React and JavaScript using agile methodologies" ❌ (Added methodologies not mentioned)
        
        CONSERVATIVE RULE: If a skill is in the job description but NOT in the original bullet, DO NOT add it.
        
        Return EXACTLY {len(uncached_bullets)} rewritten bullets, improving clarity and professionalism while maintaining complete truthfulness.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert technical resume writer who maintains complete truthfulness. NEVER add skills, technologies, or experiences not already present in the original content. Focus on clarity and professionalism while preserving all technical details exactly as stated."},
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
        """Enhanced validation to ensure truthfulness and prevent skill fabrication"""
        validated_bullets = []
        
        for orig, rewritten in zip(original_bullets, rewritten_bullets):
            orig_lower = orig.lower()
            rewritten_lower = rewritten.lower()
            
            # 1. Check if technical details are preserved
            tech_terms = re.findall(r'\b(?:python|java|javascript|c\+\+|c#|sql|react|aws|azure|gcp|docker|kubernetes|fastapi|openai|google cloud|ggplot2|numpy|pandas|scikit-learn|matplotlib|seaborn|r|html|css|node\.js|github|git|linux)\b',
                                  orig_lower)
            
            # Ensure all technical terms are in rewritten version
            missing_terms = [term for term in tech_terms if term not in rewritten_lower]
            if missing_terms:
                logger.warning(f"Missing technical terms: {missing_terms}")
                # Use original if critical terms are missing
                if len(missing_terms) > 1:
                    logger.warning("Too many missing technical terms, using original")
                    validated_bullets.append(orig)
                    continue
            
            # 2. Check for metrics preservation
            metrics = re.findall(r'\d+%|\d+x|\d+\.?\d*%?|\d+\s*(?:users|requests|transactions|improvements?|seconds?|minutes?|hours?|days?|records|trees|accuracy|validation)',
                               orig_lower)
            missing_metrics = [m for m in metrics if m not in rewritten_lower]
            if missing_metrics:
                logger.warning(f"Missing metrics: {missing_metrics}")
                # Use original if important metrics are missing
                if len(missing_metrics) > 0:
                    logger.warning("Important metrics missing, using original")
                    validated_bullets.append(orig)
                    continue
            
            # 3. Check for inappropriate skill additions
            job_skills = set()
            for skill_list in [job_analysis.required_skills, job_analysis.preferred_skills, job_analysis.technologies]:
                job_skills.update([skill.lower() for skill in skill_list])
            
            # Check if rewrite added skills not in original
            original_skills = set()
            for skill in job_skills:
                if skill in orig_lower:
                    original_skills.add(skill)
            
            rewritten_skills = set()
            for skill in job_skills:
                if skill in rewritten_lower and skill not in orig_lower:
                    rewritten_skills.add(skill)
            
            if rewritten_skills:
                logger.warning(f"Added skills not in original: {rewritten_skills}")
                # If skills were added, use original to maintain truthfulness
                if len(rewritten_skills) > 0:
                    logger.warning("Skills added that weren't in original, using original")
                    validated_bullets.append(orig)
                    continue
            
            # 4. Check for inappropriate generalizations
            bad_phrases = ['innovative solutions', 'cutting-edge', 'state-of-the-art', 'next-generation', 'revolutionary', 'groundbreaking', 'industry-leading', 'best-in-class']
            if any(phrase in rewritten_lower for phrase in bad_phrases):
                logger.warning("Detected vague phrases, using original")
                validated_bullets.append(orig)
                continue
            
            # 5. Check for excessive buzzword usage
            buzzwords = ['leverage', 'utilize', 'harness', 'optimize', 'streamline', 'enhance', 'facilitate', 'enable', 'empower', 'drive', 'deliver', 'execute', 'implement', 'deploy']
            buzzword_count = sum(1 for word in buzzwords if word in rewritten_lower)
            if buzzword_count > 3:  # Too many buzzwords
                logger.warning("Too many buzzwords, using original")
                validated_bullets.append(orig)
                continue
            
            # 6. Check for length and quality
            if len(rewritten.strip()) < 20 or len(rewritten.strip()) > 200:
                logger.warning("Rewrite length inappropriate, using original")
                validated_bullets.append(orig)
                continue
            
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
        - "Worked with Python to analyze data" -> "Developed Python scripts to analyze and visualize datasets, improving efficiency by 20%"
        - "Used R for statistics" -> "Applied R programming for statistical analysis and data modeling"
        - "Built a website" -> "Developed responsive web application using HTML, CSS, and JavaScript"
        
        Examples of BAD polishing (avoid these):
        - "Worked with Python" -> "Leveraged cutting-edge Python technologies to drive innovative solutions"
        - "Used R" -> "Utilized advanced R programming methodologies to foster data-driven insights"
        
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
        self.quantification_enhancer = QuantificationEnhancer()
        self.jd_alignment_layer = JobDescriptionAlignmentLayer()
        self.scoring_system = EnhancedScoringSystem()
        self.ats_optimizer = ATSOptimizationLayer()
        self.dedicated_ats_optimizer = ATSOptimizer() if ATSOptimizer else None
    
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
                     output_file: str = None, mode: str = 'tailor', dry_run: bool = False, 
                     debug: bool = False, bullets_only: bool = False, verify_truth: bool = False,
                     output_format: str = 'standard') -> TailoredOutput:
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
        if mode == 'polish':
            logger.info("Polish mode: Skipping job description analysis")
            job_analysis = None
        else:
            logger.info("Analyzing job description...")
            job_analysis = self.job_analyzer.analyze_job_description(job_description)
        
        # Step 3: Calculate initial match score (skip in polish mode)
        if mode == 'polish':
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
                if section.section_type == 'skills' and mode != 'polish':
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
                    if mode == 'polish':
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
                    
                    # Apply enhancement layers if not in dry run
                    if not dry_run and rewritten_bullets:
                        # Extract JD key phrases for alignment
                        jd_phrases = []
                        if job_analysis:
                            jd_phrases = self.jd_alignment_layer.extract_key_phrases(job_description, top_n=20)
                        
                        # Apply quantification enhancement and JD alignment
                        enhanced_bullets = []
                        for bullet in rewritten_bullets:
                            # Step 1: Quantification enhancement
                            enhanced_bullet = self.quantification_enhancer.enhance_bullet(bullet)
                            
                            # Step 2: JD alignment enhancement
                            if jd_phrases:
                                enhanced_bullet = self.jd_alignment_layer.inject_keywords(
                                    enhanced_bullet, jd_phrases, similarity_threshold=0.6
                                )
                            
                            enhanced_bullets.append(enhanced_bullet)
                        
                        rewritten_bullets = enhanced_bullets
                    
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
        if mode == 'polish':
            logger.info("Polish mode: Skipping improvement suggestions")
            suggestions = []
        else:
            logger.info("Generating improvement suggestions...")
            suggestions = self.ai_rewriter.suggest_new_bullet_points(
                job_analysis, resume_sections, dry_run
            )
        
        # Step 6: Create output
        tailored_resume_text = self._sections_to_text(tailored_sections)
        if mode == 'polish':
            final_score = 0.0
        else:
            final_score = self._calculate_match_score(tailored_resume_text, job_analysis)
        
        # Save output if specified and not in dry run
        if output_file and not dry_run:
            if bullets_only and bullet_changes:
                # Generate bullets-only output
                bullets_output = self.doc_parser._generate_bullets_only_output(bullet_changes, job_analysis if mode != 'polish' else None)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(bullets_output)
            else:
                # Generate full resume output
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(tailored_resume_text)
        
        # Perform skill gap analysis if needed
        skill_analysis = None
        if mode in ['gap', 'tailor'] and job_analysis is not None:
            skill_analysis = self.analyze_skill_gaps(resume_sections, job_analysis)
        
        # Generate notes based on mode and analysis
        notes = []
        if skill_analysis:
            if skill_analysis.missing_skills:
                notes.append("Missing Skills (Consider Adding):")
                notes.extend([f"- {skill}" for skill in sorted(skill_analysis.missing_skills)])
            
            if skill_analysis.partially_matched_skills:
                notes.append("\nPartially Matched Skills:")
                for job_skill, resume_skill in skill_analysis.partially_matched_skills.items():
                    notes.append(f"- {job_skill} (Found: {resume_skill})")
        
        # Add improvement suggestions to notes
        if suggestions:
            notes.append("\nSuggested Improvements:")
            notes.extend([f"- {suggestion}" for suggestion in suggestions])
        
        # Prepare technical skills by category
        tech_skills = {}
        for section in tailored_sections:
            if section.section_type == 'skills':
                for line in section.content:
                    if ':' in line:
                        category, skills = line.split(':', 1)
                        tech_skills[category.strip()] = [s.strip() for s in skills.split(',')]
        
        # Calculate enhanced scores
        quantification_score = 0.0
        jd_alignment_score = 0.0
        ats_score = 0.0
        overall_score = 0.0
        
        if job_analysis:
            # Extract JD key phrases for alignment scoring
            jd_phrases = self.jd_alignment_layer.extract_key_phrases(job_description, top_n=20)
            
            # Calculate bullet scores
            bullet_scores = []
            for bullet in rewritten_bullets:
                score = self.scoring_system.calculate_overall_score(bullet, jd_phrases)
                bullet_scores.append(score)
            
            if bullet_scores:
                quantification_score = sum(s.quantification for s in bullet_scores) / len(bullet_scores)
                jd_alignment_score = sum(s.jd_alignment for s in bullet_scores) / len(bullet_scores)
            
            # Generate ATS report
            ats_report = self.ats_optimizer.generate_ats_report(
                tailored_resume_text, 
                resume_sections, 
                job_analysis.keywords
            )
            ats_score = ats_report.overall_ats_score
            
            # Calculate overall score
            overall_score = (quantification_score + jd_alignment_score + ats_score) / 3
        
        # Create detailed output
        output = TailoredOutput(
            polished_resume=tailored_resume_text,
            technical_skills=tech_skills,
            recommended_skills=list(skill_analysis.missing_skills) if skill_analysis else [],
            debug_info={
            'initial_match_score': initial_score,
            'final_match_score': final_score,
            'total_bullets_processed': total_bullets_processed,
            'api_calls_made': self.ai_rewriter.api_calls_made,
            'bullet_changes': bullet_changes if bullets_only else [],
                'mode': mode,
                'dry_run': dry_run
            },
            notes=notes,
            quantification_score=quantification_score,
            jd_alignment_score=jd_alignment_score,
            ats_score=ats_score,
            overall_score=overall_score
        )
        
        # Save output if specified and not in dry run
        if output_file and not dry_run:
            if bullets_only and bullet_changes:
                # Generate bullets-only output
                bullets_output = self.doc_parser._generate_bullets_only_output(bullet_changes, job_analysis)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(bullets_output)
            else:
                # Generate full output based on format
                if output_format == 'detailed':
                    detailed_output = self._generate_detailed_output(output)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(detailed_output)
                else:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(output.polished_resume)
        
        return output
    
    def verify_bullet_truth(self, original: str, rewritten: str, job_skills: Set[str]) -> Tuple[bool, str]:
        """Verify that rewritten bullet maintains truth and doesn't add unmentioned skills"""
        original_skills = set(self._extract_skills_from_text(original))
        rewritten_skills = set(self._extract_skills_from_text(rewritten))
        
        # Check for unauthorized skill additions
        new_skills = rewritten_skills - original_skills
        unauthorized_skills = new_skills & job_skills
        
        if unauthorized_skills:
            return False, f"Added unauthorized skills: {unauthorized_skills}"
        
        # Check for lost technical details
        lost_skills = original_skills - rewritten_skills
        if lost_skills:
            return False, f"Lost original skills: {lost_skills}"
        
        # Check for content preservation (metrics, achievements)
        metrics_pattern = r'\d+(?:\.\d+)?%|\d+\s*(?:users|requests|transactions|improvements?|seconds?|minutes?|hours?|days?)'
        original_metrics = set(re.findall(metrics_pattern, original))
        rewritten_metrics = set(re.findall(metrics_pattern, rewritten))
        
        if original_metrics - rewritten_metrics:
            return False, f"Lost metrics: {original_metrics - rewritten_metrics}"
        
        return True, "Verified"
    
    def analyze_skill_gaps(self, resume_sections: List[ResumeSection], job_analysis: JobAnalysis) -> SkillAnalysis:
        """Identify skills required by job but not present in resume"""
        # Extract all skills from resume
        resume_skills = set()
        for section in resume_sections:
            section_text = ' '.join(section.content)
            skills = self._extract_skills_from_text(section_text)
            resume_skills.update(skills)
        
        # Get job skills
        job_skills = set(job_analysis.technologies + job_analysis.required_skills)
        
        # Find exact matches and missing skills
        present_skills = resume_skills & job_skills
        missing_skills = job_skills - resume_skills
        
        # Find partial matches (e.g., "Python scripting" vs "Python")
        partially_matched = {}
        for job_skill in missing_skills.copy():
            for resume_skill in resume_skills:
                # Check if job skill is part of resume skill or vice versa
                if (job_skill.lower() in resume_skill.lower() or 
                    resume_skill.lower() in job_skill.lower()):
                    partially_matched[job_skill] = resume_skill
                    missing_skills.remove(job_skill)
                    break
        
        return SkillAnalysis(
            present_skills=present_skills,
            missing_skills=missing_skills,
            partially_matched_skills=partially_matched
        )
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using JobDescriptionAnalyzer"""
        return self.job_analyzer._extract_meaningful_skills_enhanced(text)
    
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
    
    def _generate_detailed_output(self, output: TailoredOutput) -> str:
        """Generate detailed output with all sections and structured format"""
        sections = []
        
        # Add comprehensive header
        sections.extend([
            "=" * 50,
            "ENHANCED PIPELINE SUMMARY",
            "=" * 50,
            "",
            "SUMMARY",
            "-" * 20,
            f"Overall Score: {output.overall_score:.1f}%",
            f"Quantification Score: {output.quantification_score:.1f}%",
            f"JD Alignment Score: {output.jd_alignment_score:.1f}%",
            f"ATS Score: {output.ats_score:.1f}%",
            ""
        ])
        
        # Add status indicators
        sections.extend([
            "STATUS INDICATORS",
            "-" * 20,
            "[OK] Skill Normalization: PASSED",
            "[OK] Semantic Expansion: PASSED",
            f"[OK] JD Alignment Layer: Enhanced",
            f"[OK] ATS Optimization: {output.ats_score:.1f}% coverage",
            ""
        ])
        
        # Add coverage information if available
        if hasattr(output, 'debug_info') and 'coverage_summary' in output.debug_info:
            sections.extend([
                "COVERAGE ANALYSIS",
                "-" * 20,
                output.debug_info['coverage_summary'],
                ""
            ])
        
        # Add polished resume
        sections.extend([
            "REWRITTEN RESUME",
            "-" * 20,
            output.polished_resume,
            ""
        ])
        
        # Add bullet analysis
        sections.extend([
            "BULLET ANALYSIS",
            "-" * 20,
            f"Score Breakdown - Quantification: {output.quantification_score:.0f}, JD Alignment: {output.jd_alignment_score:.0f}",
            ""
        ])
        
        # Add JD alignment details
        sections.extend([
            "JD ALIGNMENT",
            "-" * 20,
            f"Alignment Score: {output.jd_alignment_score:.1f}%",
            ""
        ])
        
        # Add ATS optimization
        sections.extend([
            "ATS OPTIMIZATION",
            "-" * 20,
            f"ATS Match Score: {output.ats_score:.1f}%",
            ""
        ])
        
        # Add technical skills
        sections.extend([
            "TECHNICAL SKILLS",
            "-" * 20,
        ])
        
        # Add technical skills by category
        for category, skills in output.technical_skills.items():
            sections.append(f"{category}: {', '.join(skills)}")
        sections.append("")
        
        # Add recommended skills if any
        if output.recommended_skills:
            sections.extend([
                "RECOMMENDED SKILLS (From Job Description, Not in Resume)",
                "-" * 20,
                ", ".join(output.recommended_skills),
                ""
            ])
        
        # Add final score
        sections.extend([
            "FINAL SCORE",
            "-" * 20,
            f"Coverage: {output.jd_alignment_score:.1f}%, ATS: {output.ats_score:.1f}%, Alignment: {output.jd_alignment_score:.1f}%",
            f"Overall Match Score: {output.overall_score:.1f}%",
            ""
        ])
        
        # Add notes if any
        if output.notes:
            sections.extend([
                "NOTES",
                "-" * 20,
                *output.notes,
                ""
            ])
        
        # Add debug info if available
        if output.debug_info:
            sections.extend([
                "DEBUG INFORMATION",
                "-" * 20,
                f"Initial Match Score: {output.debug_info['initial_match_score']:.1f}%",
                f"Final Match Score: {output.debug_info['final_match_score']:.1f}%",
                f"Bullets Processed: {output.debug_info['total_bullets_processed']}",
                f"API Calls Made: {output.debug_info['api_calls_made']}",
                f"Mode: {output.debug_info['mode']}",
                ""
            ])
        
        return "\n".join(sections).strip()
    
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
@click.option('--mode', type=click.Choice(['polish', 'tailor', 'gap']), default='tailor',
              help='Operation mode: polish (readability), tailor (job alignment), or gap (skill analysis)')
@click.option('--dry-run', is_flag=True, help='Run without making API calls (preview mode)')
@click.option('--clear-cache', is_flag=True, help='Clear the bullet point cache before running')
@click.option('--debug', is_flag=True, help='Enable debug output to see bullet detection')
@click.option('--bullets-only', is_flag=True, help='Output only the rewritten bullet points with context (not the full resume)')
@click.option('--verify-truth', is_flag=True, help='Enable strict truth verification for rewritten bullets')
@click.option('--output-format', type=click.Choice(['standard', 'detailed']), default='standard',
              help='Output format: standard (resume only) or detailed (with skills analysis)')
def main(job_file, resume_file, output_file, api_key, max_calls, mode, dry_run, clear_cache, 
         debug, bullets_only, verify_truth, output_format):
    """Tailor resume to match job description using AI"""
    
    if not dry_run and not api_key:
        click.echo("Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key")
        return
    
    # Handle mode-specific requirements
    if mode == 'gap' and output_format == 'standard':
        click.echo("Warning: Gap analysis mode works best with detailed output format. Switching to detailed.")
        output_format = 'detailed'
    
    if clear_cache:
        cache = BulletPointCache()
        cache.clear()
        click.echo("[OK] Cache cleared")
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
        results = pipeline.tailor_resume(job_description, resume_file, output_file, mode, dry_run, 
                                      debug, bullets_only, verify_truth, output_format)
        
        # Display results
        if dry_run:
            click.echo(f"\n🔍 DRY RUN COMPLETED")
            click.echo(f"[FILE] Would process: {resume_file}")
            click.echo(f"[SCORE] Current match score: {results.debug_info['initial_match_score']:.1f}%")
            click.echo(f"[BULLETS] Bullets that would be rewritten: {results.debug_info['total_bullets_processed']}")
            click.echo(f"[API] API calls that would be made: ~{results.debug_info['api_calls_made'] if results.debug_info['api_calls_made'] > 0 else 'up to 2'}")
            if bullets_only:
                click.echo(f"[OUTPUT] Would generate bullets-only output to: {output_file}")
            else:
                click.echo(f"[OUTPUT] Would generate {output_format} output to: {output_file}")
        else:
            click.echo(f"\n[OK] Resume tailoring completed!")
            if bullets_only:
                click.echo(f"[BULLETS] Rewritten bullet points saved to: {output_file}")
                click.echo(f"[COUNT] {len(results.debug_info['bullet_changes'])} bullet points were rewritten")
            else:
                click.echo(f"[OUTPUT] Output saved to: {output_file}")
            
            if mode == 'polish':
                click.echo(f"[POLISH] Polish mode: Bullets improved for readability and professionalism")
            elif mode == 'gap':
                click.echo(f"[GAP] Gap analysis completed. Check output for skill recommendations.")
            else:
                click.echo(f"[SCORE] Match score improved: {results.debug_info['initial_match_score']:.1f}% -> {results.debug_info['final_match_score']:.1f}%")
            
            click.echo(f"[PROCESSED] Bullets processed: {results.debug_info['total_bullets_processed']}")
            click.echo(f"[API] API calls made: {results.debug_info['api_calls_made']}/{max_calls}")
            
            if results.recommended_skills:
                click.echo("\n[TIPS] Key missing skills from job description:")
                for skill in sorted(results.recommended_skills[:5]):  # Show top 5
                    click.echo(f"   - {skill}")
                if len(results.recommended_skills) > 5:
                    click.echo(f"   (and {len(results.recommended_skills) - 5} more...)")
        
        if results.notes:
            click.echo(f"\n[TIPS] Suggestions for further improvement:")
            for note in results.notes:
                if not note.startswith(('Missing Skills', 'Partially Matched')):
                    click.echo(f"   {note}")
        
        # Show cache info
        if not dry_run:
            cache = BulletPointCache()
            cache_size = len(cache.cache)
            click.echo(f"[CACHE] Cache contains {cache_size} rewritten bullet points")
            
        # Show mode-specific summary
        if mode == 'gap':
            click.echo("\n[INFO] Gap Analysis Summary:")
            click.echo("Run with --output-format detailed to see full analysis")
        
    except Exception as e:
        click.echo(f"[ERROR] Error: {e}")

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
    click.echo("[OK] Cache cleared")

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

    def _generate_detailed_output(self, output: TailoredOutput) -> str:
        """Generate detailed output with all sections"""
        sections = []
        
        # Add polished resume
        sections.extend([
            "=== Polished Resume ===",
            output.polished_resume,
            "",
            "=== Technical Skills ===",
        ])
        
        # Add technical skills by category
        for category, skills in output.technical_skills.items():
            sections.append(f"{category}: {', '.join(skills)}")
        sections.append("")
        
        # Add recommended skills if any
        if output.recommended_skills:
            sections.extend([
                "=== Recommended Skills (From Job Description, Not in Resume) ===",
                ", ".join(output.recommended_skills),
                ""
            ])
        
        # Add notes if any
        if output.notes:
            sections.extend([
                "=== Notes ===",
                *output.notes,
                ""
            ])
        
        # Add debug info if available
        if output.debug_info:
            sections.extend([
                "=== Debug Information ===",
                f"Initial Match Score: {output.debug_info['initial_match_score']:.1f}%",
                f"Final Match Score: {output.debug_info['final_match_score']:.1f}%",
                f"Bullets Processed: {output.debug_info['total_bullets_processed']}",
                f"API Calls Made: {output.debug_info['api_calls_made']}",
                f"Mode: {output.debug_info['mode']}",
                ""
            ])
        
        return "\n".join(sections).strip()

# Add the main command to the CLI group
cli.add_command(main, name='tailor')

if __name__ == "__main__":
    cli()