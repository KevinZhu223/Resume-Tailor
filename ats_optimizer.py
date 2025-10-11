#!/usr/bin/env python3
"""
Dedicated ATS Optimization Module
Enhances resumes for job postings with comprehensive ATS compatibility analysis.
"""

import re
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

@dataclass
class ATSOptimizationResult:
    """Result of ATS optimization analysis"""
    overall_score: float
    keyword_coverage: float
    formatting_score: float
    header_compliance: float
    keyword_frequency: Dict[str, int]
    missing_keywords: List[str]
    issues: List[str]
    recommendations: List[str]
    ats_optimized_resume: str

class ATSOptimizer:
    """Dedicated ATS Optimization Module for resume enhancement"""
    
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
    
    def optimize_resume_for_job(self, resume_text: str, job_description: str, 
                               resume_sections: List[Any]) -> ATSOptimizationResult:
        """Main method to optimize resume for a specific job posting"""
        
        # Extract keywords from job description
        jd_keywords = self._extract_job_keywords(job_description)
        
        # Calculate keyword coverage
        keyword_coverage = self._calculate_keyword_coverage(resume_text, jd_keywords)
        
        # Calculate keyword frequency
        keyword_frequency = self._calculate_keyword_frequency(resume_text, jd_keywords)
        
        # Identify missing keywords
        missing_keywords = self._identify_missing_keywords(resume_text, jd_keywords)
        
        # Check formatting compliance
        formatting_score, formatting_issues = self._check_formatting_compliance(resume_text)
        
        # Check header compliance
        header_score, header_issues = self._check_header_compliance(resume_sections)
        
        # Generate ATS-optimized resume
        ats_optimized_resume = self._generate_ats_optimized_resume(
            resume_text, jd_keywords, missing_keywords
        )
        
        # Calculate overall score
        overall_score = (keyword_coverage * 0.4 + formatting_score * 0.2 + 
                        header_score * 0.2 + 100 * 0.2)  # 100 for normalization
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            keyword_coverage, formatting_score, header_score, 
            missing_keywords, keyword_frequency
        )
        
        return ATSOptimizationResult(
            overall_score=overall_score,
            keyword_coverage=keyword_coverage,
            formatting_score=formatting_score,
            header_compliance=header_score,
            keyword_frequency=keyword_frequency,
            missing_keywords=missing_keywords,
            issues=formatting_issues + header_issues,
            recommendations=recommendations,
            ats_optimized_resume=ats_optimized_resume
        )
    
    def _extract_job_keywords(self, job_description: str) -> List[str]:
        """Extract relevant keywords from job description"""
        keywords = []
        
        # Technical skills patterns
        tech_patterns = [
            r'\b(?:python|java|javascript|react|angular|vue|node\.?js|typescript|sql|mongodb|postgresql|mysql|aws|azure|gcp|docker|kubernetes|git|jenkins|agile|scrum|devops|ci/cd|machine learning|artificial intelligence|data science|big data|cloud computing|microservices|rest api|graphql)\b',
            r'\b(?:software development|web development|mobile development|full stack|frontend|backend|database|testing|debugging|deployment|scaling|optimization|automation|integration|security|performance|monitoring|analytics|visualization)\b',
            r'\b(?:team collaboration|cross-functional|leadership|mentoring|project management|stakeholder|communication|problem solving|analytical|creative|innovative|detail-oriented|self-motivated)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, job_description, re.IGNORECASE)
            keywords.extend(matches)
        
        # Remove duplicates and normalize
        unique_keywords = list(dict.fromkeys(keywords))
        normalized_keywords = []
        
        for keyword in unique_keywords:
            normalized = self._normalize_keyword(keyword)
            if normalized not in normalized_keywords:
                normalized_keywords.append(normalized)
        
        return normalized_keywords[:20]  # Top 20 keywords
    
    def _normalize_keyword(self, keyword: str) -> str:
        """Normalize keyword using skill mapping"""
        keyword_lower = keyword.lower()
        
        for variant, standard in self.skill_normalization.items():
            if keyword_lower == variant.lower():
                return standard
        
        return keyword.title()
    
    def _calculate_keyword_coverage(self, resume_text: str, jd_keywords: List[str]) -> float:
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
    
    def _calculate_keyword_frequency(self, resume_text: str, jd_keywords: List[str]) -> Dict[str, int]:
        """Calculate frequency of each JD keyword in resume"""
        frequency_map = {}
        resume_lower = resume_text.lower()
        
        for keyword in jd_keywords:
            keyword_lower = keyword.lower()
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            count = len(re.findall(pattern, resume_lower))
            frequency_map[keyword] = count
        
        return frequency_map
    
    def _identify_missing_keywords(self, resume_text: str, jd_keywords: List[str]) -> List[str]:
        """Identify keywords from JD that are missing from resume"""
        missing = []
        resume_lower = resume_text.lower()
        
        for keyword in jd_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in resume_lower:
                missing.append(keyword)
        
        return missing
    
    def _check_formatting_compliance(self, resume_text: str) -> Tuple[float, List[str]]:
        """Check ATS formatting compliance"""
        issues = []
        score = 100.0
        
        # Check for special characters
        special_chars = ['●', '•', '→', '←', '↑', '↓', '★', '☆', '◆', '◇']
        for char in special_chars:
            if char in resume_text:
                issues.append(f"Contains special character: {char}")
                score -= 10
        
        # Check for tables
        if '|' in resume_text or '\t' in resume_text:
            issues.append("Contains tables or tabular formatting")
            score -= 20
        
        # Check for excessive line breaks
        if resume_text.count('\n') > len(resume_text.split()) * 0.1:
            issues.append("Excessive line breaks detected")
            score -= 10
        
        return max(score, 0), issues
    
    def _check_header_compliance(self, resume_sections: List[Any]) -> Tuple[float, List[str]]:
        """Check header compliance"""
        issues = []
        score = 100.0
        
        for section in resume_sections:
            header_lower = section.title.lower().strip()
            
            if header_lower in ['section', 'other', 'misc', 'additional']:
                issues.append(f"Generic header: {section.title}")
                score -= 15
            
            if not any(std_header in header_lower for std_header in self.standard_headers):
                if len(header_lower) > 3:
                    issues.append(f"Non-standard header: {section.title}")
                    score -= 5
        
        return max(score, 0), issues
    
    def _generate_ats_optimized_resume(self, resume_text: str, jd_keywords: List[str], 
                                     missing_keywords: List[str]) -> str:
        """Generate ATS-optimized version of resume"""
        optimized_resume = resume_text
        
        # Replace special characters with ATS-friendly alternatives
        optimized_resume = optimized_resume.replace('●', '-')
        optimized_resume = optimized_resume.replace('•', '-')
        
        # Add missing keywords naturally (this is a simplified version)
        # In a real implementation, this would be more sophisticated
        if missing_keywords:
            # Add a skills section if missing keywords are technical
            tech_keywords = [kw for kw in missing_keywords if kw.lower() in [
                'python', 'java', 'javascript', 'react', 'aws', 'azure', 'sql', 'docker'
            ]]
            
            if tech_keywords and 'technical skills' not in optimized_resume.lower():
                skills_section = f"\n\nTechnical Skills\n{', '.join(tech_keywords[:5])}\n"
                optimized_resume += skills_section
        
        return optimized_resume
    
    def _generate_recommendations(self, keyword_coverage: float, formatting_score: float,
                                header_score: float, missing_keywords: List[str],
                                keyword_frequency: Dict[str, int]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if keyword_coverage < 70:
            recommendations.append("Add more job-relevant keywords to improve ATS matching")
        
        if formatting_score < 80:
            recommendations.append("Remove special characters and tables for better ATS compatibility")
        
        if header_score < 90:
            recommendations.append("Use standard section headers (Experience, Education, Skills)")
        
        if missing_keywords:
            recommendations.append(f"Consider adding missing keywords: {', '.join(missing_keywords[:3])}")
        
        low_frequency_keywords = [kw for kw, freq in keyword_frequency.items() if freq < 2]
        if low_frequency_keywords:
            recommendations.append(f"Increase keyword density for: {', '.join(low_frequency_keywords[:3])}")
        
        return recommendations

def main():
    """Test the ATS Optimizer module"""
    optimizer = ATSOptimizer()
    
    # Sample resume and job description
    resume_text = """
    Professional Experience
    Software Engineer at Tech Corp
    • Developed React applications
    • Improved performance by 25%
    
    Education
    Bachelor of Science in Computer Science
    
    Technical Skills
    Python, JavaScript, React, AWS
    """
    
    job_description = """
    We are looking for a Software Engineer with experience in:
    - Python, JavaScript, React, Node.js
    - Cloud computing (AWS, Azure)
    - Machine learning and data science
    - Agile development methodologies
    - Team collaboration and leadership
    """
    
    # Mock resume sections
    class MockSection:
        def __init__(self, title):
            self.title = title
    
    resume_sections = [
        MockSection("Professional Experience"),
        MockSection("Education"),
        MockSection("Technical Skills")
    ]
    
    # Run optimization
    result = optimizer.optimize_resume_for_job(resume_text, job_description, resume_sections)
    
    print("ATS Optimization Results:")
    print(f"Overall Score: {result.overall_score:.1f}%")
    print(f"Keyword Coverage: {result.keyword_coverage:.1f}%")
    print(f"Formatting Score: {result.formatting_score:.1f}%")
    print(f"Header Compliance: {result.header_compliance:.1f}%")
    print(f"Missing Keywords: {result.missing_keywords}")
    print(f"Issues: {result.issues}")
    print(f"Recommendations: {result.recommendations}")

if __name__ == "__main__":
    main()
