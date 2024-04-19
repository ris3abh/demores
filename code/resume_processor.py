import docx
import fitz  # PyMuPDF
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from fuzzywuzzy import process

class ResumeProcessor:
    SIMILARITY_THRESHOLD = 80
    COMMON_HEADERS = [
        'education', 'experience', 'work history', 'professional background', 'skills', 'abilities',
        'competencies', 'projects', 'portfolio', 'certifications', 'credentials', 'licenses',
        'awards', 'honors', 'publications', 'papers', 'articles', 'interests', 'hobbies',
        'summary', 'objective', 'languages', 'activities', 'references',
        "work experience", "academic projects", "professional experience", "professional summary"
    ]

    def __init__(self, file_path):
        self.file_path = file_path
        self.full_text = self.extract_text()

    def extract_text(self):
        if self.file_path.endswith('.docx'):
            return self.extract_text_from_docx()
        elif self.file_path.endswith('.pdf'):
            return self.extract_text_from_pdf()
        else:
            raise ValueError("Unsupported file type. Please upload a .docx or .pdf file.")

    def extract_text_from_docx(self):
        doc = docx.Document(self.file_path)
        return '\n'.join(para.text.lower().replace('\t', ' ').replace('\xa0', ' ') for para in doc.paragraphs)

    def extract_text_from_pdf(self):
        doc = fitz.open(self.file_path)
        text = ''
        for page in doc:
            text += page.get_text()
        return text.lower()

    def process_text(self):
        """Consolidates the text of the resume after removing stopwords and non-alphabetic characters."""
        words = word_tokenize(self.full_text)
        filtered_words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]
        clean_text = ' '.join(filtered_words)
        return clean_text

    def is_section_header(self, line):
        line_clean = ' '.join(word for word in word_tokenize(line) if len(word) > 1).lower()
        match, similarity = process.extractOne(line_clean, self.COMMON_HEADERS)
        return similarity >= self.SIMILARITY_THRESHOLD or line.isupper() or line.endswith(':')

    def identify_section_headers(self, text):
        lines = [line for line in text.split('\n') if line.strip()]
        return [line for line in lines if self.is_section_header(line)]

    def extract_text_for_section_header(self, full_text, headers):
        sections = []
        current_section = None
        for line in full_text.split('\n'):
            line = line.strip()
            if line in headers:
                if current_section:
                    sections.append(current_section)
                current_section = {"header": line, "content": []}
            elif current_section:
                current_section["content"].append(line)
        if current_section:
            sections.append(current_section)
        return sections

    def process_resume(self):
        headers = self.identify_section_headers(self.full_text)
        sections = self.extract_text_for_section_header(self.full_text, headers)
        return {sec["header"]: self.segregate_description_and_title(sec) for sec in sections}

    def segregate_description_and_title(self, section):
        structured_content = {}
        current_title = None

        for line in section["content"]:
            line = line.strip()
            if not line:
                continue
            if len(line.split()) <= 10:
                current_title = line
                structured_content[current_title] = []
            elif current_title:
                structured_content[current_title].append(line)
        return structured_content
