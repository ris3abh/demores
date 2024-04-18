import docx
import json
from fuzzywuzzy import process
from nltk.tokenize import word_tokenize

class ResumeProcessor:
    SIMILARITY_THRESHOLD = 80
    COMMON_HEADERS = [
        'education', 'experience', 'work history', 'professional background', 'skills', 'abilities', 'competencies',
        'projects', 'portfolio', 'certifications', 'credentials', 'licenses', 'awards', 'honors', 'publications',
        'papers', 'articles', 'interests', 'hobbies', 'summary', 'objective', 'languages', 'activities',
        'references', "work experience", "academic projects", "professional experience", "professional summary"
    ]

    def __init__(self, doc_path):
        self.doc_path = doc_path

    def extract_text_from_docx(self):
        doc = docx.Document(self.doc_path)
        full_text = '\n'.join([para.text.lower().replace('\t', ' ').replace('\xa0', ' ') for para in doc.paragraphs])
        return full_text

    def is_section_header(self, line):
        line_clean = ' '.join([word for word in word_tokenize(line) if len(word) > 1]).lower()
        match, similarity = process.extractOne(line_clean, self.COMMON_HEADERS)
        if similarity >= self.SIMILARITY_THRESHOLD or line.isupper() or line.endswith(':'):
            return True
        return False

    def identify_section_headers(self, text):
        lines = [line for line in text.split('\n') if line.strip()]
        return [line for line in lines if self.is_section_header(line)]

    def extract_text_for_section_header(self, full_text, headers):
        sections = []
        current_section = None
        for line in full_text.split('\n'):
            line = line.strip()
            if line in headers:
                current_section = {"header": line, "content": []}
                sections.append(current_section)
            elif current_section:
                current_section["content"].append(line)
        return sections

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

    def process_resume(self):
        full_text = self.extract_text_from_docx()
        headers = self.identify_section_headers(full_text)
        sections = self.extract_text_for_section_header(full_text, headers)
        structured_content = {sec["header"]: self.segregate_description_and_title(sec) for sec in sections}

        with open('structured_section.json', 'w', encoding='utf-8') as file:
            json.dump(structured_content, file, ensure_ascii=False, indent=4)

        return sections, structured_content

# Example usage
processor = ResumeProcessor('RISHABH SHARMA_01.docx')
resume_sections, structured_content = processor.process_resume()

# Save sections as JSON
with open('resume_sections.json', 'w', encoding='utf-8') as file:
    json.dump(resume_sections, file, ensure_ascii=False, indent=4)

print("Resume sections saved to 'resume_sections.json'")
