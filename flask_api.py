import os
import json
import docx
import uuid
import nltk
import pandas as pd
from typing import List
from pypdf import PdfReader
from annotated_text import parameters
from scripts.similarity.get_score import get_score
from scripts.parsers import ParseJobDesc, ParseResume
from scripts.powerExtract import ResumeJobMatchingSystem
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize logging and configuration
from scripts.utils.logger import init_logging_config
init_logging_config()

# Ensure required NLTK resources are downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Helper functions
# ... (include your helper functions here) ...
def read_document(file_path, file_type):
    if file_type == 'pdf':
        return read_pdf(file_path)
    elif file_type == 'docx':
        return read_docx(file_path)
    else:
        raise ValueError("Unsupported file type for document reading.")

def read_pdf(file_path):
    text_output = []
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PdfReader(file)
            text_output = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
    except Exception as e:
        print(f"Failed to read PDF file: {str(e)}")
    return " ".join(text_output)

def read_docx(file_path):
    text_output = []
    try:
        doc = docx.Document(file_path)
        text_output = [para.text for para in doc.paragraphs if para.text]
    except Exception as e:
        print(f"Failed to read DOCX file: {str(e)}")
    return " ".join(text_output)

@app.route('/score_resume', methods=['POST'])
def score_resume():
    try:
        # Get data from the request
        data = request.get_json()
        company_name = data.get('company_name')
        job_title = data.get('job_title')
        resume_file = data.get('resume_file')
        job_description_text = data.get('job_description_text')

        if not all([company_name, job_title, resume_file, job_description_text]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Save the resume file
        unique_id = uuid.uuid4().hex
        resume_filename = f"{unique_id}_resume.pdf"  # Assuming PDF format
        resume_path = os.path.join("data/Resumes", resume_filename)
        with open(resume_path, "wb") as f:
            f.write(resume_file.read())

        # Save the job description text
        jobdesc_filename = f"{unique_id}_jobdesc.txt"
        job_desc_path = os.path.join("data/JobDescription", jobdesc_filename)
        with open(job_desc_path, "w") as f:
            f.write(job_description_text)

        # Read and parse the documents
        resume_text = read_document(resume_path, 'pdf')
        resume_data = ParseResume(resume_text).get_JSON()
        job_desc_data = ParseJobDesc(job_description_text).get_JSON()

        DescRes = ResumeJobMatchingSystem("data/job_skills.json", "data/job_skills.csv")
        if DescRes.df is not None:
            job_title = job_title.lower()
            resume_text = resume_text.lower()
            job_description_text = job_description_text.lower()
            resume_keywords = DescRes.extract_keywords(resume_text)
            job_keywords = DescRes.extract_keywords(job_description_text)
            matches = DescRes.find_job_title_matches(job_title)
            resume_optimized_keywords = DescRes.optimize_keywords(matches, resume_keywords)
            job_optimized_keywords = DescRes.optimize_keywords(matches, job_keywords)
            resume_advanced_key_words = DescRes.process_keywords(resume_optimized_keywords, matches)
            job_advanced_key_words = DescRes.process_keywords(job_optimized_keywords, matches)

            # Similarity scoring
            resume_advanced_key_words = " ".join(resume_advanced_key_words)
            job_advanced_key_words = " ".join(job_advanced_key_words)
            result = get_score(resume_advanced_key_words, job_advanced_key_words)
            similarity_score = round(result[0].score * 100, 2)

            # Return the similarity score
            return jsonify({'similarity_score': similarity_score})
        else:
            return jsonify({'error': 'Failed to load job skills data'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)