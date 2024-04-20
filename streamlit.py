import os
import json
import docx
import uuid
import nltk
import pandas as pd
import streamlit as st
from typing import List
from pypdf import PdfReader
import plotly.graph_objects as go
from annotated_text import parameters
from streamlit_extras import add_vertical_space as avs
from scripts.similarity.get_score import get_score
from scripts.parsers import ParseJobDesc, ParseResume
from scripts.powerExtract import ResumeJobMatchingSystem, analyze_job_fit_hard_skills, analyze_job_fit_soft_skills

# Initialize logging and configuration
from scripts.utils.logger import init_logging_config
init_logging_config()

# Ensure required NLTK resources are downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Streamlit UI configuration
parameters.SHOW_LABEL_SEPARATOR = False
parameters.BORDER_RADIUS = 3
parameters.PADDING = "0.5 0.25rem"

# Helper functions
def create_annotated_text(input_string: str, word_list: List[str], annotation: str, color_code: str):
    tokens = nltk.word_tokenize(input_string)
    word_set = set(word_list)
    annotated_text = []
    for token in tokens:
        if token in word_set:
            annotated_text.append((token, annotation, color_code))
        else:
            annotated_text.append(token)
    return annotated_text

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
        st.error(f"Failed to read PDF file: {str(e)}")
    return " ".join(text_output)

def read_docx(file_path):
    text_output = []
    try:
        doc = docx.Document(file_path)
        text_output = [para.text for para in doc.paragraphs if para.text]
    except Exception as e:
        st.error(f"Failed to read DOCX file: {str(e)}")
    return " ".join(text_output)

# Streamlit page configuration
st.set_page_config(page_title="Resume Scorer", layout="wide")
st.title("Resume Scorer")
st.sidebar.header("Upload Your Resume")

# Input fields for company and job title
company_name = st.sidebar.text_input("Company Name", "")
job_title = st.sidebar.text_input("Job Title", "")

# File uploaders
uploaded_resume = st.sidebar.file_uploader("Upload a Resume", type=['pdf', 'docx'], key="resume_uploader")
job_description_text = st.sidebar.text_area("Enter your job description here:", key="job_desc_text", height=200)

# Check if all fields are provided
if st.sidebar.button("Process"):
    if uploaded_resume and job_description_text and company_name and job_title:
        unique_id = uuid.uuid4().hex
        resume_filename = f"{unique_id}_resume"
        jobdesc_filename = f"{unique_id}_jobdesc.txt"

        # Saving the resume file
        resume_path = os.path.join("data/Resumes", resume_filename + os.path.splitext(uploaded_resume.name)[1])
        with open(resume_path, "wb") as f:
            f.write(uploaded_resume.getbuffer())
        
        # Saving the job description text
        job_desc_path = os.path.join("data/JobDescription", jobdesc_filename)
        with open(job_desc_path, "w") as f:
            f.write(job_description_text)
        
        # Reading and parsing the documents
        resume_text = read_document(resume_path, uploaded_resume.type.split('/')[-1])
        resume_data = ParseResume(resume_text).get_JSON()
        job_desc_data = ParseJobDesc(job_description_text).get_JSON()
        if resume_text and job_description_text:
            st.success("resume and job successfully read.")
        json_file_path = 'data/job_skills.json'
        csv_file_path = 'data/job_skills.csv'
        ss_json_file_path = 'data/soft_skills.json'
        matching_system = ResumeJobMatchingSystem(json_file_path, csv_file_path, ss_json_file_path)
        if matching_system.df is not None and matching_system.soft_skills_df is not None:
            resume_keywords, job_keywords = analyze_job_fit_hard_skills(matching_system, job_title, resume_text, job_description_text) # matching_system, job_title, resume_text, job_text
            resume_soft_keywords, job_soft_keywords = analyze_job_fit_soft_skills(matching_system, resume_text, job_description_text) # matching_system, resume_text, job_text
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Resume Keywords")
            st.write(resume_keywords)
            st.markdown("### Job Description Keywords")
            st.write(job_keywords)
        with col2:
            st.markdown("### Resume Soft Skills")
            st.write(resume_soft_keywords)
            st.markdown("### Job Description Soft Skills")
            st.write(job_soft_keywords)

        # Similarity scoring
        resume_advanced_key_words = " ".join(resume_keywords)
        job_advanced_key_words = " ".join(job_keywords)
        resume_advanced_soft_key_words = " ".join(resume_soft_keywords)
        job_advanced_soft_key_words = " ".join(job_soft_keywords)
        st.warning("Calculating similarity score...")
        result1 = get_score(resume_advanced_key_words, job_advanced_key_words)
        similarity_score = round(result1[0].score * 100, 2)
        result2 = get_score(resume_advanced_soft_key_words, job_advanced_soft_key_words)
        similarity_score_soft = round(result2[0].score * 100, 2)
        st.success("Processing complete!")

        # Display results
        score_color = "green" if similarity_score >= 75 else "orange" if similarity_score >= 60 else "red"
        st.markdown(f"**Similarity Score**: <span style='color: {score_color};'>{similarity_score}%</span>", unsafe_allow_html=True)
        score_color = "green" if similarity_score_soft >= 75 else "orange" if similarity_score_soft >= 60 else "red"
        st.markdown(f"**Similarity Score (Soft Skills)**: <span style='color: {score_color};'>{similarity_score_soft}%</span>", unsafe_allow_html=True)
    else:
        st.warning("Please provide all required fields: a resume, a job description, the company name, and the job title.")

st.markdown("## Visualizations and Further Analysis")