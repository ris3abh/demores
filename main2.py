import os
import json
import docx
import nltk
import pandas as pd
from typing import List
import streamlit as st
from pypdf import PdfReader
import plotly.express as px
import plotly.graph_objects as go
from annotated_text import parameters
from streamlit_extras import add_vertical_space
from scripts.similarity.get_score import *
from scripts.utils import get_filenames_from_dir
from scripts.utils.logger import init_logging_config
from scripts.similarity.get_score import get_score
from scripts.ReadPdf import read_single_pdf
from scripts.parsers import ParseJobDesc, ParseResume

init_logging_config()
cwd = find_path("Resume-Matcher")
config_path = os.path.join(cwd, "scripts", "similarity")

# Make sure to include this only once at the top of your code
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

parameters.SHOW_LABEL_SEPARATOR = False
parameters.BORDER_RADIUS = 3
parameters.PADDING = "0.5 0.25rem"

def create_annotated_text(
    input_string: str, word_list: List[str], annotation: str, color_code: str):
    # Tokenize the input string
    tokens = nltk.word_tokenize(input_string)

    # Convert the list to a set for quick lookups
    word_set = set(word_list)

    # Initialize an empty list to hold the annotated text
    annotated_text = []

    for token in tokens:
        # Check if the token is in the set
        if token in word_set:
            # If it is, append a tuple with the token, annotation, and color code
            annotated_text.append((token, annotation, color_code))
        else:
            # If it's not, just append the token as a string
            annotated_text.append(token)

    return annotated_text

def read_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def tokenize_string(input_string):
    tokens = nltk.word_tokenize(input_string)
    return tokens

# Function to read text from various document formats
def read_document(file_path):
    # Extract the file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            return read_pdf(file_path)
        elif file_extension == '.docx':
            return read_docx(file_path)
        elif file_extension == '.txt':
            return read_txt(file_path)
        else:
            raise ValueError("Unsupported file type. Please upload a .pdf, .docx, or .txt file.")
    except Exception as e:
        st.error(f"Failed to read file: {str(e)}")
        return None

# Read PDF using PdfReader
def read_pdf(file_path):
    text_output = []
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_output.append(text)
    except Exception as e:
        raise ValueError(f"PDF read error: {str(e)}")
    return " ".join(text_output)

# Read DOCX using python-docx
def read_docx(file_path):
    text_output = []
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            if para.text:
                text_output.append(para.text)
    except Exception as e:
        raise ValueError(f"DOCX read error: {str(e)}")
    return " ".join(text_output)

# Read TXT files
def read_txt(file_path):
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise ValueError(f"TXT read error: {str(e)}")

# Initialize Streamlit layout
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("Resume Matcher")
st.sidebar.header("Upload Your Files")

# File uploaders
uploaded_resume = st.sidebar.file_uploader("Upload a Resume", type=['pdf', 'docx', 'txt'], key="resume_uploader")
uploaded_job_desc = st.sidebar.file_uploader("Upload a Job Description", type=['pdf', 'docx', 'txt'], key="jobdesc_uploader")

# Process Files Button
if st.sidebar.button("Save and Process Files"):
    if uploaded_resume is not None and uploaded_job_desc is not None:
        # Save files
        resume_path = os.path.join("data/Resumes", uploaded_resume.name)
        job_desc_path = os.path.join("data/JobDescription", uploaded_job_desc.name)
        
        with open(resume_path, "wb") as f:
            f.write(uploaded_resume.getbuffer())
        with open(job_desc_path, "wb") as f:
            f.write(uploaded_job_desc.getbuffer())

        # Process Files
        resume_text = read_document(resume_path)
        job_desc_text = read_document(job_desc_path)
        
        # Parsing the text using custom parser scripts
        resume_data = ParseResume(resume_text).get_JSON()
        job_desc_data = ParseJobDesc(job_desc_text).get_JSON()

        resume_string = " ".join(resume_data["extracted_keywords"])
        jd_string = " ".join(job_desc_data["extracted_keywords"])
        result = get_score(resume_string, jd_string)
        similarity_score = round(result[0].score * 100, 2)
        score_color = "green"
        if similarity_score < 60:
            score_color = "red"
        elif 60 <= similarity_score < 75:
            score_color = "orange"
        st.markdown(
            f"Similarity Score obtained for the resume and job description is "
            f'<span style="color:{score_color};font-size:24px; font-weight:Bold">{similarity_score}</span>',
            unsafe_allow_html=True,
)        
    else:
        st.warning("Please upload both a resume and a job description.")

# Additional Streamlit elements
st.markdown("## Visualizations and Further Analysis")
st.markdown("Here you could add further analysis and visualizations related to the uploaded and processed data.")
