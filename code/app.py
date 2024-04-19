import streamlit as st
import pandas as pd
import os
import uuid
from resume_processor import ResumeProcessor
from resume_intiution import ResumeDataProcessor

def save_uploaded_file(uploadedfile, subdir, prefix):
    save_path = os.path.join(subdir, f"{prefix}_{uploadedfile.name}")
    with open(save_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return save_path

def save_text_file(content, filename, subdir, prefix):
    file_path = os.path.join(subdir, f"{prefix}_{filename}")
    with open(file_path, "w") as f:
        f.write(content)
    return file_path

def process_and_convert_to_csv(file_path, output_dir, prefix):
    processor = ResumeProcessor(file_path)
    structured_content = processor.process_resume()
    data_processor = ResumeDataProcessor(structured_content)
    processed_content = data_processor.process_content()
    df = pd.DataFrame.from_dict(processed_content, orient='index').transpose()
    csv_file_path = os.path.join(output_dir, f"{prefix}_processed_resume.csv")
    df.to_csv(csv_file_path, index=False)
    return df, csv_file_path

# Directory paths
upload_dir = '../data/uploads'
output_dir = '../data/outputs'
os.makedirs(upload_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

st.title('Resume CSV Generator')

# Generate a unique ID for the current session
session_id = uuid.uuid4().hex

uploaded_file = st.file_uploader("Choose a resume file (.docx, .doc, .pdf) – PS: preferably a DOC file", type=['docx', 'doc', 'pdf'])
job_description = st.text_area("Paste the job description here:", height=150)

if uploaded_file and job_description:
    st.write("Job Description:", job_description)
    job_description_path = save_text_file(job_description, "job_description.txt", output_dir, session_id)

    file_path = save_uploaded_file(uploaded_file, upload_dir, session_id)
    if st.button('Process Resume'):
        result_df, csv_file_path = process_and_convert_to_csv(file_path, output_dir, session_id)
        st.write(result_df)
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download processed CSV",
            data=csv,
            file_name=f'{session_id}_processed_resume.csv',
            mime='text/csv',
        )
        st.success(f"Processed CSV saved to {csv_file_path}")
        st.success(f"Job description saved to {job_description_path}")
