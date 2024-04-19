import streamlit as st
import pandas as pd
from resume_processor import ResumeProcessor
from resume_intiution import ResumeDataProcessor
import os

def save_uploaded_file(uploadedfile):
    with open(os.path.join("uploads", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join("uploads", uploadedfile.name)

def process_and_convert_to_csv(file_path):
    processor = ResumeProcessor(file_path)
    structured_content = processor.process_resume()
    data_processor = ResumeDataProcessor(structured_content)
    processed_content = data_processor.process_content()
    df = pd.DataFrame.from_dict(processed_content, orient='index').transpose()
    return df

# Create directories if they don't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists('outputs'):
    os.makedirs('outputs')

st.title('Resume CSV Generator')

uploaded_file = st.file_uploader("Choose a resume file (.docx)", type='docx')
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if st.button('Process Resume'):
        result_df = process_and_convert_to_csv(file_path)
        st.write(result_df)
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download processed CSV",
            data=csv,
            file_name='processed_resume.csv',
            mime='text/csv',
        )
