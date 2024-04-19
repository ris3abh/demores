from resume_processor import ResumeProcessor
from resume_intiution import ResumeDataProcessor
from jd import JobDescriptionProcessing
from key_word_extractor import ResumeJobMatcher

def main():
    doc_path = '../data/3.docx'
    # ask user for job description path
    job_description_path = '../data/sample_job_description.txt'
    output_csv_path = '../data/output.csv'
    
    # Initialize the processor with the doc path and process the resume
    processor = ResumeProcessor(doc_path)
    cleaned_text = processor.process_text()
    structured_content = processor.process_resume()

    # Create an instance of ResumeDataProcessor and use it
    data_processor = ResumeDataProcessor(structured_content)
    processed_content = data_processor.process_content()
    data_processor.write_to_csv(processed_content, output_csv_path)
    jd = JobDescriptionProcessing(job_description_path)
    job_description = jd.process_text()
    matcher = ResumeJobMatcher()
    score, keyword_score, semantic_score, keywords, job_desc_keywords, resume_keywords = matcher.match_resume_to_job_description(cleaned_text, job_description)
    print(f"Overall Match Score: {score:.2f}")
    print(f"Keyword Match Score: {keyword_score:.2f}")
    print(f"Semantic Similarity Score: {semantic_score:.2f}")
    print("Common Keywords:", keywords)
    print("Job Description Keywords:", job_desc_keywords)
    print("Resume Keywords:", resume_keywords)
    print(f"The similarity score between the job description and the resume is: {score:.2f}")

if __name__ == '__main__':
    main()