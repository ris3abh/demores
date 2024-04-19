from resume_processor import ResumeProcessor
from resume_intiution import ResumeDataProcessor

def main():
    doc_path = '../data/2.docx'
    output_csv_path = '../data/output.csv'
    
    # Initialize the processor with the doc path and process the resume
    processor = ResumeProcessor(doc_path)
    structured_content = processor.process_resume()

    # Create an instance of ResumeDataProcessor and use it
    data_processor = ResumeDataProcessor(structured_content)
    processed_content = data_processor.process_content()
    data_processor.write_to_csv(processed_content, output_csv_path)

if __name__ == '__main__':
    main()

