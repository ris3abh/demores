import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

nltk.download('punkt')
nltk.download('stopwords')

class JobDescriptionProcessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.full_text = self.extract_text()
        self.stop_words = set(stopwords.words('english'))  # Cache stopwords to improve efficiency

    def extract_text(self):
        """
        Extracts text from a .txt file.
        """
        if self.file_path.endswith('.txt'):
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return file.read().lower()  # Convert text to lower case
        else:
            raise ValueError("Unsupported file type. Please upload a .txt file.")

    def process_text(self):
        """
        Processes the extracted text by tokenizing, removing stopwords, and non-alphabetical tokens.
        """
        words = word_tokenize(self.full_text)
        filtered_words = [word for word in words if word.isalpha() and word not in self.stop_words]
        clean_text = ' '.join(filtered_words)
        return clean_text

# Example usage:
# Assuming you have a valid path to a .txt file containing a job description
# job_processor = JobDescriptionProcessing('path_to_job_description.txt')
# processed_text = job_processor.process_text()
# print(processed_text)

    
