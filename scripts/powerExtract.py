import json
import nltk
import pandas as pd
import numpy as np
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure necessary resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

class ResumeJobMatchingSystem:
    def __init__(self, json_filepath, csv_filepath):
        self.json_filepath = json_filepath
        self.csv_filepath = csv_filepath
        self.df = self.convert_json_to_csv()

    def convert_json_to_csv(self):
        try:
            with open(self.json_filepath, 'r') as file:
                data = json.load(file)
            if isinstance(data, dict):
                df = pd.json_normalize(data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise ValueError("Unsupported JSON format")
            df.to_csv(self.csv_filepath, index=False)
            print(f"CSV file has been created at {self.csv_filepath}")
            return df
        except FileNotFoundError:
            print(f"Error: The file {self.json_filepath} does not exist.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def find_job_title_matches(self, job_title, limit=3):
        self.df['id'] = self.df['id'].astype(str)
        results = process.extract(job_title, self.df['id'], limit=limit)
        return results

    def optimize_keywords(self, job_titles, user_keywords):
        relevant_skills = self.get_relevant_skills(job_titles)
        optimized_keywords = self.replace_keywords(user_keywords, relevant_skills)
        return optimized_keywords

    def get_relevant_skills(self, job_titles):
        relevant_skills = []
        for title, _, _ in job_titles:
            skills = self.df.loc[self.df['id'] == title, 'skills'].values[0]
            relevant_skills.extend(skills)
        return relevant_skills

    def replace_keywords(self, user_keywords, relevant_skills):
        optimized_keywords = user_keywords.copy()
        for keyword in user_keywords:
            matches = process.extract(keyword, relevant_skills, scorer=fuzz.token_sort_ratio)
            best_match, best_score = max(matches, key=lambda x: x[1])
            if best_score > 95:  # Threshold for replacement
                if keyword in optimized_keywords:
                    optimized_keywords.remove(keyword)
                optimized_keywords.append(best_match)
        return optimized_keywords

    def process_keywords(self, optimized_key_words, job_titles):
        actual_key_words = self.get_actual_key_words(job_titles)
        vectors = np.concatenate([np.array([text for text in optimized_key_words]),
                                np.array([text for text in actual_key_words.keys()])])

        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(vectors).toarray()

        opt_vectors = tfidf_matrix[:len(optimized_key_words)]
        act_vectors = tfidf_matrix[len(optimized_key_words):]

        opt_norms = np.linalg.norm(opt_vectors, axis=1, keepdims=True)
        act_norms = np.linalg.norm(act_vectors, axis=1, keepdims=True)
        similarity_scores = np.dot(opt_vectors, act_vectors.T) / (opt_norms * act_norms.T)
        advanced_key_words = {}
        for i, score_row in enumerate(similarity_scores):
            for j, similarity_score in enumerate(score_row):
                if similarity_score >= 0.90:
                    act_keyword = list(actual_key_words.keys())[j]
                    advanced_key_words[act_keyword] = actual_key_words[act_keyword]
        return advanced_key_words

    def get_actual_key_words(self, job_titles):
        actual_key_words = {}
        for title, _, _ in job_titles:
            skills = self.df.loc[self.df['id'] == title, 'skills'].values[0]
            for skill in skills:
                actual_key_words[skill] = actual_key_words.get(skill, 0) + 1
        return actual_key_words

    @staticmethod
    def extract_keywords(text):
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word not in stop_words and word.isalnum()]

        # Generate n-grams
        bigrams = [' '.join(gram) for gram in ngrams(filtered_words, 2) if not any(stop in gram for stop in stop_words)]
        trigrams = [' '.join(gram) for gram in ngrams(filtered_words, 3) if not any(stop in gram for stop in stop_words)]

        # Combine single words and n-grams
        full_keywords = filtered_words + bigrams + trigrams
        return full_keywords

def main():
    json_file_path = '../data/job_skills.json'
    csv_file_path = '../data/job_skills.csv'
    matching_system = ResumeJobMatchingSystem(json_file_path, csv_file_path)

    if matching_system.df is not None:
        job_title = "Data Scientist"
        resume_text = "Extensive experience in data analysis, machine learning and deep learning."
        job_text = "We are looking for a Data Scientist with experience in machine learning and data analysis."
        resume_keywords = matching_system.extract_keywords(resume_text)
        job_keywords = matching_system.extract_keywords(job_text)
        matches = matching_system.find_job_title_matches(job_title)
        print("Job title matches found:", matches)
        resume_optimized_keywords = matching_system.optimize_keywords(matches, resume_keywords)
        job_optimized_keywords = matching_system.optimize_keywords(matches, job_keywords)
        resume_advanced_key_words = matching_system.process_keywords(resume_optimized_keywords, matches)
        job_advanced_key_words = matching_system.process_keywords(job_optimized_keywords, matches)
        print("Advanced Key Words (Resume):", resume_advanced_key_words)
        print("Advanced Key Words (Job Description):", job_advanced_key_words)

if __name__ == "__main__":
    main()