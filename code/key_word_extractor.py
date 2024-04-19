import spacy
import textacy.extract
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class ResumeJobMatcher:
    def __init__(self):
        # Load a pre-trained Sentence Transformer model for semantic comparison
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load("en_core_web_trf")

    def preprocess_text(self, text):
        # Utilize Spacy for tokenization and removal of stopwords
        doc = self.nlp(text.lower())
        filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return " ".join(filtered_tokens)

    def extract_keywords(self, text):
        doc = self.nlp(text)
        keywords = set()

        # Extract entities and noun chunks as keywords
        keywords.update([entity.text.lower() for entity in textacy.extract.entities(doc)])
        keywords.update([chunk.text.lower() for chunk in textacy.extract.noun_chunks(doc) if len(chunk.text) > 2])

        # Use TF-IDF to extract terms as backup/reinforcement
        tfidf = TfidfVectorizer(max_features=30, stop_words='english')
        try:
            features = tfidf.fit_transform([text])
            scores = np.array(features.sum(axis=0)).flatten()
            tfidf_keywords = [word for word, score in zip(tfidf.get_feature_names_out(), scores) if score > 0.2]
            keywords.update(tfidf_keywords)
        except:
            pass
        return list(keywords)

    def calculate_semantic_similarity(self, text1, text2):
        embedding1 = self.model.encode(text1, convert_to_tensor=True)
        embedding2 = self.model.encode(text2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        return similarity

    def match_resume_to_job_description(self, resume_text, job_description):
        processed_resume = self.preprocess_text(resume_text)
        processed_job_desc = self.preprocess_text(job_description)

        resume_keywords = self.extract_keywords(processed_resume)
        job_desc_keywords = self.extract_keywords(processed_job_desc)

        common_keywords = set(resume_keywords).intersection(set(job_desc_keywords))
        keyword_match_score = len(common_keywords) / max(len(set(job_desc_keywords)), 1)  # Avoid division by zero

        semantic_similarity = self.calculate_semantic_similarity(processed_resume, processed_job_desc)

        final_score = 0.5 * keyword_match_score + 0.5 * semantic_similarity
        return final_score, keyword_match_score, semantic_similarity, list(common_keywords), job_desc_keywords, resume_keywords
