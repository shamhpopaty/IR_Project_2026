import joblib
import numpy as np
import pandas as pd

from preprocessor import Preprocessor
from sklearn.metrics.pairwise import cosine_similarity
from database_service import DataBaseService
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFService:
    def __init__(self):
        try:
            self.load_models()
        except FileNotFoundError:
            print("Model files not found. Please run train() first.")
            self.vectorizer = None
            self.tfidf_matrix = None
            self.doc_ids = None

    def load_models(self):
        self.vectorizer = joblib.load("results/tfidf_vectorizer.joblib")
        self.tfidf_matrix = joblib.load("results/tfidf_matrix.joblib")
        self.doc_ids = joblib.load("results/tfidf_doc_ids.joblib")

    def train(self):
        database = DataBaseService()
        docs_collection = database.documents_processed
        docs = list(docs_collection.find({}))
        docs_df = pd.DataFrame([{
                "doc_id": doc["_id"],
                "tokens_text": " ".join(doc["tokens"])}
            for doc in docs
        ])
        doc_ids = docs_df["doc_id"].tolist()
        corpus = docs_df["tokens_text"].tolist()
        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b"
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
        joblib.dump(tfidf_matrix, "tfidf_matrix.joblib")
        joblib.dump(doc_ids, "tfidf_doc_ids.joblib")
        print("mission complete!!")
        
    def search(self, query, top_k=10):
        if self.vectorizer is None:
            self.load_models()
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append(self.doc_ids[idx])
        return results

    def test(self):
        preprocessor = Preprocessor()
        q = preprocessor.process("are zebra loaches safe with shrimp?")
        query = " ".join(q)
        results = self.search(query, top_k=10)
        for r in results:
            print(r)

