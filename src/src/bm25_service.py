import os
import joblib
import pandas as pd

from database_service import DataBaseService
from preprocessor import Preprocessor
from rank_bm25 import BM25Okapi


class BM25Service:
    def __init__(self):
        try:
            self.load_models()
        except FileNotFoundError:
            print("Model files not found. Please run train() first.")
            self.bm25 = None
            self.doc_ids = None

    def load_models(self):
        self.bm25 = joblib.load("results/bm25_model.pkl")
        self.doc_ids = joblib.load("results/bm25_doc_ids.pkl")

    def train(self):
        database = DataBaseService()
        docs_collection = database.documents_processed
        docs = list(docs_collection.find({}))
        docs_df = pd.DataFrame([
            {
                "doc_id": doc["_id"],
                "text_clean": " ".join(doc["tokens"])
            }
            for doc in docs
        ])
        doc_ids = docs_df["doc_id"].tolist()
        tokenized_corpus = [text.split() for text in docs_df["text_clean"]]
        # tried:
        # bm25 = BM25Okapi(tokenized_corpus, k1=1.4, b=0.64)
        bm25 = BM25Okapi(tokenized_corpus, k1=1.2, b=0.5)
        joblib.dump(bm25, "results/bm25_model.pkl")
        joblib.dump(doc_ids, "results/bm25_doc_ids.pkl")
        print("BM25 model saved successfully!")
        print("Documents indexed:", len(doc_ids))

    def search(self, query, top_k=10):
        if self.bm25 is None:
            self.load_models()
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append(self.doc_ids[idx])
        return results

    def test(self):
        preprocessor = Preprocessor()
        q = preprocessor.process("are zebra loaches safe with shrimp?")
        query = " ".join(q)
        results = self.search(
            query,
            top_k=10
        )
        for r in results:
            print(r)
            