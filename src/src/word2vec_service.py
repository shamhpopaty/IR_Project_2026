import itertools
import logging
import os

import joblib
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from database_service import DataBaseService
from preprocessor import Preprocessor

class Word2VecService:
    def __init__(self):
        try:
            self.load_models()
        except FileNotFoundError:
            self.model = None
            self.doc_vectors = None
            self.doc_ids = None
            print('models not found')
    
    def load_models(self):
        self.model = Word2Vec.load("results/word2vec_model.model")
        self.doc_vectors = joblib.load(
            "results/word2vec_doc_vectors.pkl"
        )
        self.doc_ids = joblib.load(
            "results/word2vec_doc_ids.pkl"
        )

    def document_vector(self, model, tokens):
        vectors = [
            model.wv[word]
            for word in tokens
            if word in model.wv
        ]
        if len(vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

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
        tokenized_docs = [
            text.split()
            for text in docs_df["text_clean"]
        ]
        model = Word2Vec(
            sentences=tokenized_docs, 
            vector_size=150,   
            window=4,           
            min_count=5,        
            workers=4,          
            sg=0,               
            negative=5,         
            epochs=10
        )
        doc_vectors = np.array([
            self.document_vector(model, tokens)
            for tokens in tokenized_docs
        ])
        model.save("word2vec_model.model")
        joblib.dump(doc_vectors, "word2vec_doc_vectors.pkl")
        joblib.dump(doc_ids, "word2vec_doc_ids.pkl")
        print("mission completed.")

    def query_vector(self, text):
        tokens = text.split()
        vectors = [
            self.model.wv[word]
            for word in tokens
            if word in self.model.wv
        ]
        if len(vectors) == 0:
            return np.zeros(self.model.vector_size)
        return np.mean(vectors, axis=0)

    def search(self, query, top_k=10):
        qvec = self.query_vector(query)
        scores = cosine_similarity(
            [qvec],
            self.doc_vectors
        )[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
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

    def grid_search(self):
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
        tokenized_docs = [
            text.split()
            for text in docs_df["text_clean"]
        ]

        logging.basicConfig(level=logging.WARNING)
        param_grid = {
            'vector_size': [100, 150],
            'window': [4, 5, 6],
            'sg': [0, 1], 
            'epochs': [9]
        }

        static_params = {
            'sentences': tokenized_docs,
            'min_count': 5,
            'workers': 4,
            'negative': 5
        }

        keys, values = zip(*param_grid.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print(f"Total combinations to test: {len(experiments)}\n")

        best_loss = float('inf')
        best_params = None

        for i, params in enumerate(experiments, 1):
            current_config = {**static_params, **params}
            
            print(f"Running experiment {i}/{len(experiments)} with params: {params}")
            
            model = Word2Vec(
                **current_config,
                compute_loss=True
            )
            
            training_loss = model.get_latest_training_loss()
            print(f"-> Training Loss: {training_loss:.2f}")
            
            if training_loss < best_loss:
                best_loss = training_loss
                best_params = params


        print("\n" + "="*40)
        print("GRID SEARCH COMPLETE")
        print("="*40)
        print(f"Best Parameters Found: {best_params}")
        print(f"Lowest Training Loss: {best_loss:.2f}")

        ## after running the results were:
        # Best Parameters Found: {'vector_size': 150, 'window': 4, 'sg': 0, 'epochs': 9}
        # Lowest Training Loss: 49414876.00
