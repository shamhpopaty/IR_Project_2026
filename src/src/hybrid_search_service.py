import numpy as np

from bm25_service import BM25Service
from word2vec_service import Word2VecService
from sklearn.metrics.pairwise import cosine_similarity


class HybridParallelBM25Word2Vec:
    def __init__(self, bm25_search, word2vec_search, a = 0.5, b = 0.5):
        self.a = a
        self.b = b
        self.bm25_search = bm25_search
        self.word2vec_search = word2vec_search
        self.doc_ids = self.bm25_search.doc_ids
    
    def search(self, query, top_k=10):
        bm25_tokens = query.lower().split()
        bm25_scores = self.bm25_search.bm25.get_scores(bm25_tokens)
        
        w2v_query_vec = self.word2vec_search.query_vector(query)
        w2v_scores = cosine_similarity(
            [w2v_query_vec],
            self.word2vec_search.doc_vectors
        )[0]
        
        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        w2v_scores_norm = (w2v_scores - w2v_scores.min()) / (w2v_scores.max() - w2v_scores.min() + 1e-10)
        
        combined_scores = self.a * bm25_scores_norm + self.b * w2v_scores_norm
        
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        results = [self.doc_ids[idx] for idx in top_indices]
        
        return results


class HybridSerialBM25Word2Vec:
    def __init__(self, bm25_search, word2vec_search, candidate_k=100):
        self.bm25_search = bm25_search
        self.word2vec_search = word2vec_search
        self.candidate_k = candidate_k
        self.doc_ids = self.bm25_search.doc_ids
    
    def search(self, query, top_k=10):
        bm25_candidates = self.bm25_search.search(query, top_k=self.candidate_k)
        
        query_embedding = self.word2vec_search.query_vector(query)
        
        candidate_indices = [self.doc_ids.index(doc_id) for doc_id in bm25_candidates]
        candidate_embeddings = self.word2vec_search.doc_vectors[candidate_indices]
        
        word2vec_scores = cosine_similarity([query_embedding], candidate_embeddings)[0]
        
        sorted_indices = np.argsort(word2vec_scores)[::-1][:top_k]
        results = [bm25_candidates[idx] for idx in sorted_indices]
        
        return results
