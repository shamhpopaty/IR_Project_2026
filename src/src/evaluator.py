from collections import defaultdict
import math
import os
import joblib

from database_service import DataBaseService
from hybrid_search_service import HybridParallelBM25Word2Vec, HybridSerialBM25Word2Vec


class Evaluator:
    def __init__(self, tfidf, word2vec, bm25, query_refiner):
        self.tfidf_search = tfidf
        self.word2vec_search = word2vec
        self.bm25_search = bm25
        self.query_refiner = query_refiner
        self.evaluation_results = {}
        self.hybrid_parallel_1 = HybridParallelBM25Word2Vec(
            bm25_search=self.bm25_search, 
            word2vec_search=self.word2vec_search,
            a=0.5,
            b=0.5
        )
        self.hybrid_parallel_2 = HybridParallelBM25Word2Vec(
            bm25_search=self.bm25_search,
            word2vec_search=self.word2vec_search,
            a=0.35,
            b=0.65
        )
        self.hybrid_serial_1 = HybridSerialBM25Word2Vec(
            bm25_search=self.bm25_search,
            word2vec_search=self.word2vec_search,
            candidate_k=50
        )
        self.hybrid_serial_2 = HybridSerialBM25Word2Vec(
            bm25_search=self.bm25_search,
            word2vec_search=self.word2vec_search,
            candidate_k=100
        )
        self.hybrid_serial_3 = HybridSerialBM25Word2Vec(
            bm25_search=self.bm25_search,
            word2vec_search=self.word2vec_search,
            candidate_k=200
        )

    def compute_query_metrics(self, ranked_list: list, qrel_list: list, top_k_p: int = 10) -> dict:
        total_relevant = sum(1 for rel in qrel_list)
        
        if total_relevant == 0:
            return None 
        
        top_k_docs = ranked_list[:top_k_p]
        rel_in_top_k = sum(1 for doc_id in top_k_docs if doc_id in qrel_list)
        precision_at_k = rel_in_top_k / top_k_p

        rel_retrieved = sum(1 for doc_id in ranked_list if doc_id in qrel_list)
        recall = rel_retrieved / total_relevant

        ap_sum = 0.0
        num_rel_found = 0
        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id in qrel_list:
                num_rel_found += 1
                precision_at_rank = num_rel_found / rank
                ap_sum += precision_at_rank
        ap = ap_sum / total_relevant

        dcg = 0.0
        for rank, doc_id in enumerate(ranked_list, start=1):
            rel = 1 if doc_id in qrel_list else 0
            dcg += (2**rel - 1) / math.log2(rank + 1)

        ideal_rels = sorted([1 if doc_id in qrel_list else 0 for doc_id in ranked_list], reverse=True)
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels[:len(ranked_list)], start=1):
            idcg += (2**rel - 1) / math.log2(rank + 1)

        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            "p_at_k": precision_at_k,
            "recall": recall,
            "ap": ap,
            "ndcg": ndcg
        }
    
    def evalute(self):
        print("Loading qrels and preprocessed queries from MongoDB...")
        db = DataBaseService()
        qrels_cursor = db.qrels.find({})
        qrels_lookup = defaultdict(list)
        for doc in qrels_cursor:
            qid = doc.get("query_id")
            did = doc.get("doc_id")
            qrels_lookup[qid].append(did)
        queries_cursor = list(db.queries_processed.find({}))
        queries = [
            {"query_id": q.get("_id"), "text_clean": " ".join(q.get("tokens", []))}
            for q in queries_cursor
        ]
        print(f"Loaded {len(queries)} queries and {len(qrels_lookup)} qrel mappings.\n")
        models_to_evaluate = {
            "TF-IDF": self.tfidf_search.search,
            "Word2Vec": self.word2vec_search.search,
            "BM25": self.bm25_search.search,
            "Parallel (BM25+W2V) 0.5/0.5": self.hybrid_parallel_1.search,
            "Parallel (BM25+W2V) 0.35/0.65": self.hybrid_parallel_2.search,
            "Serial (BM25→W2V) k=50": self.hybrid_serial_1.search,
            "Serial (BM25→W2V) k=100": self.hybrid_serial_2.search,
            "Serial (BM25→W2V) k=200": self.hybrid_serial_3.search,
        }

        final_results = {}

        for model_name, retrieve_fn in models_to_evaluate.items():
            print(f"Running evaluation for model: {model_name}...")
            
            total_p_at_10 = 0.0
            total_recall = 0.0
            total_ap = 0.0
            total_ndcg = 0.0
            evaluated_queries_count = 0

            for q in queries:
                q_id = q['query_id']
                q_text = q['text_clean']
                q_text = self.query_refiner.refine(q_text)
                ranked_results = retrieve_fn(query=q_text, top_k=100)
                
                if not ranked_results:
                    ranked_results = []

                metrics = self.compute_query_metrics(ranked_results, qrels_lookup[q_id], top_k_p=10)
                
                if metrics:
                    total_p_at_10 += metrics["p_at_k"]
                    total_recall += metrics["recall"]
                    total_ap += metrics["ap"]
                    total_ndcg += metrics["ndcg"]
                    evaluated_queries_count += 1

            if evaluated_queries_count > 0:
                final_results[model_name] = {
                    "MAP": total_ap / evaluated_queries_count,
                    "Recall": total_recall / evaluated_queries_count,
                    "Precision@10": total_p_at_10 / evaluated_queries_count,
                    "nDCG": total_ndcg / evaluated_queries_count
                }
            else:
                final_results[model_name] = {"MAP": 0.0, "Recall": 0.0, "Precision@10": 0.0, "nDCG": 0.0}

        print('RESULSTS:')
        print(f"{'Model':<20} | {'MAP':<10} | {'Recall':<10} | {'Precision@10':<12} | {'nDCG':<10}")
        print("-" * 75)
        for model_name, metrics in final_results.items():
            print(f"{model_name:<20} | {metrics['MAP']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['Precision@10']:<12.4f} | {metrics['nDCG']:<10.4f}")

        self.evaluation_results = final_results
        try:
            os.makedirs("results", exist_ok=True)
            joblib.dump(final_results, "results/evaluation_results.pkl")
            print("Saved evaluation results to results/evaluation_results.pkl")
        except Exception as e:
            print(f"Failed to save evaluation results: {e}")

        return final_results

    def get_results(self):
        return self.evaluation_results

    def load_results_from_file(self, path: str = "results/evaluation_results.pkl"):
        try:
            results = joblib.load(path)
            self.evaluation_results = results
            return results
        except FileNotFoundError:
            print(f"Evaluation results file not found: {path}")
            return None
        except Exception as e:
            print(f"Failed to load evaluation results from {path}: {e}")
            return None

