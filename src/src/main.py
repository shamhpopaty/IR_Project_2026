# import os

# from flask import Flask, jsonify, render_template, request

# from evaluator import Evaluator
# from query_refinement import QueryRefiner
# from database_service import DataBaseService
# from bm25_service import BM25Service
# from hybrid_search_service import HybridParallelBM25Word2Vec, HybridSerialBM25Word2Vec
# from tfidf_service import TFIDFService
# from word2vec_service import Word2VecService


# def load_services():
#     bm25 = BM25Service()
#     word2vec = Word2VecService()
#     services = {
#         "bm25": bm25,
#         "tfidf": TFIDFService(),
#         "hybrid_serial_search": HybridSerialBM25Word2Vec(bm25_search=bm25, word2vec=word2vec),
#         "hybrid_parallel_search": HybridParallelBM25Word2Vec(bm25_search=bm25, word2vec_search=word2vec),
#         "word2vec": word2vec,
#     }
#     print('loaded all services')
#     return services

# services = load_services()
# query_refiner = QueryRefiner()
# db = DataBaseService()
# from bm25_service import BM25Service
# from evaluator import Evaluator
# from tfidf_service import TFIDFService
# from word2vec_service import Word2VecService


from bm25_service import BM25Service
from evaluator import Evaluator
from query_refinement import QueryRefiner
from tfidf_service import TFIDFService
from word2vec_service import Word2VecService


evaluator = Evaluator(
    tfidf=TFIDFService(), 
    word2vec=Word2VecService(),
    bm25=BM25Service(),
    query_refiner=QueryRefiner()
)
evaluator.evalute()
