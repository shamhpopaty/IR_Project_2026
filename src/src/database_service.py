from itertools import islice

import ir_datasets
from pymongo import MongoClient

class DataBaseService:
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["lottesearch"]
        
        self.dataset = ir_datasets.load("lotte/lifestyle/dev/search")
        self.qrels = self.dataset.qrels_iter()
        self.queries = self.dataset.queries_iter()
        self.documents = self.db["documents"]
        self.queries = self.db["queries"]
        self.qrels = self.db["qrels"]
        self.documents_processed = self.db["documents_processed"]
        self.queries_processed = self.db["queries_processed"]
    
    def store_documents(self):
        self.documents.delete_many({})
        self.queries.delete_many({})
        self.qrels.delete_many({})

        docs_batch = []

        for doc in self.dataset.docs_iter():
            docs_batch.append({
                "_id": doc.doc_id,
                "text": doc.text
            })

            if len(docs_batch) >= 1000:
                self.documents.insert_many(docs_batch)
                docs_batch = []

        if docs_batch:
            self.documents.insert_many(docs_batch)

        print("Documents stored")

        queries_batch = []

        for query in self.dataset.queries_iter():
            queries_batch.append({
                "_id": query.query_id,
                "text": query.text
            })

            if len(queries_batch) >= 1000:
                self.queries.insert_many(queries_batch)
                queries_batch = []

        if queries_batch:
            self.queries.insert_many(queries_batch)

        print("Queries stored")

        qrels_batch = []

        for qrel in self.dataset.qrels_iter():
            qrels_batch.append({
                "query_id": qrel.query_id,
                "doc_id": qrel.doc_id,
                "relevance": qrel.relevance,
                "iteration": qrel.iteration
            })

            if len(qrels_batch) >= 1000:
                self.qrels.insert_many(qrels_batch)
                qrels_batch = []

        if qrels_batch:
            self.qrels.insert_many(qrels_batch)

        print("Qrels stored")
        print("Done")

    def summary(self):
        print("Documents:", self.documents.count_documents({}))
        print("Queries:", self.queries.count_documents({}))
        print("Qrels:", self.qrels.count_documents({}))
        print("Documents Processed:", self.documents_processed.count_documents({}))
        print("Queries Processed:", self.queries_processed.count_documents({}))
