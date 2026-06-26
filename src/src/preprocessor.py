import re

import nltk

from nltk.corpus import stopwords
from database_service import DataBaseService

class Preprocessor:
    def __init__(self, remove_stopwords=True):
        nltk.download('stopwords')
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.database = DataBaseService()

    def normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)  
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str):
        return text.split()

    def remove_stopwords_fn(self, tokens):
        if not self.remove_stopwords:
            return tokens
        return [t for t in tokens if t not in self.stop_words]

    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def process(self, text: str):
        text = self.normalize(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords_fn(tokens)
        tokens = self.lemmatize(tokens)
        return tokens
    
    def process_documents(self, documents, docs_proc):
        docs_proc.delete_many({}) 
        batch = []
        for doc in documents.find({}):
            batch.append({
                "_id": doc["_id"],
                "tokens": self.process(doc["text"])
            })
            if len(batch) >= 1000:
                docs_proc.insert_many(batch)
                batch = []
        if batch:
            docs_proc.insert_many(batch)
        print("Processed documents stored")

    def process_queries(self, quries, queries_proc):
        queries_proc.delete_many({})
        batch = []
        for q in quries.find({}):
            batch.append({
                "_id": q["_id"],
                "tokens": self.process(q["text"])
            })
            if len(batch) >= 1000:
                queries_proc.insert_many(batch)
                batch = []
        if batch:
            queries_proc.insert_many(batch)
        print("Processed queries stored")

    def process_data(self):
        self.process_documents(self.database.documents, self.database.documents_processed)
        self.process_queries(self.database.queries, self.database.queries_processed)
