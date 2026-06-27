import math
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

class ClusteringService:
    def __init__(self):  # تم إصلاح الخطأ هنا (كان مكتوباً init)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2)
        )

    def _get_text(self, doc):
        if isinstance(doc, dict):
            if "text_clean" in doc: return doc["text_clean"]
            if "text" in doc: return doc["text"]
            if "content" in doc: return doc["content"]
            if "tokens" in doc and isinstance(doc["tokens"], list): return " ".join(doc["tokens"])
            if "title" in doc: return doc["title"]
        return str(doc)

    def _auto_k(self, n_docs):
        if n_docs < 6:
            return 2
        return min(6, max(2, int(math.sqrt(n_docs))))

    def _extract_cluster_labels(self, kmeans, feature_names, top_n=6):
        labels = {}
        for cluster_id, center in enumerate(kmeans.cluster_centers_):
            top_indices = center.argsort()[::-1][:top_n]
            top_terms = [feature_names[i] for i in top_indices]
            labels[cluster_id] = ", ".join(top_terms)
        return labels

    def cluster_results(self, results, n_clusters=None):
        if results is None or len(results) < 2:
            return {"n_clusters": 0, "clusters": [], "message": "Not enough results"}

        documents = list(results)
        texts = [self._get_text(doc) for doc in documents]

        filtered_docs = []
        filtered_texts = []
        for doc, text in zip(documents, texts):
            if text and len(text.strip()) > 5:
                filtered_docs.append(doc)
                filtered_texts.append(text)

        if len(filtered_docs) < 2:
            return {"n_clusters": 0, "clusters": [], "message": "Not enough valid text"}

        X = self.vectorizer.fit_transform(filtered_texts)

        if n_clusters is None or n_clusters == "auto":
            n_clusters = self._auto_k(len(filtered_docs))

        n_clusters = min(int(n_clusters), len(filtered_docs))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(X)

        feature_names = self.vectorizer.get_feature_names_out()
        cluster_labels = self._extract_cluster_labels(kmeans, feature_names)

        grouped = defaultdict(list)
        for index, cluster_id in enumerate(cluster_ids):
            doc = filtered_docs[index]
            doc_copy = doc.copy() if isinstance(doc, dict) else {"text": str(doc)}
            doc_copy["original_rank"] = index + 1
            grouped[int(cluster_id)].append(doc_copy)

        clusters = []
        for cluster_id, docs in grouped.items():
            clusters.append({
                "cluster_id": cluster_id,
                "label": cluster_labels.get(cluster_id, f"Cluster {cluster_id}"),
                "documents_count": len(docs),
                "documents": docs
            })

        clusters = sorted(clusters, key=lambda c: c["documents_count"], reverse=True)

        # تم إصلاح المسافة البادئة هنا (كانت خارج النطاق)
        return {
            "n_clusters": n_clusters,
            "clusters": clusters
        }