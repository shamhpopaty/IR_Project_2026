import streamlit as st

from bm25_service import BM25Service
from database_service import DataBaseService
# from search.bert import BertSearch
from hybrid_search_service import HybridParallelBM25Word2Vec, HybridSerialBM25Word2Vec
from preprocessor import Preprocessor
from query_refinement import QueryRefiner
from tfidf_service import TFIDFService
from word2vec_service import Word2VecService
from evaluator import Evaluator
import pandas as pd
import altair as alt

MODEL_LABELS = {
    "bm25": "BM25",
    "tfidf": "TF-IDF",
    "word2vec": "Word2Vec",
    "hybrid_serial_search": "Serial Hybrid BM25 + Word2Vec",
    "hybrid_parallel_search": "Parallel Hybrid BM25 + Word2Vec",
}

MODEL_ORDER = [
    "bm25",
    "tfidf",
    "word2vec",
    "hybrid_serial_search",
    "hybrid_parallel_search",
]

preprocessor = Preprocessor()
query_refiner = QueryRefiner()

@st.cache_resource
def load_services():
    bm25 = BM25Service()
    tfidf = TFIDFService()
    word2vec = Word2VecService()
    hybrid_serial = HybridSerialBM25Word2Vec(bm25_search=bm25, word2vec_search=word2vec)
    hybrid_parallel = HybridParallelBM25Word2Vec(bm25_search=bm25, word2vec_search=word2vec)
    return {
        "bm25": bm25,
        "tfidf": tfidf,
        "word2vec": word2vec,
        "hybrid_serial_search": hybrid_serial,
        "hybrid_parallel_search": hybrid_parallel,
    }


@st.cache_resource
def load_db():
    return DataBaseService()


def fetch_documents_text(db, doc_ids):
    if not doc_ids:
        return []

    documents = list(db.documents.find({"_id": {"$in": doc_ids}}, {"_id": 1, "text": 1}))
    documents_by_id = {doc["_id"]: doc for doc in documents}
    return [documents_by_id.get(doc_id, {"text": "[document not found]"}) for doc_id in doc_ids]


def run_search(services, query, selected_models, top_k=10):
    results = {}
    query = preprocessor.process(query)
    query = " ".join(query)
    query = query_refiner.refine(query)
    for model_name in selected_models:
        service = services.get(model_name)
        if service is None:
            continue
        try:
            results[model_name] = service.search(query, top_k=top_k)
        except Exception as exc:
            results[model_name] = []
            st.error(f"Error running {MODEL_LABELS.get(model_name, model_name)}: {exc}")
    return results


def main():
    st.title("Information Retrieval Search UI")
    st.write("Enter a query, choose one or more models, and click Search to view retrieved documents.")

    query = st.text_input("Search query", value="")
    selected_model = st.selectbox(
        "Choose model",
        options=MODEL_ORDER,
        format_func=lambda key: MODEL_LABELS.get(key, key),
        index=0,
    )
    top_k = st.number_input("Top k results", min_value=1, max_value=50, value=10, step=1)

    if st.button("Load saved evaluation results"):
        services = load_services()
        evaluator = Evaluator(
            tfidf=services["tfidf"],
            word2vec=services["word2vec"],
            bm25=services["bm25"],
            query_refiner=query_refiner,
        )
        results = evaluator.load_results_from_file()
        if results is None:
            st.warning("No saved evaluation results were found.")
        else:
            st.success("Loaded evaluation results from results/evaluation_results.pkl")
            df = pd.DataFrame.from_dict(results, orient="index")
            df.index.name = "Model"
            df = df.reset_index()
            metric_columns = [col for col in df.columns if col != "Model"]
            for col in metric_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            st.subheader("Evaluation Results")
            styled_df = df.style.highlight_max(subset=metric_columns, color="rgba(210, 235, 255, 0.45)")
            st.dataframe(styled_df, use_container_width=True)

            if not df.empty:
                st.subheader("Comparison Charts")
                columns = st.columns(len(metric_columns))
                for col_name, col in zip(metric_columns, columns):
                    metric_df = df[["Model", col_name]].rename(columns={col_name: "Score"})
                    chart = alt.Chart(metric_df).mark_bar().encode(
                        x=alt.X("Model:N", sort=None, title="Model"),
                        y=alt.Y("Score:Q", title=col_name),
                        color=alt.value("#1f77b4"),
                        tooltip=["Model", "Score"]
                    ).properties(height=300)
                    col.subheader(col_name)
                    col.altair_chart(chart, use_container_width=True)

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a query before searching.")
            return
        if not selected_model:
            st.warning("Please select a model.")
            return

        services = load_services()
        db = load_db()

        with st.spinner("Running search..."):
            model_results = run_search(services, query, [selected_model], top_k=top_k)

        doc_ids = model_results.get(selected_model, [])
        label = MODEL_LABELS.get(selected_model, selected_model)
        st.header(f"{label} Results")
        if not doc_ids:
            st.info("No results returned.")
        else:
            documents = fetch_documents_text(db, doc_ids)
            for idx, (doc_id, document) in enumerate(zip(doc_ids, documents), start=1):
                st.subheader(f"{idx}. Document ID: {doc_id}")
                st.write(document.get("text", "[no text field found]"))
                st.markdown("---")


if __name__ == "__main__":
    main()
