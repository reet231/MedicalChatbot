"""
retriever.py

A TF-IDF + cosine-similarity retriever over the MedQuAD-style QA dataset.

Why TF-IDF instead of an embedding model? It needs no internet access, no
model download, and no GPU, so the app works fully offline out of the box.
The public interface (`Retriever.search`) is small enough to swap in a
sentence-embedding retriever (e.g. sentence-transformers + FAISS) later
without touching the rest of the app.
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Retriever:
    def __init__(self, df: pd.DataFrame):
        """
        df must have at least: question, answer, focus columns.
        We index over "focus + question" so a query mentioning the disease
        name or the phrasing of the question can both match.
        """
        required = {"question", "answer"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        self.df = df.reset_index(drop=True).copy()
        if "focus" not in self.df.columns:
            self.df["focus"] = ""

        corpus_text = (
            self.df["focus"].fillna("").astype(str) + " " + self.df["question"].fillna("").astype(str)
        ).apply(_normalize)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(corpus_text)

    def search(self, query, top_k=5, min_score=0.05):
        """
        Returns a list of dicts sorted by descending similarity:
            [{"question", "answer", "focus", "score", ...other columns}, ...]
        """
        query_norm = _normalize(query)
        query_vec = self.vectorizer.transform([query_norm])
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        top_idx = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < min_score:
                continue
            row = self.df.iloc[idx].to_dict()
            row["score"] = score
            results.append(row)
        return results
