"""
ml_pipeline.py — Hybrid Movie Recommendation ML Pipeline
Trains and saves the content-based model from Movies_Recommendation.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_PATH   = "Movies_Recommendation.csv"
MODEL_DIR   = "models"
MODEL_FILE  = os.path.join(MODEL_DIR, "recommender.pkl")

# ─── Step 1: Load Data ────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    print("[1/5] Loading dataset...")
    df = pd.read_csv(path)
    print(f"      → {len(df)} movies loaded.")
    return df

# ─── Step 2: Clean & Preprocess ───────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Preprocessing...")

    df = df.copy()

    # Fill nulls
    text_cols = ["Movie_Genre", "Movie_Director", "Movie_Cast",
                 "Movie_Keywords", "Movie_Overview", "Movie_Tagline"]
    for col in text_cols:
        df[col] = df[col].fillna("")

    df["Movie_Vote"]       = pd.to_numeric(df["Movie_Vote"], errors="coerce").fillna(0)
    df["Movie_Vote_Count"] = pd.to_numeric(df["Movie_Vote_Count"], errors="coerce").fillna(0)
    df["Movie_Popularity"] = pd.to_numeric(df["Movie_Popularity"], errors="coerce").fillna(0)

    # Weighted rating score (IMDb formula): WR = (v/(v+m)) * R + (m/(v+m)) * C
    m = df["Movie_Vote_Count"].quantile(0.70)
    C = df["Movie_Vote"].mean()
    df["weighted_score"] = (
        (df["Movie_Vote_Count"] / (df["Movie_Vote_Count"] + m)) * df["Movie_Vote"] +
        (m / (df["Movie_Vote_Count"] + m)) * C
    )

    # Build combined feature string for TF-IDF
    df["Movie_Genre"]    = df["Movie_Genre"].str.replace(" ", "_")
    df["Movie_Director"] = df["Movie_Director"].str.strip()

    df["soup"] = (
        df["Movie_Genre"]    * 3 + " " +   # Genre gets 3x weight
        df["Movie_Director"] * 2 + " " +   # Director 2x
        df["Movie_Cast"]           + " " +
        df["Movie_Keywords"]       + " " +
        df["Movie_Tagline"]
    )

    print(f"      → Preprocessed {len(df)} rows.")
    return df.reset_index(drop=True)

# ─── Step 3: Build TF-IDF Matrix ─────────────────────────────────────────────

def build_tfidf(df: pd.DataFrame):
    print("[3/5] Building TF-IDF matrix...")
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=15000,
        ngram_range=(1, 2)
    )
    matrix = tfidf.fit_transform(df["soup"])
    print(f"      → Matrix shape: {matrix.shape}")
    return tfidf, matrix

# ─── Step 4: Compute Cosine Similarity ───────────────────────────────────────

def build_similarity(matrix):
    print("[4/5] Computing cosine similarity...")
    sim = cosine_similarity(matrix, matrix)
    print(f"      → Similarity matrix: {sim.shape}")
    return sim

# ─── Step 5: Save Model ───────────────────────────────────────────────────────

def save_model(df, tfidf, sim_matrix):
    print("[5/5] Saving model artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "df":         df,
        "tfidf":      tfidf,
        "sim_matrix": sim_matrix,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(payload, f)
    size_mb = os.path.getsize(MODEL_FILE) / 1_000_000
    print(f"      → Saved to '{MODEL_FILE}' ({size_mb:.1f} MB)")

# ─── Load Model ───────────────────────────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_FILE}'. Run ml_pipeline.py first."
        )
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

# ─── Recommend Function ───────────────────────────────────────────────────────

def recommend(
    title: str,
    df: pd.DataFrame,
    sim_matrix,
    top_n: int = 10,
    genre_filter: str = None,
    min_vote: float = 0.0,
) -> pd.DataFrame:
    """
    Content-based recommendations for a given movie title.
    Optionally filter by genre and minimum vote score.
    """
    titles = df["Movie_Title"].str.lower()
    matches = df[titles == title.lower()]

    if matches.empty:
        # Fuzzy fallback: partial match
        matches = df[titles.str.contains(title.lower(), na=False)]
        if matches.empty:
            return pd.DataFrame()

    idx = matches.index[0]

    # Content similarity scores
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(i, s) for i, s in sim_scores if i != idx]

    # Extract candidates
    candidate_idx = [i for i, _ in sim_scores[:top_n * 5]]
    candidates    = df.iloc[candidate_idx].copy()
    candidates["content_score"] = [s for _, s in sim_scores[:top_n * 5]]

    # Apply filters
    if genre_filter and genre_filter != "All":
        candidates = candidates[
            candidates["Movie_Genre"].str.contains(genre_filter, case=False, na=False)
        ]
    if min_vote > 0:
        candidates = candidates[candidates["Movie_Vote"] >= min_vote]

    # Hybrid score: content similarity + weighted popularity
    scaler = MinMaxScaler()
    if len(candidates) > 1:
        candidates["pop_score"] = scaler.fit_transform(
            candidates[["weighted_score"]]
        )
    else:
        candidates["pop_score"] = 0

    candidates["hybrid_score"] = (
        0.65 * candidates["content_score"] +
        0.35 * candidates["pop_score"]
    ).round(4)

    result = candidates.sort_values("hybrid_score", ascending=False).head(top_n)
    return result[[
        "Movie_Title", "Movie_Genre", "Movie_Director",
        "Movie_Vote", "Movie_Popularity", "Movie_Overview",
        "Movie_Cast", "Movie_Release_Date",
        "content_score", "hybrid_score"
    ]].reset_index(drop=True)


def search_movies(query: str, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Search movies by partial title match."""
    mask = df["Movie_Title"].str.contains(query, case=False, na=False)
    results = df[mask].sort_values("weighted_score", ascending=False)
    return results[["Movie_Title", "Movie_Genre", "Movie_Director",
                    "Movie_Vote", "Movie_Release_Date"]].head(top_n)


def get_popular_movies(df: pd.DataFrame, genre: str = None, top_n: int = 20) -> pd.DataFrame:
    """Return top movies by weighted score, optionally filtered by genre."""
    filtered = df.copy()
    if genre and genre != "All":
        filtered = filtered[
            filtered["Movie_Genre"].str.contains(genre, case=False, na=False)
        ]
    return filtered.sort_values("weighted_score", ascending=False).head(top_n)[
        ["Movie_Title", "Movie_Genre", "Movie_Director",
         "Movie_Vote", "Movie_Vote_Count", "Movie_Popularity",
         "Movie_Overview", "Movie_Release_Date", "weighted_score"]
    ].reset_index(drop=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🎬 Movie Recommendation ML Pipeline")
    print("=" * 45)

    df         = load_data()
    df         = preprocess(df)
    tfidf, mat = build_tfidf(df)
    sim        = build_similarity(mat)
    save_model(df, tfidf, sim)

    print("\n✅ Pipeline complete! Testing recommendations...\n")

    # Quick smoke test
    payload = load_model()
    recs = recommend("The Dark Knight", payload["df"], payload["sim_matrix"], top_n=5)
    if not recs.empty:
        print("Top 5 recommendations for 'The Dark Knight':")
        for i, row in recs.iterrows():
            print(f"  {i+1}. {row['Movie_Title']} ({row['Movie_Genre']}) — ⭐ {row['Movie_Vote']} | Score: {row['hybrid_score']}")
    else:
        print("  Movie not found in dataset.")

    print("\n🚀 Ready! Run: streamlit run app.py\n")