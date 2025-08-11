
import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Content-Based Movie Recommender")
st.write("Type a movie you like and get similar titles based on genres, keywords, tagline, cast, and director.")

@st.cache_data(show_spinner=True)
def load_movies(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure required columns exist
    required = ['title','genres','keywords','tagline','cast','director']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Your movies.csv is missing columns: {missing}")
    # Normalize NaNs
    for c in required:
        df[c] = df[c].fillna('')
    # Ensure an index column exists for lookup; create if missing
    if 'index' not in df.columns:
        df = df.reset_index(drop=False).rename(columns={'index':'index'})
    return df

@st.cache_data(show_spinner=True)
def build_model(df: pd.DataFrame):
    combined = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(combined)
    sim = cosine_similarity(vectors)
    return sim

def recommend(title: str, df: pd.DataFrame, sim_matrix, top_k: int = 10):
    titles = df['title'].tolist()
    close = difflib.get_close_matches(title, titles, n=1, cutoff=0.4)
    if not close:
        return [], None
    match = close[0]
    idx = df[df.title == match]['index'].values[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    results = []
    for s_idx, s_val in scores_sorted:
        rec_title = df.iloc[s_idx]['title']
        if rec_title == match:
            continue
        results.append((rec_title, float(s_val)))
        if len(results) >= top_k:
            break
    return results, match

csv_path = Path("movies.csv")
if not csv_path.exists():
    st.warning("⚠️ Place your **movies.csv** next to this app file. It must include columns: "
               "`title, genres, keywords, tagline, cast, director` (and optional `index`).")
else:
    try:
        movies_df = load_movies(csv_path)
        sim = build_model(movies_df)
        movie_name = st.text_input("Enter a movie you like:", value="Avatar")
        top_k = st.slider("How many recommendations?", 5, 30, 10, 1)
        if st.button("Recommend") or movie_name:
            recs, matched = recommend(movie_name, movies_df, sim, top_k=top_k)
            if matched is None:
                st.error("Couldn't find a close match. Try another title (e.g., 'Avatar', 'The Dark Knight').")
            else:
                st.success(f"Close match: **{matched}**")
                if not recs:
                    st.info("No similar movies found.")
                else:
                    st.subheader("Movies suggested for you")
                    for rank, (t, score) in enumerate(recs, start=1):
                        st.write(f"{rank}. {t}  —  similarity: {score:.3f}")
    except Exception as e:
        st.exception(e)
