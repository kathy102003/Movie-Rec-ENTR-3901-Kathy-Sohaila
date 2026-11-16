
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page setup
st.set_page_config(
    page_title="CineMatch - Netflix Style",
    page_icon="🍿",
    layout="wide"
)

# Custom styling for Dark Netflix theme
st.markdown("""
<style>
    .main {
        background-color: #141414;
        color: #FFFFFF;
    }
    h1, h2, h3, h4 {
        color: #E50914 !important;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        border: none;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #B20710;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load movie database
@st.cache_data
def load_movies():
    df = pd.read_csv("cinematch_movies.csv")
    text_columns = [
        "title", "overview", "genres", "actors",
        "directors", "keywords", "language",
        "certification", "poster"
    ]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    for col in ["year", "rating", "vote_count", "runtime"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["combined_features"] = (
        df["overview"] + " " +
        df["genres"] + " " +
        df["keywords"] + " " +
        df["actors"] + " " +
        df["directors"]
    )
    return df

movies = load_movies()

# Create recommendation model
@st.cache_resource
def build_model(features):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(features)
    similarity = cosine_similarity(matrix)
    return similarity

cosine_sim = build_model(movies["combined_features"].tolist())

# Netflix App Header
st.title("🍿 CineMatch")
st.write("Netflix-style personalized streaming experience powered by Content-Based Filtering & NLP Engine")

# 1. Netflix Hero Banner
if not movies.empty:
    hero_movie = movies.sort_values(by="rating", ascending=False).iloc[0]
    st.image(hero_movie["poster"] if hero_movie["poster"].startswith("http") else "https://via.placeholder.com/1200x400/141414/ffffff?text=CineMatch+Spotlight", use_container_width=True)
    st.markdown(f"## **Spotlight:** {hero_movie['title']} ({int(hero_movie['year'])})")
    st.markdown(f"⭐ Rating: {hero_movie['rating']}/10  |  🇣 Genres: {hero_movie['genres']}")
    st.write(hero_movie["overview"])
st.divider()

# 2. Dynamic Personalization ("Because you liked...")
st.sidebar.header("👤 Netflix Personalization Profile")
user_history_title = st.sidebar.selectbox("Set your active watch history:", sorted(movies["title"].tolist()))

if user_history_title:
    st.header(f"🎬 Because you watched: **{user_history_title}**")
    
    idx = movies[movies["title"] == user_history_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    rec_indices = [i[0] for i in sim_scores if i[0] != idx][:4]
    
    rec_cols = st.columns(4)
    for j, r_idx in enumerate(rec_indices):
        rec_movie = movies.iloc[r_idx]
        with rec_cols[j]:
            if rec_movie["poster"].startswith("http"):
                st.image(rec_movie["poster"], use_container_width=True)
            st.markdown(f"##### **{rec_movie['title']}**")
            st.caption(f"{int(rec_movie['year'])} | ⭐ {rec_movie['rating']} | Similarity: {sim_scores[j+1][1]*100:.1f}%")

st.divider()

# 3. Categorized Horizontal Rows (Netflix Shelves)
st.header("🔥 Trending Now")
trending_cols = st.columns(4)
trending_movies = movies.sample(min(4, len(movies)))
for idx, (_, m) in enumerate(trending_movies.iterrows()):
    with trending_cols[idx]:
        if m["poster"].startswith("http"):
            st.image(m["poster"], use_container_width=True)
        st.markdown(f"##### **{m['title']}**")
        st.caption(f"{int(m['year'])} | ⭐ {m['rating']}")

st.divider()

st.header("🏆 Critical Acclaim (Top Rated)")
top_cols = st.columns(4)
top_movies = movies.sort_values(by="rating", ascending=False).head(4)
for idx, (_, m) in enumerate(top_movies.iterrows()):
    with top_cols[idx]:
        if m["poster"].startswith("http"):
            st.image(m["poster"], use_container_width=True)
        st.markdown(f"##### **{m['title']}**")
        st.caption(f"{int(m['year'])} | ⭐ {m['rating']}")
