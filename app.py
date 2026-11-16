
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set wide-screen page layout
st.set_page_config(
    page_title="CineMatch - Netflix Stream",
    page_icon="🍿",
    layout="wide"
)

# Redesigned dark-theme Netflix CSS injector
st.markdown("""
<style>
    .main {
        background-color: #141414;
        color: #FFFFFF;
    }
    h1, h2, h3, h4, h5 {
        color: #E50914 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        border: none;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #B20710;
        color: white;
    }
    .movie-card {
        background-color: #181818;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #222;
        margin-bottom: 20px;
    }
    .match-tag {
        color: #46d369;
        font-weight: bold;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Load database safely
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

# Fit TF-IDF model
@st.cache_resource
def get_tfidf_resources(features):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(features)
    return vectorizer, matrix

vectorizer, tfidf_matrix = get_tfidf_resources(movies["combined_features"].tolist())

# Application Headers
st.title("🍿 CineMatch")
st.write("Experience search, browsing, and discovery fully tailored to your preferences.")
st.divider()

# Sidebar Filters
st.sidebar.header("🔎 Discovery Controls")

# 1. Text Filters
search_title = st.sidebar.text_input("Title Search", "")
search_director = st.sidebar.text_input("Director", "")

# 2. Year Range Slider
min_year, max_year = int(movies["year"].min()), int(movies["year"].max())
year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))

# 3. Categorical Selectors
genres_list = sorted(list(set([g.strip() for sublist in movies["genres"].str.split(",") for g in sublist if g.strip()])))
selected_genre = st.sidebar.selectbox("Genre", ["All"] + genres_list)

languages_list = sorted(movies["language"].unique().tolist())
selected_lang = st.sidebar.selectbox("Language", ["All"] + languages_list)

# 4. Semantic Search Prompt
st.sidebar.divider()
st.sidebar.subheader("🧠 Semantic Search Prompt")
user_prompt = st.sidebar.text_area("Describe your ideal movie mood:", placeholder="e.g., a dark psychological crime drama set in New York")

# Apply Filter Pipeline
filtered_df = movies.copy()

if search_title.strip():
    filtered_df = filtered_df[filtered_df["title"].str.contains(search_title, case=False, na=False)]
    
if search_director.strip():
    filtered_df = filtered_df[filtered_df["directors"].str.contains(search_director, case=False, na=False)]

filtered_df = filtered_df[
    (filtered_df["year"] >= year_range[0]) &
    (filtered_df["year"] <= year_range[1])
]

if selected_genre != "All":
    filtered_df = filtered_df[filtered_df["genres"].str.contains(selected_genre, case=False, na=False)]

if selected_lang != "All":
    filtered_df = filtered_df[filtered_df["language"] == selected_lang]

# 1. Spotlight Feature Banner (Dynamic or Highest Rated)
if not filtered_df.empty:
    spotlight_movie = filtered_df.sort_values(by="rating", ascending=False).iloc[0]
    cols = st.columns([1, 2])
    with cols[0]:
        if spotlight_movie["poster"].startswith("http"):
            st.image(spotlight_movie["poster"], use_container_width=True)
    with cols[1]:
        st.markdown(f"# **Spotlight: {spotlight_movie['title']}** ({int(spotlight_movie['year'])})")
        st.markdown(f"⭐⭐ {spotlight_movie['rating']}/10  |  🎬 Directed by: **{spotlight_movie['directors']}**")
        st.markdown(f"🎭 *{spotlight_movie['genres']}*  |  🌍 Lang: **{spotlight_movie['language'].upper()}**")
        st.write(spotlight_movie["overview"])
else:
    st.warning("No movie matches the filters in the sidebar.")

st.divider()

# 2. Recommendations via Natural Language Prompt
if user_prompt.strip():
    st.header("🎯 Tailored Matches for your Mood")
    
    # Calculate similarity with prompt
    prompt_vector = vectorizer.transform([user_prompt])
    sim_scores = cosine_similarity(prompt_vector, tfidf_matrix).flatten()
    
    # Combine with dataframe
    movies_with_scores = movies.copy()
    movies_with_scores["match_percent"] = sim_scores * 100
    
    # Sort & display
    prompt_matches = movies_with_scores.sort_values(by="match_percent", ascending=False).head(4)
    
    rec_cols = st.columns(4)
    for j, (_, row) in enumerate(prompt_matches.iterrows()):
        with rec_cols[j]:
            st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
            if row["poster"].startswith("http"):
                st.image(row["poster"], use_container_width=True)
            st.markdown(f"##### {row['title']}")
            st.markdown(f"<span class='match-tag'>{row['match_percent']:.1f}% Match</span>", unsafe_allow_html=True)
            st.caption(f"{int(row['year'])} | ⭐ {row['rating']} | {row['genres']}")
            st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

# 3. Traditional Netflix Shelf Rows
st.header("🔥 Trending Discoveries")
if not filtered_df.empty:
    trending_movies = filtered_df.sample(min(4, len(filtered_df)))
    cols = st.columns(4)
    for idx, (_, m) in enumerate(trending_movies.iterrows()):
        with cols[idx]:
            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
            if m["poster"].startswith("http"):
                st.image(m["poster"], use_container_width=True)
            st.markdown(f"##### {m['title']}")
            st.caption(f"{int(m['year'])} | ⭐ {m['rating']} | {m['genres']}")
            st.markdown("</div>", unsafe_allow_html=True)
