
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

# Dark Theme CSS Injector
st.markdown("""
<style>
    .main {
        background-color: #141414;
        color: #FFFFFF;
    }
    h1, h2, h3, h4, h5 {
        color: #E50914 !important;
    }
    .movie-card {
        background-color: #1c1c1c;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2d2d2d;
        margin-bottom: 15px;
    }
    .match-tag {
        color: #46d369;
        font-weight: bold;
        font-size: 15px;
        margin-top: 5px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Safe Metadata Loading
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

# Fit TF-IDF resources for Semantic Querying
@st.cache_resource
def get_tfidf_resources(features):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(features)
    return vectorizer, matrix

vectorizer, tfidf_matrix = get_tfidf_resources(movies["combined_features"].tolist())

# Layout Header
st.title("🎬 CineMatch")
st.write("Experience advanced semantic search, multi-criteria filtering, and transparent similarity explanations.")
st.divider()

# Sidebar Filters
st.sidebar.header("🔎 Advanced Search & Filters")

# Text-based matching
search_title = st.sidebar.text_input("Title contains:", "")
search_director = st.sidebar.text_input("Director contains:", "")
search_actor = st.sidebar.text_input("Actor contains:", "")

# Year Slider
min_year = int(movies["year"].min()) if not movies.empty else 1970
max_year = int(movies["year"].max()) if not movies.empty else 2026
if min_year == max_year: min_year -= 1
year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))

# Dynamic Categorical Selectors
genres_list = sorted(list(set([g.strip() for sublist in movies["genres"].str.split(",") for g in sublist if g.strip()])))
selected_genre = st.sidebar.selectbox("Genre", ["All"] + genres_list)

languages_list = sorted(movies["language"].unique().tolist())
selected_lang = st.sidebar.selectbox("Language", ["All"] + languages_list)

certifications_list = sorted([x for x in movies["certification"].unique() if x])
selected_cert = st.sidebar.selectbox("Certification", ["All"] + certifications_list)

# Semantic Input
st.sidebar.divider()
st.sidebar.subheader("🧠 Semantic Match Engine")
user_prompt = st.sidebar.text_area("Describe your movie vibe:", placeholder="e.g. a dark, mind-bending space adventure with a dramatic relationship")

# Apply Filtering Pipeline
filtered_df = movies.copy()
if search_title.strip():
    filtered_df = filtered_df[filtered_df["title"].str.contains(search_title, case=False, na=False)]
if search_director.strip():
    filtered_df = filtered_df[filtered_df["directors"].str.contains(search_director, case=False, na=False)]
if search_actor.strip():
    filtered_df = filtered_df[filtered_df["actors"].str.contains(search_actor, case=False, na=False)]

filtered_df = filtered_df[
    (filtered_df["year"] >= year_range[0]) & 
    (filtered_df["year"] <= year_range[1])
]

if selected_genre != "All":
    filtered_df = filtered_df[filtered_df["genres"].str.contains(selected_genre, case=False, na=False)]
if selected_lang != "All":
    filtered_df = filtered_df[filtered_df["language"] == selected_lang]
if selected_cert != "All":
    filtered_df = filtered_df[filtered_df["certification"] == selected_cert]

# Displays results
st.header("🍿 Discover Movies")
if filtered_df.empty:
    st.warning("No movies matched your filters. Try widening your criteria.")
else:
    cols = st.columns(3)
    for idx, (_, movie) in enumerate(filtered_df.iterrows()):
        with cols[idx % 3]:
            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
            if movie["poster"].startswith("http"):
                st.image(movie["poster"], use_container_width=True)
            st.markdown(f"#### {movie['title']} ({int(movie['year'])})")
            st.markdown(f"⭐ **Rating:** {movie['rating']}/10  |  ⏱️ **Runtime:** {int(movie['runtime'])} mins")
            st.write(movie["overview"])
            st.markdown("</div>", unsafe_allow_html=True)

# Semantic Recommendations Block
if user_prompt.strip():
    st.divider()
    st.header("🎯 Intelligent Semantic Recommendations")
    prompt_vector = vectorizer.transform([user_prompt])
    scores = cosine_similarity(prompt_vector, tfidf_matrix).flatten()
    
    rec_movies = movies.copy()
    rec_movies["match_percent"] = scores * 100
    matched_results = rec_movies.sort_values(by="match_percent", ascending=False).head(3)
    
    rec_cols = st.columns(3)
    for j, (_, row) in enumerate(matched_results.iterrows()):
        with rec_cols[j]:
            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
            if row["poster"].startswith("http"):
                st.image(row["poster"], use_container_width=True)
            st.markdown(f"#### {row['title']}")
            st.markdown(f"<span class='match-tag'>{row['match_percent']:.1f}% Match</span>", unsafe_allow_html=True)
            st.markdown(f"<br>🎬 **Director:** {row['directors']}<br>🎭 **Genres:** {row['genres']}", unsafe_allow_html=True)
            st.write(row["overview"])
            st.markdown("</div>", unsafe_allow_html=True)
