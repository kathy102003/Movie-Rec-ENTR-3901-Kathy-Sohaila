
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page setup
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

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

# Website heading
st.title("🎬 CineMatch")
st.subheader("Your Personalized Movie Finder")
st.write("Discover movies, explore genres, and find your next favourite film with explainable recommendations!")

st.divider()

# Sidebar filters
st.sidebar.header("🔎 Find Your Movie")
search = st.sidebar.text_input("Search movie title")

genre_options = sorted({
    genre.strip()
    for genres in movies["genres"]
    for genre in genres.split(",")
    if genre.strip()
})

selected_genres = st.sidebar.multiselect(
    "Choose genres",
    genre_options
)

min_year = int(movies["year"].min())
max_year = int(movies["year"].max())

year_range = st.sidebar.slider(
    "Release year",
    min_year,
    max_year,
    (min_year, max_year)
)

min_rating = st.sidebar.slider(
    "Minimum rating",
    0.0, 10.0, 0.0, 0.5
)

min_votes = st.sidebar.number_input(
    "Minimum votes",
    min_value=0,
    value=0
)

max_runtime = st.sidebar.number_input(
    "Maximum runtime (minutes)",
    min_value=0,
    value=0
)

language_options = sorted(
    movies["language"].unique().tolist()
)

selected_language = st.sidebar.selectbox(
    "Language",
    ["All"] + language_options
)

certification_options = sorted(
    [x for x in movies["certification"].unique() if x]
)

selected_certification = st.sidebar.selectbox(
    "Certification",
    ["All"] + certification_options
)

actor_search = st.sidebar.text_input("Actor name")
director_search = st.sidebar.text_input("Director name")

# Apply filters
filtered = movies.copy()

if search:
    filtered = filtered[
        filtered["title"].str.contains(search, case=False, na=False)
    ]

if selected_genres:
    filtered = filtered[
        filtered["genres"].apply(
            lambda value: all(g in value for g in selected_genres)
        )
    ]

filtered = filtered[
    (filtered["year"] >= year_range[0]) &
    (filtered["year"] <= year_range[1]) &
    (filtered["rating"] >= min_rating) &
    (filtered["vote_count"] >= min_votes)
]

if max_runtime > 0:
    filtered = filtered[filtered["runtime"] <= max_runtime]

if selected_language != "All":
    filtered = filtered[filtered["language"] == selected_language]

if selected_certification != "All":
    filtered = filtered[
        filtered["certification"] == selected_certification
    ]

if actor_search:
    filtered = filtered[
        filtered["actors"].str.contains(actor_search, case=False, na=False)
    ]

if director_search:
    filtered = filtered[
        filtered["directors"].str.contains(director_search, case=False, na=False)
    ]

# Display movie results
st.header("🍿 Explore Movies")
st.write(f"Found **{len(filtered)} movies**")

if filtered.empty:
    st.warning("No movies found. Try changing your filters.")
else:
    columns = st.columns(3)
    for i, (_, movie) in enumerate(filtered.iterrows()):
        with columns[i % 3]:
            if movie["poster"].startswith("http"):
                st.image(movie["poster"], use_container_width=True)
            st.subheader(movie["title"])
            st.write(f"📅 Year: {int(movie['year']) if movie['year'] else 'Unknown'}")
            st.write(f"⭐ Rating: {movie['rating']}/10")
            st.write(f"🎭 Genres: {movie['genres']}")
            st.write(f"⏱️ Runtime: {int(movie['runtime'])} minutes")
            st.write(movie["overview"] or "No description available.")
            st.divider()

# Similar movie recommendations with clear matching metrics
st.header("✨ Find Similar Movies (Explainable AI Engine)")

movie_titles = sorted(movies["title"].dropna().unique())
selected_movie = st.selectbox(
    "Choose a movie you like",
    movie_titles
)

number_to_show = st.slider(
    "Number of recommendations",
    1, 10, 5
)

if st.button("Recommend Movies"):
    selected_row = movies[movies["title"] == selected_movie].iloc[0]
    movie_index = movies[movies["title"] == selected_movie].index[0]
    
    scores = list(enumerate(cosine_sim[movie_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    recommendations = [
        item for item in scores if item[0] != movie_index
    ][:number_to_show]
    
    st.subheader(f"Movies similar to {selected_movie}")
    
    rec_columns = st.columns(3)
    for i, (index, score) in enumerate(recommendations):
        rec_movie = movies.iloc[index]
        
        # Explainer calculations:
        source_genres = set([g.strip().lower() for g in selected_row["genres"].split(",") if g.strip()])
        rec_genres = set([g.strip().lower() for g in rec_movie["genres"].split(",") if g.strip()])
        shared_genres = source_genres.intersection(rec_genres)
        
        source_actors = set([a.strip().lower() for a in selected_row["actors"].split(",") if a.strip()])
        rec_actors = set([a.strip().lower() for a in rec_movie["actors"].split(",") if a.strip()])
        shared_actors = source_actors.intersection(rec_actors)
        
        source_keywords = set([k.strip().lower() for k in selected_row["keywords"].split(",") if k.strip()])
        rec_keywords = set([k.strip().lower() for k in rec_movie["keywords"].split(",") if k.strip()])
        shared_keywords = source_keywords.intersection(rec_keywords)
        
        shared_director = False
        if selected_row["directors"] and rec_movie["directors"]:
            s_dirs = set([d.strip().lower() for d in selected_row["directors"].split(",") if d.strip()])
            r_dirs = set([d.strip().lower() for d in rec_movie["directors"].split(",") if d.strip()])
            shared_director = len(s_dirs.intersection(r_dirs)) > 0
            
        with rec_columns[i % 3]:
            if rec_movie["poster"].startswith("http"):
                st.image(rec_movie["poster"], use_container_width=True)
            st.write(f"### **{rec_movie['title']}**")
            st.write(f"⭐ Rating: {rec_movie['rating']}/10")
            st.write(f"🎭 {rec_movie['genres']}")
            st.write(f"**Similarity match:** {score * 100:.1f}%")
            
            # Dynamic explanation block
            st.info("🔍 **Why is this recommended?**")
            reasons = []
            if shared_genres:
                reasons.append(f"• Shares genres: **{', '.join(list(shared_genres)).title()}**")
            if shared_director:
                reasons.append(f"• Directorial overlap: Directed by the same director (**{rec_movie['directors']}**)")
            if shared_actors:
                reasons.append(f"• Actor overlap: Features shared cast: **{', '.join(list(shared_actors)).title()}**")
            if shared_keywords:
                reasons.append(f"• Story elements: Both stories involve themes of **{', '.join(list(shared_keywords)[:4]).title()}**")
                
            if not reasons:
                reasons.append("• Matched via contextual plot/overview semantics.")
                
            st.markdown("\n".join(reasons))
            st.divider()

# Footer
st.divider()
st.caption(
    "CineMatch | This product uses the TMDB API but is not endorsed "
    "or certified by TMDB."
)
