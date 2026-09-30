import os
import streamlit as st 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure your ml_pipeline.py is in the same directory
from ml_pipeline import load_model, recommend, search_movies, get_popular_movies

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Professional Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Styles */
    .main { background-color: #050505; color: #ffffff; }
    
    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background-color: rgba(20, 20, 20, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid #333;
    }

    /* Movie Card Styling */
    .movie-card {
        background: #111111;
        border: 1px solid #222;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .movie-card:hover {
        transform: translateY(-5px);
        border-color: #e50914;
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.2);
    }

    /* Text & Badges */
    .movie-title { color: #ffffff; font-size: 1.2rem; font-weight: 800; margin-bottom: 5px; }
    .movie-meta { color: #888; font-size: 0.9rem; margin-bottom: 8px; font-family: 'Helvetica', sans-serif; }
    
    .score-badge {
        background: linear-gradient(90deg, #e50914 0%, #b20710 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
    }
    
    .genre-tag {
        background: #222;
        color: #e50914;
        border: 1px solid #e50914;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 10px;
        margin-right: 5px;
        text-transform: uppercase;
        font-weight: 600;
    }

    .overview-text { 
        color: #bbbbbb; 
        font-size: 0.85rem; 
        margin-top: 12px; 
        line-height: 1.6;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: white;
        border: 1px solid #333;
        border-radius: 8px;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background: #111;
        border: 1px solid #222;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Powering up the recommendation engine...")
def get_model():
    return load_model()

@st.cache_data
def get_all_genres(_df):
    genres = set()
    for g in _df["Movie_Genre"].dropna():
        for part in g.split():
            genres.add(part.strip())
    return sorted(genres)

# ─── Render Movie Card ────────────────────────────────────────────────────────
def render_movie_card(row, show_score=True, rank=None):
    title    = row.get("Movie_Title", "Unknown")
    genre    = row.get("Movie_Genre", "")
    director = row.get("Movie_Director", "N/A")
    vote     = row.get("Movie_Vote", 0)
    
    # Robust year extraction
    raw_date = str(row.get("Movie_Release_Date", ""))
    year = raw_date[-4:] if len(raw_date) >= 4 else "N/A"
    
    overview = row.get("Movie_Overview", "")
    cast     = row.get("Movie_Cast", "")
    score    = row.get("hybrid_score", row.get("weighted_score", 0))

    stars = "⭐" * int(round(vote / 2)) if vote else ""
    rank_html = f'<span style="color:#e50914">{rank}.</span> ' if rank else ""

    genre_tags = "".join(
        f'<span class="genre-tag">{g.replace("_", " ")}</span>'
        for g in genre.split()[:3]
    )
    
    score_html = f'<div class="score-badge">Match Score: {score:.1%}</div>' if (show_score and score > 0) else ""
    
    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-title">{rank_html}{title} <span style="color:#666; font-weight:400">({year})</span></div>
        <div class="movie-meta"><b>{director}</b> &nbsp;•&nbsp; {stars} {vote}/10</div>
        <div style="margin: 8px 0">{genre_tags}</div>
        <div class="overview-text">{overview}</div>
        {score_html}
    </div>
    """, unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150) # Example logo
    st.markdown("### **AI Recommender**")
    st.markdown("---")

    page = st.selectbox(
        "Navigation",
        ["🔍 Discovery", "🏆 Trending", "📊 Insights"]
    )
    
    st.markdown("---")

    try:
        model   = get_model()
        df      = model["df"]
        sim_mat = model["sim_matrix"]
        genres  = ["All Genres"] + get_all_genres(df)
        st.success(f"Library: {len(df):,} Movies")
    except Exception as e:
        st.error(f"Engine Error: {e}")
        st.stop()

    st.markdown("#### 🛠️ Refine Results")
    genre_filter = st.selectbox("Genre Preference", genres)
    min_vote     = st.slider("Minimum Rating", 0.0, 10.0, 5.0)
    top_n        = st.slider("Display Limit", 5, 30, 12)

# ─── Page: Discovery ──────────────────────────────────────────────────────────
if page == "🔍 Discovery":
    st.title("Find Your Next Obsession")
    
    query = st.text_input("Search for a movie you love:", placeholder="Inception, Interstellar, The Godfather...")

    if query:
        search_results = search_movies(query, df, top_n=8)

        if search_results.empty:
            st.warning(f"We couldn't find '{query}'. Try another title!")
        else:
            col_sel, col_info = st.columns([1, 1])
            
            with col_sel:
                selected_title = st.selectbox("Pick the closest match:", search_results["Movie_Title"].tolist())
            
            movie_info = df[df["Movie_Title"] == selected_title].iloc[0]
            
            st.markdown("---")
            st.subheader(f"Recommendations based on {selected_title}")

            recs = recommend(
                selected_title, df, sim_mat,
                top_n=top_n,
                genre_filter=genre_filter if "All" not in genre_filter else None,
                min_vote=min_vote,
            )

            if recs.empty:
                st.info("No matches found. Try lowering the 'Minimum Rating' in the sidebar.")
            else:
                # Layout cards in a grid
                idx = 0
                for _ in range(len(recs) // 2 + 1):
                    cols = st.columns(2)
                    for col in cols:
                        if idx < len(recs):
                            with col:
                                render_movie_card(recs.iloc[idx], show_score=True, rank=idx+1)
                            idx += 1

    else:
        st.markdown("### 🍿 Trending Picks")
        trending = get_popular_movies(df, top_n=6)
        t_cols = st.columns(3)
        for i, (_, row) in enumerate(trending.iterrows()):
            with t_cols[i % 3]:
                render_movie_card(row, show_score=False)

# ─── Page: Trending ───────────────────────────────────────────────────────────
elif page == "🏆 Trending":
    st.title("🏆 Critically Acclaimed")
    
    g = genre_filter if "All" not in genre_filter else None
    top_movies = get_popular_movies(df, genre=g, top_n=top_n)

    # Key Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Library Size", f"{len(df):,}")
    m2.metric("Avg Quality", f"{df['Movie_Vote'].mean():.1f}/10")
    m3.metric("Top Genre", genre_filter)

    st.markdown("---")
    
    # Grid display
    idx = 0
    for _ in range(len(top_movies) // 3 + 1):
        cols = st.columns(3)
        for col in cols:
            if idx < len(top_movies):
                with col:
                    render_movie_card(top_movies.iloc[idx], show_score=False, rank=idx+1)
                idx += 1

# ─── Page: Insights ───────────────────────────────────────────────────────────
elif page == "📊 Insights":
    st.title("📊 Cinematic Data Analytics")
    
    c1, c2 = st.columns(2)

    with c1:
        # Rating Dist
        fig = px.histogram(
            df[df["Movie_Vote"] > 0], x="Movie_Vote", 
            nbins=30, title="Global Rating Distribution",
            color_discrete_sequence=["#e50914"], template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Genre Distribution
        genre_counts = {}
        for g in df["Movie_Genre"].dropna():
            for part in g.split():
                genre_counts[part] = genre_counts.get(part, 0) + 1
        genre_df = pd.DataFrame(list(genre_counts.items()), columns=["Genre", "Count"]).sort_values("Count", ascending=False).head(10)
        
        fig = px.pie(genre_df, values='Count', names='Genre', title='Top 10 Genres',
                     color_discrete_sequence=px.colors.sequential.Reds_r, hole=.4)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Year Trend
    df["year"] = pd.to_numeric(df["Movie_Release_Date"].str[-4:], errors="coerce")
    year_counts = df[df["year"] >= 1980]["year"].value_counts().sort_index()
    fig_line = px.area(x=year_counts.index, y=year_counts.values, title="Releases Over Time (Post-1980)",
                       labels={'x': 'Year', 'y': 'Movies'}, color_discrete_sequence=["#e50914"])
    fig_line.update_layout(template="plotly_dark")
    st.plotly_chart(fig_line, use_container_width=True)

    st.info("💡 Data is processed using a TF-IDF Vectorizer and Cosine Similarity for content matching.")