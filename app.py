import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import re
import streamlit as st

# ===============================
# Load Data
# ===============================
movies = pd.read_csv("data/movies.csv")           # movieId, title
ratings = pd.read_csv("data/ratings.csv")         # userId, movieId, rating, timestamp
final_dataset = pd.read_csv("data/cleaned_data.csv")  # Pivoted user × movie ratings

# movieId is the index for the final datset 
if "movieId" in final_dataset.columns:
    final_dataset.set_index("movieId", inplace=True)

# CSR matrix for KNN . CSR matrix is more efficient for sparse data it uses less memory and is faster for matrix operations
sr_data = csr_matrix(final_dataset.values)

# ===============================
# Train KNN model
# ===============================
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(sr_data)

# ===============================
# Calculate average rating and votes per movie
# ===============================
avg_ratings = ratings.groupby("movieId")["rating"].mean().round(1)
num_votes = ratings.groupby("movieId")["rating"].count()

# ===============================
# Helper function: get recommendations
# ===============================
def get_movie_recommendation(movie_name, n_movies_to_reccomend=5):
    #clean the input
    movie_name = movie_name.strip().lower()
    ##ignore the input if its less than 2 characters or contains only special characters
    if len(movie_name) < 2 or re.match(r'^[\W_]+$', movie_name):
        return None
##search for the movie in the movies dataset

    movie_list = movies[movies['title'].str.lower().str.contains(movie_name, regex=True)]
    
    if len(movie_list) == 0:
        return None
    
    ## take the first matched movie
    movie_id = movie_list.iloc[0]['movieId']
    
##make sure the movie exists in the final_dataset
    if movie_id not in final_dataset.index:
        return None
    ## get the index of the movie in the final_dataset
    movie_idx = final_dataset.index.get_loc(movie_id)
    
## find the nearest neighbors for the movie
    distances, indices = knn.kneighbors(sr_data[movie_idx], n_neighbors=n_movies_to_reccomend+1)
    
    rec_movie_indices = indices.squeeze().tolist()[1:] # Exclude the first one as it is the movie itself
    
    recommend_frame = []
    for idx in rec_movie_indices:
        rec_movie_id = final_dataset.index[idx]
        title = movies[movies['movieId'] == rec_movie_id]['title'].values[0]
        rating = avg_ratings.get(rec_movie_id, 0)
        votes = num_votes.get(rec_movie_id, 0)
        recommend_frame.append({'Title': title, 'Rating': rating, 'Votes': votes})
    
    return pd.DataFrame(recommend_frame, index=range(1, n_movies_to_reccomend+1))

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Dark theme CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
        color: #ffffff;
    }
    .movie-card {
        background-color: #1f2a38;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        text-align: center;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.6);
        min-width: 180px;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: scale(1.05);
    }
    .movie-title {
        font-size: 18px;
        font-weight: bold;
        margin-top: 8px;
        color: #f5f5f5;
    }
    .movie-rating {
        font-size: 16px;
        color: #ffd700;
        margin-top: 5px;
    }
    .movie-votes {
        font-size: 12px;
        color: #cccccc;
        margin-top: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🎬 Movie Recommender System")
st.subheader("🔎 Search for a movie")
movie_name = st.text_input("", placeholder="Type movie title here...")

# Results
if movie_name:
    recs = get_movie_recommendation(movie_name, n_movies_to_reccomend=5)
    if recs is not None and not recs.empty:
        st.subheader("📌 Recommended Movies:")
        cols = st.columns(5)
        for i, row in recs.iterrows():
            with cols[i % 5]:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="movie-title">{row["Title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="movie-rating">⭐ {row["Rating"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="movie-votes">Votes: {row["Votes"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("⚠️ No movies found. Please try another title.")
