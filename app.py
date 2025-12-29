import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD

#Page config
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("Hybrid Movie Recommender System")
st.markdown("""
This application combines **Content-Based Filtering** (genres) ja **Collaborative Filtering** (user-data) 
models to offer the best possible movie recommendations.
""")

#Data loading and model training from cache
@st.cache_resource
def load_data_and_train_models():
    try:
        movies = pd.read_csv('./data/movies.csv')
        ratings = pd.read_csv('./data/ratings.csv')
    except FileNotFoundError:
        st.error(".csv files not found.")
        return None, None, None, None, None

    #Content based (TF-IDF)
    movies['genres_str'] = movies['genres'].str.replace('|', ' ')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres_str'])
    
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    #Collaborative (SVD)
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    
    #Train SVD (Matrix Factorization)
    svd = TruncatedSVD(n_components=20, random_state=42)
    matrix_svd = svd.fit_transform(user_movie_matrix)
    
    corr_matrix = np.corrcoef(matrix_svd)

    return movies, ratings, indices, cosine_sim, user_movie_matrix, corr_matrix

with st.spinner('Training models...'):
    movies, ratings, indices, cosine_sim, user_movie_matrix, corr_matrix = load_data_and_train_models()

def get_content_recommendations(title, movies, indices, cosine_sim):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

def get_collaborative_recommendations(user_id, user_movie_matrix, corr_matrix, movies, top_k=5):
    if user_id not in user_movie_matrix.index:
        return []
    
    user_idx = user_movie_matrix.index.get_loc(user_id)
    similar_users = list(enumerate(corr_matrix[user_idx]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
    
    top_similar_users_indices = [x[0] for x in similar_users[1:11]]
    
    similar_users_ratings = user_movie_matrix.iloc[top_similar_users_indices]
    mean_ratings = similar_users_ratings.mean(axis=0)
    top_movies = mean_ratings.sort_values(ascending=False)

    user_seen_movies = user_movie_matrix.loc[user_id]
    seen_mask = user_seen_movies > 0
    top_movies = top_movies[~seen_mask]
    
    recommended_movie_ids = top_movies.head(top_k).index
    return movies[movies['movieId'].isin(recommended_movie_ids)]['title'].tolist()

def hybrid_recommendation(user_id):
    collab_recs = get_collaborative_recommendations(user_id, user_movie_matrix, corr_matrix, movies)
    
    user_ratings = ratings[ratings['userId'] == user_id]
    if user_ratings.empty:
         return ["New user , not enough data"], "No data"

    user_ratings = user_ratings.sort_values('rating', ascending=False)
    top_movie_id = user_ratings.iloc[0]['movieId']
    fav_movie_title = movies[movies['movieId'] == top_movie_id]['title'].values[0]
    
    content_recs = get_content_recommendations(fav_movie_title, movies, indices, cosine_sim)
    
    combined = list(set(collab_recs + content_recs))
    return combined, fav_movie_title

#Ui

if movies is not None:
    st.sidebar.header("Selections")
    
    all_users = ratings['userId'].unique()
    selected_user = st.sidebar.selectbox("Choose user (ID)", all_users[:50])

    if st.sidebar.button("Recommend"):
        st.subheader(f"Results for user {selected_user}")
        
        recs, fav_movie = hybrid_recommendation(selected_user)
        
        st.info(f"Based on analysis the user's favourite movie is: **{fav_movie}**")
        
        st.write("### Recommended movies:")
        
        for i, movie in enumerate(recs, 1):
            st.success(f"{i}. {movie}")

    with st.expander("Show dataset statistics"):
        st.write(f"Movies in total: {len(movies)}")
        st.write(f"Ratings in total: {len(ratings)}")
        st.write("Example of data:")
        st.dataframe(movies.head())