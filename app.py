import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_data():
    movie_titles = pd.read_csv("movie_titles.csv")
    sample_ratings = pd.read_csv("sample_ratings.csv")
    return movie_titles, sample_ratings

@st.cache_resource
def load_model():
    with open("netflix_svd_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def get_recommendations_for_user(user_id, model, ratings_data, movie_data, n=10):
    all_movie_ids = ratings_data["movie_id"].unique()

    watched_movies = ratings_data[
        ratings_data["customer_id"] == user_id
    ]["movie_id"].unique()

    movies_to_predict = [
        movie_id for movie_id in all_movie_ids
        if movie_id not in watched_movies
    ]

    predictions = []

    for movie_id in movies_to_predict:
        predicted_rating = model.predict(user_id, movie_id).est
        predictions.append((movie_id, predicted_rating))

    recommendations = pd.DataFrame(
        predictions,
        columns=["movie_id", "predicted_rating"]
    )

    recommendations = recommendations.merge(
        movie_data,
        on="movie_id",
        how="left"
    )

    return recommendations.sort_values(
        by="predicted_rating",
        ascending=False
    ).head(n)

st.title("🎬 Netflix Recommendation System")
st.write("Personalized movie recommendations using collaborative filtering and SVD.")

movie_titles, sample_ratings = load_data()
model = load_model()

user_ids = sorted(sample_ratings["customer_id"].unique())

selected_user = st.selectbox(
    "Select a Netflix user ID",
    user_ids
)

n_recommendations = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

if st.button("Recommend Movies"):
    recommendations = get_recommendations_for_user(
        user_id=selected_user,
        model=model,
        ratings_data=sample_ratings,
        movie_data=movie_titles,
        n=n_recommendations
    )

    st.subheader("Top Movie Recommendations")
    st.dataframe(recommendations)