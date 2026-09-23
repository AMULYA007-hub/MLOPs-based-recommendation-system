import pandas as pd
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.tokenizer import genre_tokenizer
# --------------------------------------------------
# 1. Load processed data
# --------------------------------------------------

data = pd.read_csv("data/processed/movies_ratings.csv")

print("Data loaded successfully!")
print("Shape:", data.shape)


# --------------------------------------------------
# 2. Create movie-level dataset
# --------------------------------------------------

movies = data[["movie_id", "title", "genres"]].drop_duplicates(
    subset="movie_id"
).reset_index(drop=True)

print("Number of unique movies:", len(movies))


# --------------------------------------------------
# 3. Clean genres
# --------------------------------------------------

movies["genres"] = movies["genres"].fillna("")


# --------------------------------------------------
# 4. Convert genres into numerical features
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    tokenizer=genre_tokenizer,
    token_pattern=None
)

genre_matrix = vectorizer.fit_transform(movies["genres"])

print("Genre matrix shape:", genre_matrix.shape)


# --------------------------------------------------
# 5. Calculate movie similarity
# --------------------------------------------------

similarity_matrix = cosine_similarity(genre_matrix)

print("Similarity matrix created!")
print("Similarity matrix shape:", similarity_matrix.shape)


# --------------------------------------------------
# 6. Create movie index
# --------------------------------------------------

movie_indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()


# --------------------------------------------------
# 7. Recommendation function
# --------------------------------------------------

def recommend_movies(title, number_of_recommendations=10):

    if title not in movie_indices:
        return []

    index = movie_indices[title]

    similarity_scores = list(
        enumerate(similarity_matrix[index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:number_of_recommendations + 1]

    movie_indices_list = [
        score[0] for score in similarity_scores
    ]

    recommendations = movies.iloc[
        movie_indices_list
    ][["movie_id", "title", "genres"]]

    return recommendations.to_dict("records")


# --------------------------------------------------
# 8. Test the recommendation system
# --------------------------------------------------

test_movie = "Toy Story (1995)"

recommendations = recommend_movies(
    test_movie,
    10
)

print("\nRecommendations for:", test_movie)

for movie in recommendations:
    print(
        movie["movie_id"],
        "-",
        movie["title"],
        "|",
        movie["genres"]
    )


# --------------------------------------------------
# 9. Save model artifacts
# --------------------------------------------------

Path("models").mkdir(
    parents=True,
    exist_ok=True
)

model_data = {
    "movies": movies,
    "vectorizer": vectorizer,
    "similarity_matrix": similarity_matrix,
    "movie_indices": movie_indices
}

with open("models/recommender.pkl", "wb") as file:
    pickle.dump(model_data, file)


print("\nModel saved successfully!")
print("Location: models/recommender.pkl")