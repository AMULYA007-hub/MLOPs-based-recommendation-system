import pickle
import numpy as np

from sqlalchemy import text
from database.connection import engine


# Load collaborative model
with open(
    "models/collaborative_model.pkl",
    "rb"
) as file:
    model_data = pickle.load(file)


user_latent_matrix = model_data["user_latent_matrix"]
movie_latent_matrix = model_data["movie_latent_matrix"]

user_to_index = model_data["user_to_index"]
movie_to_index = model_data["movie_to_index"]
movie_ids = model_data["movie_ids"]


def recommend_for_user(
    user_id,
    number_of_recommendations=10
):

    # Check whether user exists in the trained model
    if user_id not in user_to_index:
        return []

    user_index = user_to_index[user_id]

    # Calculate predicted preference scores
    scores = np.dot(
        user_latent_matrix[user_index],
        movie_latent_matrix.T
    )

    # Get movies already rated by this user
    query = text("""
        SELECT movie_id
        FROM ratings
        WHERE user_id = :user_id
    """)

    with engine.connect() as connection:
        result = connection.execute(
            query,
            {"user_id": user_id}
        )

        rated_movie_ids = {
            row[0]
            for row in result
        }

    # Sort movies by predicted score
    recommended_indices = np.argsort(
        scores
    )[::-1]

    recommendations = []

    for index in recommended_indices:

        movie_id = int(movie_ids[index])

        # Don't recommend movies already rated
        if movie_id in rated_movie_ids:
            continue

        recommendations.append({
            "movie_id": movie_id,
            "predicted_score": round(
                float(scores[index]),
                3
            )
        })

        if len(recommendations) >= number_of_recommendations:
            break

    # Get movie details from PostgreSQL
    if not recommendations:
        return []

    recommended_movie_ids = [
        movie["movie_id"]
        for movie in recommendations
    ]

    movie_query = text("""
        SELECT movie_id, title, genres
        FROM movies
        WHERE movie_id = ANY(:movie_ids)
    """)

    with engine.connect() as connection:
        result = connection.execute(
            movie_query,
            {"movie_ids": recommended_movie_ids}
        )

        movie_details = {
            row.movie_id: {
                "title": row.title,
                "genres": row.genres
            }
            for row in result
        }

    # Add title and genres to recommendations
    for movie in recommendations:

        details = movie_details.get(
            movie["movie_id"]
        )

        if details:
            movie["title"] = details["title"]
            movie["genres"] = details["genres"]

    return recommendations


# Test
test_user = 1

recommendations = recommend_for_user(
    test_user,
    10
)

print(
    "Recommendations for user:",
    test_user
)

for movie in recommendations:
    print(
        movie["movie_id"],
        "|",
        movie["title"],
        "|",
        movie["genres"],
        "| Score:",
        movie["predicted_score"]
    )