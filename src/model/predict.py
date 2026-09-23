import pickle

from src.utils.tokenizer import genre_tokenizer


# --------------------------------------------------
# 1. Load trained model
# --------------------------------------------------

with open("models/recommender.pkl", "rb") as file:
    model_data = pickle.load(file)


movies = model_data["movies"]
similarity_matrix = model_data["similarity_matrix"]


# --------------------------------------------------
# 2. Recommendation function
# --------------------------------------------------

def recommend_movies(title, number_of_recommendations=10):

    # Find the movie row directly
    matching_movies = movies[
        movies["title"] == title
        ]

    if matching_movies.empty:
        return []

    index = matching_movies.index[0]

    # Calculate similarity scores
    similarity_scores = list(
        enumerate(similarity_matrix[index])
    )

    # Sort from highest similarity to lowest
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the movie itself
    similarity_scores = [
        score for score in similarity_scores
        if score[0] != index
    ]

    # Take requested number
    similarity_scores = similarity_scores[
        :number_of_recommendations
    ]

    # Get movie indices
    movie_indices_list = [
        score[0]
        for score in similarity_scores
    ]

    # Get recommendations
    recommendations = movies.iloc[
        movie_indices_list
    ][
        ["movie_id", "title", "genres"]
    ]

    return recommendations.to_dict("records")


# --------------------------------------------------
# 3. Test recommendation
# --------------------------------------------------

test_movie = "Toy Story (1995)"

recommendations = recommend_movies(
    test_movie,
    5
)

print("Recommendations for:", test_movie)

for movie in recommendations:
    print(
        movie["movie_id"],
        "-",
        movie["title"],
        "|",
        movie["genres"]
    )