import pickle
import numpy as np

from src.model.predict import recommend_movies


# -----------------------------
# Load Collaborative Model
# -----------------------------

with open(
    "models/collaborative_model.pkl",
    "rb"
) as file:
    collaborative_model = pickle.load(file)


user_latent_matrix = collaborative_model["user_latent_matrix"]
movie_latent_matrix = collaborative_model["movie_latent_matrix"]

user_to_index = collaborative_model["user_to_index"]
movie_ids = collaborative_model["movie_ids"]
def get_collaborative_scores(user_id):

    if user_id not in user_to_index:
        return {}

    user_index = user_to_index[user_id]

    scores = np.dot(
        user_latent_matrix[user_index],
        movie_latent_matrix.T
    )

    return {
        int(movie_ids[index]): float(scores[index])
        for index in range(len(movie_ids))
    }
def get_content_recommendations(movie_title, limit=20):

    recommendations = recommend_movies(
        movie_title,
        limit
    )

    return recommendations
def hybrid_recommend(
    user_id,
    movie_title,
    limit=10,
    collaborative_weight=0.5,
    content_weight=0.5
):

    collaborative_scores = get_collaborative_scores(user_id)

    content_recommendations = get_content_recommendations(
        movie_title,
        limit=20
    )

    if not collaborative_scores:
        return content_recommendations[:limit]

    if not content_recommendations:
        return []


    # Normalize collaborative scores

    collab_values = np.array(
        list(collaborative_scores.values())
    )

    collab_min = collab_values.min()
    collab_max = collab_values.max()

    if collab_max != collab_min:

        normalized_collab = {
            movie_id:
            (score - collab_min) /
            (collab_max - collab_min)

            for movie_id, score
            in collaborative_scores.items()
        }

    else:

        normalized_collab = {
            movie_id: 0.0
            for movie_id
            in collaborative_scores.items()
        }


    # Create hybrid scores

    hybrid_scores = []

    for movie in content_recommendations:

        movie_id = movie["movie_id"]

        content_score = movie.get(
            "similarity_score",
            0.0
        )

        collaborative_score = normalized_collab.get(
            movie_id,
            0.0
        )

        hybrid_score = (
            collaborative_weight *
            collaborative_score
            +
            content_weight *
            content_score
        )

        hybrid_scores.append({

            "movie_id": movie_id,

            "title": movie.get(
                "title"
            ),

            "genres": movie.get(
                "genres"
            ),

            "hybrid_score": round(
                float(hybrid_score),
                4
            )

        })


    # Sort recommendations

    hybrid_scores.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )


    return hybrid_scores[:limit]
if __name__ == "__main__":

    test_user = 1
    test_movie = "Toy Story (1995)"

    recommendations = hybrid_recommend(
        test_user,
        test_movie,
        limit=5
    )

    print("\nHybrid Recommendations:")

    for movie in recommendations:

        print(
            movie["movie_id"],
            "|",
            movie["title"],
            "|",
            movie["genres"],
            "| Hybrid Score:",
            movie["hybrid_score"]
        )
