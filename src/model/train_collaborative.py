import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

class CollaborativeRecommendationModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):

        with open(
            context.artifacts["model"],
            "rb"
        ) as file:

            self.model_data = pickle.load(file)

    def predict(
        self,
        context,
        model_input
    ):

        return model_input
# -----------------------------
# MLflow Configuration
# -----------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Movie_Recommendation_Collaborative")


# -----------------------------
# Load Training Data
# -----------------------------

print("Loading ratings data...")

ratings = pd.read_csv(
    "data/processed/ratings_training.csv"
)

print("Ratings shape:", ratings.shape)


# -----------------------------
# Train-Test Split
# -----------------------------

train_ratings, test_ratings = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

print("Training ratings:", train_ratings.shape)
print("Testing ratings:", test_ratings.shape)


# -----------------------------
# Create User-Movie Matrix
# -----------------------------

user_ids = train_ratings["user_id"].unique()
movie_ids = train_ratings["movie_id"].unique()

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}


rows = train_ratings["user_id"].map(user_to_index)
cols = train_ratings["movie_id"].map(movie_to_index)
values = train_ratings["rating"]


rating_matrix = csr_matrix(
    (
        values,
        (rows, cols)
    ),
    shape=(
        len(user_ids),
        len(movie_ids)
    )
)


print(
    "User-movie matrix shape:",
    rating_matrix.shape
)


# -----------------------------
# Train Model
# -----------------------------

n_components = 50


with mlflow.start_run():

    mlflow.log_param(
        "n_components",
        n_components
    )

    mlflow.log_param(
        "number_of_users",
        len(user_ids)
    )

    mlflow.log_param(
        "number_of_movies",
        len(movie_ids)
    )

    mlflow.log_param(
        "training_ratings",
        len(train_ratings)
    )

    mlflow.log_param(
        "testing_ratings",
        len(test_ratings)
    )

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )

    user_latent_matrix = svd.fit_transform(
        rating_matrix
    )

    movie_latent_matrix = svd.components_.T


    explained_variance = (
        svd.explained_variance_ratio_.sum()
    )


    mlflow.log_metric(
        "explained_variance",
        float(explained_variance)
    )


    print("SVD training completed!")

    print(
        "User latent matrix:",
        user_latent_matrix.shape
    )

    print(
        "Movie latent matrix:",
        movie_latent_matrix.shape
    )

    print(
        "Explained variance:",
        explained_variance
    )


    # -----------------------------
    # Save Model
    # -----------------------------

    model_data = {
        "svd": svd,
        "user_latent_matrix": user_latent_matrix,
        "movie_latent_matrix": movie_latent_matrix,
        "user_to_index": user_to_index,
        "movie_to_index": movie_to_index,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "test_ratings": test_ratings
    }


    with open(
        "models/collaborative_model.pkl",
        "wb"
    ) as file:

        pickle.dump(
            model_data,
            file
        )
    mlflow.log_artifact(
        "models/collaborative_model.pkl",
        artifact_path="model"
    )
    mlflow.pyfunc.log_model(
        artifact_path="registered_model",
        python_model=CollaborativeRecommendationModel(),
        artifacts={
            "model": "models/collaborative_model.pkl"
        },
        registered_model_name="MovieRecommendationCollaborative"
    )


    print("Collaborative model saved!")

    print(
        "Location:",
        "models/collaborative_model.pkl"
    )


print("\nMLflow tracking completed successfully!")