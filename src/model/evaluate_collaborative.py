import pickle
import json
from pathlib import Path

import numpy as np
import mlflow

from sklearn.metrics import mean_squared_error, mean_absolute_error


# -----------------------------
# MLflow Configuration
# -----------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment(
    "Movie_Recommendation_Collaborative"
)


# -----------------------------
# Load Collaborative Model
# -----------------------------

print("Loading collaborative model...")

with open(
    "models/collaborative_model.pkl",
    "rb"
) as file:
    model_data = pickle.load(file)


user_latent_matrix = model_data["user_latent_matrix"]
movie_latent_matrix = model_data["movie_latent_matrix"]

user_to_index = model_data["user_to_index"]
movie_to_index = model_data["movie_to_index"]

test_ratings = model_data["test_ratings"]


print(
    "Test ratings:",
    test_ratings.shape
)


# -----------------------------
# Calculate Predictions
# -----------------------------

actual_ratings = []
predicted_ratings = []


print("Calculating predictions...")


for row in test_ratings.itertuples():

    user_id = row.user_id
    movie_id = row.movie_id
    actual_rating = row.rating

    if user_id not in user_to_index:
        continue

    if movie_id not in movie_to_index:
        continue

    user_index = user_to_index[user_id]
    movie_index = movie_to_index[movie_id]

    predicted_rating = np.dot(
        user_latent_matrix[user_index],
        movie_latent_matrix[movie_index]
    )

    actual_ratings.append(actual_rating)
    predicted_ratings.append(predicted_rating)


# -----------------------------
# Calculate Metrics
# -----------------------------

rmse = np.sqrt(
    mean_squared_error(
        actual_ratings,
        predicted_ratings
    )
)

mae = mean_absolute_error(
    actual_ratings,
    predicted_ratings
)


# -----------------------------
# Save Metrics for DVC
# -----------------------------

Path("metrics").mkdir(
    parents=True,
    exist_ok=True
)

metrics = {
    "rmse": float(rmse),
    "mae": float(mae)
}

with open(
    "metrics/collaborative_metrics.json",
    "w"
) as file:
    json.dump(
        metrics,
        file,
        indent=4
    )

print(
    "\nMetrics saved for DVC:",
    "metrics/collaborative_metrics.json"
)


# -----------------------------
# Log Evaluation to MLflow
# -----------------------------

with mlflow.start_run():

    mlflow.log_param(
        "evaluation_type",
        "test_set"
    )

    mlflow.log_param(
        "test_ratings",
        len(test_ratings)
    )

    mlflow.log_metric(
        "test_rmse",
        float(rmse)
    )

    mlflow.log_metric(
        "test_mae",
        float(mae)
    )

    mlflow.log_metric(
        "ratings_evaluated",
        len(actual_ratings)
    )


# -----------------------------
# Display Results
# -----------------------------

print("\nEvaluation completed!")

print(
    "Ratings evaluated:",
    len(actual_ratings)
)

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "MAE:",
    round(mae, 4)
)

print("\nMLflow evaluation metrics logged successfully!")