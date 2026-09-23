import pandas as pd

from database.connection import engine


print("Loading ratings from PostgreSQL...")

query = """
SELECT
    user_id,
    movie_id,
    rating,
    timestamp
FROM ratings
"""

ratings = pd.read_sql(
    query,
    engine
)

print("Ratings loaded successfully!")
print("Shape:", ratings.shape)

print("\nFirst 5 rows:")
print(ratings.head())

ratings.to_csv(
    "data/processed/ratings_training.csv",
    index=False
)

print("\nTraining dataset saved!")
print("Location: data/processed/ratings_training.csv")