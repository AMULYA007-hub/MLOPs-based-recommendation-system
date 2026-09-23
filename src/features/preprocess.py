import pandas as pd
from pathlib import Path

# Load ratings
ratings = pd.read_csv(
    "data/raw/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Load movies
movies = pd.read_csv(
    "data/raw/ml-1m/movies.dat",
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    encoding="latin-1"
)

# Merge ratings with movie information
data = ratings.merge(movies, on="movie_id")

# Remove missing values
data = data.dropna()

# Create processed directory if it doesn't exist
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Save processed data
data.to_csv(
    "data/processed/movies_ratings.csv",
    index=False
)

print("Preprocessing completed successfully!")
print("Processed data shape:", data.shape)
print("\nColumns:")
print(data.columns.tolist())
print("\nFirst 5 rows:")
print(data.head())