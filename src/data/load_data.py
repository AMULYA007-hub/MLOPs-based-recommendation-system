import pandas as pd

ratings = pd.read_csv(
    "data/raw/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "data/raw/ml-1m/movies.dat",
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    encoding="latin-1"
)

print("Ratings:")
print(ratings.head())

print("\nMovies:")
print(movies.head())

print("\nRatings shape:", ratings.shape)
print("Movies shape:", movies.shape)