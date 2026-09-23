import os
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.engine import URL


# -----------------------------
# PostgreSQL connection
# -----------------------------

DB_USER = "postgres"
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "movie_recommendation"

database_url = URL.create(
    drivername="postgresql+psycopg",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME
)

engine = create_engine(database_url)


# -----------------------------
# File paths
# -----------------------------

users_file = "data/raw/ml-1m/users.dat"
movies_file = "data/raw/ml-1m/movies.dat"
ratings_file = "data/raw/ml-1m/ratings.dat"


# -----------------------------
# Load users
# -----------------------------

print("Loading users...")

users = pd.read_csv(
    users_file,
    sep="::",
    engine="python",
    names=[
        "user_id",
        "gender",
        "age",
        "occupation",
        "zip_code"
    ],
    encoding="latin-1"
)

users.to_sql(
    "users",
    engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=1000
)

print(f"Users inserted: {len(users)}")


# -----------------------------
# Load movies
# -----------------------------

print("Loading movies...")

movies = pd.read_csv(
    movies_file,
    sep="::",
    engine="python",
    names=[
        "movie_id",
        "title",
        "genres"
    ],
    encoding="latin-1"
)

movies.to_sql(
    "movies",
    engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=1000
)

print(f"Movies inserted: {len(movies)}")


# -----------------------------
# Load ratings
# -----------------------------

print("Loading ratings...")

ratings = pd.read_csv(
    ratings_file,
    sep="::",
    engine="python",
    names=[
        "user_id",
        "movie_id",
        "rating",
        "timestamp"
    ],
    encoding="latin-1"
)

ratings.to_sql(
    "ratings",
    engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=5000
)

print(f"Ratings inserted: {len(ratings)}")


print("\n================================")
print("MovieLens data loaded successfully!")
print("================================")