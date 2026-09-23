from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.model.predict import recommend_movies
from src.model.predict_collaborative import recommend_for_user

from database.session import get_db
from database.models import Movie, Rating

class RatingRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float

app = FastAPI(
    title="Movie Recommendation API",
    description="MLOps-based Movie Recommendation System",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Movie Recommendation API is running"
    }


@app.get("/recommend/{movie_title}")
def recommend(movie_title: str, limit: int = 5):

    recommendations = recommend_movies(
        movie_title,
        limit
    )

    if not recommendations:
        return {
            "movie": movie_title,
            "recommendations": [],
            "message": "Movie not found"
        }

    return {
        "movie": movie_title,
        "recommendations": recommendations
    }


@app.get("/database/test")
def database_test(db: Session = Depends(get_db)):

    movie_count = db.query(Movie).count()

    return {
        "database": "PostgreSQL",
        "status": "connected",
        "movie_count": movie_count
    }
@app.get("/movies/search")
def search_movies(
    title: str,
    db: Session = Depends(get_db)
):

    movies = (
        db.query(Movie)
        .filter(Movie.title.ilike(f"%{title}%"))
        .limit(20)
        .all()
    )

    return {
        "query": title,
        "results": [
            {
                "movie_id": movie.movie_id,
                "title": movie.title,
                "genres": movie.genres
            }
            for movie in movies
        ]
    }
@app.post("/ratings")
def add_rating(
    rating_data: RatingRequest,
    db: Session = Depends(get_db)
):

    new_rating = Rating(
        user_id=rating_data.user_id,
        movie_id=rating_data.movie_id,
        rating=rating_data.rating
    )

    db.add(new_rating)
    db.commit()
    db.refresh(new_rating)

    return {
        "message": "Rating added successfully",
        "rating_id": new_rating.rating_id,
        "user_id": new_rating.user_id,
        "movie_id": new_rating.movie_id,
        "rating": float(new_rating.rating)
    }
@app.get("/recommend/user/{user_id}")
def recommend_for_user_api(
    user_id: int,
    limit: int = 10
):

    recommendations = recommend_for_user(
        user_id,
        limit
    )

    if not recommendations:
        return {
            "user_id": user_id,
            "recommendations": [],
            "message": "User not found or no recommendations available"
        }

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }