from sqlalchemy import BigInteger, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    gender: Mapped[str] = mapped_column(
        String(1)
    )

    age: Mapped[int] = mapped_column(
        Integer
    )

    occupation: Mapped[int] = mapped_column(
        Integer
    )

    zip_code: Mapped[str] = mapped_column(
        String(10)
    )


class Movie(Base):
    __tablename__ = "movies"

    movie_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    genres: Mapped[str] = mapped_column(
        String(255)
    )


class Rating(Base):
    __tablename__ = "ratings"

    rating_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id"),
        nullable=False
    )

    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.movie_id"),
        nullable=False
    )

    rating: Mapped[float] = mapped_column(
        Numeric(2, 1),
        nullable=False
    )

    timestamp: Mapped[int] = mapped_column(
        BigInteger
    )
