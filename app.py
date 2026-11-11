import os
import re
from dotenv import load_dotenv
import requests
from flask import Flask, render_template, request, jsonify
import pandas as pd

load_dotenv()

app = Flask(__name__)

# Load movies and similarity matrices
movies = pd.read_pickle("movies.pkl").reset_index(drop=True)
similarity = pd.read_pickle("similarity.pkl")

# Cached unique list of titles for fast suggestion lookups
movie_titles = list(dict.fromkeys(movies["title"].dropna().tolist()))


def find_movie_by_id(movie_id):
    """Fetch movie details and poster from TMDB API with graceful fallback."""
    token = os.getenv("TMDB_API_READ_ACCESS_TOKEN")
    if not token:
        return {
            "id": int(movie_id),
            "poster_link": None,
            "rating": None,
            "year": None,
            "overview": None,
        }

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            poster_link = (
                f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            )
            release_date = data.get("release_date") or ""
            year = release_date.split("-")[0] if release_date else None
            vote_average = data.get("vote_average")
            rating = (
                round(vote_average, 1)
                if vote_average is not None and vote_average > 0
                else None
            )

            return {
                "id": int(movie_id),
                "title": data.get("title") or data.get("original_title"),
                "poster_link": poster_link,
                "rating": rating,
                "year": year,
                "overview": data.get("overview"),
            }
    except Exception as e:
        print(f"Error fetching movie {movie_id} from TMDB: {e}")

    return {
        "id": int(movie_id),
        "poster_link": None,
        "rating": None,
        "year": None,
        "overview": None,
    }


def movie_recommender(movie_name, top_n=5):
    """Find recommendations for a movie title."""
    if not movie_name or not isinstance(movie_name, str):
        return None, []

    movie_name_clean = movie_name.strip()

    # 1. Exact match (case-insensitive)
    matches = movies[
        movies["title"].str.strip().str.lower() == movie_name_clean.lower()
    ]

    # 2. If no exact match, try contains match
    if matches.empty:
        matches = movies[
            movies["title"]
            .str.strip()
            .str.lower()
            .str.contains(re.escape(movie_name_clean.lower()), regex=True)
        ]

    if matches.empty:
        return None, []

    movie_index = matches.index[0]
    matched_title = movies.iloc[movie_index]["title"]

    distances = similarity[movie_index]
    # Top recommendations excluding the searched movie itself
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[
        1 : top_n + 1
    ]

    recommended_movies = []
    for i in movies_list:
        rec_index = i[0]
        rec_id = movies.iloc[rec_index]["id"]
        rec_title = movies.iloc[rec_index]["title"]

        movie_details = find_movie_by_id(rec_id)
        # Ensure title is populated even if TMDB fails
        if not movie_details.get("title"):
            movie_details["title"] = rec_title

        recommended_movies.append(movie_details)

    return matched_title, recommended_movies


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/suggestions")
def get_suggestions():
    """Return autocomplete suggestions based on movie title."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"suggestions": []})

    q_lower = q.lower()
    # Prioritize titles starting with query, then titles containing query
    starts_with = [t for t in movie_titles if t.lower().startswith(q_lower)]
    contains = [
        t
        for t in movie_titles
        if q_lower in t.lower() and not t.lower().startswith(q_lower)
    ]

    results = (starts_with + contains)[:10]
    return jsonify({"suggestions": results})


@app.route("/api/recommend", methods=["GET", "POST"])
def recommend_api():
    """API endpoint to get movie recommendations."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        movie_name = data.get("movie") or request.form.get("movie")
    else:
        movie_name = request.args.get("movie")

    if not movie_name or not movie_name.strip():
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Please provide a movie title to get recommendations.",
                }
            ),
            400,
        )

    matched_title, recommendations = movie_recommender(movie_name.strip(), top_n=5)

    if not recommendations and matched_title is None:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Movie '{movie_name}' was not found in our database. Please select from the suggestions.",
                }
            ),
            404,
        )

    return jsonify(
        {
            "success": True,
            "searched_movie": matched_title or movie_name,
            "recommendations": recommendations,
            "count": len(recommendations),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
