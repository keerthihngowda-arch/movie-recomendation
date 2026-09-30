# CineMatch AI

A movie recommendation app by **Keerthi Gowda**. It suggests films from a title you already like, and it also shows trending picks and simple library insights.

Recommendations blend two signals:

- **Content match** — TF-IDF over genre, director, cast, keywords, and tagline, scored with cosine similarity
- **Quality** — an IMDb-style weighted rating so popular, well-rated films rank higher

## Features

- Search a movie and get ranked recommendations
- Filter by genre, minimum rating, and how many results to show
- Trending list based on weighted score
- Charts for rating spread, top genres, and releases over time

## Project layout

| File | Role |
| --- | --- |
| `app.py` | Streamlit interface |
| `ml_pipeline.py` | Data cleaning, model training, and recommendation logic |
| `requirments.txt` | Python dependencies |
| `Movies_Recommendation.csv` | Source dataset (kept local, not committed) |

## Setup

Use Python 3.10 or newer.

```bash
python -m venv movieenv
movieenv\Scripts\activate
pip install -r requirments.txt
```

Place `Movies_Recommendation.csv` in the project folder. The file is listed in `.gitignore`, so it stays on your machine.

Train the model, then start the app:

```bash
python ml_pipeline.py
streamlit run app.py
```

The training step writes `models/recommender.pkl`. The app reads that file on startup.

## Author

Keerthi Gowda
