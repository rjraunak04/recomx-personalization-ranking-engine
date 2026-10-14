FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RECOMX_ENV=production \
    RECOMX_HOST=0.0.0.0 \
    RECOMX_PORT=8000

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Copy only artifacts required by the serving runtime. In particular, never
# copy data/processed wholesale: that directory also contains the locked test.
RUN mkdir -p data/processed data/interim reports/ranking/ml reports/ranking
COPY data/processed/ratings_train.parquet ./data/processed/ratings_train.parquet
COPY data/interim/movies.parquet ./data/interim/movies.parquet
COPY reports/ranking/ml/model.npz ./reports/ranking/ml/model.npz
COPY reports/ranking/validation_ranking_features.parquet ./reports/ranking/validation_ranking_features.parquet

RUN useradd --create-home --uid 10001 recomx
USER recomx

EXPOSE 8000

CMD ["uvicorn", "recomx.serving.production:create_production_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
