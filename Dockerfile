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

COPY data/processed ./data/processed
COPY data/interim ./data/interim
COPY reports/ranking ./reports/ranking

RUN useradd --create-home --uid 10001 recomx
USER recomx

EXPOSE 8000

CMD ["uvicorn", "recomx.serving.production:app", "--host", "0.0.0.0", "--port", "8000"]
