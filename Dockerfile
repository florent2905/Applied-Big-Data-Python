# syntax=docker/dockerfile:1

FROM python:3.13-slim AS base
WORKDIR /app

# Builder stage: install dependencies in a venv
FROM base AS builder

# Install system dependencies for scientific packages and MariaDB connector
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmariadb3 \
    build-essential \
    libmariadb-dev \
    libmariadb-dev-compat \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy 1_createDB.py only (no .git, no secrets)
COPY --link 1_createDB.py ./

# Create requirements.txt inline for reproducibility
RUN echo "yfinance\nmatplotlib\npandas\ngdeltdoc\nvaderSentiment\nscikit-learn\nseaborn\nscipy\nmariadb\nsqlalchemy\n" > requirements.txt

# Create and populate virtual environment
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install -r requirements.txt

# Final stage: minimal runtime image
FROM base AS final

# Installer la bibliothèque MariaDB client dans l'image finale
RUN apt-get update && apt-get install -y libmariadb3 && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser
USER appuser

WORKDIR /app

# Copy 1_createDB.py and venv from builder
COPY --link --from=builder /app/1_createDB.py ./
COPY --link --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["python", "1_createDB.py"]
