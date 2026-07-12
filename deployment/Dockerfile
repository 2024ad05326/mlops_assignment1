# Use a lightweight official Python image
FROM python:3.10-slim

WORKDIR /app

# Install uv directly into the container for fast dependency handling
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# Copy environment files first to leverage build cache
COPY pyproject.toml uv.lock ./

# Install project dependencies globally inside the container
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy project directories
COPY src/ ./src/
COPY models/ ./models/

# Expose FastAPI's standard communication port
EXPOSE 8000

# Run the app worker server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
