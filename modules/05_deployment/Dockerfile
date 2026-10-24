# Base image
FROM python:3.8.12-slim

# Copy the uv program from uv's official image into this image
# Format: COPY --from=<source image> <file in that image> <destination in this image>
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create a folder called /app inside the image and work from there
WORKDIR /app

# Copy the dependency files into the image
# Format: COPY ["file1", "file2", "destination_folder"]
COPY ["pyproject.toml", "uv.lock", "./"]

# Install the exact dependency versions from uv.lock
RUN uv sync --locked --no-dev --no-install-project

# Add the venv's bin folder to PATH so the container can find gunicorn
ENV PATH="/app/.venv/bin:$PATH"

# Copy the app and model into /app
# because the container can only use files inside the image
COPY ["modules/05_deployment/predict.py", "modules/05_deployment/model_C=1.0.bin", "./"]

# Note that the app uses port 9696
EXPOSE 9696

# The command that starts the app when the container runs
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
