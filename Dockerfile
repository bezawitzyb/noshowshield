# Use Python 3.10.6 buster image (compatible with TensorFlow 2.10.0)
FROM python:3.10.6-buster

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libgomp1 \
#     libffi-dev \
#     libssl-dev \
#     libopenblas-dev \
#     && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /prod

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY eda_package ./eda_package
COPY api ./api

# Copy data and models
COPY models ./models
COPY raw_data ./raw_data

# Set environment variables
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD uvicorn api.fast_copy:app --host 0.0.0.0 --port $PORT

#For docker run
# docker run -e PORT=8000 -p 8080:8000 $GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$DOCKER_REPO_NAME/$DOCKER_IMAGE_NAME:0.1
# http://localhost:8080/docs
# make build push deploy
