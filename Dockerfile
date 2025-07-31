# Dockerfile

# 1. Use an official Python runtime as a parent image
# Using a slim version keeps the final image size smaller.
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Set environment variables
# This tells Python not to buffer stdout and stderr, making logs appear in real-time.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 4. Install system dependencies (if any)
# Some Python packages might need system libraries. Add them here if needed.
# RUN apt-get update && apt-get install -y ...

# 5. Copy the requirements file and install dependencies
# This step is separated to leverage Docker's layer caching.
# If requirements.txt doesn't change, this layer won't be rebuilt.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code into the container
COPY ./app /app/app
COPY ./main.py /app/main.py

# 7. Expose the port the app runs on
# Cloud Run expects the container to listen for requests on the port defined by the PORT environment variable.
# Uvicorn will automatically use this variable if it's set. We expose 8080 as a default.
EXPOSE 8080

# 8. Define the command to run your application
# This command starts the Uvicorn server.
# --host 0.0.0.0 makes the server accessible from outside the container.
# --port 8080 is the default port Cloud Run will use.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
