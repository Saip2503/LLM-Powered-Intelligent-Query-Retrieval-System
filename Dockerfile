# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- THIS IS THE FIX ---
# 4. Install system dependencies required by OpenCV (a dependency of unstructured)
RUN apt-get update && apt-get install -y libgl1-mesa-glx
# ----------------------

# 5. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code into the container
COPY ./app /app/app
COPY ./main.py /app/main.py

# 7. Expose the port the app runs on
EXPOSE 8080

# 8. Define the command to run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
