# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install netcat (specifically netcat-openbsd) and other necessary system utilities
RUN apt-get update && \
    apt-get install -y netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

# Copy the Python requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords
RUN python -m nltk.downloader stopwords

# Copy the entire application into the container
COPY . /app

# Copy the 'elasticsearch' folder specifically into '/app/elasticsearch'
COPY elasticsearch /app/elasticsearch
