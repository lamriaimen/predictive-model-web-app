# Use Python 3.10 as base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Start the Flask app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
