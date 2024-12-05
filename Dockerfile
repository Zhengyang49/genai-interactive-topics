# Use official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8050 (default for Dash)
EXPOSE 8050

# Run the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "app:app"]