FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make sure the entrypoint script is executable
RUN chmod +x run_api_server.py

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the API port
EXPOSE 8000

# Run the API server
CMD ["python", "run_api_server.py"]