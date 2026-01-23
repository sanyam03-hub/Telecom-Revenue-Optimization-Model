# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Command to run the dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]
