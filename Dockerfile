FROM python:3.10-slim

# Create working dir
WORKDIR /app

# Copy requirements first (better cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Expose the port Spaces expects
EXPOSE 7860

# Run the app. Use the module:variable syntax.
# main:app means "from main import app" where app is FastAPI instance.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
