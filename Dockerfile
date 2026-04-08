FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Hugging Face's default port
EXPOSE 7860

# Start the web server instead of the inference script
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]