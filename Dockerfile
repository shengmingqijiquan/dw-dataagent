FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nl2insight/ nl2insight/
COPY data/ data/
COPY config.yaml .

EXPOSE 8000

CMD ["uvicorn", "nl2insight.api:app", "--host", "0.0.0.0", "--port", "8000"]
