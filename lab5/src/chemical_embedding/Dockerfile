FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5006

CMD ["panel", "serve", "app.py", "--address", "0.0.0.0", "--port", "5006", "--allow-websocket-origin=*"]
