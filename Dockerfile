FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git docker.io && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY forge_bot/ ./forge_bot/
COPY prompts/ ./prompts/

RUN groupadd -g 999 docker-host && \
    useradd -m -u 1000 -G docker-host forgebot
USER forgebot

EXPOSE 8080
CMD ["uvicorn", "forge_bot.server:app", "--host", "0.0.0.0", "--port", "8080"]
