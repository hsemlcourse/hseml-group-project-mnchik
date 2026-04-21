# Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Устанавливаем системные зависимости при необходимости
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы с зависимостями
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

ENV PYTHONUNBUFFERED=1

# По умолчанию: запуск EDA-ноутбука через jupyter (или пайплайна)
CMD ["python", "-m", "src.models.modeling"]