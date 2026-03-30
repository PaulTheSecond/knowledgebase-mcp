FROM python:3.11-slim

# Рабочая директория
WORKDIR /app

# Копирование файла зависимостей
COPY requirements.txt .

# Устанавливаем CPU-only версию PyTorch (экономия ~1.5 ГБ RAM и диска по сравнению с полным CUDA-пакетом)
# Затем остальные зависимости
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY knowledge_mcp/ /app/knowledge_mcp/

# Создание директории для базы данных (volume mount point)
RUN mkdir -p /data
ENV DB_PATH=/data/knowledge.db

# Настройка по умолчанию: старт HTTP-сервера триггеров.
# Для использования агентами через stdio нужно будет запускать с командой `mcp`
CMD ["python", "-m", "knowledge_mcp.main", "--db-path", "/data/knowledge.db", "serve", "--host", "0.0.0.0", "--port", "8000"]
