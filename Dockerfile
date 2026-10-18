# ── Stage 1: Сборка RoslynParser ───────────────────────────────────────────
# Используем официальный SDK-образ — не нужен wget + bash-скрипт (~210 MB)
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS dotnet-builder

WORKDIR /roslyn
COPY RoslynParser/ .
# publish создаёт плоский каталог со всеми DLL-зависимостями Roslyn
RUN dotnet publish -c Release -o /roslyn/out

# ── Stage 2: Сборка Angular 19 Frontend ─────────────────────────────────────
# Собираем современный SPA дашборд с темами оформления и мультиязычностью
FROM node:22-alpine AS frontend-builder

WORKDIR /frontend
# Копируем package*.json отдельно для эффективного кэширования npm ci
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ── Stage 3: Финальный образ ────────────────────────────────────────────────
# Используем dotnet SDK как базу и доустанавливаем Python.
# dotnet/runtime:8.0 НЕ подходит: MSBuildLocator ищет SDK при запуске и падает.
FROM mcr.microsoft.com/dotnet/sdk:8.0

# Устанавливаем Python 3.11 и libicu (нужна для Roslyn Unicode-обработки)
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip libicu-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# ── Python зависимости (отдельные слои = максимальный кеш) ─────────────────
COPY requirements.txt .

# torch в отдельном RUN — инвалидируется только при смене версии torch,
# а не при любом изменении requirements.txt
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Остальные зависимости — инвалидируются только при изменении requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем свой путь для кэша HF, чтобы скачанная модель не перетиралась volume из docker-compose
ENV HF_HOME=/app/models

# Предзагрузка NLP-модели, чтобы не скачивать её при каждом старте контейнера
RUN /opt/venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# ── Код приложения и артефакты сборки ──────────────────────────────────────
# Копируется последним — изменение кода не инвалидирует кеш pip и dotnet
COPY knowledge_mcp/ /app/knowledge_mcp/

# Копируем собранный RoslynParser из stage 1
COPY --from=dotnet-builder /roslyn/out /app/RoslynParser/bin/Release/net8.0/

# Копируем скомпилированный Angular SPA из stage 2
COPY --from=frontend-builder /frontend/dist/dashboard/browser /app/frontend/dist/dashboard/browser

RUN mkdir -p /data
ENV DB_PATH=/data/knowledge_roslyn.db
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "knowledge_mcp.main", "serve", "--host", "0.0.0.0", "--port", "8000"]
