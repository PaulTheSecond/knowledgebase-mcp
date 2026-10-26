# ── Stage 1: Сборка RoslynParser (.NET 8) ───────────────────────────────────
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS dotnet-builder

WORKDIR /roslyn
# Копируем сначала только .csproj для эффективного кэширования NuGet пакетов
COPY RoslynParser/RoslynParser.csproj ./
RUN dotnet restore

# Копируем исходники и собираем плоский каталог со всеми зависимостями Roslyn
COPY RoslynParser/ ./
RUN dotnet publish -c Release -o /roslyn/out --no-restore

# ── Stage 2: Сборка Angular 21 Frontend ─────────────────────────────────────
FROM node:22-alpine AS frontend-builder

WORKDIR /frontend
# Кэширование npm ci
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ── Stage 3: Финальный образ (Python 3.11 Slim Bookworm) ─────────────────────
# Базируемся на легковесном образе (~150 МБ вместо 2.1 ГБ SDK)
FROM python:3.11-slim-bookworm

# Устанавливаем минимально необходимые runtime-пакеты (libicu72 для Unicode-парсинга Roslyn, git, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libicu72 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Переносим .NET 8 runtime и MSBuild SDK из stage 1 (без тяжелых шаблонов и powershell)
COPY --from=dotnet-builder /usr/share/dotnet/dotnet /usr/share/dotnet/dotnet
COPY --from=dotnet-builder /usr/share/dotnet/host /usr/share/dotnet/host
COPY --from=dotnet-builder /usr/share/dotnet/shared/Microsoft.NETCore.App /usr/share/dotnet/shared/Microsoft.NETCore.App
COPY --from=dotnet-builder /usr/share/dotnet/sdk /usr/share/dotnet/sdk
COPY --from=dotnet-builder /usr/share/dotnet/packs/Microsoft.NETCore.App.Ref /usr/share/dotnet/packs/Microsoft.NETCore.App.Ref

RUN ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet

# Настраиваем изолированное виртуальное окружение Python
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# ── Python зависимости ───────────────────────────────────────────────────────
COPY requirements.txt .

# Установка PyTorch (CPU) и зависимостей с последующей чисткой кэшей байткода
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type f -name "*.pyc" -delete || true

# ── Предзагрузка модели эмбеддингов ──────────────────────────────────────────
ENV HF_HOME=/app/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')" \
    && rm -rf /app/models/hub/.locks /root/.cache

# ── Код приложения и артефакты сборки ────────────────────────────────────────
COPY knowledge_mcp/ /app/knowledge_mcp/
COPY --from=dotnet-builder /roslyn/out /app/RoslynParser/bin/Release/net8.0/
COPY --from=frontend-builder /frontend/dist/dashboard/browser /app/frontend/dist/dashboard/browser

RUN mkdir -p /data
ENV DB_PATH=/data/knowledge.db
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV PYTHONUNBUFFERED=1

# Проверка работоспособности сервиса (SRE Best Practice)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/ -o /dev/null || exit 1

CMD ["python", "-m", "knowledge_mcp.main", "serve", "--host", "0.0.0.0", "--port", "8000"]
