# Container Registry Publication & Distribution Guide

This document outlines the container distribution strategy for `knowledgebase-mcp`, instructions for automated publication via GitHub Actions to Docker Hub (or domestic alternatives like Yandex Cloud), and setup instructions for end users (including mirrors for users in Russia).

---

## 1. Overview & Registry Strategy

The knowledge base server is packaged as a self-contained multi-stage Docker image containing:
- .NET 8.0 SDK & RoslynParser binaries
- Angular 21 Single Page Application Web Dashboard
- Python 3.11 with PyTorch (CPU) & pre-cached Hugging Face embedding model (`intfloat/multilingual-e5-small`)

### Registry Options

| Registry | Accessibility in RU | CI/CD Integration | Notes |
| :--- | :--- | :--- | :--- |
| **Docker Hub** (`docker.io`) | Requires mirror / VPN for some ISPs (403 errors) | Secrets `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` | Standard international public registry |
| **GitHub Container Registry** (`ghcr.io`) | Can experience throttling / DPI issues | Automatic via `GITHUB_TOKEN` | Native GitHub packages |
| **Yandex Cloud Registry** (`cr.yandex`) | 100% accessible in Russia, high bandwidth | Service account key (`json_key`) | Supports public unauthenticated pulls |

---

## 2. Docker Hub Automated Publication (GitHub Actions)

A ready-to-use workflow is configured in [`.github/workflows/docker-publish.yml`](../.github/workflows/docker-publish.yml).

### Step-by-Step Setup:

1. **Create Docker Hub Personal Access Token (PAT):**
   - Navigate to [hub.docker.com](https://hub.docker.com) → **Account Settings** → **Security**.
   - Click **New Access Token**.
   - Description: `github-actions-knowledgebase-mcp`.
   - Access Permissions: **Read & Write**.
   - Copy the generated token (`dckr_pat_...`).

2. **Add GitHub Repository Secrets:**
   - In your GitHub repository: **Settings** → **Secrets and variables** → **Actions** → **New repository secret**.
   - Secret 1: `DOCKERHUB_USERNAME` (your Docker Hub handle, e.g. `paulthesecond`).
   - Secret 2: `DOCKERHUB_TOKEN` (the copied `dckr_pat_...` token).

3. **Triggering Builds:**
   - Push to `master` branch: Automatically tags and pushes `<username>/knowledgebase-mcp:latest`.
   - Release tags: Pushing a tag `v1.x.x` creates matching semver tags (`:1.x.x`, `:1.x`).
   - Manual: Via GitHub UI **Actions** tab → **Build and Push Docker Image to Docker Hub** → **Run workflow**.

---

## 3. End-User Quickstart with Pre-Built Image

Users do not need to install .NET SDK, Node.js, or build the container locally.

### Option A: `docker run`
```bash
docker run -d \
  --name knowledge-mcp \
  -p 8000:8000 \
  -v ./data:/data \
  -v /path/to/local/repos:/repos:ro \
  <username>/knowledgebase-mcp:latest
```

### Option B: `docker-compose.yml`
```yaml
services:
  knowledge-mcp:
    image: <username>/knowledgebase-mcp:latest
    container_name: knowledge-mcp
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
      - ${REPOS_DIR:-./src}:/repos:ro
    environment:
      - DB_PATH=/data/knowledge.db
```

---

## 4. Docker Hub Mirrors for Users in Russia

If users in the Russian Federation experience HTTP 403 Forbidden or connection timeouts when pulling from `docker.io`, they can configure a domestic caching mirror:

Edit `/etc/docker/daemon.json` (Linux) or **Docker Desktop → Settings → Docker Engine** (Windows/macOS):

```json
{
  "registry-mirrors": [
    "https://dockerhub.timeweb.cloud"
  ]
}
```

Restart Docker daemon. `docker pull <username>/knowledgebase-mcp:latest` will route through the domestic mirror seamlessly.

---

## 5. Alternative: Yandex Cloud Container Registry (Domestic Alternative)

If domestic cloud hosting is required for corporate/offline compliance:
1. Create a registry in Yandex Cloud console and toggle **Public access** (Публичный доступ) to ON.
2. Create a service account with `container-registry.images.pusher` role and create an authorized key JSON.
3. Authenticate in GitHub Actions:
   ```bash
   echo '${{ secrets.YC_SA_KEY_JSON }}' | docker login --username json_key --password-stdin cr.yandex
   ```
4. Push to `cr.yandex/<registry-id>/knowledge-mcp:latest`.
5. Users in Russia can pull directly without logging in:
   ```bash
   docker pull cr.yandex/<registry-id>/knowledge-mcp:latest
   ```
