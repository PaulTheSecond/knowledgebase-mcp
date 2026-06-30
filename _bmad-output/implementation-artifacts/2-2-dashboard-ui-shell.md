# Story 2.2: Dashboard UI Shell

Status: ready-for-dev

## Story

As a разработчик,
I want иметь красивый, современный веб-интерфейс в темных тонах,
so that мне было приятно и удобно работать с системой в браузере.

## Acceptance Criteria

1. **Given** статический SPA фронтенд (чистый HTML5, Vanilla CSS, Vanilla JS) в папке `dashboard/static/`,
   **When** я открываю корень дашборда в браузере,
   **Then** отображается страница с премиум-дизайном (темная тема, curated HSL-палитра, микро-анимации переходов).
2. **And** навигационное меню позволяет переключаться между вкладками "Repositories", "Sync Queue" и "RAG Chat" без перезагрузки страницы (SPA).

## Tasks / Subtasks

- [ ] Создание разметки HTML5 (AC: 1, 2)
  - [ ] Создать файл `knowledge_mcp/dashboard/static/index.html`.
  - [ ] Спроектировать сайдбар навигации и контейнер основного контента с уникальными DOM ID.
- [ ] Оформление стилей CSS3 (AC: 1)
  - [ ] Создать файл `knowledge_mcp/dashboard/static/css/styles.css`.
  - [ ] Подключить шрифт Google Fonts (например, `Inter` или `Outfit`).
  - [ ] Настроить CSS-переменные для темной темы (HSL оттенки угольного, графитового и неоново-синего акцента).
  - [ ] Добавить плавные переходы (`transition: all 0.3s ease`) для интерактивных элементов.
- [ ] SPA Маршрутизация на Vanilla JS (AC: 2)
  - [ ] Создать файл `knowledge_mcp/dashboard/static/js/app.js`.
  - [ ] Реализовать клиентский роутер по переключению видимости секций `<section id="repositories-view">`, `<section id="queue-view">`, `<section id="chat-view">`.
  - [ ] Добавить автозаполнение Basic Auth диалогового окна браузера при открытии страницы.
- [ ] Интеграция статики в FastAPI (AC: 1)
  - [ ] Смонтировать статику в `knowledge_mcp/daemon/app.py` через `FastAPI.mount("/static", StaticFiles(directory="knowledge_mcp/dashboard/static"), name="static")`.
  - [ ] Отдавать `index.html` при запросе к корню `/`.

## Dev Notes

Интерфейс должен выглядеть крайне премиально (эффект "glassmorphism", скругления углов `border-radius: 12px`, мягкие неоновые тени).
Пример CSS переменных темы:
```css
:root {
  --bg-primary: #0d0e12;
  --bg-secondary: #161820;
  --accent-color: #3b82f6;
  --accent-glow: rgba(59, 130, 246, 0.15);
  --text-primary: #f3f4f6;
  --text-secondary: #9ca3af;
}
```

### References

- [Архитектурные требования: ADR-005-web-dashboard-architecture.md](file:///c:/Repos/knowledgebase-mcp/docs/adr-005-web-dashboard-architecture.md)

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
