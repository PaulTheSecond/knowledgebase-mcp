// SPA Маршрутизатор панели управления (Story 2.2)

let activeProgresses = {};
let wsConn = null;

document.addEventListener('DOMContentLoaded', () => {
    // Настраиваем SPA маршрутизацию
    initRouter();
    
    // Подгружаем первоначальные статусы
    updateSystemStatusIndicator();

    // Подключаемся к WebSocket для обновлений в реальном времени
    connectWebSocket();
});

/**
 * Инициализирует SPA роутер на основе хэшей URL.
 */
function initRouter() {
    const navItems = {
        '#repositories': {
            btnId: 'nav-repositories',
            viewId: 'repositories-view',
            title: 'Управление репозиториями'
        },
        '#queue': {
            btnId: 'nav-queue',
            viewId: 'queue-view',
            title: 'Очередь синхронизации'
        },
        '#chat': {
            btnId: 'nav-chat',
            viewId: 'chat-view',
            title: 'Интерактивный RAG Чат'
        }
    };

    function handleRoute() {
        let hash = window.location.hash || '#repositories';
        
        // Корректируем хэш, если его нет в списке
        if (!navItems[hash]) {
            hash = '#repositories';
        }

        // Обновляем визуальное состояние меню и секций
        Object.keys(navItems).forEach(key => {
            const config = navItems[key];
            const btn = document.getElementById(config.btnId);
            const view = document.getElementById(config.viewId);

            if (key === hash) {
                btn.classList.add('active');
                view.classList.add('active');
                document.getElementById('page-title').innerText = config.title;
            } else {
                btn.classList.remove('active');
                view.classList.remove('active');
            }
        });

        // Запуск специфической логики при переходе на вкладку
        if (hash === '#repositories') {
            loadRepositories();
        } else if (hash === '#queue') {
            loadQueue();
        }
    }

    // Слушаем изменение хэша
    window.addEventListener('hashchange', handleRoute);
    
    // Запускаем первый раз при загрузке
    handleRoute();

    // Настраиваем кнопку обновления
    document.getElementById('btn-refresh').addEventListener('click', () => {
        const hash = window.location.hash || '#repositories';
        if (hash === '#repositories') {
            loadRepositories();
        } else if (hash === '#queue') {
            loadQueue();
        }
        updateSystemStatusIndicator();
    });

    // Настраиваем кнопку повтора упавших задач
    const btnRetry = document.getElementById('btn-retry-failed');
    if (btnRetry) {
        btnRetry.addEventListener('click', async () => {
            try {
                const res = await fetch('/api/queue/retry-failed', { method: 'POST' });
                if (!res.ok) throw new Error('Ошибка сброса задач');
                const data = await res.json();
                loadQueue();
                updateSystemStatusIndicator();
                alert(`Сброшено задач для повторной попытки: ${data.count}`);
            } catch (err) {
                alert(`Ошибка: ${err.message}`);
            }
        });
    }
}

/**
 * Обновляет индикатор системного статуса в футере сайдбара.
 */
async function updateSystemStatusIndicator() {
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const workerBadge = document.getElementById('worker-state-badge');
    
    try {
        const response = await fetch('/api/status');
        if (!response.ok) throw new Error('Auth or API error');
        
        const data = await response.json();
        
        // Скрытие / показ чата в сайдбаре (Story 3.3)
        const navChat = document.getElementById('nav-chat');
        if (navChat) {
            if (data.chat_enabled === false) {
                navChat.style.display = 'none';
            } else {
                navChat.style.display = '';
            }
        }
        
        // Мапим статус воркера
        if (data.status === 'idle') {
            statusDot.className = 'status-dot green';
            statusText.innerText = 'Воркер: Ожидание';
            if (workerBadge) {
                workerBadge.className = 'badge-status success';
                workerBadge.innerText = 'Воркер: Ожидание';
            }
        } else if (data.status === 'processing') {
            statusDot.className = 'status-dot green animate-pulse'; // пульсирующий эффект
            statusText.innerText = 'Воркер: Индексация';
            if (workerBadge) {
                workerBadge.className = 'badge-status info';
                workerBadge.innerText = 'Воркер: Индексация';
            }
        } else if (data.status === 'halted') {
            statusDot.className = 'status-dot red';
            statusText.innerText = 'Воркер: Остановлен';
            if (workerBadge) {
                workerBadge.className = 'badge-status danger';
                workerBadge.innerText = 'Воркер: Остановлен (Критическая ошибка)';
            }
        } else {
            statusDot.className = 'status-dot yellow';
            statusText.innerText = `Воркер: ${data.status}`;
            if (workerBadge) {
                workerBadge.className = 'badge-status warning';
                workerBadge.innerText = `Воркер: ${data.status}`;
            }
        }
    } catch (err) {
        statusDot.className = 'status-dot red';
        statusText.innerText = 'Ошибка соединения';
    }
}

/**
 * Загрузка списка репозиториев.
 */
async function loadRepositories() {
    const tbody = document.getElementById('repos-table-body');
    tbody.innerHTML = '<tr><td colspan="5" class="text-center">Загрузка...</td></tr>';
    
    try {
        const response = await fetch('/api/status');
        if (!response.ok) throw new Error('API request failed');
        
        const data = await response.json();
        const repos = data.repositories || [];
        
        if (repos.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">Нет настроенных репозиториев</td></tr>';
            return;
        }
        
        tbody.innerHTML = '';
        repos.forEach(repo => {
            const tr = document.createElement('tr');
            tr.id = `repo-row-${repo.id}`;
            
            // Защита от XSS через textContent
            const tdId = document.createElement('td');
            tdId.textContent = repo.id;
            
            const tdUrl = document.createElement('td');
            tdUrl.textContent = repo.url;
            
            const tdBranch = document.createElement('td');
            tdBranch.textContent = repo.branch;
            
            const tdStats = document.createElement('td');
            tdStats.id = `repo-stats-${repo.id}`;
            
            const active = activeProgresses[repo.id];
            if (active) {
                tdStats.innerHTML = `
                    <div class="progress-container">
                        <div class="progress-bar" style="width: ${active.progress}%; transition: width 0.2s ease-in-out;"></div>
                        <span class="progress-label">${active.status} (${active.progress}%)</span>
                    </div>
                `;
            } else {
                tdStats.innerHTML = `<span class="badge-status info">Файлов: ${repo.files_count}</span> <span class="badge-status success">Чанков: ${repo.chunks_count}</span>`;
            }
            
            const tdActions = document.createElement('td');
            const syncBtn = document.createElement('button');
            syncBtn.className = 'btn btn-primary btn-sm';
            syncBtn.textContent = 'Синхронизировать';
            syncBtn.addEventListener('click', () => triggerSync(repo.id));
            tdActions.appendChild(syncBtn);
            
            tr.appendChild(tdId);
            tr.appendChild(tdUrl);
            tr.appendChild(tdBranch);
            tr.appendChild(tdStats);
            tr.appendChild(tdActions);
            
            tbody.appendChild(tr);
        });
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="5" class="text-center text-danger">Не удалось загрузить данные: ${err.message}</td></tr>`;
    }
}

/**
 * Загрузка очереди задач.
 */
async function loadQueue() {
    const tbody = document.getElementById('queue-table-body');
    tbody.innerHTML = '<tr><td colspan="6" class="text-center">Загрузка...</td></tr>';
    
    try {
        const response = await fetch('/api/queue');
        if (!response.ok) throw new Error('API request failed');
        
        const data = await response.json();
        const jobs = data.jobs || [];
        
        // Обновляем бейдж в боковом меню
        const queueBadge = document.getElementById('queue-badge');
        const activeJobsCount = jobs.filter(j => j.status === 'pending' || j.status === 'processing').length;
        if (activeJobsCount > 0) {
            queueBadge.textContent = activeJobsCount;
            queueBadge.style.display = 'inline-block';
        } else {
            queueBadge.style.display = 'none';
        }
        
        if (jobs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">Очередь задач пуста</td></tr>';
            return;
        }
        
        tbody.innerHTML = '';
        jobs.forEach(job => {
            const tr = document.createElement('tr');
            
            const tdId = document.createElement('td');
            tdId.textContent = `#${job.id}`;
            
            const tdRepo = document.createElement('td');
            tdRepo.textContent = job.repo_id;
            
            const tdStatus = document.createElement('td');
            let statusClass = 'info';
            if (job.status === 'completed') statusClass = 'success';
            if (job.status === 'failed') statusClass = 'danger';
            if (job.status === 'processing') statusClass = 'warning';
            tdStatus.innerHTML = `<span class="badge-status ${statusClass}">${job.status}</span>`;
            
            const tdAttempts = document.createElement('td');
            tdAttempts.textContent = `${job.attempts}/3`;
            
            const tdError = document.createElement('td');
            if (job.last_error) {
                // Выводим ошибку безопасно
                const errorSpan = document.createElement('span');
                errorSpan.className = 'text-danger';
                errorSpan.style.cursor = 'help';
                errorSpan.title = job.last_error;
                errorSpan.textContent = job.last_error.length > 30 ? job.last_error.substring(0, 30) + '...' : job.last_error;
                tdError.appendChild(errorSpan);
            } else {
                tdError.innerHTML = '<span class="text-muted">—</span>';
            }
            
            const tdTime = document.createElement('td');
            tdTime.textContent = new Date(job.created_at).toLocaleString();
            
            tr.appendChild(tdId);
            tr.appendChild(tdRepo);
            tr.appendChild(tdStatus);
            tr.appendChild(tdAttempts);
            tr.appendChild(tdError);
            tr.appendChild(tdTime);
            
            tbody.appendChild(tr);
        });
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="6" class="text-center text-danger">Не удалось загрузить очередь: ${err.message}</td></tr>`;
    }
}

/**
 * Запуск синхронизации.
 */
async function triggerSync(repoId) {
    try {
        const response = await fetch(`/api/sync/${repoId}`, {
            method: 'POST'
        });
        if (!response.ok) throw new Error('Failed to trigger sync');
        
        const data = await response.json();
        alert(`Синхронизация репозитория ${repoId} успешно запущена! Задача #${data.job_id} добавлена в очередь.`);
        
        // Перенаправляем на вкладку очереди
        window.location.hash = '#queue';
    } catch (err) {
        alert(`Ошибка при запуске синхронизации: ${err.message}`);
    }
}

/**
 * Подключение к WebSockets для получения статуса синхронизации в реальном времени.
 */
async function connectWebSocket() {
    try {
        // Сначала запрашиваем короткоживущий токен для вебсокетов
        const response = await fetch('/api/auth/token', { method: 'POST' });
        if (!response.ok) {
            console.error('Failed to obtain WebSocket token');
            setTimeout(connectWebSocket, 5000);
            return;
        }
        
        const data = await response.json();
        const token = data.token;
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/status?token=${token}`;
        
        wsConn = new WebSocket(wsUrl);
        
        wsConn.onopen = () => {
            console.log('WebSocket status connection established.');
        };
        
        wsConn.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                if (message.type === 'sync_progress') {
                    handleSyncProgressUpdate(message.data);
                }
            } catch (err) {
                console.error('Error handling WebSocket message:', err);
            }
        };
        
        wsConn.onclose = (event) => {
            console.log(`WebSocket connection closed (code: ${event.code}). Reconnecting in 3s...`);
            setTimeout(connectWebSocket, 3000);
        };
        
        wsConn.onerror = (err) => {
            console.error('WebSocket error:', err);
            wsConn.close();
        };
    } catch (err) {
        console.error('Failed to initialize WebSocket connection:', err);
        setTimeout(connectWebSocket, 5000);
    }
}

/**
 * Обработка события изменения прогресса синхронизации из вебсокета.
 */
function handleSyncProgressUpdate(data) {
    const { repo_id, progress, status } = data;
    
    if (progress === 100 || status === 'completed' || status === 'failed') {
        // Задача завершилась — удаляем ее из активного прогресса
        delete activeProgresses[repo_id];
        // Полностью перерисовываем таблицы для обновления статических метрик
        const hash = window.location.hash || '#repositories';
        if (hash === '#repositories') {
            loadRepositories();
        } else if (hash === '#queue') {
            loadQueue();
        }
    } else {
        activeProgresses[repo_id] = { progress, status };
        renderActiveProgress(repo_id);
        
        // Если открыта вкладка очереди, обновляем ее тоже
        const hash = window.location.hash || '#repositories';
        if (hash === '#queue') {
            loadQueue();
        }
    }
    
    updateSystemStatusIndicator();
}

/**
 * Реактивное обновление прогресс-бара конкретного репозитория в таблице.
 */
function renderActiveProgress(repoId) {
    const tdStats = document.getElementById(`repo-stats-${repoId}`);
    if (!tdStats) return; // Репозиторий может быть не на текущей странице/экране
    
    const active = activeProgresses[repoId];
    if (!active) return;
    
    tdStats.innerHTML = `
        <div class="progress-container">
            <div class="progress-bar" style="width: ${active.progress}%; transition: width 0.2s ease-in-out;"></div>
            <span class="progress-label">${active.status} (${active.progress}%)</span>
        </div>
    `;
}
