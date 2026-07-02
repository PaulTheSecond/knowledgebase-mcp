// SPA Маршрутизатор панели управления (Story 2.2)

document.addEventListener('DOMContentLoaded', () => {
    // Настраиваем SPA маршрутизацию
    initRouter();
    
    // Подгружаем первоначальные статусы
    updateSystemStatusIndicator();
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
}

/**
 * Обновляет индикатор системного статуса в футере сайдбара.
 */
function updateSystemStatusIndicator() {
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    
    // Временная заглушка, реальные данные будут в Story 2.3 и 2.4
    statusDot.className = 'status-dot green';
    statusText.innerText = 'Система готова';
}

/**
 * Загрузка списка репозиториев (заглушка для Story 2.3).
 */
function loadRepositories() {
    const tbody = document.getElementById('repos-table-body');
    // Заглушка, наполнение будет в 2.3
    tbody.innerHTML = `
        <tr>
            <td>test-repo</td>
            <td>https://github.com/PaulTheSecond/knowledgebase-mcp.git</td>
            <td>master</td>
            <td><span class="text-muted">Нет данных</span></td>
            <td>
                <button class="btn btn-primary btn-sm" onclick="triggerSync('test-repo')">Синхронизировать</button>
            </td>
        </tr>
    `;
}

/**
 * Загрузка очереди задач (заглушка для Story 2.3).
 */
function loadQueue() {
    const tbody = document.getElementById('queue-table-body');
    // Заглушка, наполнение будет в 2.3
    tbody.innerHTML = `
        <tr>
            <td colspan="6" class="text-center text-muted">Очередь задач пуста</td>
        </tr>
    `;
}

/**
 * Запуск синхронизации (заготовка для Story 2.3).
 */
function triggerSync(repoId) {
    console.log(`Triggering sync for ${repoId}...`);
}
