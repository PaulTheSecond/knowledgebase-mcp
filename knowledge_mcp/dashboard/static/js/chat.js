// Клиент RAG Чат (Story 3.3)

let chatHistory = [];
let isGenerating = false;

document.addEventListener('DOMContentLoaded', () => {
    const btnSend = document.getElementById('btn-send-chat');
    const inputChat = document.getElementById('chat-input');

    if (btnSend && inputChat) {
        btnSend.addEventListener('click', sendMessage);
        inputChat.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }

    // Обработчик закрытия модального окна (Story 3.4)
    const modal = document.getElementById('source-modal');
    const btnCloseModal = document.getElementById('btn-close-modal');
    if (btnCloseModal && modal) {
        btnCloseModal.addEventListener('click', () => {
            modal.style.display = 'none';
        });
        window.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
    }
});

/**
 * Отправляет сообщение пользователя в чат.
 */
async function sendMessage() {
    if (isGenerating) return;

    const inputChat = document.getElementById('chat-input');
    const query = inputChat.value.trim();
    if (!query) return;

    // Сбрасываем инпут
    inputChat.value = '';

    // Выводим сообщение пользователя в UI
    appendMessage('user', query);

    // Добавляем в историю
    chatHistory.push({ role: 'user', content: query });

    isGenerating = true;

    // Выводим индикатор "думания"
    const thinkingBubble = appendThinkingIndicator();

    // Создаем контейнер для ответа ассистента
    const chatMessages = document.getElementById('chat-messages');
    const assistantMessageDiv = document.createElement('div');
    assistantMessageDiv.className = 'message assistant';
    
    // Создаем внутренний элемент для самого текста ответа
    const textSpan = document.createElement('span');
    assistantMessageDiv.appendChild(textSpan);
    
    chatMessages.appendChild(assistantMessageDiv);
    scrollToBottom();

    let fullResponseText = '';
    let tokenReceived = false;

    try {
        // 1. Получаем короткоживущий токен авторизации
        const tokenResponse = await fetch('/api/auth/token', { method: 'POST' });
        if (!tokenResponse.ok) throw new Error('Не удалось авторизоваться на сервере');
        const tokenData = await tokenResponse.json();
        const token = tokenData.token;

        // 2. Делаем POST-запрос к стриминговому эндпоинту чата
        const response = await fetch(`/api/chat/stream?token=${token}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ messages: chatHistory })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Ошибка генерации ответа');
        }

        // 3. Читаем SSE-поток асинхронно
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Оставляем последний незавершенный кусок

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;

                if (trimmed.startsWith('data: ')) {
                    const dataContent = trimmed.slice(6);
                    if (dataContent.trim() === '[DONE]') continue;

                    try {
                        const parsed = JSON.parse(dataContent);
                        
                        if (parsed.token) {
                            if (!tokenReceived) {
                                // Удаляем индикатор загрузки при первом токене
                                removeThinkingIndicator(thinkingBubble);
                                tokenReceived = true;
                            }
                            fullResponseText += parsed.token;
                            // Временный простой рендеринг текста
                            textSpan.innerText = fullResponseText;
                            scrollToBottom();
                        } else if (parsed.sources) {
                            // Рендерим блок источников (Story 3.4)
                            appendSources(assistantMessageDiv, parsed.sources);
                            scrollToBottom();
                        } else if (parsed.error) {
                            throw new Error(parsed.error);
                        }
                    } catch (err) {
                        console.warn('Ошибка парсинга чанка стрима:', err);
                    }
                }
            }
        }

        // Рендерим Markdown в HTML после окончания стрима (Story 3.4)
        if (typeof marked !== 'undefined') {
            textSpan.innerHTML = marked.parse(fullResponseText);
        } else {
            textSpan.innerText = fullResponseText;
        }

        // Подсвечиваем синтаксис кода (Story 3.4)
        if (typeof Prism !== 'undefined') {
            Prism.highlightAllUnder(assistantMessageDiv);
        }

        // Добавляем ответ ассистента в историю диалога
        chatHistory.push({ role: 'assistant', content: fullResponseText });

    } catch (err) {
        console.error(err);
        removeThinkingIndicator(thinkingBubble);
        if (!tokenReceived) {
            textSpan.innerHTML = `<span style="color: var(--danger-color); font-weight: bold;">Ошибка: ${err.message}</span>`;
        } else {
            const errDiv = document.createElement('div');
            errDiv.style.color = 'var(--danger-color)';
            errDiv.style.marginTop = '8px';
            errDiv.innerText = `[Поток прерван: ${err.message}]`;
            assistantMessageDiv.appendChild(errDiv);
        }
        scrollToBottom();
    } finally {
        isGenerating = false;
        // Удаляем индикатор, если он вдруг остался
        removeThinkingIndicator(thinkingBubble);
    }
}

/**
 * Добавляет обычное текстовое сообщение в UI.
 */
function appendMessage(role, content) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const p = document.createElement('p');
    p.innerText = content;
    messageDiv.appendChild(p);

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Выводит индикатор "думания" модели.
 */
function appendThinkingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    const bubble = document.createElement('div');
    bubble.className = 'thinking-bubble';
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'thinking-dot';
        bubble.appendChild(dot);
    }

    chatMessages.appendChild(bubble);
    scrollToBottom();
    return bubble;
}

/**
 * Удаляет индикатор "думания".
 */
function removeThinkingIndicator(bubble) {
    if (bubble && bubble.parentNode) {
        bubble.parentNode.removeChild(bubble);
    }
}

/**
 * Рендерит блок источников под ответом ассистента.
 */
function appendSources(containerDiv, sources) {
    if (!sources || sources.length === 0) return;

    const sourcesContainer = document.createElement('div');
    sourcesContainer.className = 'sources-container';

    const title = document.createElement('div');
    title.className = 'sources-title';
    title.innerText = 'Источники контекста:';
    sourcesContainer.appendChild(title);

    const list = document.createElement('div');
    list.className = 'sources-list';

    sources.forEach(src => {
        const tag = document.createElement('span');
        tag.className = 'source-tag';
        tag.innerText = `${src.filepath} (строки ${src.lines})`;
        
        tag.addEventListener('click', () => {
            openFileSource(src.filepath, src.lines);
        });
        list.appendChild(tag);
    });

    sourcesContainer.appendChild(list);
    containerDiv.appendChild(sourcesContainer);
}

/**
 * Запрашивает содержимое файла с бэкенда и выводит его в модальном окне (Story 3.4).
 */
async function openFileSource(filepath, linesRange) {
    const modal = document.getElementById('source-modal');
    const modalFilename = document.getElementById('modal-filename');
    const modalCodeBlock = document.getElementById('modal-code-block');

    if (!modal || !modalFilename || !modalCodeBlock) return;

    modalFilename.innerText = `Загрузка: ${filepath}...`;
    modalCodeBlock.textContent = 'Пожалуйста, подождите...';
    modal.style.display = 'flex';

    // Парсим диапазон строк (например "10-25")
    let startLine = 1;
    let endLine = null;
    if (linesRange && linesRange.includes('-')) {
        const parts = linesRange.split('-');
        startLine = parseInt(parts[0], 10) || 1;
        endLine = parseInt(parts[1], 10) || null;
    }

    try {
        const url = `/api/chat/file?filepath=${encodeURIComponent(filepath)}&start_line=${startLine}` + 
                    (endLine ? `&end_line=${endLine}` : '');
                    
        const response = await fetch(url);
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Не удалось загрузить файл');
        }

        const data = await response.json();
        
        modalFilename.innerText = `${filepath} (строки ${data.start_line}-${data.end_line})`;
        modalCodeBlock.textContent = data.content;

        // Определяем язык для Prism подсветки по расширению
        modalCodeBlock.className = '';
        const ext = filepath.split('.').pop().toLowerCase();
        if (ext === 'py') {
            modalCodeBlock.classList.add('language-python');
        } else if (ext === 'js') {
            modalCodeBlock.classList.add('language-javascript');
        } else if (ext === 'css') {
            modalCodeBlock.classList.add('language-css');
        } else if (ext === 'html') {
            modalCodeBlock.classList.add('language-html');
        } else if (ext === 'yaml' || ext === 'yml') {
            modalCodeBlock.classList.add('language-yaml');
        } else if (ext === 'json') {
            modalCodeBlock.classList.add('language-json');
        } else {
            modalCodeBlock.classList.add('language-clike');
        }

        if (typeof Prism !== 'undefined') {
            Prism.highlightElement(modalCodeBlock);
        }

    } catch (err) {
        modalFilename.innerText = 'Ошибка загрузки';
        modalCodeBlock.textContent = `Не удалось загрузить исходный код:\n${err.message}`;
    }
}

/**
 * Плавно прокручивает чат к последнему сообщению.
 */
function scrollToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTo({
        top: chatMessages.scrollHeight,
        behavior: 'smooth'
    });
}
