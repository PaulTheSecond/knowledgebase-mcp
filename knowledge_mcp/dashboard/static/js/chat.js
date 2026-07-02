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
        // Заглушка клика — в Story 3.4 сделаем открытие файла
        tag.addEventListener('click', () => {
            alert(`Открытие файла: ${src.filepath}, строки: ${src.lines}`);
        });
        list.appendChild(tag);
    });

    sourcesContainer.appendChild(list);
    containerDiv.appendChild(sourcesContainer);
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
