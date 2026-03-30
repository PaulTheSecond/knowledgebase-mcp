import logging

logger = logging.getLogger(__name__)

class LocalEmbedder:
    """
    Вычисляет вектора(embeddings) прямо внутри приложения (in-process)
    через CPU или GPU без необходимости скачивать/поднимать внешнюю Ollama.
    
    ВАЖНО: import sentence_transformers (который тянет за собой torch ~30 сек)
    выполняется лениво — только при первом вызове, а не при старте процесса.
    Это критически важно для MCP-режима, где handshake должен пройти мгновенно.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        logger.info(f"Loading local embedding model: {model_name}... (this may take a while on first run)")
        # Ленивый импорт: sentence_transformers тянет за собой torch (~30 сек на импорт),
        # поэтому импортируем только когда реально нужна модель.
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        logger.info("Local model loaded successfully.")
        
    def embed_text(self, text: str) -> list[float]:
        try:
            # encode возвращает numpy array, конвертируем в Python list
            vector = self.model.encode(text)
            return vector.tolist()
        except Exception as e:
            logger.error(f"Failed to compute local embedding: {e}")
            return []
