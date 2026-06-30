import logging

logger = logging.getLogger(__name__)

class LocalEmbedder:
    """
    Вычисляет вектора(embeddings) прямо внутри приложения (in-process)
    через CPU или GPU.
    
    Использует модель intfloat/multilingual-e5-small:
    - Супер-быстрая на CPU (размерность 384)
    - Отличное качество для русского и английского языков
    - Требует префиксов 'query: ' и 'passage: ' для асимметричного поиска.
    """
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        logger.info(f"Loading local embedding model: {model_name}... (this may take a while on first run)")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        logger.info("Local model loaded successfully.")
        
    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        """Векторизация текста.
        is_query=True для пользовательских вопросов.
        is_query=False (default) для фрагментов кода и документов.
        """
        prefix = "query: " if is_query else "passage: "
        try:
            # Префиксы критически важны для качества моделей семейства E5
            full_text = prefix + text
            vector = self.model.encode(full_text)
            return vector.tolist()
        except Exception as e:
            logger.error(f"Failed to compute local embedding: {e}")
            return []

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Пакетная векторизация для документов (всегда префикс 'passage: ')."""
        if not texts:
            return []
        try:
            # Добавляем префикс к каждому элементу батча
            prefixed_texts = ["passage: " + t for t in texts]
            vectors = self.model.encode(prefixed_texts, batch_size=batch_size, show_progress_bar=False)
            return vectors.tolist()
        except Exception as e:
            logger.warning(f"Batch embedding failed ({e}), falling back to per-item processing...")
            results = []
            for i, text in enumerate(texts):
                results.append(self.embed_text(text, is_query=False))
            return results
