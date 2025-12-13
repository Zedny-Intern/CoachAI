"""Simple Cohere embeddings client wrapper."""

from typing import List, Optional, Any

try:
    import cohere
except Exception:
    cohere = None

from coachai.core.config import Config


class CohereClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or Config.COHERE_API_KEY
        # Prefer a 384-dim embed model by default to match PGVECTOR_DIMENSION=384.
        cfg_model = Config.COHERE_MODEL
        if isinstance(cfg_model, str) and cfg_model.strip().lower() in ('small', 'medium', 'large'):
            cfg_model = ''
        self.model = model or cfg_model or 'embed-multilingual-light-v3.0'
        self._client = None
        self._init_error: Optional[str] = None

        if not self.api_key:
            self._init_error = 'COHERE_API_KEY not set'
            return
        if cohere is None:
            self._init_error = 'cohere package not installed or import failed'
            return

        # Try to initialize client across cohere SDK versions.
        try:
            self._client = cohere.Client(self.api_key)
            return
        except Exception as e:
            self._init_error = f'cohere.Client init failed: {type(e).__name__}: {e}'

        try:
            if hasattr(cohere, 'ClientV2'):
                self._client = cohere.ClientV2(self.api_key)
                self._init_error = None
                return
        except Exception as e:
            self._init_error = f'cohere.ClientV2 init failed: {type(e).__name__}: {e}'

        try:
            # Some versions expect keyword api_key
            self._client = cohere.Client(api_key=self.api_key)
            self._init_error = None
            return
        except Exception as e:
            self._init_error = f'cohere.Client(api_key=...) init failed: {type(e).__name__}: {e}'

    def is_available(self) -> bool:
        return self._client is not None

    def diagnostics(self) -> str:
        return self._init_error or ''

    def embed(self, texts: List[str], input_type: str = 'search_document') -> List[List[float]]:
        if not self.is_available():
            raise RuntimeError('Cohere client is not available or COHERE_API_KEY not set')
        # Cohere SDK (cohere>=5) uses client.embed with required input_type for v3 models.
        resp: Any = self._client.embed(texts=texts, model=self.model, input_type=input_type)

        vectors: Any = None
        if isinstance(resp, dict):
            vectors = resp.get('embeddings')
        else:
            vectors = getattr(resp, 'embeddings', None)

        # Newer SDK can return embeddings_by_type
        if vectors is not None and not isinstance(vectors, list):
            try:
                vectors = getattr(vectors, 'float', None) or getattr(vectors, 'floats', None)
            except Exception:
                vectors = None

        if not vectors:
            raise RuntimeError('Cohere embed returned no embeddings')

        dim = getattr(Config, 'PGVECTOR_DIMENSION', 384) or 384
        if vectors and len(vectors[0]) != int(dim):
            raise RuntimeError(f'Cohere embedding dimension {len(vectors[0])} does not match PGVECTOR_DIMENSION={dim}. Update COHERE_MODEL or PGVECTOR_DIMENSION.')
        return vectors
