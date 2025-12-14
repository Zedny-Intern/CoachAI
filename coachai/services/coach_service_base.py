from typing import Optional

from coachai.core.config import Config
from coachai.repositories.knowledge_repository import KnowledgeRepository
from coachai.services.model_handler import ModelHandler


class CoachServiceBase:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.knowledge_repo = KnowledgeRepository(self.config.EMBED_MODEL_NAME)
        self.model_handler = ModelHandler(self.config)
        self.current_user_id: Optional[str] = None

    def set_user_context(self, user_id: Optional[str], access_token: Optional[str] = None, refresh_token: Optional[str] = None) -> None:
        self.current_user_id = str(user_id) if user_id else None
        self.knowledge_repo.set_user_context(self.current_user_id, access_token=access_token, refresh_token=refresh_token)

    def initialize(self) -> bool:
        try:
            self.knowledge_repo.load()
        except Exception:
            pass
        return bool(self.model_handler.load_model())

    def find_relevant(self, query: str, top_k: Optional[int] = None):
        return self.knowledge_repo.search(query, top_k=top_k or self.config.TOP_K)
