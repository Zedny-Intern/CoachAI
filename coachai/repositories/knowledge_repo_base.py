from typing import List, Dict, Any, Optional
from pathlib import Path

from coachai.client.supabase_client import SupabaseClient
from coachai.client.postgres_client import PostgresClient
from coachai.client.cohere_client import CohereClient


class KnowledgeRepositoryBase:
    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2"):
        self._supabase_user: Optional[SupabaseClient] = None
        self._supabase_service: Optional[SupabaseClient] = None
        self._pg: Optional[PostgresClient] = None
        self._user_id: Optional[str] = None

        try:
            self._cohere = CohereClient()
        except Exception:
            self._cohere = None

        self.embed_model_name = embed_model_name
        self.lessons: List[Dict[str, Any]] = []

        try:
            Path('logs').mkdir(exist_ok=True)
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        try:
            with open('logs/knowledge_repository.log', 'a', encoding='utf-8') as lf:
                lf.write('---\n')
                lf.write(msg + '\n')
        except Exception:
            pass

    def _vector_literal(self, vector: List[float]) -> str:
        return '[' + ','.join(f'{float(x):.8f}' for x in vector) + ']'

    def set_user_context(self, user_id: Optional[str], access_token: Optional[str] = None, refresh_token: Optional[str] = None) -> None:
        self._user_id = str(user_id) if user_id else None
        if access_token:
            self._supabase_user = SupabaseClient(access_token=access_token, refresh_token=refresh_token)
        else:
            self._supabase_user = None

    def _get_supabase(self) -> Optional[SupabaseClient]:
        if self._supabase_user is not None:
            return self._supabase_user
        try:
            return SupabaseClient()
        except Exception:
            return None

    def _get_supabase_service(self) -> Optional[SupabaseClient]:
        if self._supabase_service is None:
            try:
                self._supabase_service = SupabaseClient(use_service_role=True)
            except Exception:
                self._supabase_service = None
        return self._supabase_service

    def _user_bucket(self, owner_id: str) -> str:
        return f"user-{str(owner_id).lower()}"

    def _get_postgres(self) -> Optional[PostgresClient]:
        if self._pg is None:
            try:
                self._pg = PostgresClient()
            except Exception:
                self._pg = None
        return self._pg
