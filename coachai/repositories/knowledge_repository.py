"""Repository layer for knowledge base operations (Supabase/Postgres-backed)."""

from coachai.repositories.knowledge_repo_base import KnowledgeRepositoryBase
from coachai.repositories.knowledge_repo_embeddings import KnowledgeRepositoryEmbeddingsMixin
from coachai.repositories.knowledge_repo_search import KnowledgeRepositorySearchMixin
from coachai.repositories.knowledge_repo_lessons import KnowledgeRepositoryLessonsMixin
from coachai.repositories.knowledge_repo_attachments import KnowledgeRepositoryAttachmentsMixin


class KnowledgeRepository(
    KnowledgeRepositoryBase,
    KnowledgeRepositoryEmbeddingsMixin,
    KnowledgeRepositorySearchMixin,
    KnowledgeRepositoryLessonsMixin,
    KnowledgeRepositoryAttachmentsMixin,
):
    pass
