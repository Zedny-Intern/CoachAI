"""External system adapters (Supabase, Postgres, Cohere, etc.).

This package is the canonical location for outbound integrations.
"""

from coachai.client.cohere_client import CohereClient
from coachai.client.mistral_client import MistralClient
from coachai.client.postgres_client import PostgresClient
from coachai.client.supabase_client import SupabaseClient

__all__ = [
    'CohereClient',
    'MistralClient',
    'PostgresClient',
    'SupabaseClient',
]
