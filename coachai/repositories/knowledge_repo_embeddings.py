from typing import List, Dict, Any, Optional


class KnowledgeRepositoryEmbeddingsMixin:
    def embed_texts(self, texts: List[str], input_type: str = 'search_document') -> List[List[float]]:
        if not getattr(self, '_cohere', None) or not self._cohere.is_available():
            diag = ''
            try:
                if self._cohere is not None and hasattr(self._cohere, 'diagnostics'):
                    diag = str(self._cohere.diagnostics() or '')
            except Exception:
                diag = ''
            if diag:
                raise RuntimeError(f'Cohere embeddings not available: {diag}')
            raise RuntimeError('Cohere embeddings not available. Ensure `cohere` is installed and COHERE_API_KEY is set.')
        return self._cohere.embed(texts, input_type=input_type)

    def add_embedding_for_lesson(self, lesson_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        pg = self._get_postgres()
        if pg:
            try:
                eid = pg.insert_embedding('lessons', lesson_id, embedding, metadata)
                if eid:
                    return eid
            except Exception as e:
                self._log(f'add_embedding_for_lesson: postgres insert failed: {repr(e)} lesson_id={lesson_id}')

        svc = self._get_supabase_service()
        if not svc:
            return None
        try:
            rec = {
                'source_table': 'lessons',
                'source_id': lesson_id,
                'embedding': self._vector_literal(embedding),
                'metadata': metadata or {},
                'lesson_id': lesson_id,
            }
            res = svc.table_insert('embeddings', rec)
            if res and getattr(res, 'data', None):
                return res.data[0].get('id')
        except Exception as e:
            self._log(f'add_embedding_for_lesson: supabase insert failed: {repr(e)} lesson_id={lesson_id}')
        return None

    def add_embedding_for_source(self, source_table: str, source_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        pg = self._get_postgres()
        if pg:
            try:
                eid = pg.insert_embedding(source_table, source_id, embedding, metadata)
                if eid:
                    return eid
            except Exception as e:
                self._log(
                    f'add_embedding_for_source: postgres insert failed: {repr(e)} source_table={source_table} source_id={source_id}'
                )

        svc = self._get_supabase_service()
        if not svc:
            return None
        try:
            rec: Dict[str, Any] = {
                'source_table': source_table,
                'source_id': source_id,
                'embedding': self._vector_literal(embedding),
                'metadata': metadata or {},
            }
            if source_table == 'lessons':
                rec['lesson_id'] = source_id
            elif source_table == 'user_queries':
                rec['query_id'] = source_id
            elif source_table == 'generated_questions':
                rec['generated_question_id'] = source_id

            res = svc.table_insert('embeddings', rec)
            if res and getattr(res, 'data', None):
                return res.data[0].get('id')
        except Exception as e:
            self._log(
                f'add_embedding_for_source: supabase insert failed: {repr(e)} source_table={source_table} source_id={source_id}'
            )
        return None
