from typing import List, Dict, Any

import numpy as np


class KnowledgeRepositorySearchMixin:
    def load(self) -> bool:
        sup = self._get_supabase()
        if not sup:
            self.lessons = []
            return True

        try:
            res = sup.table_select('lessons', limit=5000)
            self.lessons = res.data if res and getattr(res, 'data', None) else []
            return True
        except Exception:
            self.lessons = []
            return False

    def all(self) -> List[Dict[str, Any]]:
        return self.lessons

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        pg = self._get_postgres()
        if pg:
            rows: List[Dict[str, Any]] = []
            try:
                emb = self.embed_texts([query], input_type='search_query')[0]
                rows = pg.vector_search(emb, source_table='lessons', top_k=top_k)
            except Exception:
                self._log('search: pgvector path failed (embed/vector_search threw)')
                rows = []

            sup = self._get_supabase()
            results = []
            for r in rows:
                sid = r.get('source_id')
                lesson = None
                if sup:
                    try:
                        res = sup.table_select('lessons', {'id': sid}, returning='*')
                        if res and getattr(res, 'data', None):
                            lesson = res.data[0]
                    except Exception:
                        lesson = None

                if not lesson:
                    lesson = next((l for l in self.lessons if l.get('id') == sid or str(l.get('id')) == str(sid)), None)

                if lesson:
                    lesson_copy = dict(lesson)
                    dist = r.get('distance')
                    lesson_copy['distance'] = dist
                    try:
                        lesson_copy['similarity'] = 1.0 / (1.0 + float(dist))
                    except Exception:
                        lesson_copy['similarity'] = None
                    results.append(lesson_copy)

            if results:
                return results

            self._log('search: pgvector returned 0 results, attempting Supabase RPC fallback')

        try:
            emb = self.embed_texts([query], input_type='search_query')[0]
            sup = self._get_supabase()
            if sup:
                res = sup.rpc('match_lessons', {'query_embedding': self._vector_literal(emb), 'match_count': int(top_k)})
                rows = getattr(res, 'data', None) if res is not None else None
                if rows:
                    out: List[Dict[str, Any]] = []
                    for r in rows:
                        item = dict(r)
                        dist = item.get('distance')
                        item['distance'] = dist
                        try:
                            item['similarity'] = 1.0 / (1.0 + float(dist))
                        except Exception:
                            item['similarity'] = None
                        out.append(item)
                    return out
        except Exception as e:
            self._log(f'search: supabase rpc match_lessons failed: {repr(e)}')

        if not self.lessons:
            self.load()

        texts = [f"{l.get('topic', '')}: {l.get('content', '')}" for l in self.lessons]
        if not texts:
            return []

        try:
            embeddings = self.embed_texts(texts, input_type='search_document')
            query_emb = self.embed_texts([query], input_type='search_query')[0]
        except Exception:
            return []

        embeddings_np = np.array(embeddings, dtype=float)
        query_np = np.array(query_emb, dtype=float)

        sims = (embeddings_np @ query_np.T).squeeze() / ((np.linalg.norm(embeddings_np, axis=1) * np.linalg.norm(query_np)) + 1e-10)
        ranked_idx = list(np.argsort(sims)[-min(top_k, len(sims)):][::-1])

        results = []
        for idx in ranked_idx:
            lesson_copy = dict(self.lessons[idx])
            lesson_copy['similarity'] = float(sims[idx])
            results.append(lesson_copy)
        return results
