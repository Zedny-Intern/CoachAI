from typing import Dict, Any, Optional, List
import uuid

from coachai.client.supabase_client import SupabaseClient
from coachai.core.config import Config


class KnowledgeRepositoryLessonsMixin:
    def add(self, topic: str, content: str, subject: str, level: str, owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        sup = self._get_supabase()
        rec = {
            'owner_id': owner_id,
            'title': topic,
            'topic': topic,
            'subject': subject,
            'level': level,
            'content': content,
            'visibility': 'private'
        }

        if not sup:
            new_rec = dict(rec)
            new_rec['id'] = str(uuid.uuid4())
            self.lessons.append(new_rec)
            return new_rec

        try:
            res = sup.table_insert('lessons', rec)
            err = getattr(res, 'error', None)
            if err:
                return None

            if res and getattr(res, 'data', None):
                new_id = res.data[0].get('id')
                self.load()

                try:
                    emb = self.embed_texts([content], input_type='search_document')[0]
                    metadata = {'topic': topic, 'subject': subject, 'owner_id': owner_id}
                    if new_id:
                        eid = self.add_embedding_for_lesson(new_id, emb, metadata)
                        if not eid:
                            self._log(f'add: embedding insert returned empty id for lesson_id={new_id}')
                            try:
                                svc = self._get_supabase_service()
                                if svc:
                                    svc.table_delete('lessons', 'id', new_id)
                            except Exception as e:
                                self._log(f'add: cleanup delete failed: {repr(e)} lesson_id={new_id}')
                            return None
                except Exception as e:
                    self._log(f'add: embedding failed for lesson_id={new_id}: {repr(e)}')
                    try:
                        svc = self._get_supabase_service()
                        if svc and new_id:
                            svc.table_delete('lessons', 'id', new_id)
                    except Exception as e:
                        self._log(f'add: cleanup delete failed: {repr(e)} lesson_id={new_id}')
                    return None

                return res.data[0]
        except Exception:
            return None

        return None

    def upsert_lesson_to_supabase(self, lesson: Dict[str, Any], owner_id: Optional[str] = None) -> Optional[str]:
        sup = self._get_supabase()
        if not sup:
            return None

        rec = {
            'owner_id': owner_id,
            'title': lesson.get('topic'),
            'topic': lesson.get('topic'),
            'subject': lesson.get('subject'),
            'level': lesson.get('level'),
            'content': lesson.get('content'),
            'visibility': lesson.get('visibility', 'private')
        }
        res = sup.table_insert('lessons', rec)
        if res and getattr(res, 'data', None):
            self.load()
            return res.data[0].get('id')
        return None

    def delete_lesson(self, lesson_id: str) -> bool:
        sup = self._get_supabase()
        pg = self._get_postgres()

        def _attempt_delete(sup_client) -> bool:
            try:
                res = sup_client.table_delete('lessons', 'id', lesson_id)
                err = getattr(res, 'error', None)
                if err:
                    return False
                try:
                    self.load()
                except Exception:
                    pass
                return True
            except Exception:
                return False

        if sup:
            ok = _attempt_delete(sup)
            if not ok and Config.SUPABASE_SERVICE_ROLE_KEY:
                try:
                    svc = SupabaseClient(key=Config.SUPABASE_SERVICE_ROLE_KEY)
                    ok = _attempt_delete(svc)
                except Exception:
                    ok = False
            if not ok:
                return False
        else:
            self.lessons = [l for l in self.lessons if str(l.get('id')) != str(lesson_id)]

        if pg:
            try:
                pg.delete_embeddings_for_source('lessons', lesson_id)
            except Exception:
                pass

        return True
