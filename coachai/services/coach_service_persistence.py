from typing import Optional
import uuid


class CoachServicePersistenceMixin:
    def store_user_query(self, user_id: str, text_query: str, image_bytes_list: Optional[list] = None, content_types: Optional[list] = None) -> Optional[str]:
        try:
            attachment_ids = []
            if image_bytes_list:
                for i, b in enumerate(image_bytes_list):
                    bucket = self.config.SUPABASE_STORAGE_BUCKET
                    # Per-user bucket is already unique; keep object names simple.
                    path = f"attachments/{uuid.uuid4().hex}_{i}.png"
                    att = self.knowledge_repo.upload_attachment(
                        user_id,
                        bucket,
                        path,
                        b,
                        content_type=(content_types[i] if content_types and i < len(content_types) else 'image/png')
                    )
                    if att and att.get('id'):
                        attachment_ids.append(att.get('id'))

            emb = self.knowledge_repo.embed_texts([text_query], input_type='search_query')[0]

            sup = self.knowledge_repo._get_supabase()
            qid = None
            if sup:
                rec = {'user_id': user_id, 'text_query': text_query, 'image_attachment_ids': attachment_ids}
                res = sup.table_insert('user_queries', rec)
                if res and getattr(res, 'data', None):
                    qid = res.data[0].get('id')

            # Backfill query_id + metadata on attachments now that query exists.
            if sup and qid and attachment_ids:
                for idx, aid in enumerate(attachment_ids):
                    try:
                        md = {
                            'source': 'user_query',
                            'query_id': str(qid),
                            'user_id': str(user_id),
                            'index': idx,
                            'content_type': (content_types[idx] if content_types and idx < len(content_types) else None),
                        }
                        sup.table_update('attachments', {'query_id': qid, 'metadata': md}, 'id', aid)
                    except Exception as e:
                        try:
                            self.knowledge_repo._log(
                                f'store_user_query: failed to update attachment id={aid} query_id={qid}: {repr(e)}'
                            )
                        except Exception:
                            pass

            if qid:
                self.knowledge_repo.add_embedding_for_source('user_queries', qid, emb, {'source': 'user_query'})

            return qid
        except Exception:
            return None

    def store_generated_question(self, lesson_id: Optional[str], query_id: Optional[str], question_text: str, author_model: str = '') -> Optional[str]:
        sup = self.knowledge_repo._get_supabase()
        if not sup:
            return None
        rec = {'lesson_id': lesson_id, 'query_id': query_id, 'author_model': author_model, 'question_text': question_text}
        res = sup.table_insert('generated_questions', rec)
        if res and getattr(res, 'data', None):
            return res.data[0].get('id')
        return None
