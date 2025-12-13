"""Service layer for CoachAI."""

from typing import List, Dict, Any, Optional
import uuid
import re

from coachai.core.config import Config
from coachai.repositories.knowledge_repository import KnowledgeRepository
from coachai.services.model_handler import ModelHandler


class CoachService:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.knowledge_repo = KnowledgeRepository(self.config.EMBED_MODEL_NAME)
        self.model_handler = ModelHandler(self.config)
        self.current_user_id: Optional[str] = None

    def _filter_relevant_to_user(self, relevant: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.current_user_id:
            return relevant
        out: List[Dict[str, Any]] = []
        for r in relevant or []:
            try:
                if str(r.get('owner_id') or '') == str(self.current_user_id):
                    out.append(r)
            except Exception:
                pass
        return out

    def _format_retrieved_section(self, relevant: List[Dict[str, Any]], max_chars: int = 900) -> str:
        if not relevant:
            return 'Retrieved documents: none available.'

        retrieved_lines: List[str] = []
        for l in relevant:
            lid = l.get('id')
            topic = l.get('topic', '')
            subject = l.get('subject', '')
            sim = l.get('similarity', None)
            sim_str = f"{float(sim):.4f}" if sim is not None else "N/A"
            content_text = (l.get('content', '') or '')
            content_text = content_text[:max_chars]
            retrieved_lines.append(
                f"ID: {lid}\nTopic: {topic}\nSubject: {subject}\nSimilarity: {sim_str}\n{content_text}\n---"
            )
        return 'Retrieved documents:\n' + "\n".join(retrieved_lines)

    def _postprocess_math_markdown(self, text: str) -> str:
        if not text:
            return text

        out = str(text)

        # Convert bracketed math like: [ a^2 + b^2 = c^2 ] into display math.
        def _bracket_to_display(m: re.Match) -> str:
            inner = (m.group(1) or '').strip()
            if not inner:
                return m.group(0)

            # Only treat as math if it contains typical math tokens.
            math_tokens = ['=', '^', '\\sqrt', '\\frac', '+', '-', '*', '/', '\\', '_']
            if not any(tok in inner for tok in math_tokens):
                return m.group(0)

            # Avoid producing nested $$ blocks.
            if inner.startswith('$$') and inner.endswith('$$'):
                return m.group(0)

            return f"\n\n$$\n{inner}\n$$\n\n"

        out = re.sub(r"\[\s*([^\]]+?)\s*\]", _bracket_to_display, out)

        # Convert parenthesized inline math like: ( a = 5 ) or ( c ) into inline math.
        def _paren_to_inline(m: re.Match) -> str:
            inner = (m.group(1) or '').strip()
            if not inner:
                return m.group(0)

            # Only convert if the content looks like a short math expression.
            if len(inner) > 40:
                return m.group(0)

            if not re.fullmatch(r"[A-Za-z0-9\s=+\-*/^_\\{}\.]+", inner):
                return m.group(0)

            math_tokens = ['=', '^', '\\', '_']
            if not any(tok in inner for tok in math_tokens) and not re.fullmatch(r"[A-Za-z]", inner):
                return m.group(0)

            return f"${inner}$"

        out = re.sub(r"\(\s*([^\)]+?)\s*\)", _paren_to_inline, out)

        return out

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

    def generate_explanation(self, query: str, relevant: List[Dict[str, Any]], image=None):
        if not relevant:
            try:
                relevant = self.find_relevant(query, top_k=self.config.TOP_K)
            except Exception:
                relevant = []

        relevant = self._filter_relevant_to_user(relevant)

        if relevant:
            retrieved_lines = []
            for l in relevant:
                lid = l.get('id')
                topic = l.get('topic', '')
                content_text = l.get('content', '')
                sim = l.get('similarity', None)
                sim_str = f"{float(sim):.4f}" if sim is not None else "N/A"
                retrieved_lines.append(f"ID: {lid}\nTopic: {topic}\nSimilarity: {sim_str}\n{content_text}\n---")
            retrieved_section = "Retrieved documents:\n" + "\n".join(retrieved_lines)
        else:
            retrieved_section = "Retrieved documents: none available."

        system_prompt = (
            "You are an expert learning coach with advanced visual analysis capabilities. "
            "Use the retrieved documents to ground answers and cite document IDs when relevant. "
            "When writing math/science equations, format them in LaTeX. "
            "Use $$ ... $$ for standalone centered equations and $ ... $ for inline math."
        )
        user_prompt = (
            f"{retrieved_section}\n\n"
            f"Question: {query}\n\n"
            "Provide a clear, educational explanation that directly uses the retrieved documents."
        )

        content = []
        if image is not None:
            content.append({
                'type': 'image',
                'image': image,
                'min_pixels': self.config.MIN_PIXELS,
                'max_pixels': self.config.MAX_PIXELS,
            })

        content.append({'type': 'text', 'text': user_prompt})

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': content}
        ]

        resp = self.model_handler.generate(messages)
        return self._postprocess_math_markdown(resp)

    def generate_practice_question(self, topic: str):
        lesson_text = ''
        try:
            for l in self.knowledge_repo.all() or []:
                if str(l.get('topic', '')).strip().lower() == str(topic).strip().lower():
                    if self.current_user_id and str(l.get('owner_id') or '') != str(self.current_user_id):
                        continue
                    lesson_text = str(l.get('content') or '')
                    break
        except Exception:
            lesson_text = ''

        retrieval_query = lesson_text if lesson_text else topic
        try:
            relevant = self.find_relevant(retrieval_query, top_k=self.config.TOP_K)
        except Exception:
            relevant = []
        relevant = self._filter_relevant_to_user(relevant)

        retrieved_section = self._format_retrieved_section(relevant, max_chars=1200)
        system_prompt = (
            "You are a learning coach. You must ONLY use the retrieved documents as your source of truth. "
            "Do not introduce concepts, facts, or terminology that are not present in the retrieved documents. "
            "If the retrieved documents are insufficient to create a good question, say: INSUFFICIENT_MATERIAL. "
            "When writing math/science equations, format them in LaTeX. "
            "Use $$ ... $$ for standalone centered equations and $ ... $ for inline math."
        )
        user_prompt = (
            f"{retrieved_section}\n\n"
            f"Task: Create ONE practice question that can be answered using ONLY the retrieved documents.\n"
            f"Target topic label: {topic}\n\n"
            "Requirements:\n"
            "- The question must be tightly grounded in the lesson wording and scope.\n"
            "- Avoid broad/general textbook questions not covered in the documents.\n"
            "- Provide only the question text (no explanation, no answer)."
        )

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': [{'type': 'text', 'text': user_prompt}]},
        ]
        q = self.model_handler.generate(messages, max_new_tokens=256, temperature=0.8)
        q = self._postprocess_math_markdown(q)

        # Persist generated question (best-effort) when authenticated.
        try:
            if self.current_user_id:
                # Try to resolve lesson_id from cached lessons
                lesson_id = None
                for l in self.knowledge_repo.all():
                    if str(l.get('topic', '')).strip().lower() == str(topic).strip().lower():
                        lesson_id = l.get('id')
                        break
                self.store_generated_question(lesson_id=lesson_id, query_id=None, question_text=q, author_model=getattr(self.config, 'MODEL_NAME', ''))
        except Exception:
            pass

        return q

    def evaluate_answer(self, question: str, student_answer: str, correct_concept: str):
        retrieval_query = f"Question: {question}\nStudent answer: {student_answer}".strip()
        try:
            relevant = self.find_relevant(retrieval_query, top_k=self.config.TOP_K)
        except Exception:
            relevant = []
        relevant = self._filter_relevant_to_user(relevant)

        retrieved_section = self._format_retrieved_section(relevant, max_chars=1400)

        system_prompt = (
            "You are a strict grader. Grade ONLY against the retrieved documents. "
            "Do not reward knowledge that is not present in the retrieved documents. "
            "If the retrieved documents do not contain enough information to grade reliably, "
            "say: INSUFFICIENT_MATERIAL_TO_GRADE. "
            "When writing math/science equations, format them in LaTeX. "
            "Use $$ ... $$ for standalone centered equations and $ ... $ for inline math."
        )

        correct_concept_text = (correct_concept or '')
        if len(correct_concept_text) > 600:
            correct_concept_text = correct_concept_text[:600]

        user_prompt = (
            f"{retrieved_section}\n\n"
            "Task: Evaluate the student's answer using ONLY the retrieved documents.\n"
            f"Question: {question}\n"
            f"Student answer: {student_answer}\n"
            f"Instructor reference (optional, may be incomplete): {correct_concept_text}\n\n"
            "Output format (plain text):\n"
            "Score: <0-10>\\10"
            "\nFeedback: <2-6 sentences>\n"
            "Model answer (grounded): <1-5 sentences>\n"
            "Citations: <comma-separated document IDs used, or 'none'>"
        )

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': [{'type': 'text', 'text': user_prompt}]},
        ]
        resp = self.model_handler.generate(messages, max_new_tokens=512)
        resp = self._postprocess_math_markdown(resp)

        try:
            sup = self.knowledge_repo._get_supabase()
            if sup and self.current_user_id:
                rec = {
                    'question_id': None,
                    'user_id': self.current_user_id,
                    'user_answer': student_answer,
                    'model_answer': resp,
                    'grade': None,
                    'feedback': None
                }
                sup.table_insert('answers', rec)
        except Exception:
            pass

        return resp

    def store_user_query(self, user_id: str, text_query: str, image_bytes_list: Optional[list] = None, content_types: Optional[list] = None) -> Optional[str]:
        try:
            attachment_ids = []
            if image_bytes_list:
                for i, b in enumerate(image_bytes_list):
                    bucket = self.config.SUPABASE_STORAGE_BUCKET
                    # Per-user bucket is already unique; keep object names simple.
                    path = f"attachments/{uuid.uuid4().hex}_{i}.png"
                    att = self.knowledge_repo.upload_attachment(user_id, bucket, path, b, content_type=(content_types[i] if content_types and i < len(content_types) else 'image/png'))
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
                            self.knowledge_repo._log(f'store_user_query: failed to update attachment id={aid} query_id={qid}: {repr(e)}')
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
