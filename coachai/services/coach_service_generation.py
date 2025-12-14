from typing import List, Dict, Any


class CoachServiceGenerationMixin:
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
                lesson_id = None
                for l in self.knowledge_repo.all():
                    if str(l.get('topic', '')).strip().lower() == str(topic).strip().lower():
                        lesson_id = l.get('id')
                        break
                self.store_generated_question(
                    lesson_id=lesson_id,
                    query_id=None,
                    question_text=q,
                    author_model=getattr(self.config, 'MODEL_NAME', '')
                )
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
