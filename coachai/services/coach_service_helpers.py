from typing import List, Dict, Any
import re


class CoachServiceHelpersMixin:
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
