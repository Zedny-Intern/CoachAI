"""
Knowledge base management with embeddings for the Multimodal Learning Coach
"""
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class KnowledgeBase:
    """Manages lessons and embeddings"""

    def __init__(self, embed_model_name="all-MiniLM-L6-v2"):
        # Use a lightweight sentence transformer model
        self.embed_model = SentenceTransformer(embed_model_name, device='cpu')
        self.lessons = []
        self.embeddings = None

    def load_lessons(self, filepath):
        """Load lessons from JSON"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.lessons = json.load(f)
        else:
            self.lessons = self._get_default_lessons()
            self.save_lessons(filepath)

        self._compute_embeddings()

    def save_lessons(self, filepath):
        """Save lessons to JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.lessons, f, indent=2, ensure_ascii=False)

    def _compute_embeddings(self):
        """Compute embeddings for all lessons"""
        texts = [f"{l['topic']}: {l['content']}" for l in self.lessons]
        self.embeddings = self.embed_model.encode(texts, convert_to_numpy=True)

    def retrieve_relevant_lessons(self, query, top_k=3):
        """Find most relevant lessons"""
        if not self.lessons:
            return []

        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Ensure top_k doesn't exceed available lessons
        actual_top_k = min(top_k, len(self.lessons))
        top_indices = np.argsort(similarities)[-actual_top_k:][::-1]

        results = []
        for idx in top_indices:
            if idx < len(self.lessons):  # Additional safety check
                results.append({
                    **self.lessons[idx],
                    'similarity': float(similarities[idx])
                })
        return results

    def add_lesson(self, topic, content, subject, level):
        """Add new lesson"""
        self.lessons.append({
            "topic": topic,
            "content": content,
            "subject": subject,
            "level": level
        })
        self._compute_embeddings()

    @staticmethod
    def _get_default_lessons():
        """Default knowledge base"""
        return [
            {
                "topic": "Pythagorean Theorem",
                "content": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse (c) equals the sum of squares of the other two sides: a² + b² = c²",
                "subject": "Mathematics",
                "level": "Middle School"
            },
            {
                "topic": "Photosynthesis",
                "content": "Photosynthesis is the process by which plants convert light energy into chemical energy. The equation is: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂",
                "subject": "Biology",
                "level": "High School"
            },
            {
                "topic": "Newton's Second Law",
                "content": "Newton's second law states that Force equals mass times acceleration: F = ma. This fundamental principle describes the relationship between force, mass, and motion.",
                "subject": "Physics",
                "level": "High School"
            },
            {
                "topic": "Quadratic Formula",
                "content": "The quadratic formula solves equations of form ax² + bx + c = 0. The solution is x = (-b ± √(b²-4ac)) / 2a",
                "subject": "Mathematics",
                "level": "High School"
            },
            {
                "topic": "Cell Structure",
                "content": "Eukaryotic cells contain a nucleus, mitochondria, endoplasmic reticulum, and other organelles. The nucleus contains DNA and controls cell activities.",
                "subject": "Biology",
                "level": "High School"
            }
        ]
