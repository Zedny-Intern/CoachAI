"""
Main agent logic for the Multimodal Learning Coach
"""
import streamlit as st
from PIL import Image
from .image_processor import ImageProcessor
from .knowledge_base import KnowledgeBase
from .model_handler import ModelHandler


class LearningCoachAgent:
    """Main agent"""

    def __init__(self, config):
        self.config = config
        self.image_processor = ImageProcessor()
        self.knowledge_base = KnowledgeBase(config.EMBED_MODEL_NAME)
        self.model_handler = ModelHandler(config)

    def initialize(self):
        """Initialize components"""
        self.knowledge_base.load_lessons(self.config.KNOWLEDGE_BASE_PATH)
        return self.model_handler.load_model()

    def process_query(self, text_query=None, image=None, image_type="General Text"):
        """Process query with enhanced image processing"""
        ocr_text = ""
        if image is not None:
            # Select processing mode based on image type
            processing_mode = "general"
            if image_type in ["Math Equations", "Handwritten Notes"]:
                processing_mode = "math"

            with st.spinner("🔍 Analyzing image..."):
                ocr_text = self.image_processor.extract_text(image, mode=processing_mode)

            if ocr_text:
                # Clean and display OCR text
                clean_ocr = ImageProcessor._post_process_physics_text(ocr_text)
                display_text = clean_ocr[:300] + "..." if len(clean_ocr) > 300 else clean_ocr
                st.info(f"📝 Extracted text: {display_text}")

                # Show processing mode and detected concepts
                mode_msg = "🔢 Math-enhanced OCR" if processing_mode == "math" else "📄 Standard OCR"
                if "🔍 Detected physics concepts:" in clean_ocr:
                    concepts = clean_ocr.split("🔍 Detected physics concepts:")[1].strip()
                    st.success(f"🎯 Detected physics concepts: {concepts}")
                st.caption(f"Processing mode: {mode_msg}")
            else:
                st.warning("⚠️ No text detected in image. The AI will analyze the visual content directly.")

        combined_query = text_query or ""
        if ocr_text:
            combined_query = f"{combined_query} {ocr_text}".strip()

        if not combined_query and image is None:
            return None, None, None

        # If we have an image but no text, create a better vision-only query
        if not combined_query and image is not None:
            if image_type == "Math Equations":
                combined_query = "This image contains a mathematics or physics problem with equations and formulas. Carefully analyze the mathematical symbols, variables, and relationships shown. Explain the physics or mathematical concepts, solve any equations visible, and provide step-by-step reasoning about Newton's laws, forces, motion, or other physics principles shown in the image."
            elif image_type == "Diagram/Chart":
                combined_query = "This image contains a physics diagram, motion graph, or force diagram. Analyze the visual elements, arrows, labels, and relationships. Explain what physical concepts are being illustrated, how forces or motion are represented, and what the diagram teaches about physics principles."
            elif image_type == "Handwritten Notes":
                combined_query = "This image contains handwritten physics notes about Newton's laws, forces, motion, or other physics concepts. Carefully examine the writing and symbols. Explain the physics concepts mentioned, any formulas or equations written, and provide a clear educational explanation of the physics principles covered in these notes. Focus on Newton's laws, force-mass-acceleration relationships, or other classical mechanics concepts."
            else:
                combined_query = "Please analyze this image and provide an explanation of any physics, mathematics, or scientific content visible, focusing on concepts like Newton's laws, forces, motion, energy, or other physics principles."

        with st.spinner("🔍 Finding relevant knowledge..."):
            relevant_lessons = self.knowledge_base.retrieve_relevant_lessons(
                combined_query,
                top_k=self.config.TOP_K
            )

            # Enhanced physics boosting for handwritten content
            has_physics_content = (
                image_type in ["Math Equations", "Handwritten Notes"] or
                (ocr_text and any(term in ocr_text.lower() for term in
                    ['newton', 'force', 'mass', 'acceleration', 'velocity', 'gravity', 'physics', 'law', 'inertia', 'energy', 'work', 'power']))
            )

            needs_physics_boost = (
                len(relevant_lessons) < 2 and has_physics_content
            ) or (
                # Boost if OCR contained physics concepts but top results aren't physics
                has_physics_content and len(relevant_lessons) > 0 and
                not any('physics' in lesson['subject'].lower() or
                       any(physics_term in lesson['topic'].lower() for physics_term in
                           ['newton', 'force', 'mass', 'acceleration', 'velocity', 'gravity', 'law', 'motion', 'energy'])
                       for lesson in relevant_lessons[:2])  # Check top 2 results
            )

            if needs_physics_boost:
                # More aggressive physics boosting
                physics_boost_terms = [
                    "physics mechanics force mass acceleration velocity newton",
                    "newton's laws f=ma motion energy work power",
                    "classical mechanics kinematics dynamics thermodynamics"
                ]

                all_physics_lessons = []
                for boost_term in physics_boost_terms:
                    boost_query = combined_query + " " + boost_term
                    physics_results = self.knowledge_base.retrieve_relevant_lessons(
                        boost_query, top_k=self.config.TOP_K * 2
                    )
                    all_physics_lessons.extend(physics_results)

                # Merge with original results and deduplicate by topic
                all_lessons = relevant_lessons + all_physics_lessons
                seen_topics = set()
                deduplicated = []
                for lesson in all_lessons:
                    topic_key = lesson['topic'].lower().strip()
                    if topic_key not in seen_topics:
                        seen_topics.add(topic_key)
                        # Boost physics-related lessons
                        if any(physics_term in topic_key for physics_term in
                               ['newton', 'force', 'mass', 'acceleration', 'physics', 'law', 'motion', 'energy', 'work', 'power', 'inertia']):
                            lesson['similarity'] = min(lesson['similarity'] * 1.5, 1.0)  # Boost similarity
                        deduplicated.append(lesson)

                # Prioritize physics lessons when physics content is detected
                if has_physics_content:
                    physics_lessons = []
                    other_lessons = []

                    for lesson in deduplicated:
                        is_physics = (
                            'physics' in lesson['subject'].lower() or
                            any(physics_term in lesson['topic'].lower() for physics_term in
                                ['newton', 'force', 'mass', 'acceleration', 'velocity', 'gravity', 'law', 'motion', 'energy', 'work', 'power', 'inertia'])
                        )
                        if is_physics:
                            physics_lessons.append(lesson)
                        else:
                            other_lessons.append(lesson)

                    # Sort physics lessons by similarity, then add other lessons
                    physics_lessons.sort(key=lambda x: x['similarity'], reverse=True)
                    other_lessons.sort(key=lambda x: x['similarity'], reverse=True)

                    # Take more physics lessons, fewer others
                    physics_count = min(len(physics_lessons), self.config.TOP_K - 1)
                    other_count = min(len(other_lessons), self.config.TOP_K - physics_count)

                    relevant_lessons = physics_lessons[:physics_count] + other_lessons[:other_count]
                else:
                    # Normal sorting for non-physics content
                    deduplicated.sort(key=lambda x: x['similarity'], reverse=True)
                    relevant_lessons = deduplicated[:self.config.TOP_K]

        return relevant_lessons, combined_query, ocr_text

    def generate_explanation(self, query, relevant_lessons, image=None, ocr_text=None):
        """Generate explanation"""
        context = "\n\n".join([
            f"**{l['topic']}**:\n{l['content']}"
            for l in relevant_lessons
        ])

        system_prompt = """You are an expert learning coach with advanced visual analysis capabilities. When analyzing images, carefully examine all visual content including handwritten text, diagrams, equations, and symbols. Provide clear, comprehensive explanations tailored to the student's level."""

        # Create more detailed prompt based on available information
        user_prompt = f"""Context from knowledge base:\n{context}\n\n"""

        if image is not None:
            image_type = getattr(st.session_state, 'image_type', 'General Text')
            user_prompt += f"""Image Analysis Request: {query}\n\n"""
            user_prompt += f"""Image Content Type: {image_type}\n\n"""

            if ocr_text:
                user_prompt += f"""OCR Extracted Text: "{ocr_text}"\n\n"""
                user_prompt += """Please analyze both the extracted text and the visual content of the image to provide a comprehensive explanation.\n\n"""
            else:
                user_prompt += """Note: OCR could not extract text from this image. Please analyze the visual content directly and explain what you see in the image.\n\n"""

            if image_type == "Math Equations":
                user_prompt += """Focus on mathematical symbols, equations, formulas, and problem-solving steps visible in the image.\n\n"""
            elif image_type == "Diagram/Chart":
                user_prompt += """Describe the diagram/chart, explain relationships shown, and interpret any data or concepts illustrated.\n\n"""
            elif image_type == "Handwritten Notes":
                user_prompt += """Carefully read the handwritten content and provide a clear explanation of the concepts, problems, or notes.\n\n"""

        else:
            user_prompt += f"""Question: {query}\n\n"""

        user_prompt += """Provide a clear, educational explanation."""

        content = []
        if image is not None:
            content.append({
                "type": "image",
                "image": image,
                "min_pixels": self.config.MIN_PIXELS,
                "max_pixels": self.config.MAX_PIXELS,
            })

        content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]

        return self.model_handler.generate(messages)

    def generate_practice_question(self, topic):
        """Generate practice question"""
        prompt = f"Create one practice question about {topic}. Make it challenging but appropriate. Provide only the question."
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.model_handler.generate(messages, max_new_tokens=256, temperature=0.8)

    def evaluate_answer(self, question, student_answer, correct_concept):
        """Evaluate answer"""
        prompt = f"""Evaluate this answer:
Question: {question}
Answer: {student_answer}
Key Concept: {correct_concept}

Provide constructive feedback."""

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.model_handler.generate(messages, max_new_tokens=512)
