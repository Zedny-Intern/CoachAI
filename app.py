"""
Multimodal Learning Coach - Streamlit Application
Run with: streamlit run app.py
"""
import os
# Force TensorFlow to use CPU only before any other imports
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
from PIL import Image

# Import from modular components
from src import Config, ImageProcessor, LearningCoachAgent


def main():
    config = Config()
    
    st.set_page_config(
        page_title="🎓 Multimodal Learning Coach",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Multimodal Learning Coach")
    st.markdown(f"""
    Powered by **{config.MODEL_NAME}**
    - Ask questions • Upload images • Get explanations • Practice
    """)

    # Initialize session state for operation control
    if 'operation_running' not in st.session_state:
        st.session_state.operation_running = False
    if 'operation_type' not in st.session_state:
        st.session_state.operation_type = None
    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False

    # Initialize
    if 'agent' not in st.session_state:
        with st.spinner("Loading..."):
            st.session_state.agent = LearningCoachAgent(config)
            if not st.session_state.agent.initialize():
                st.error("Failed to load model. Check path in config.")
                st.stop()

    agent = st.session_state.agent

    # Show operation status after initialization
    if st.session_state.operation_running:
        st.warning(f"🔄 {st.session_state.operation_type} in progress... Use the stop button in the sidebar to cancel.")
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Knowledge Base")
        st.write(f"Lessons: {len(agent.knowledge_base.lessons)}")

        st.markdown("---")
        st.info(f"**Model:** {config.MODEL_NAME}")
        st.info(f"**Device:** {agent.model_handler.device}")

        # Stop button for ongoing operations
        if st.session_state.operation_running:
            st.error(f"⚠️ {st.session_state.operation_type} in progress...")
            if st.button("⏹️ Stop Operation", type="secondary"):
                st.session_state.stop_requested = True
                st.rerun()

        st.markdown("---")

        if st.button("📖 View Topics"):
            for l in agent.knowledge_base.lessons:
                st.markdown(f"**{l['topic']}** - {l['subject']}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Ask", "📝 Practice", "📊 Manage"])
    
    # Ask Question
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_query = st.text_area("❓ Question:", height=100)
        
        with col2:
            st.markdown("### 📤 Image Upload")
            uploaded_file = st.file_uploader(
                "Upload image (PNG, JPG, JPEG)",
                type=['png', 'jpg', 'jpeg'],
                help="Upload images of math problems, diagrams, or text for analysis"
            )

            if uploaded_file:
                image = Image.open(uploaded_file)

                # Validate image
                if ImageProcessor.validate_image(image):
                    st.image(image, use_container_width=True, caption="Uploaded Image")

                    # Image type selector for better processing
                    image_type = st.selectbox(
                        "📋 Image Content Type:",
                        ["General Text", "Math Equations", "Diagram/Chart", "Handwritten Notes"],
                        help="Select the type of content for optimized processing"
                    )

                    # Store image type in session state for processing
                    st.session_state.image_type = image_type

                    # Show processing hints based on type
                    hints = {
                        "General Text": "📄 Text extraction optimized for printed documents",
                        "Math Equations": "🔢 Enhanced processing for mathematical symbols and formulas",
                        "Diagram/Chart": "📊 Visual understanding optimized for diagrams and charts",
                        "Handwritten Notes": "✍️ OCR tuned for handwritten text recognition"
                    }
                    st.info(f"ℹ️ {hints[image_type]}")

                    # Resize if necessary
                    original_size = image.size
                    image = ImageProcessor.resize_image(image)
                    if image.size != original_size:
                        st.info(f"📏 Image resized from {original_size[0]}×{original_size[1]} to {image.size[0]}×{image.size[1]} for optimal processing")

                    # Show image info
                    st.caption(f"📏 Processed image size: {image.size[0]}×{image.size[1]} pixels")
                else:
                    st.error("❌ Please upload a valid image file")
        
        if st.button("🚀 Get Explanation", type="primary", disabled=st.session_state.operation_running):
            if not text_query and not uploaded_file:
                st.warning("Enter a question or upload an image!")
            elif st.session_state.operation_running:
                st.warning("Another operation is already running. Please wait or stop it first.")
            else:
                # Set operation status
                st.session_state.operation_running = True
                st.session_state.operation_type = "Generating Explanation"
                st.session_state.stop_requested = False

                try:
                    with st.spinner("Analyzing..."):
                        image = Image.open(uploaded_file) if uploaded_file else None
                        image_type = getattr(st.session_state, 'image_type', 'General Text')
                        relevant, query, ocr = agent.process_query(text_query, image, image_type)

                        if st.session_state.stop_requested:
                            st.warning("❌ Operation cancelled by user")
                            st.session_state.operation_running = False
                            st.session_state.operation_type = None
                            st.rerun()

                        if relevant is None:
                            st.warning("Please provide a question!")
                            st.session_state.operation_running = False
                            st.session_state.operation_type = None
                            st.stop()

                        st.success("✅ Found relevant material!")

                        with st.expander("📚 Relevant Lessons"):
                            for l in relevant:
                                st.markdown(f"**{l['topic']}** - {l['similarity']:.2%}\n\n{l['content']}")

                        with st.spinner("Generating..."):
                            explanation = agent.generate_explanation(query, relevant, image, ocr)
                            if st.session_state.stop_requested:
                                st.warning("❌ Explanation generation cancelled")
                            else:
                                st.markdown("### 💡 Explanation")
                                st.markdown(explanation)

                finally:
                    # Reset operation status
                    st.session_state.operation_running = False
                    st.session_state.operation_type = None
                    st.session_state.stop_requested = False
    
    # Practice
    with tab2:
        st.header("📝 Practice")
        
        topic = st.selectbox("Topic:", [l['topic'] for l in agent.knowledge_base.lessons])
        
        if st.button("Generate Question", disabled=st.session_state.operation_running):
            if st.session_state.operation_running:
                st.warning("Another operation is already running. Please wait or stop it first.")
            else:
                # Set operation status
                st.session_state.operation_running = True
                st.session_state.operation_type = "Generating Question"
                st.session_state.stop_requested = False

                try:
                    with st.spinner("Creating..."):
                        question = agent.generate_practice_question(topic)
                        if st.session_state.stop_requested:
                            st.warning("❌ Question generation cancelled")
                        else:
                            st.session_state.practice_question = question
                            st.session_state.topic = topic
                finally:
                    # Reset operation status
                    st.session_state.operation_running = False
                    st.session_state.operation_type = None
                    st.session_state.stop_requested = False
        
        if 'practice_question' in st.session_state:
            st.info(st.session_state.practice_question)
            answer = st.text_area("Your Answer:", height=150)
            
            if st.button("Submit", disabled=st.session_state.operation_running):
                if st.session_state.operation_running:
                    st.warning("Another operation is already running. Please wait or stop it first.")
                elif answer:
                    # Set operation status
                    st.session_state.operation_running = True
                    st.session_state.operation_type = "Evaluating Answer"
                    st.session_state.stop_requested = False

                    try:
                        lesson = next((l['content'] for l in agent.knowledge_base.lessons
                                     if l['topic'] == st.session_state.topic), "")

                        with st.spinner("Evaluating..."):
                            feedback = agent.evaluate_answer(
                                st.session_state.practice_question, answer, lesson
                            )
                            if st.session_state.stop_requested:
                                st.warning("❌ Answer evaluation cancelled")
                            else:
                                st.success(feedback)
                    finally:
                        # Reset operation status
                        st.session_state.operation_running = False
                        st.session_state.operation_type = None
                        st.session_state.stop_requested = False
    
    # Manage
    with tab3:
        st.header("📊 Manage Knowledge")
        
        with st.form("add"):
            new_topic = st.text_input("Topic")
            new_content = st.text_area("Content")
            new_subject = st.text_input("Subject")
            new_level = st.selectbox("Level", ["Elementary", "Middle School", "High School", "College"])
            
            if st.form_submit_button("Add"):
                if new_topic and new_content:
                    agent.knowledge_base.add_lesson(new_topic, new_content, new_subject, new_level)
                    agent.knowledge_base.save_lessons(config.KNOWLEDGE_BASE_PATH)
                    st.success("✅ Added!")
                    st.rerun()
        
        st.markdown("---")
        for l in agent.knowledge_base.lessons:
            with st.expander(f"{l['topic']} - {l['subject']}"):
                st.write(l['content'])


if __name__ == "__main__":
    main()
