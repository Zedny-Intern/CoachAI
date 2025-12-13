import streamlit as st
from PIL import Image

from coachai.ui.image_processor import ImageProcessor


def render_ask_tab(agent) -> None:
    col1, col2 = st.columns([2, 1])

    with col1:
        text_query = st.text_area("❓ Question:", height=100)

    uploaded_file = None
    with col2:
        st.markdown("### 📤 Image Upload")
        uploaded_file = st.file_uploader(
            "Upload image (PNG, JPG, JPEG)",
            type=['png', 'jpg', 'jpeg'],
            help="Upload images for Qwen3-VL analysis - supports math, diagrams, and handwritten content",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)

            if ImageProcessor.validate_image(image):
                try:
                    st.image(image, width='stretch', caption="Uploaded Image")
                except Exception:
                    st.image(image, use_container_width=True, caption="Uploaded Image")

                image_type = st.selectbox(
                    "📋 Image Content Type:",
                    ["General Text", "Math Equations", "Diagram/Chart", "Handwritten Notes"],
                    help="Select the type of content for optimized processing",
                )

                st.session_state.image_type = image_type

                hints = {
                    "General Text": "📄 Optimized for general document analysis and text understanding",
                    "Math Equations": "🔢 Enhanced mathematical symbol and equation recognition",
                    "Diagram/Chart": "📊 Advanced visual analysis for diagrams and data charts",
                    "Handwritten Notes": "✍️ Specialized recognition for handwritten content",
                }
                st.info(f"ℹ️ {hints[image_type]}")

                original_size = image.size
                image = ImageProcessor.resize_image(image)
                if image.size != original_size:
                    st.info(
                        f"📏 Image resized from {original_size[0]}×{original_size[1]} to {image.size[0]}×{image.size[1]} for optimal processing"
                    )

                st.caption(f"📏 Processed image size: {image.size[0]}×{image.size[1]} pixels")
            else:
                st.error("❌ Please upload a valid image file")

    if st.button("🚀 Get Explanation", type="primary", disabled=st.session_state.get('operation_running', False)):
        if not text_query and not uploaded_file:
            st.warning("Enter a question or upload an image!")
            return

        if st.session_state.get('operation_running', False):
            st.warning("Another operation is already running. Please wait or stop it first.")
            return

        st.session_state.operation_running = True
        st.session_state.operation_type = "Generating Explanation"
        st.session_state.stop_requested = False

        try:
            with st.spinner("Analyzing..."):
                image = Image.open(uploaded_file) if uploaded_file else None
                image_type = getattr(st.session_state, 'image_type', 'General Text')
                relevant, query, _ = agent.process_query(text_query, image, image_type)

                try:
                    uid = st.session_state.get('user_id')
                    if uid:
                        image_bytes_list = [uploaded_file.getvalue()] if uploaded_file else None
                        content_types = [getattr(uploaded_file, 'type', None) or 'image/png'] if uploaded_file else None
                        st.session_state.last_query_id = agent.service.store_user_query(
                            uid,
                            query or text_query or '',
                            image_bytes_list=image_bytes_list,
                            content_types=content_types,
                        )
                except Exception:
                    pass

                if st.session_state.get('stop_requested'):
                    st.warning("❌ Operation cancelled by user")
                    return

                if relevant is None:
                    st.warning("Please provide a question!")
                    return

                st.success("✅ Found relevant material!")

                with st.expander("📚 Relevant Lessons"):
                    for l in relevant:
                        try:
                            st.markdown(f"**{l['topic']}** - {l['similarity']:.2%}\n\n{l['content']}")
                        except Exception:
                            st.markdown(f"**{l.get('topic')}**\n\n{l.get('content')}")

                with st.spinner("Generating..."):
                    explanation = agent.generate_explanation(query, relevant, image)
                    if st.session_state.get('stop_requested'):
                        st.warning("❌ Explanation generation cancelled")
                    else:
                        st.markdown("### 💡 Explanation")
                        st.markdown(explanation)
        finally:
            st.session_state.operation_running = False
            st.session_state.operation_type = None
            st.session_state.stop_requested = False
