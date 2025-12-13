import streamlit as st

from coachai.ui.learning_coach_agent import LearningCoachAgent


def init_page(config) -> None:
    st.set_page_config(
        page_title="🎓 Multimodal Learning Coach",
        page_icon="🎓",
        layout="wide",
    )

    st.title("🎓 Multimodal Learning Coach")
    st.markdown(
        f"""
    Powered by **{config.MODEL_NAME}**
    - Ask questions • Upload images • Get explanations • Practice
    """
    )


def init_operation_state() -> None:
    if 'operation_running' not in st.session_state:
        st.session_state.operation_running = False
    if 'operation_type' not in st.session_state:
        st.session_state.operation_type = None
    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False


def safe_rerun() -> None:
    try:
        st.experimental_rerun()
    except Exception:
        st.session_state['_rerun_toggle'] = not st.session_state.get('_rerun_toggle', False)
        try:
            st.stop()
        except Exception:
            return


def get_agent(config) -> LearningCoachAgent:
    if 'agent' not in st.session_state:
        with st.spinner("Loading..."):
            st.session_state.agent = LearningCoachAgent(config)
            if not st.session_state.agent.initialize():
                st.error("Failed to load model. Check path in config.")
                st.stop()
    return st.session_state.agent


def render_operation_status() -> None:
    if st.session_state.get('operation_running'):
        st.warning(
            f"🔄 {st.session_state.get('operation_type')} in progress... Use the stop button in the sidebar to cancel."
        )
