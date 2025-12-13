"""
Multimodal Learning Coach - Streamlit Application
Run with: streamlit run app.py
"""
import os
# Reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st

from coachai.core.config import Config
from coachai.ui.sidebar import render_sidebar
from coachai.ui.streamlit_utils import init_page, init_operation_state, get_agent, render_operation_status
from coachai.ui.tabs.ask_tab import render_ask_tab
from coachai.ui.tabs.practice_tab import render_practice_tab
from coachai.ui.tabs.manage_tab import render_manage_tab


def main():
    config = Config()
    init_page(config)
    init_operation_state()

    agent = get_agent(config)

    render_operation_status()

    with st.sidebar:
        render_sidebar(config, agent)
    
    tab1, tab2, tab3 = st.tabs(["💬 Ask", "📝 Practice", "📊 Manage"])

    with tab1:
        render_ask_tab(agent)

    with tab2:
        render_practice_tab(agent)

    with tab3:
        render_manage_tab(agent)


if __name__ == "__main__":
    main()
