import streamlit as st


def render_practice_tab(agent) -> None:
    st.header("📝 Practice")

    uid = st.session_state.get('user_id')
    available_topics = (
        [l.get('topic') for l in agent.knowledge_repo.all() if l.get('owner_id') and str(l.get('owner_id')) == str(uid)]
        if uid
        else []
    )

    topic = st.selectbox("Topic:", available_topics)

    if st.button("Generate Question", disabled=st.session_state.get('operation_running', False)):
        if st.session_state.get('operation_running', False):
            st.warning("Another operation is already running. Please wait or stop it first.")
        else:
            st.session_state.operation_running = True
            st.session_state.operation_type = "Generating Question"
            st.session_state.stop_requested = False

            try:
                with st.spinner("Creating..."):
                    question = agent.generate_practice_question(topic)
                    if st.session_state.get('stop_requested'):
                        st.warning("❌ Question generation cancelled")
                    else:
                        st.session_state.practice_question = question
                        st.session_state.topic = topic
            finally:
                st.session_state.operation_running = False
                st.session_state.operation_type = None
                st.session_state.stop_requested = False

    if 'practice_question' in st.session_state:
        st.markdown(st.session_state.practice_question)
        answer = st.text_area("Your Answer:", height=150)

        if st.button("Submit", disabled=st.session_state.get('operation_running', False)):
            if st.session_state.get('operation_running', False):
                st.warning("Another operation is already running. Please wait or stop it first.")
            elif answer:
                st.session_state.operation_running = True
                st.session_state.operation_type = "Evaluating Answer"
                st.session_state.stop_requested = False

                try:
                    with st.spinner("Evaluating..."):
                        feedback = agent.evaluate_answer(
                            st.session_state.practice_question,
                            answer,
                            st.session_state.topic,
                        )
                        if st.session_state.get('stop_requested'):
                            st.warning("❌ Answer evaluation cancelled")
                        else:
                            st.markdown(feedback)
                finally:
                    st.session_state.operation_running = False
                    st.session_state.operation_type = None
                    st.session_state.stop_requested = False
