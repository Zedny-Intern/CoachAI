import streamlit as st

from coachai.ui.streamlit_utils import safe_rerun


def render_manage_tab(agent) -> None:
    st.header("📊 Manage Knowledge")

    with st.form("add"):
        new_topic = st.text_input("Topic")
        new_content = st.text_area("Content")
        new_subject = st.text_input("Subject")
        new_level = st.selectbox("Level", ["Elementary", "Middle School", "High School", "College"])

        if st.form_submit_button("Add"):
            if not st.session_state.get('user_id'):
                st.warning("You must be signed in to add a topic.")
            elif new_topic and new_content:
                added = agent.knowledge_repo.add(
                    new_topic,
                    new_content,
                    new_subject,
                    new_level,
                    owner_id=st.session_state.get('user_id'),
                )
                if not added:
                    st.error("❌ Failed to add topic to Supabase. Check server logs for details.")
                else:
                    try:
                        agent.knowledge_repo.load()
                    except Exception:
                        pass
                    st.success("✅ Added!")
                    st.rerun()

    st.markdown("---")

    uid = st.session_state.get('user_id')
    if not uid:
        st.info("Sign in to view your saved topics")
        return

    owned = [l for l in agent.knowledge_repo.all() if l.get('owner_id') and str(l.get('owner_id')) == str(uid)]
    for l in owned:
        lid = l.get('id')
        with st.expander(f"{l.get('topic')} - {l.get('subject')}"):
            st.markdown(l.get('content') or '')
            cols = st.columns([1, 1, 3])

            with cols[0]:
                if st.button("🗑️ Delete", key=f"delete_{lid}"):
                    st.session_state.delete_pending = lid
                    st.session_state['delete_topic'] = l.get('topic')
                    safe_rerun()

            with cols[1]:
                if st.button("Make Public", key=f"pub_{lid}"):
                    try:
                        sup = agent.knowledge_repo._get_supabase()
                        if sup:
                            sup.table_update('lessons', {'visibility': 'public'}, 'id', lid)
                            agent.knowledge_repo.load()
                            st.success("Topic made public")
                        else:
                            st.warning("Supabase not configured")
                    except Exception:
                        st.error("Failed to update visibility")

            with cols[2]:
                st.write("")

    if st.session_state.get('delete_pending'):
        pending_id = st.session_state.get('delete_pending')
        pending_topic = st.session_state.get('delete_topic', '')
        modal = getattr(st, 'modal', None)
        if modal:
            with st.modal("Confirm delete"):
                st.warning(f"Are you sure you want to delete '**{pending_topic}**' ? This action cannot be undone.")
                c1, c2 = st.columns(2)
                if c1.button("Confirm Delete", key=f"confirm_delete_{pending_id}"):
                    success = agent.knowledge_repo.delete_lesson(pending_id)
                    if success:
                        st.success("Deleted topic")
                        st.session_state.pop('delete_pending', None)
                        st.session_state.pop('delete_topic', None)
                        agent.knowledge_repo.load()
                        safe_rerun()
                    else:
                        st.error("Failed to delete topic. Check server logs or permissions.")
                if c2.button("Cancel", key=f"cancel_delete_{pending_id}"):
                    st.session_state.pop('delete_pending', None)
                    st.session_state.pop('delete_topic', None)
                    safe_rerun()
        else:
            with st.container():
                st.warning(f"Confirm deletion of '**{pending_topic}**' ? This action cannot be undone.")
                c1, c2 = st.columns(2)
                if c1.button("Confirm Delete", key=f"confirm_delete_{pending_id}"):
                    success = agent.knowledge_repo.delete_lesson(pending_id)
                    if success:
                        st.success("Deleted topic")
                        st.session_state.pop('delete_pending', None)
                        st.session_state.pop('delete_topic', None)
                        agent.knowledge_repo.load()
                        safe_rerun()
                    else:
                        st.error("Failed to delete topic. Check server logs or permissions.")
                if c2.button("Cancel", key=f"cancel_delete_{pending_id}"):
                    st.session_state.pop('delete_pending', None)
                    st.session_state.pop('delete_topic', None)
                    safe_rerun()
