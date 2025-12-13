import streamlit as st

from coachai.client.supabase_client import SupabaseClient


def render_sidebar(config, agent) -> None:
    st.header("🔐 Account")

    sup = None
    try:
        sup = SupabaseClient()
    except Exception:
        sup = None

    if sup:
        if 'user_id' not in st.session_state:
            st.markdown("**Sign in / Sign up**")
            email = st.text_input("Email", key='auth_email')
            password = st.text_input("Password", type='password', key='auth_password')
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Sign In"):
                    try:
                        resp = sup.auth_sign_in(email, password)
                        user = resp.get('user') or getattr(resp, 'user', None) or resp.get('data')
                        session = resp.get('session')

                        uid = None
                        if isinstance(user, dict):
                            uid = user.get('id') or user.get('user', {}).get('id') if user.get('user') else user.get('id')
                        elif user is not None:
                            uid = getattr(user, 'id', None) or getattr(user, 'user', None) and getattr(user.user, 'id', None)

                        if uid:
                            st.session_state.user_id = uid

                            access_token = None
                            refresh_token = None
                            if isinstance(session, dict):
                                access_token = session.get('access_token')
                                refresh_token = session.get('refresh_token')
                            elif session is not None:
                                access_token = getattr(session, 'access_token', None)
                                refresh_token = getattr(session, 'refresh_token', None)

                            st.session_state.supabase_access_token = access_token
                            st.session_state.supabase_refresh_token = refresh_token

                            try:
                                agent.service.set_user_context(uid, access_token=access_token, refresh_token=refresh_token)
                                agent.knowledge_repo.load()
                            except Exception:
                                pass

                            st.success("Signed in")
                        else:
                            st.error("Sign in failed or user id not available")
                    except Exception as e:
                        st.error(f"Sign in error: {e}")
            with col2:
                if st.button("Sign Up"):
                    try:
                        resp = sup.auth_sign_up(email, password)
                        user = resp.get('user') or getattr(resp, 'user', None) or resp.get('data')
                        session = resp.get('session')

                        uid = None
                        if isinstance(user, dict):
                            uid = user.get('id') or user.get('user', {}).get('id') if user.get('user') else user.get('id')
                        elif user is not None:
                            uid = getattr(user, 'id', None) or getattr(user, 'user', None) and getattr(user.user, 'id', None)

                        if uid:
                            st.session_state.user_id = uid

                            access_token = None
                            refresh_token = None
                            if isinstance(session, dict):
                                access_token = session.get('access_token')
                                refresh_token = session.get('refresh_token')
                            elif session is not None:
                                access_token = getattr(session, 'access_token', None)
                                refresh_token = getattr(session, 'refresh_token', None)

                            st.session_state.supabase_access_token = access_token
                            st.session_state.supabase_refresh_token = refresh_token

                            try:
                                agent.service.set_user_context(uid, access_token=access_token, refresh_token=refresh_token)
                                agent.knowledge_repo.load()
                            except Exception:
                                pass

                            st.success("Signed up")
                        else:
                            st.info("Check email for confirmation link if required")
                    except Exception as e:
                        st.error(f"Sign up error: {e}")
        else:
            st.markdown(f"Signed in: `{st.session_state.get('user_id')}`")
            if st.button("Sign Out"):
                st.session_state.pop('user_id', None)
                st.session_state.pop('supabase_access_token', None)
                st.session_state.pop('supabase_refresh_token', None)
                try:
                    agent.service.set_user_context(None)
                    agent.knowledge_repo.load()
                except Exception:
                    pass
                st.success("Signed out")

    st.markdown("---")
    st.header("📚 Knowledge Base")
    st.write(f"Lessons: {len(agent.knowledge_repo.all())}")

    st.markdown("---")
    st.info(f"**Model:** {config.MODEL_NAME}")
    st.info(f"**Device:** {agent.model_handler.device}")

    if st.session_state.get('operation_running'):
        st.error(f"⚠️ {st.session_state.get('operation_type')} in progress...")
        if st.button("⏹️ Stop Operation", type="secondary"):
            st.session_state.stop_requested = True
            st.rerun()

    st.markdown("---")

    if st.button("📖 View Topics"):
        uid = st.session_state.get('user_id')
        if not uid:
            st.info("Sign in to view your topics")
        else:
            owned = [l for l in agent.knowledge_repo.all() if l.get('owner_id') and str(l.get('owner_id')) == str(uid)]
            if not owned:
                st.info("You have no saved topics yet.")
            for l in owned:
                st.markdown(f"**{l.get('topic')}** - {l.get('subject')}")
