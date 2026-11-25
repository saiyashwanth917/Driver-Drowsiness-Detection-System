import streamlit as st

from app.database import (
    initialize_database,
    register_user,
    authenticate_user,
)


def _logout():
    processor = st.session_state.get("drowsiness_processor")
    if processor is not None:
        try:
            processor.close()
        except Exception:
            pass

    st.session_state["drowsiness_processor"] = None
    st.session_state.pop("authenticated_user", None)
    st.rerun()


def require_login():
    initialize_database()

    user = st.session_state.get("authenticated_user")

    if user:
        left, right = st.columns([5, 1])
        with left:
            st.caption(
                f"Signed in as **{user['name']}** · {user['email']}"
            )
        with right:
            if st.button("LOG OUT", use_container_width=True):
                _logout()
        return

    st.markdown("""
        <div style="text-align:center;padding:35px 10px 15px">
            <h1>DRIVEGUARD AI</h1>
            <p>HYBRID DRIVER SAFETY MONITORING SYSTEM</p>
        </div>
    """, unsafe_allow_html=True)

    login_tab, register_tab = st.tabs(["LOGIN", "CREATE ACCOUNT"])

    with login_tab:
        with st.form("driveguard_login"):
            email = st.text_input("Email address")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "LOGIN", type="primary", use_container_width=True
            )

        if submitted:
            user = authenticate_user(email, password)
            if user is None:
                st.error("Invalid email address or password.")
            else:
                st.session_state["authenticated_user"] = user
                st.rerun()

    with register_tab:
        with st.form("driveguard_register"):
            name = st.text_input("Full name")
            new_email = st.text_input("Account email")
            emergency_email = st.text_input(
                "Emergency contact email",
                help="Critical-risk alerts will be sent to this address."
            )
            new_password = st.text_input(
                "Create password", type="password"
            )
            confirm_password = st.text_input(
                "Confirm password", type="password"
            )
            create = st.form_submit_button(
                "CREATE ACCOUNT",
                type="primary",
                use_container_width=True
            )

        if create:
            if new_password != confirm_password:
                st.error("The passwords do not match.")
            else:
                try:
                    user = register_user(
                        name, new_email, new_password, emergency_email
                    )
                    st.session_state["authenticated_user"] = user
                    st.success("Account created successfully.")
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))

    st.stop()
