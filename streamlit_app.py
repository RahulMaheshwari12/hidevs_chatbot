"""
HiDevs Support Chat - Streamlit front end.

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file = streamlit_app.py
"""

import os

import streamlit as st


def _load_streamlit_secrets_into_env() -> None:
    """Copy Streamlit secrets into environment variables.

    On Streamlit Community Cloud, secrets are set in the app settings
    rather than in a .env file. The rest of the project reads its
    configuration with os.getenv, so secrets are copied there. Existing
    environment variables (e.g. from .env) are not overwritten.
    """
    try:
        secrets = dict(st.secrets)
    except Exception:
        # No secrets file present (normal for local runs using .env).
        return

    for key, value in secrets.items():
        if isinstance(value, (str, int, float, bool)) and not os.getenv(key):
            os.environ[key] = str(value)


_load_streamlit_secrets_into_env()

from agent.hidevs_agent import HiDevsSupportAgent  # noqa: E402


st.set_page_config(
    page_title="HiDevs Support",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource(show_spinner="Connecting to the knowledge base...")
def get_agent() -> HiDevsSupportAgent:
    """Create the agent once per server process and reuse it."""
    return HiDevsSupportAgent()


st.title("HiDevs Support")
st.caption("Ask me anything about HiDevs programs, projects and platform features.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                for index, source in enumerate(message["sources"], start=1):
                    section = source["metadata"].get("section_title", "Section")
                    st.markdown(f"**{index}. {section}**")
                    st.text(source["content"][:500])

prompt = st.chat_input("Type your question")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching HiDevs knowledge base..."):
                response = get_agent().process_message(prompt)
            answer = response["answer"]
            sources = response["sources"]
        except ValueError as exc:
            answer = str(exc)
            sources = []
        except Exception:
            answer = (
                "Sorry, something went wrong while answering your question. "
                "Please try again in a moment."
            )
            sources = []

        st.markdown(answer)
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for index, source in enumerate(sources, start=1):
                    section = source["metadata"].get("section_title", "Section")
                    st.markdown(f"**{index}. {section}**")
                    st.text(source["content"][:500])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
