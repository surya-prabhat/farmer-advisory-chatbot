import streamlit as st
from rag.chain import build_chain
from translation.translator import translate_from_english, translate_to_english

st.set_page_config(
    page_title="Farmer Advisory Chatbot",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Farmer Advisory Chatbot")

st.caption("Ask your Queries")

@st.cache_resource
def get_chain():
    return build_chain()


chain = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            english_query, detected_lang = translate_to_english(prompt)
            response = chain.invoke(english_query)
            final_response = translate_from_english(response, detected_lang)

        st.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})