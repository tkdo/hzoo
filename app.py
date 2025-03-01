import streamlit as st
import os

st.title('Chat with Groq')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for messages in st.session_state.messages:
    with chat_message(messages['role']):
        st.markdown(messages['content'], unsafe_allow_html=True)

model_max_tokens = {
    'llama3-70b-8192': 8192,
    'llama3-8b-8192': 8192,
    'mixtral-8x7b-32768': 32768,
    'gemma-7b-it': 8192
}

with st.sidebar:
    model: str = st.selectbox('Choose a LLM to chat', ('llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'))
    system_prompt: str = st.text_area('System Prompt', value='You are a helpful assistant.')
    max_tokens: int = st.slider('Max Tokens', 1, model_max_tokens[model], model_max_tokens[model], step=1)
    temperature: float = st.slider('Temperature', 0.00, 2.00, 0.75, step=0.01)
    top_p: float = st.slider('Top P', 0.00, 1.00, 1.00, step=0.01)

    if st.button('New Button'):
        st.session_state.messages = []
        st.experimental_rerun()


user_input = st.chat_input('Say something...')

if user_input:
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.markdown('user_input', unsafe_allow_html=True)

    messages = [{'role': 'system', 'content': system_prompt}] + st.session_state.messages

    
    response = "hello"
    st.session_state.messages.append({'role': 'assistant', 'content': response})

    with chat_message('assistant'):
        st.markdown('response', unsafe_allow_html=True)

