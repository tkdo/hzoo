# -*- coding: utf-8 -*-
import streamlit as st
from openai import OpenAI

# ==================== 初始化配置 ====================
st.set_page_config(
    page_title="智能对话系统",
    page_icon="🤖",
    layout="centered"
)  # ‌:ml-citation{ref="2,4" data="citationList"}

# ==================== 侧边栏配置 ====================
with st.sidebar:
    st.header("⚙️ 控制面板")
    api_key = st.text_input("API密钥", type="password")
    client = OpenAI(api_key=api_key) if api_key else None  # ‌:ml-citation{ref="2" data="citationList"}
    
    st.divider()
    temperature = st.slider("创造力", 0.0, 2.0, 0.7)
    max_tokens = st.number_input("最大输出长度", 512, 4096, 2048)  # ‌:ml-citation{ref="2,4" data="citationList"}
    
    if st.button("🧹 清空对话"):
        st.session_state.messages = []
        st.rerun()  # ‌:ml-citation{ref="3" data="citationList"}

# ==================== 主界面 ====================
st.title("💬 智能对话系统")
st.caption("支持GPT-4 Turbo的流式对话系统")

# 初始化消息容器
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "请问有什么可以帮您？"}
    ]  # ‌:ml-citation{ref="2,3" data="citationList"}

# 显示历史消息
for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])  # ‌:ml-citation{ref="2,3" data="citationList"}

# ==================== 对话处理 ====================
if prompt := st.chat_input("输入消息..."):
    # 用户消息处理
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)  # ‌:ml-citation{ref="2,3" data="citationList"}

    # AI响应处理
    with st.chat_message("assistant", avatar="🤖"):
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=st.session_state.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True  # ‌:ml-citation{ref="1,2" data="citationList"}
        )
        
        # 流式输出实现
        response_container = st.empty()
        full_response = ""
        for chunk in response:
            if chunk.choices.delta.content:
                full_response += chunk.choices.delta.content
                response_container.markdown(full_response + "▌")  # ‌:ml-citation{ref="1,2" data="citationList"}
        
        response_container.markdown(full_response)
    
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )  # ‌:ml-citation{ref="2,3" data="citationList"}
