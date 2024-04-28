import os

from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


from retriever import generateRetrievalModel

## .env読み込み
load_dotenv()

retriever = generateRetrievalModel()

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=chat, chain_type="refine", retriever=retriever)

st.title("StreamlitのChatサンプル")

# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"


def response_retriever(
    user_msg: str,
):
    """ChatGPTのレスポンスを取得

    Args:
        user_msg (str): ユーザーメッセージ。
    """
    response = qa_chain(
       f'Stockについて次の問いに日本語で回答してください。もしわからなければ無理に回答を生成せずわからないと答えてください。 質問：{user_msg}'
    )
    return response


# チャットログを保存したセッション情報を初期化
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []


user_msg = st.chat_input("ここにメッセージを入力")
if user_msg:
    # 以前のチャットログを表示
    for chat in st.session_state.chat_log:
        with st.chat_message(chat["name"]):
            st.write(chat["msg"])

    # 最新のメッセージを表示
    with st.chat_message(USER_NAME):
        st.write(user_msg)

    # アシスタントのメッセージを表示
    with st.chat_message(ASSISTANT_NAME):
        response = response_retriever(user_msg)
        assistant_response_area = st.empty()
        assistant_response_area.write(response['result'])
    #     assistant_msg = ""
    #     assistant_response_area = st.empty()
    #     for chunk in response:
    #         if chunk.choices[0].finish_reason is not None:
    #             break
    #         # 回答を逐次表示
    #         assistant_msg += chunk.choices[0].delta.content
    #         assistant_response_area.write(assistant_msg)

    # セッションにチャットログを追加
    st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
    st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": response['result']})