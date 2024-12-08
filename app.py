import streamlit as st
import json
from model import LLMModel
from db import DB
from rag import RAGPipeline
import re
# CSS 스타일 추가
def add_custom_css():
    st.markdown(
        """
        <style>
        /* 전체 배경 흐릿한 회색 */
        body {
            background-color: #f8f8f8;
        }
        /* 강조된 입력창 스타일 */
        .highlight-input-container input {
            background-color: #ffffff !important;
            border: 2px solid #007bff !important;
            border-radius: 5px;
            padding: 10px;
            outline: none;
            box-shadow: 0 4px 8px rgba(0, 123, 255, 0.2);
        }
        /* 입력창 외부 흐릿하게 처리 */
        .blur-container {
            opacity: 0.5;
        }
        .highlight-input-container {
            opacity: 1.0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def initialize_db_and_vectorstore(config):
    db = DB(
        path=config["folder_path"],
        embed_model=config["embed_model"],
        milvus_uri=config["milvus_uri"],
        dir_list=config["directories"]
    )
    db.process_multiple_directories()
    return db.get_vectorstore()

@st.cache_resource
def initialize_llm(config_path):
    llm_model = LLMModel(config_path=config_path)
    return llm_model.get_llm()
import streamlit as st
from langchain_milvus import Milvus


def main():
    add_custom_css()
    st.title("Phone Manual Query")
    # 세션 상태 초기화
    if "rag_pipeline" not in st.session_state:
        st.sidebar.header("Configuration")
        config_path = "config.json"
        with st.spinner("Initializing resources..."):  # 초기화 중 로딩 표시
            with open(config_path, "r") as config_file:
                config = json.load(config_file)

            st.sidebar.text("Initializing database and vectorstore...")
            vectorstore = initialize_db_and_vectorstore(config)

            st.sidebar.text("Initializing LLM...")
            llm = initialize_llm(config_path)

            st.session_state["rag_pipeline"] = RAGPipeline(vectorstore=vectorstore, llm=llm)
    # 세션 상태 초기화
    if "phone_type" not in st.session_state:
        st.session_state["phone_type"] = None  # 폰 종류 초기화

    # 폰 종류를 먼저 물어봄
    if st.session_state["phone_type"] is None:
        st.subheader("어떤 폰을 사용하시나요?")
        phone_type = st.selectbox("폰 종류를 선택하세요", ["Samsung", "Apple"])

        if st.button("확인"):
            # 선택한 폰 종류를 세션 상태에 저장
            st.session_state["phone_type"] = phone_type.lower()
            st.success(f"{phone_type} 매뉴얼을 선택했습니다.")
    else:
        # 선택된 폰 종류를 기반으로 매뉴얼 검색
        st.subheader(f"{st.session_state['phone_type'].lower()} 폰 매뉴얼 검색")

        user_query = st.text_input("검색할 기능을 입력하세요:")
        if user_query:
            with st.spinner("검색 중..."):
                # 선택된 폰 종류로 데이터 필터링
                results = st.session_state["rag_pipeline"].run(
                    user_query,
                    filters={"phone_type": st.session_state["phone_type"]}
                )

                # 결과 출력
                if results:
                    st.success("검색 결과:")
                    # 불필요한 텍스트 필터링
                    clean_results = re.sub(r"[|]+", "", results)  # 특수 기호 제거
                    st.write(clean_results)
                else:
                    st.warning("관련된 매뉴얼을 찾을 수 없습니다.")


if __name__ == "__main__":
    main()
