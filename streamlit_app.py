import streamlit as st
import tiktoken             # text, token간 변환

from loguru import logger   # loguru 라이브러리에서 logger 객체 호출
import os       # 운영체제와 상호작용
import tempfile # 임시 파일 및 임시 디렉터리를 생성하고 관리


from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain_community.document_loaders import TextLoader
# 9/5 txt파일 인식을 위해 추가 (클로드)

from langchain.text_splitter import RecursiveCharacterTextSplitter
# 긴 텍스트를 자연스러운 범위 내에서 분할
from langchain_community.embeddings import HuggingFaceEmbeddings
# Hugging Face의 임베딩 모델을 사용하여 텍스트를 벡터로 변환

from langchain.memory import ConversationBufferMemory
# 대화 메모리 관리. 대화의 기록을 저장, 대화의 문맥을 기억
from langchain_community.vectorstores import FAISS
# 문서나 텍스트 임베한 후, 벡터 저장, 주어진 질문과 가장 유사한 벡터를 빠르게 검색

# from langchain_community.callbacks import get_openai_callback
# OpenAI API를 호출하는 작업에서 필요한 이벤트를 추적하거나
# 특정 작업이 완료되었을 때 후속 작업을 처리하는 데 사용
from langchain.memory import StreamlitChatMessageHistory
# 채팅 메시지 기록을 관리하는 클래스


# .env 파일에 저장된 환경 변수를 파이썬에서 사용할 수 있도록 메모리에 불러옴
# 9/3 코칭시 추가
from dotenv import load_dotenv
load_dotenv()

@st.cache_data
def load_files(_data_folder, _files_to_load):
    files_text = []
    for filename in _files_to_load:
        file_path = os.path.join(_data_folder, filename)
        if os.path.exists(file_path):
            files_text.extend(load_document(file_path))
        else:
            st.warning(f"파일을 찾을 수 없습니다: {filename}")
    return files_text

@st.cache_resource
def initialize_conversation(_files_text, openai_api_key):
        text_chunks = get_text_chunks(_files_text)
        vetorestore = get_vectorstore(text_chunks)
        return get_conversation_chain(vetorestore, openai_api_key)

def main():

    st.set_page_config(page_title="AI마케터")

    st.title("AI마케터 유통업무 Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None


    # 9/20 사이드바 및 gpt모델 강제 선택. 원 코드 주석처리 및 새 코드 삽입
    # with st.sidebar:
    #    model_selection = st.selectbox(
    #        "Choose the language model",
    #        ("gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o", "bnksys/yanolja-eeve-korean-instruct-10.8b"),
    #        key="model_selection"
    #    )   

    # 9/20 아래 라인 추가    
    st.session_state.model_selection = "gpt-4o-mini"

    openai_api_key = os.getenv("OPEN_API_KEY")
    # api 환경변수, UI 관련 코드는 삭제하였음
       

        
    
    if "conversation" not in st.session_state:
        if not openai_api_key:
                st.info("Please add all necessary API keys and project information to continue.")
                st.stop()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(current_dir, "data")
            

        files_to_load = ["1.개요.docx", "2.매뉴얼.docx"]
        files_text = load_files(data_folder, files_to_load)

        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True


    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! AI마케터 유통업무 chatbot 입니다. 궁금한 점을 물어보세요."}]
        

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):      # avatar를 넣는 등 variation 가능
            st.markdown(message["content"])

    # Chat logic
    
    def process_user_input(query):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.conversation({"question": query})
        
            response = result['answer']
            source_documents = result.get('source_documents', [])
        
            st.markdown(response)
        
            if source_documents:
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": response})

    if query := st.chat_input("Message to chatbot"):
        process_user_input(query)
        
        
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


# 파일 자동 업로드
def load_document(file_path):
    if file_path.endswith('.pdf'):
        return PyPDFLoader(file_path).load_and_split()
    elif file_path.endswith('.docx'):
        return Docx2txtLoader(file_path).load_and_split()
    elif file_path.endswith('.pptx'):
        return UnstructuredPowerPointLoader(file_path).load_and_split()
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=100,
            length_function=tiktoken_len
        )
        return text_splitter.split_documents(documents)
    else:
        return []  # 지원되지 않는 파일 유형


def get_text(docs):
    doc_list = []
    for doc in docs:
        doc_list.extend(load_document(doc))
    return doc_list

@st.cache_data
def get_text_chunks(_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(_text)
    return chunks

@st.cache_resource
def get_vectorstore(_text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(_text_chunks, embeddings)
    return vectordb


@st.cache_resource
def get_conversation_chain(_vetorestore, _openai_api_key):
    llm = ChatOpenAI(openai_api_key=_openai_api_key, model_name="gpt-4o-mini", temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=_vetorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain



if __name__ == '__main__':
    main()