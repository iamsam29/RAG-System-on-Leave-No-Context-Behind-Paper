import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from langchain_text_splitters import NLTKTextSplitter

st.header('RAG System on “Leave No Context Behind” Paper')

loader = PyPDFLoader("D:/Codes_VS/New folder/rag_langchain_app/leave_no_context_behind.pdf")

pages = loader.load_and_split()

f = open('D:/Codes_VS/New folder/rag_langchain_app/key.txt')        #save your Google API key in a text file
key = f.read()
genai.configure(api_key=key)

text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key = key, model = "models/embedding-001")

db = Chroma.from_documents(chunks, embedding_model, persist_directory = "./chroma_db_")
db.persist()
db_connection = Chroma(persist_directory = "./chroma_db_", embedding_function = embedding_model)

retriever = db_connection.as_retriever(search_kwargs={"k":5})

model = genai.GenerativeModel('gemini-1.5-pro-latest')

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.chat = model.start_chat(history=st.session_state.history)
    st.session_state.user_input_value = ""  # Initialize user_input_value state

def clear_history():
    st.session_state.history = []
    st.session_state.chat = model.start_chat(history=st.session_state.history)
    st.session_state.user_input_value = ""  # Reset user_input_value state
    st.experimental_rerun()

with st.sidebar:
    if st.button("Clear Chat Window", use_container_width=True, type="primary"):
        clear_history()

user_input_value = st.session_state.user_input_value
user_input = st.text_input('Enter Your Question here : ', key='user_input', value=user_input_value)

if st.button('Send'):
    if user_input:
        response = st.session_state.chat.send_message(user_input)
        st.session_state.history = st.session_state.chat.history
        for message in st.session_state.history:
            role = "assistant" if message.role == 'model' else message.role
            with st.chat_message(role):
                st.markdown(message.parts[0].text)
        st.session_state.user_input_value = ""  # Reset user_input_value state
