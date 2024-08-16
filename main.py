import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from gtts import gTTS
import tempfile
import shutil

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN= "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"
#os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.set_page_config(page_title="Conversational RAG with PDF Uploads and Chat History", layout="centered")
st.markdown("""
<style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #000000;
        color: #ff0000;
        margin: 0;
        padding: 0;
    }
    .stApp {
        background-color: #000000;
        color: #ff0000;
        border-radius: 8px;
        padding: 20px;
        margin: 20px auto;
        max-width: 800px;
    }
    .stButton button {
        background-color: #ff0000;
        color: #000;
        border: none;
        border-radius: 4px;
        padding: 12px 24px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #cc0000;
        transition: background-color 0.3s;
    }
    .stTextInput input {
        background-color: #202020;
        color: #ff0000;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 10px;
        font-size: 16px;
    }
    .stTextInput input:focus {
        border-color: #ff0000;
        outline: none;
    }
    .title-container {
        text-align: center;
        margin-bottom: 20px;
    }
    .title-container .title {
        font-size: 28px;
        font-weight: bold;
        cursor: pointer;
        transition: color 0.3s;
    }
    .title-container .title:hover {
        color: #00ff00;
    }
    .stAudio audio {
        width: 100%;
        border: 1px solid #444;
        border-radius: 4px;
    }
</style>

""", unsafe_allow_html=True)

st.title("Conversational RAG With PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content")

api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    session_id = st.text_input("Session ID", value="default_session")
    
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if st.button("Submit"):
            if user_input:
                session_history = get_session_history(session_id)
        
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": session_id}
                    },
                )

                if 'answer' in response:
                    answer = response['answer']
                    st.write("Assistant:", answer)

                    # Convert the answer text to speech and provide a link to play it
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                        tts = gTTS(text=answer, lang='en')
                        tts.save(temp_file.name)
                        temp_file_path = temp_file.name
                    
                    st.audio(temp_file_path, format="audio/mp3")

                else:
                    st.error("Could not retrieve the answer from the response.")

                st.write("Chat History:", session_history.messages)

else:
    st.warning("Please enter the Groq API Key")
