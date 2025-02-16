import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from streamlit import session_state
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import SentenceTransformer
from htmlTemplates import css, bot_template, user_template

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_text_from_url(url):
    """Fetches and extracts text from a webpage."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")  # Extract paragraph texts
            text = "\n".join([p.get_text() for p in paragraphs])
            return text
        else:
            st.error(f"Failed to fetch {url}. Status code: {response.status_code}")
            return ""
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return ""

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct",
        task="text-generation"
    )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def handle_user_input(user_question):
    response = session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    st.set_page_config(page_title="Chat with PDFs & Links", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs & Links :books:")
    user_question = st.text_input("Ask a question about your documents")

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your Documents")

        # Upload PDFs
        pdf_docs = st.file_uploader("Upload PDFs:", accept_multiple_files=True)

        # Input URLs
        url_input = st.text_area("Enter URLs (one per line):")

        if st.button("Process"):
            with st.spinner("Processing..."):
                all_text = ""

                # Extract text from PDFs
                if pdf_docs:
                    all_text += get_pdf_text(pdf_docs) + "\n"

                # Extract text from URLs
                if url_input:
                    urls = url_input.split("\n")
                    for url in urls:
                        url_text = get_text_from_url(url.strip())
                        all_text += url_text + "\n"

                # Split into chunks & vectorize
                text_chunks = get_text_chunks(all_text)
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()
