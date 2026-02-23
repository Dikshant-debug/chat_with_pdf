import os
import streamlit as st
from dotenv import load_dotenv
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.write("Upload a PDF and ask any question about it!")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:

    with st.spinner("Reading your PDF..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()

    st.success(f"PDF loaded! Total characters: {len(text)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    st.info(f"Split into {len(chunks)} chunks")

    with st.spinner("Understanding your PDF..."):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = Chroma.from_texts(chunks, embeddings)

    st.success("Ready! Ask your question below 👇")

    question = st.text_input("Ask anything about your PDF:")

    if question:
        with st.spinner("Thinking..."):
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            answer = qa_chain.invoke(question)

        st.write("### 💬 Answer:")
        st.write(answer["result"])