import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

load_dotenv()

st.title("News Research Tool")

st.sidebar.title("Paste URLS below")

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

processed_url_clicked = st.sidebar.button("Process URLS")
file_path = 'faiss_store_openai.pkl'

llm = OpenAI(temperature = 0.7, max_tokens=500, openai_api_key= os.getenv("OPEN_AI_API"))

main_placefolder = st.empty()
if processed_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Analyzing URLS")
    data=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placefolder.text("Chunking ...")
    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore_ai=faiss.FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embedding ...")
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_ai,f)

query = main_placefolder.text_input("Questions : ...")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm , retriever = vectorstore.as_retriever())
            result = chain({"question" : query}, return_only_outputs = True)
            st.header("Answer")
            st.write(result["answer"]) 


            sources = result.get("sources","")
            if sources: 
                st.subheader("Sources : ")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)