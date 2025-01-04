import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())
google_api_key = os.environ["GOOGLE_API_KEY"]

st.title("RAG QA APPLICATION USING GEMINI PRO")

loader = PyPDFLoader("Generative AI with LangChain.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
docs = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

chromadb.api.client.SharedSystemClient.clear_system_cache()

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

query = st.chat_input("Ask me anything: ")
prompt = query

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrived context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    print(response["answer"])
    
    st.write(response["answer"])

