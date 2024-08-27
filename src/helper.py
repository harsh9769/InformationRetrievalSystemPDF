import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI

from dotenv import load_dotenv


load_dotenv()

PineCone_API_KEY=os.getenv("Pinecone_API_KEY")
os.environ["PineCone_API_KEY"]=PineCone_API_KEY

PineCone_API_ENV=os.getenv("Pinecone_API_ENV")
os.environ["PineCone_API_ENV"]=PineCone_API_ENV

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vectore_store(chunks):
    embeddings=GooglePalmEmbeddings()
    index_name="lama2web"
    vectore_store=PineconeVectorStore(index_name=index_name,embedding=embeddings,pinecone_api_key=PineCone_API_KEY)
    vectore_store.from_texts(chunks,embedding=embeddings,index_name=index_name)
    return vectore_store

def conversational_chain(vector_store):
    
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=GOOGLE_API_KEY, temperature=0.1)
    memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                output_key='answer',
                return_messages=True)
    conversation_chain =ConversationalRetrievalChain.from_llm(llm=llm,retriever=vector_store.as_retriever(),memory=memory)
    return conversation_chain
