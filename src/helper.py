import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
#from langchain_google_genai import GoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings

from dotenv import load_dotenv


load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")
os.environ["MISTRAL_API_KEY"]=MISTRAL_API_KEY

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
    embeddings=MistralAIEmbeddings(model="mistral-embed",api_key=MISTRAL_API_KEY)
    index_name="harsh12"
    vectore_store=PineconeVectorStore(index_name=index_name,embedding=embeddings,pinecone_api_key=PINECONE_API_KEY)
    vectore_store.from_texts(chunks,embedding=embeddings,index_name=index_name)
    return vectore_store

def conversational_chain(vector_store):
    llm = ChatMistralAI(model="mistral-large-latest",temperature=0,max_retries=2,api_key=MISTRAL_API_KEY)
    #llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=GOOGLE_API_KEY, temperature=0.1)
    memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                output_key='answer',
                return_messages=True)

    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
    except Exception as e:
        print(f"Error setting up conversation chain: {str(e)}")
        return None
        
    #conversation_chain =ConversationalRetrievalChain.from_llm(llm=llm,retriever=vector_store.as_retriever(),memory=memory)
    return conversation_chain
