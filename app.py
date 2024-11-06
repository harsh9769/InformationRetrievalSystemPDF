import streamlit as st
from src.helper import get_pdf_text,get_text_chunks,get_vectore_store,conversational_chain
import time
import httpx


def user_input(user_question):
    try:
       
        response = st.session_state.conversation({'question': user_question})

        
        if 'chat_history' in response:
            st.session_state.chatHistory = response['chat_history']
        else:
            st.warning("No chat history found in the response.")
   
        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:  
                st.write("User: ", message.content)
            else:  
                st.write("Reply: ", message.content)

    except KeyError as e:
        st.error(f"Missing key in response: {e}")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:  # Rate limit error
            st.warning("Rate limit exceeded. Please try again later.")
        else:
            st.error(f"HTTP error occurred: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        
def main():
    st.set_page_config("Information Retrieval")
    st.header("Information-Retreival-System")

    user_question=st.text_input("Ask a question from pdf files")

    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory=None
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload your PDF files",accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):

                raw_text=get_pdf_text(pdf_docs)
                chunks=get_text_chunks(raw_text)
                vector_store=get_vectore_store(chunks=chunks)
                st.session_state.conversation = conversational_chain(vector_store)
                
                st.success("Done")

if __name__ == '__main__':
    main()