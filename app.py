import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv



# to see environment variable
load_dotenv()

# configuring the api key to whatever GOOGLE_API_KEY we have given in the .env 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"Model Name: {m.name}, Supported Methods: {m.supported_generation_methods}")
except Exception as e:
    print(f"Error listing models: {e}")


# --- Function to extract text from PDFs ---
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        except Exception as e:
            st.error(f"Failed to read {pdf.name}: {str(e)}")
    return text


# --- Function to split text into chunks ---
def get_text_chunks(text):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks= text_splitter.split_text(text)
    return chunks


# --- Function to create and save vector store ---
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



# --- Function to create QA chain ---
def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not provided just say, "answer is not available in the context", don't provide the wrong answer \n
   
     Context: \n {context}?\n
    Question: \n {question}\n

    Answer: 

    """
    model=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)

    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain= load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



# --- Function to handle user input and generate answer ---
def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db= FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs= new_db.similarity_search(user_question)

    chain= get_conversational_chain()

    response=chain(
        {
            "input_documents": docs, 
            "question": user_question
        }, 
        return_only_outputs=True
    )
        
    st.session_state.chat_history.append(("You", user_question))
    st.session_state.chat_history.append(("Bot", response["output_text"]))

    # Display response
    print(response)
    st.write("Reply:", response["output_text"])

    # Display sources
    with st.expander("Sources used"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}**")
            st.write(doc.page_content)




# --- Main Streamlit app ---
def main():
    st.set_page_config(page_title="Chat With Multiple PDFs", layout="wide")

    st.header("Chat with Multiple PDF using Gemini")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        try:
            user_input(user_question)
        except Exception as e:
            st.error(f"Error while processing the question: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("###Chat History")
        for speaker, msg in st.session_state.chat_history:
            st.markdown(f"**{speaker}:** {msg}")
    
    
    with st.sidebar:
        st.title("Menu: ")
        pdf_docs = st.file_uploader("Upload your PDF files and click on the submit button", accept_multiple_files=True)
        
        if st.button("Submit"):
            if not pdf_docs:
                    st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."): 
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text could be extracted from uploaded PDFs.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done.")

if __name__=="__main__":
    main()