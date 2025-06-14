Chat With Multiple PDFs


GOAL:

    The goal is to build an interactive application that allows users to upload multiple PDFs, process their contents, and ask questions that the system answers using context from the uploaded documents—powered by Gemini AI and vector-based retrieval.
    
Approach for the project

    1. PDF Upload and Text Extraction
        - Users upload multiple PDF files via Streamlit’s file_uploader.
        - PDFs are read using PyPDF2.PdfReader and plain text is extracted from all pages.
    
    2. Text Chunking
        - Extracted text is split into manageable chunks using RecursiveCharacterTextSplitter.
        - Chunking ensures that each vector embedding has a meaningful, focused context and fits within token limits.
    
    3. Embeddings and Vector Store Creation
        - Each chunk is embedded using GoogleGenerativeAIEmbeddings.
        - Chunks and their vectors are stored locally using FAISS (vector_store.save_local()).
    
    4. Conversational Chain Setup
        - A custom prompt is used to instruct Gemini to answer only using the context.
        - Gemini (ChatGoogleGenerativeAI) is used with a load_qa_chain and PromptTemplate to create the conversational flow.
    
    5. Question Answering
        When a user asks a question:
            - FAISS searches for the most relevant text chunks using similarity search.
            - Retrieved chunks are passed into the QA chain.
            - Gemini generates a detailed, context-aware answer.
            - The response and source chunks are shown in the UI.
    
    6. Chat History Maintenance
        - Past Q&A pairs are saved in st.session_state.chat_history.
        - Displayed dynamically in the UI for user reference.


    
Technologies: 

    - streamlit
    - google-generativeai
    - python-dotenv
    - langchain
    - PyPDF2
    - faiss-cpu
    - langchain_google_genai
    - langchain-community
    - langchain-core
    - langchain-cli
    - pydantic

Notebook

https://github.com/pheonix2109/ChatWithMultiplePDFs/blob/main/app.py


