import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

# App Title
st.title("Knowledge Management Chatbot")

# Initialize session state to store chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Upload file
uploaded_file = st.file_uploader("Upload a file", type=["pdf"])

if uploaded_file is not None:
    with open("HRPolicy_Test.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("File uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader("HRPolicy_Test.pdf")
    docs = loader.load()

    # Text Splitting into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=15)
    documents = text_splitter.split_documents(docs)

    # Initialize embeddings and LLM
    hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    llm = ChatGroq(groq_api_key="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", model_name='llama3-70b-8192', temperature=0, top_p=0.2)

    # Vector database storage
    vector_db = FAISS.from_documents(documents, hf_embedding)

    # Craft ChatPrompt Template
    prompt = ChatPromptTemplate.from_template("""
    You are a Bandhan Life Insurance specialist. Answer the queries from an insurance specialist perspective who wants to resolve customer queries as asked.
    Answer the following questions based only on the provided context, previous responses, and the uploaded documents.
    Think step by step before providing a detailed answer.
    Wherever required, answer in a point-wise format.
    Do not answer any unrelated questions which are not in the provided documents, please be careful on this.
    I will tip you with a $1000 if the answer provided is helpful.
    
    <context>
    {context}
    </context>
    Conversation History:
    {chat_history}

    Question: {input}
    """)

    # Stuff Document Chain Creation
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retriever from vector store
    retriever = vector_db.as_retriever()

    # Create a retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Chat interface
    user_question = st.text_input("Ask a question about the relevant document", key="input")

    if user_question:
        # Build conversation history
        conversation_history = ""
        for chat in st.session_state['chat_history']:
            conversation_history += f"You: {chat['user']}\nBot: {chat['bot']}\n"

        # Get response from the retrieval chain with context
        response = retrieval_chain.invoke({
            "input": user_question,
            "chat_history": conversation_history
        })

        # Add the user's question and the model's response to chat history
        st.session_state.chat_history.append({"user": user_question, "bot": response['answer']})

    # Display chat history with a conversational format
    if st.session_state['chat_history']:
        for chat in st.session_state['chat_history']:
            st.markdown(f"<div style='padding: 10px; border-radius: 10px; background-color: #DCF8C6;'><strong>You:</strong> {chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='padding: 10px; border-radius: 10px; background-color: #ECECEC; margin-top: 5px;'><strong>Bot:</strong> {chat['bot']}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
