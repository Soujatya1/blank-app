import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings
import boto3

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

    # Initialize embeddings
    hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

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

    # Initialize Bedrock client
    client = boto3.client(
        'bedrock',
        aws_access_key_id='YOUR_ACCESS_KEY',
        aws_secret_access_key='YOUR_SECRET_KEY',
        region_name='YOUR_REGION'  # Set the correct region
    )

    # Define a function to interact with Bedrock LLM
    def invoke_bedrock_model(user_input, chat_history):
        # Request for LLM model
        response = client.invoke_model(
            modelId='bedrock-model-id',  # Replace with your Bedrock model ID
            prompt=prompt.format(context=user_input, chat_history=chat_history),
            maxTokens=100  # Adjust as needed
        )
        return response['completions'][0]['text']

    # Retriever from vector store
    retriever = vector_db.as_retriever()

    # Chat interface
    user_question = st.text_input("Ask a question about the relevant document", key="input")

    if user_question:
        # Build conversation history
        conversation_history = ""
        for chat in st.session_state['chat_history']:
            conversation_history += f"You: {chat['user']}\nBot: {chat['bot']}\n"

        # Get response from Bedrock LLM with context
        bedrock_response = invoke_bedrock_model(user_question, conversation_history)

        # Add the user's question and the model's response to chat history
        st.session_state.chat_history.append({"user": user_question, "bot": bedrock_response})

    # Display chat history with a conversational format
    if st.session_state['chat_history']:
        for chat in st.session_state['chat_history']:
            st.markdown(f"<div style='padding: 10px; border-radius: 10px; background-color: #DCF8C6;'><strong>You:</strong> {chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='padding: 10px; border-radius: 10px; background-color: #ECECEC; margin-top: 5px;'><strong>Bot:</strong> {chat['bot']}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
