import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI  # Only import what exists
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from io import BytesIO

# Load environment variables from a .env file
load_dotenv()

# Configure the Google Generative AI API with an API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define a custom embeddings class that will handle text embeddings
class GoogleGenAIEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts):
        # Embeds a list of documents (texts) using a custom logic
        embeddings = []
        for text in texts:
            # Custom logic to generate embeddings for each text
            embedding = self._custom_embedding_logic(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text):
        # Embeds a single query text
        return self._custom_embedding_logic(text)
    
    def _custom_embedding_logic(self, text):
        # Placeholder for actual embedding logic, here we return dummy data
        return [0.0] * 512  # Dummy embedding, replace with actual logic

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # Wrap bytes of each PDF file in BytesIO to treat as file-like object
        pdf_file = BytesIO(pdf.read())
        pdf_reader = PdfReader(pdf_file)
        # Extract text from each page of the PDF
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split large text into smaller chunks using a text splitter
def get_text_chunks(text):
    # Split text into chunks of size 10,000 characters with 1,000 characters overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a vector store from the text chunks
def get_vector_store(text_chunks):
    # Initialize embeddings using the custom GoogleGenAIEmbeddings class
    embeddings = GoogleGenAIEmbeddings(model="models/embedding-001")  # Use custom class
    # Create a FAISS vector store from the text chunks and embeddings
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    # Save the vector store locally for later use
    vector_store.save_local("faiss_index")

# Function to set up a conversational chain for answering questions
def get_conversational_chain():
    # Define a prompt template for question answering
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in the provided context just say, "answer not available in the context", don't provide the wrong answer.
        Context:
        {context}?
        Question:
        {question}

        Answer:
    """
    # Initialize the Google Generative AI model for chat with specific parameters
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    # Create a prompt using the defined template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Load a question answering chain using the model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input and generate a response
def user_input(user_question):
    # Load the saved FAISS vector store
    embeddings = GoogleGenAIEmbeddings(model="models/embedding-001")  # Use custom class
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Perform similarity search to find relevant documents based on the user's question
    docs = new_db.similarity_search(user_question)
    # Get the conversational chain for answering the question
    chain = get_conversational_chain()
    # Generate a response based on the found documents and user's question
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    # Display the response in the Streamlit app
    st.write("Reply: ", response["output_text"])

# Main function to run the Streamlit app
def main():
    # Set the page title and header
    st.set_page_config(page_title="Chat with multiple PDF")
    st.header("Chat with multiple PDF using Gemini")
    # Get a question input from the user
    user_question = st.text_input("Ask a question from the PDF files")
    # If the user asks a question, process it using the user_input function
    if user_question:
        user_input(user_question)

    # Create a sidebar with a menu and file uploader
    with st.sidebar:
        st.title("Menu:")
        # Allow users to upload multiple PDF files
        pdf_docs = st.file_uploader("Upload your PDF files and click on the submit & process", accept_multiple_files=True, type="pdf")
        # Button to trigger processing of uploaded files
        if st.button("Submit and Process"):
            with st.spinner("Processing..."):
                # Extract text from the uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                # Split the extracted text into chunks
                text_chunks = get_text_chunks(raw_text)
                # Create and save a vector store from the text chunks
                get_vector_store(text_chunks)
                st.success("Done")

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
