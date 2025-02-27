from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv
import os
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents

extracted_data = load_pdf("data/")
# print(extracted_data)

# Use RecursiveCharacterTextSplitter to split the extracted PDF text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=50
)

doc_splits = text_splitter.split_documents(extracted_data)

# print(f"Total Chunks: {len(doc_splits)}")
# print(doc_splits[100])

# Embedding Model
# Configure Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#Initializing the Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = 'agentic-medical-ai-chatbot'
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=768, # according to embedding model
        metric='cosine',
        spec=spec
    )

index = pc.Index(index_name)

#Creating Embeddings for Each of The Text Chunks & storing in pinecone DB , see in pinecone website
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# vectorstore.add_texts([t.page_content for t in doc_splits])

retriever = vectorstore.as_retriever()

