from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"]

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0)

# Pull the RAG prompt from LangChain Hub
prompt = hub.pull("rlm/rag-prompt")

# Create the generation chain
generation_chain = prompt | llm | StrOutputParser()

# Example usage
# response = generation_chain.invoke({"document": "Machine learning enhances AI capabilities.", "question": "What is machine learning?"})
# print(response)
