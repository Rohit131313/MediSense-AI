import os
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel , Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"]

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ..., 
        description="Route the user query to the vectorstore or web search. Available options are 'vectorstore' or 'web_search'."
    )

# Initialize Gemini-2 Flash model with API key
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0)

# Enable structured output
structured_llm_router = llm.with_structured_output(RouteQuery)

# Define the routing prompt
message = """You are an expert at routing a user question to a vectorstore or web search.  
The vectorstore contains a comprehensive medical textbook covering a wide range of medical topics.  
Use the vectorstore for questions related to medical knowledge. For ANY other question, choose the web-search route."""

router_prompt = ChatPromptTemplate.from_messages(
    [("system", message), ("human", "{question}")]
)

# Create the router
question_router = router_prompt | structured_llm_router

# Example usage
# response = question_router.invoke({"question": "What is an adversarial attack?"})
# print(response)  # Expected Output: {"datasource": "vectorstore"}
