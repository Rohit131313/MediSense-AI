from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"]

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0)

class GradeDocuments(BaseModel):
    """Binary score for the relevance check of retrieved documents"""

    binary_score: str = Field(
        description="Documents are relevant to the question? 'yes' or 'no'",
    )

# Enable structured output for Gemini model
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Define the prompt for grading document relevance
message = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", message),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Combine the prompt with the structured LLM grader
retrieval_grader = grade_prompt | structured_llm_grader

# Example usage
# response = retrieval_grader.invoke({"document": "Machine learning improves data analysis.", "question": "What is machine learning?"})
# print(response)
