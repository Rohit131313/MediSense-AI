from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"]

class GradeAnswer(BaseModel):
    """Binary score for assessing if the answer is relevant to the question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'."
    )

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0)

# Enable structured output
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Grading prompt
message = """You are a grader assessing whether an answer addresses / resolves a question. 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", message),
        ("human", "User question: {question} \n\n LLM generated answer: {generation}"),
    ]
)

# Create the grading chain
answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
