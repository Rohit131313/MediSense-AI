from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI  
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"]

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generated answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'."
    )

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0)

# Enable structured output
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Define grading prompt
message = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", message),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# Combine prompt with structured output
hallucination_grader = hallucination_prompt | structured_llm_grader
