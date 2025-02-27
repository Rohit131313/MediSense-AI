from graph.graph import app
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"]

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0)

if __name__ == "__main__":
    question = "What are the symptoms of diabetes?"
    # question = "What is the recommended treatment for hypertension in elderly patients?"
    # question = "Is there a cure for Alzheimer's disease?"
    # question = "Who won the Nobel Prize in Medicine in 2024?"

    output = app.invoke(input={"question": question, "generation": "", "use_web_search": False, "documents": []})
    print(output["generation"])

