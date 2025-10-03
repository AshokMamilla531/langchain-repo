import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from variable import INFORMATION

# Load environment variables
load_dotenv()

# Get Gemini API key
GEMINI_API_KEY = os.getenv("GEMINIAI_API_KEY")

def main():
    """Generate a summary and interesting facts about a person using Gemini AI."""
    
    if not GEMINI_API_KEY:
        print("Error: GEMINIAI_API_KEY not found in environment variables")
        return
    
    # Define the prompt template
    template = """Given the information {information} about a person, create:
1. A short summary
2. Two interesting facts about them"""
    
    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["information"],
        template=template
    )
    
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=GEMINI_API_KEY, 
        temperature=0
    )
    
    # Create and run the chain
    chain = prompt | llm
    response = chain.invoke({"information": INFORMATION})
    
    # Print the response
    print(response.content)

if __name__ == "__main__":
    main()
