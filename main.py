import os
import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Get Gemini API key
GEMINI_API_KEY = os.getenv("GEMINIAI_API_KEY")

def load_config():
    """Load configuration from JSON file."""
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """Generate a summary and interesting facts about a person using Gemini AI."""
    
    if not GEMINI_API_KEY:
        print("Error: GEMINIAI_API_KEY not found in environment variables")
        return
    
    # Load configuration from JSON file
    config = load_config()
    
    # Create prompt template from config
    prompt = PromptTemplate(
        input_variables=config["prompt_template"]["input_variables"],
        template=config["prompt_template"]["template"]
    )
    
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=GEMINI_API_KEY, 
        temperature=0
    )
    
    # Create and run the chain
    chain = prompt | llm
    response = chain.invoke({"information": config["information"]})
    
    # Print the response
    print(response.content)

if __name__ == "__main__":
    main()
