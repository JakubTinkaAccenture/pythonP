import os
import chromadb
import json
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import JSONLoader
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from pprint import pprint



# Load environment variables from a .env file
_ = load_dotenv(find_dotenv(), override=True)

# Initialize the AzureOpenAI client with API key, version, and endpoint from environment variables
llm = AzureChatOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
)

# Initialize ChromaDB client
chroma_client = chromadb.Client()

def get_recipe_suggestions(ingredients, dietary_preferences):
    prompt = f"Given the ingredients {ingredients} and dietary preferences {dietary_preferences}, suggest creative variations or improvements to existing recipes."
    print(prompt)
    response = llm.invoke(
        input=prompt,
        max_tokens=150
    )

    suggestions = response
    return suggestions

def print_json_file():
    file_path='./recipes.json'
    data = json.loads(Path(file_path).read_text())
    # pprint(data)

def store_suggestions_in_chromadb(suggestions, ingredients, dietary_preferences):
    # Create a unique ID for the suggestion
    suggestion_id = f"{ingredients}-{dietary_preferences}"
    
    # Create a vector representation of the suggestion
    vector = llm.Embedding.create(input=suggestions)["data"][0]["embedding"]
    
    # Upsert the suggestion into ChromaDB
    chroma_client.upsert(
        collection_name="recipe_suggestions",
        documents=[{
            "id": suggestion_id,
            "embedding": vector,
            "metadata": {
                "suggestions": suggestions,
                "ingredients": ingredients,
                "dietary_preferences": dietary_preferences
            }
        }]
    )

def main():
    print("Welcome to the Food Improviser!")
    ingredients = input("Enter the ingredients you have (comma-separated): ")
    dietary_preferences = input("Enter your dietary preferences (e.g., vegan, gluten-free): ")
    print("\n")
    print_json_file()
    suggestions = get_recipe_suggestions(ingredients, dietary_preferences)
    print("\nHere are some creative recipe suggestions for you:")
    print(suggestions)

    # Store the suggestions in ChromaDB
    store_suggestions_in_chromadb(suggestions, ingredients, dietary_preferences)

if __name__ == "__main__":
    main()