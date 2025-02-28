import os
import chromadb
import json
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import JSONLoader
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from pprint import pprint
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.prompts import PromptTemplate
from cookRetriever import OpenAIRetrievalAgent
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)



# Load environment variables from a .env file
_ = load_dotenv(find_dotenv(), override=True)

# Initialize the AzureOpenAI client with API key, version, and endpoint from environment variables
llm = AzureChatOpenAI(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
)

embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME4"),
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
    file_path='./recipes'
    loader = DirectoryLoader(file_path, glob='**/*.json', show_progress=True, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.content'})

    documents = loader.load()

    print(f'document count: {len(documents)}')
    print(documents[0] if len(documents) > 0 else None)

    vectorstore = InMemoryVectorStore.from_documents(
        documents= documents,
        embedding= embeddings
    )

    # Create a retriever
    # similarity search with score
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Create memory
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10
    )
    print(documents[0] if len(documents) > 0 else None)
    # ConversationalRetrievalChain
        #   - Look at conversation context (memory)
        #   - Condense the user query if needed
        #   - Retrieve relevant docs from vectorstore
        #   - Combine them with the system prompt
        #   - Apply a custom prompt for retrieval
        #   - Provide a final answer
    # qa_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=retriever,
    #     memory=memory,
    #     condense_question_llm=llm,
    #     combine_docs_chain_kwargs={
    #         "prompt": chat_prompt, 
    #     },
    #     )
    
    

    # data = json.loads(Path(file_path).read_text())
    # loader = JSONLoader(
    #     file_path='./recipes.json',
    #     jq_schema='.recipes[]',
    #     # content_key=".name",
    #     is_content_key_jq_parsable=True,
    #     text_content=False
    # )
    # data = loader.load()
    # splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # splitter = RecursiveJsonSplitter(max_chunk_size = 200, min_chunk_size= 200)
    # all_splits = splitter.split_json(json_data = data)
    # docs = splitter.split_documents(data)
    # print("\n")
    # for doc in docs[:5]:
    #     print(doc)
    # embeddings.embed_query(data)
    # db = chroma_client.from_documents(docs, AzureOpenAIEmbeddings())


    # pprint(data)
    # docs = splitter.create_documents(data)



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
    system_prompt_path = "recipes/homeCookAi_prompt.md"

    # ingredients = input("Enter the ingredients you have (comma-separated): ")
    # dietary_preferences = input("Enter your dietary preferences (e.g., vegan, gluten-free): ")
    print("Start chatting with the retrieval-enabled assistant (type 'exit' to stop):")
    cook = OpenAIRetrievalAgent(system_prompt_path)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        try:
            response = cook.send_message(user_input)
            if response.lower() == "exit":
                break
            print(f"Assistant: {response}")
        except Exception as e:
            print(e)
    # ingredients = input("Enter the ingredients you have (comma-separated): ")
    # dietary_preferences = input("Enter your dietary preferences (e.g., vegan, gluten-free): ")
    print("\n")
    print_json_file()
    # suggestions = get_recipe_suggestions(ingredients, dietary_preferences)
    # print("\nHere are some creative recipe suggestions for you:")
    # print(suggestions)

    # Store the suggestions in ChromaDB
    # store_suggestions_in_chromadb(suggestions, ingredients, dietary_preferences)

if __name__ == "__main__":
    main()