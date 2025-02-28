import os
import chromadb
import json
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import JSONLoader
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from pprint import pprint
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.prompts import PromptTemplate
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


class OpenAIRetrievalAgent:
    def __init__(self, cookAI_prompt_path):
        # Load json recipe files from directory
        file_path = './recipes'
        loader = DirectoryLoader(file_path, glob='**/*.json', show_progress=True, loader_cls=JSONLoader, loader_kwargs={'jq_schema': '.content'})
        documents = loader.load()

        # Read the system prompt from file
        self.cookAI_prompt = self._load_prompt(cookAI_prompt_path)

        # Turn "system_prompt" string into a system message
        cookAI_message_prompt = SystemMessagePromptTemplate.from_template(
            self.cookAI_prompt  # inject it at runtime
        )

        # Turn the doc context + user question into a user message
        # The placeholders {context} and {question} will be inserted by the chain
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            """Here is the relevant recipe context (if any): 
            {context}

            User's question:
            {question}
            """
        )

        # Combine them into a chat prompt
        chat_prompt = ChatPromptTemplate.from_messages([
            cookAI_message_prompt,
            human_message_prompt,
        ])

        print(f'document count: {len(documents)}')
        print(documents[0] if len(documents) > 0 else None)

        vectorstore = InMemoryVectorStore.from_documents(
            documents=documents,
            embedding=embeddings
        )

        # Create a retriever
        # similarity search with score
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # ConversationalRetrievalChain
        # - Look at conversation context (memory)
        # - Condense the user query if needed
        # - Retrieve relevant docs from vectorstore
        # - Combine them with the system prompt
        # - Apply a custom prompt for retrieval
        # - Provide a final answer
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            condense_question_llm=llm,
            combine_docs_chain_kwargs={
                "prompt": chat_prompt,
            },
        )

    def _load_prompt(self, path: str) -> str:
        """Simple helper to load the system prompt from a file."""
        with open(path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    def send_message(self, user_message: str) -> str:
        """
        Sends a user message to the retrieval chain.
        If there's relevant content in docs, it gets used.
        Otherwise, the LLM uses its own knowledge.
        """
        chain_input = {
            "question": user_message,
            "chat_history": self.qa_chain.memory.load_memory_variables({})["chat_history"]
        }
        result = self.qa_chain(chain_input)
        return result["answer"]