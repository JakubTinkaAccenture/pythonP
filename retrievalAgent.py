import os
from dotenv import load_dotenv, find_dotenv

# LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders.directory import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class OpenAIRetrievalAgent:
    def __init__(self, system_prompt_path: str, docs_dir: str = "../docs/recipes"):
        # Load environment variables from a .env file
        _ = load_dotenv(find_dotenv(), override=True)

        # Read the system prompt from file
        self.system_prompt = self._load_prompt(system_prompt_path)

        # Turn "system_prompt" string into a system message
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_prompt  # inject it at runtime
        )

        # Turn the doc context + user question into a user message
        #    The placeholders {context} and {question} will be inserted by the chain
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            """Here is the relevant recipe context (if any): 
        {context}

        User's question:
        {question}
        """
        )

        # Combine them into a chat prompt
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt, 
            human_message_prompt,
        ])

        # Initialize the LLM for normal conversation:
        self.model = AzureChatOpenAI(
            temperature=0,
            azure_endpoint=os.getenv["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.getenv["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.getenv["AZURE_OPENAI_API_VERSION"],
            model_name="gpt-4o-mini",
        )

        # Load documents from docs/recipes (or wherever you store them)
        self.docs = self._load_docs(docs_dir)

        # Embed them
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv["AZURE_OPENAI_ENDPOINT"],
            openai_api_version=os.getenv["AZURE_OPENAI_API_VERSION"],
            openai_api_key=os.getenv["AZURE_OPENAI_API_KEY"],
            model="text-embedding-3-large"

        )

        # Create a vector store
        self.vectorstore = InMemoryVectorStore.from_documents(
            documents=self.docs,
            embedding=self.embeddings
        )

        # Create a retriever
        # similarity search with score
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Create memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10
        )

        # ConversationalRetrievalChain
        #   - Look at conversation context (memory)
        #   - Condense the user query if needed
        #   - Retrieve relevant docs from vectorstore
        #   - Combine them with the system prompt
        #   - Apply a custom prompt for retrieval
        #   - Provide a final answer
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.model,
            retriever=self.retriever,
            memory=self.memory,
            condense_question_llm=self.model,
            combine_docs_chain_kwargs={
                "prompt": chat_prompt, 
            },
        )

    def _load_prompt(self, path: str) -> str:
        """Simple helper to load the system prompt from a file."""
        with open(path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    def _load_docs(self, docs_dir: str):
        """
        Load all text files from the given directory as Documents.
        If your recipes are in different formats (like .txt or .md),
        you can adapt as needed.
        """
        loader = DirectoryLoader(docs_dir, glob="*.txt", loader_cls=TextLoader)
        # loader = DirectoryLoader(docs_dir, glob="*.md", loader_cls=TextLoader)
        
        documents = loader.load()
        # todo: cleanup the documents

        return documents

    def send_message(self, user_message: str) -> str:
        """
        Sends a user message to the retrieval chain. 
        If there's relevant content in docs, it gets used. 
        Otherwise, the LLM uses its own knowledge.
        """
        chain_input = {
            "question": user_message,
            "chat_history": self.memory.load_memory_variables({})["chat_history"]
        }
        result = self.qa_chain(chain_input)
        return result["answer"]
