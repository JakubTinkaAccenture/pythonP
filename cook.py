from dotenv import load_dotenv
import os
import warnings

from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.chains import LLMChain

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

class OpenAIAgent:
    def __init__(self, system_prompt_path):
        load_dotenv()
        self.model = AzureChatOpenAI(
            temperature=0,
            azure_endpoint=os.getenv["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.getenv["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.getenv["AZURE_OPENAI_API_VERSION"],
            model_name="gp-4o-mini",
        )

        # Load system prompt from file
        self.system_prompt = self.load_prompt(system_prompt_path)
        
        # Set up memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10
        )

        # Use a structured ChatPromptTemplate
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # Create chain
        self.chain = LLMChain(
            llm=self.model,
            prompt=self.prompt_template,
            memory=self.memory
        )

    def load_prompt(self, path):
        with open(path, 'r') as file:
            return file.read().strip()

    def send_message(self, user_message):
        response = self.chain.run(input=user_message)
        return response
