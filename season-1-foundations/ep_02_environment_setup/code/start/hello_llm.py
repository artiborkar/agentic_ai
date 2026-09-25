import os
from dotenv import load_dotenv
from rich import print 
from langchain.chat_models import init_chat_model

load_dotenv()

MODEL_NAME = "llama3.2"
MODEL_PROVIDER = "ollama"
llm = init_chat_model(model=MODEL_NAME , model_provider=MODEL_PROVIDER)
response = llm.invoke("Hi. How Are you?")
print(response)
print(response.content)