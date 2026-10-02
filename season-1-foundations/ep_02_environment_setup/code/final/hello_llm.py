import os
from dotenv import load_dotenv
from rich import print 
from langchain.chat_models import init_chat_model

load_dotenv()

def call_model(model_id: str,prompt:str) ->None:
    print(f"\n[bold cyan]Model:[/bold cyan] {model_id}")
    try:
        #step 2a--init_chat_model turn "provider:model" into a ready model.try
        llm = init_chat_model(model_id)
        #step 2b --.invoke(prompt) sends the prompt and returns the reply.
        response = llm.invoke(prompt)
        print(f"[green]Reply:[/green]{response.content}")
    except Exception as exc:
        #step 2c--show  safe generic error (never leak internals or keys).
        print(f"[red] Could not call {model_id} : [/red] {type(exc).__init__} - "
        f"check the provider key in .env (or that the model is pulled in Ollama).")


def main()->None:
    prompt = "In one sentence , explain what an AI agent is to a beginner."

    #step 3 - Cloud models: only call the ones whose key is present in .env.
    if os.getenv("OPENAI_API_KEY"):
        call_model("openai:gpt-5.6-luna",prompt)
    if os.getenv("GOOGLE_API_KEY"):
        call_model("google_genai:gemini-3.5-flash-lite",prompt)
    if os.getenv("ANTHROPIC_API_KEY"):
        call_model("Claude Opus 5.5",prompt)
    if os.getenv("GROQ_API_KEY"):
        call_model("groq:openzi/gpt-oss-20b",prompt)


    #step4  - free local model , Same code; only the , model id string changed.
    #(needs Ollama running +  'ollama pull llama3.2'.')
    call_model("ollama:llama3.2",prompt)


if __name__ == "__main__":
    main()