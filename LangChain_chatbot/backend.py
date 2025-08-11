# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

# Step1: Setup Pydantic Model (Schema Validation)
from pydantic import BaseModel
from typing import List
import logging
from fastapi import Request

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

# Step2: Setup AI Agent from FrontEnd Request
from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

ALLOWED_MODEL_NAMES = ["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "gpt-4o-mini"]

app = FastAPI(title="LangGraph AI Agent")

@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.post("/chat")
def chat_endpoint(request: RequestState):
    logging.info(f"Received request: {request}")
    try:
        response = get_response_from_ai_agent(
            query=request.messages,
            allow_search=request.allow_search,
            system_prompt=request.system_prompt,
            model_name=request.model_name,
            provider=request.model_provider,
        )
        logging.info(f"Response: {response}")
        return {"response": response}
    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}", exc_info=True)
        return {"error": "Internal server error. Check backend logs for details."}


# Step3: Run app & Explore Swagger UI Docs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999, reload=True)
