from dotenv import load_dotenv
load_dotenv()

import os

# Correct environment variable name for Groq key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # You can keep or remove this if not needed

print("GROQ_API_KEY:", GROQ_API_KEY)
print("TAVILY_API_KEY:", TAVILY_API_KEY)
print("OPENAI_API_KEY:", OPENAI_API_KEY)

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt = "Act as an AI chatbot who is smart and friendly"

# Setup Groq LLM with your API key explicitly
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

search_tool = TavilySearchResults(max_results=2)

def get_response_from_ai_agent(query, allow_search=True, system_prompt=None, model_name=None, provider=None):
    # You can use system_prompt, model_name, provider inside here

    # For example, update system prompt if provided
    prompt = system_prompt or "Act as an AI chatbot who is smart and friendly"

    # Setup Groq LLM with your API key explicitly
    # Use model_name argument if provided, else default
    model = model_name or "llama-3.3-70b-versatile"
    groq_llm = ChatGroq(model=model, api_key=GROQ_API_KEY)

    # Setup tools conditionally
    tools = [search_tool] if allow_search else []

    agent = create_react_agent(
        model=groq_llm,
        tools=tools,
    )

    # Assuming you want to pass system prompt as an initial system message,
    # modify state accordingly
    state = {"messages": [prompt] + query if isinstance(query, list) else [prompt, query]}

    response = agent.invoke(state)
    messages = response.get("messages", [])
    ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
    return ai_messages[-1] if ai_messages else "No response from AI."


if __name__ == "__main__":
    user_query = "Hello, how are you?"
    response = get_response_from_ai_agent(user_query)
    print("AI Response:", response)
