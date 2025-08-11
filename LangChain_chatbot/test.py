import os
from dotenv import load_dotenv
load_dotenv()
print("GROQ_API_KEY:", os.environ.get("GROQ_API_KEY"))
print("TAVILY_API_KEY:", os.environ.get("TAVILY_API_KEY"))
print("OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))
