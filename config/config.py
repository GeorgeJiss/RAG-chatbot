import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_groq_api_key():
    """Retrieve the Groq API Key from environment variables."""
    return os.environ.get("GROQ_API_KEY", "")
