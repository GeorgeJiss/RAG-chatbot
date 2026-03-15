import os
import sys
from langchain_groq import ChatGroq
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.config import get_groq_api_key

def get_chatgroq_model():
    """Initialize and return the Groq chat model"""
    api_key = get_groq_api_key()
    if not api_key:
        return None
        
    try:
        # Initialize the Groq chat model with a fast model
        groq_model = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant",
        )
        return groq_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")