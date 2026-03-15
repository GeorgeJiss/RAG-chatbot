from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """Initialize and return the HuggingFace local embedding model"""
    try:
        # Use a lightweight, fast local model for document embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")
