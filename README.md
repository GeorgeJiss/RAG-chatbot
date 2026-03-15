# Academic Study Companion

A powerful, AI-proof Streamlit web application designed for education and research. This application allows students, researchers, and educators to upload their study materials (PDFs), process them using Retrieval-Augmented Generation (RAG), and interactively converse with the materials.

## Features

- **Document Processing (RAG)**: Upload course lecture slides, textbooks, or research papers in PDF format. The app accurately digests the content into a local vector store, permitting instant retrieval of relevant contexts.
- **Local Embeddings**: Ensures lightweight and offline-capable vectorization using reliable HuggingFace local models (`all-MiniLM-L6-v2`), preserving memory and latency while offering accurate document relevance.
- **Performant LLM Inference**: Built with the `langchain-groq` integration, taking advantage of Llama 3 for exceptionally fast, context-aware answers.
- **Neutral, "AI-Proof" Interface**: The prompts, avatars, and UI are specifically tuned to act as a neutral study companion, presenting information concisely without identifying as a virtual assistant or outputting informal emojis.
- **Live Web Search (Optional)**: Supplements document context with up-to-date web access via DuckDuckGo if the provided documents lack the answer.
- **Adjustable Detail Levels**: Customize the output format to be either "Concise" (short bullet points) or "Detailed" (in-depth lecture-style explanations).

## Tech Stack

- **Frontend**: Streamlit
- **LLM Orchestration**: LangChain, LangChain Community, LangChain Groq
- **Vector Store**: FAISS (CPU)
- **Embeddings**: HuggingFace (`sentence-transformers`)
- **Web Search**: DuckDuckGo Search API

## Setup & Installation

1. **Clone the repository** (or download the source).
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have also installed `langchain-text-splitters`)*
4. **Environment Variables**:
   Create a `.env` file in the root directory and add your Groq API Key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage for Education

1. **Upload**: Use the sidebar to upload a PDF (e.g., a textbook chapter).
2. **Process**: Click "Process Documents" to chunk and embed the text.
3. **Learn**: Ask questions in the chat interface. The system will retrieve relevant chunks and provide a structured, synthesized answer.
