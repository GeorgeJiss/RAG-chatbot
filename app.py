import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.llm import get_chatgroq_model
from utils.rag import get_pdf_text, get_text_chunks, create_vector_store, get_rag_context
from utils.search import perform_web_search
from config.config import get_groq_api_key

def get_chat_response(chat_model, messages, system_prompt, query, mode, use_web_search, vectorstore):
    """Get response from the chat model with RAG and Web Search context"""
    try:
        # Build context
        context = ""
        
        # 1. RAG Context
        if vectorstore:
            rag_context = get_rag_context(query, vectorstore)
            if rag_context:
                context += f"\n\n--- Document Context ---\n{rag_context}\n"
                
        # 2. Web Search Context
        if use_web_search:
            web_context = perform_web_search(query)
            if web_context:
                context += f"\n\n--- Live Web Search Results ---\n{web_context}\n"
        
        # Prepare system prompt
        final_system_prompt = system_prompt
        if context:
            final_system_prompt += f"\n\nPlease use the following context to answer the user's question. If the answer is not in the context, but you know it, you can answer it. If you don't know, say you don't know.\n{context}"
            
        formatted_messages = [SystemMessage(content=final_system_prompt)]
        
        # Add conversation history
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        
        # Get response from model
        response = chat_model.invoke(formatted_messages)
        return response.content
    
    except Exception as e:
        return f"Error getting response: {str(e)}"

def chat_page():
    st.title("Academic Researcher & Study Companion")
    
    # Sidebar configurations
    with st.sidebar:
        st.header("Configuration")
        
        # API Key handling
        api_key = get_groq_api_key()
        if not api_key:
            st.warning("No Groq API key found in environment.")
            user_api_key = st.text_input("Enter Groq API Key to Start:", type="password")
            if user_api_key:
                os.environ["GROQ_API_KEY"] = user_api_key
                st.success("API Key activated for this session!")
        else:
            st.success("Groq API Key loaded!")
            
        st.divider()
        
        st.header("Study Materials (RAG)")
        pdf_docs = st.file_uploader("Upload your lecture slides or textbook (PDF)", accept_multiple_files=True)
        if st.button("Process Documents"):
            with st.spinner("Processing your study material..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = create_vector_store(text_chunks)
                    st.session_state.vectorstore = vectorstore
                    st.success("Documents digested successfully! You can now ask questions about them.")
                else:
                    st.warning("Please upload a PDF first.")
                    
        st.divider()
        
        st.header("Response Settings")
        mode = st.radio("Explanation Detail:", ["Concise", "Detailed"], index=1, help="Concise returns short bullet points. Detailed gives an in-depth lecture-style explanation.")
        use_web_search = st.checkbox("Enable Live Web Search", value=False, help="Check this to allow the search for recent news or information outside the PDF.")
        
    # Initialize chat history & vectorstore
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
        
    # Set System Prompt based on Mode
    if mode == "Concise":
        system_prompt = "You are an expert Academic AI Assistant. Always answer concisely using bullet points and brief summaries. Get straight to the point."
    else:
        system_prompt = "You are an expert Academic AI Assistant. Always provide detailed, comprehensive, and well-explained answers, suitable for university-level research. Use clear structure, examples, and deep dives into the concepts."
        
    # Try to Initialize Model (fails gracefully if API key is not yet set)
    try:
        chat_model = get_chatgroq_model()
    except Exception as e:
        chat_model = None
        
    # Display chat messages
    for message in st.session_state.messages:
        avatar_icon = "https://ui-avatars.com/api/?name=User&background=random" if message["role"] == "user" else "https://ui-avatars.com/api/?name=System&background=random"
        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])
            
    # Chat Input
    if not chat_model and not os.environ.get("GROQ_API_KEY"):
        st.info("Please enter your Groq API Key in the sidebar to start.")
    elif prompt := st.chat_input("Ask a question about your documents, or any topic..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="https://ui-avatars.com/api/?name=User&background=random"):
            st.markdown(prompt)
            
        with st.chat_message("assistant", avatar="https://ui-avatars.com/api/?name=System&background=random"):
            with st.spinner("Processing..."):
                response = get_chat_response(
                    chat_model, 
                    st.session_state.messages, 
                    system_prompt, 
                    prompt, 
                    mode, 
                    use_web_search, 
                    st.session_state.vectorstore
                )
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.set_page_config(page_title="Academic Study Companion", layout="wide")
    
    with st.sidebar:
        st.title("Navigation")
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
    chat_page()

if __name__ == "__main__":
    main()