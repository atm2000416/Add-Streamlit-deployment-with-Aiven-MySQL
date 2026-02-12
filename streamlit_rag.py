import streamlit as st
import os
import sys
from io import StringIO

# Import your existing modules
# (All your imports from input.py would go here)
import requests
import json
from typing import List, Dict

# Configure page
st.set_page_config(
    page_title="Camp Chatbot",
    page_icon="🏕️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .camp-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="camp-header">
    <h1>🏕️ Camp Discovery Chatbot</h1>
    <p>Find the perfect camp in Canada for your child!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with instructions
with st.sidebar:
    st.header("📋 How to Use")
    st.markdown("""
    Share these details to get personalized recommendations:
    
    - 📍 **Region** in Canada
    - 🎯 **Type of camp** (STEM, sports, arts, etc.)
    - 👶 **Age and gender** of camper
    - 🏕️ **Day camp or overnight?**
    - 💸 **Your budget**
    
    **Examples:**
    - "Show me STEM camps in Ontario for 12-year-old boys under $500"
    - "What overnight camps in BC focus on outdoor adventures?"
    - "List day camps with swimming in Toronto"
    """)
    
    st.divider()
    st.caption("Powered by AI • Data refreshed daily")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": """Hi! I'm your camp chatbot 🤖

Please share:

📍 Region in Canada you're interested in
🎯 Type of camp (STEM, sports, arts, etc.)
👶 Age and gender of the camper
🏕️ Day camp or overnight?
💸 Your budget

Got other questions? Just ask! 💬"""
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about camps..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show assistant response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching for the best camps..."):
            try:
                # TODO: Integrate your actual RAG logic here
                # For now, placeholder that shows the structure
                
                # Get API keys from Streamlit secrets
                gemini_key = st.secrets.get("GEMINI_API_KEY", "")
                pinecone_key = st.secrets.get("PINECONE_API_KEY", "")
                
                # Call your router logic (from input.py)
                # response = process_query(prompt, gemini_key, pinecone_key)
                
                # Placeholder response
                response = f"Processing your query: '{prompt}'\n\n*Integration with your RAG system pending...*"
                
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Clear chat button in sidebar
with st.sidebar:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
