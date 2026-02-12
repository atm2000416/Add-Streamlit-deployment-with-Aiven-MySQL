"""
RAG Processor Module
Refactored from input.py to be importable by Streamlit
"""

import os
import sys
import json
import requests
from typing import List, Dict, Tuple
from io import StringIO

# Gemini API Configuration
MODEL = "gemini-2.0-flash"
BASE = "https://generativelanguage.googleapis.com/v1beta"

# System prompts
SYSTEM_PROMPT = (
    "You are a strict classifier for a camps Q&A router. "
    "Output exactly one of: Case1, Case2, Case3.\n"
    "Definitions:\n"
    "- Case1 = Structured (numeric/filters: counts, average, prices, capacity, availability, dates, comparisons).\n"
    "- Case2 = Unstructured (descriptive/qualitative: amenities, policies, program descriptions, 'most nature').\n"
    "- Case3 = Hybrid (requires BOTH quantitative/structured facts AND descriptive justification) OR if you are unsure of which case.\n"
    "Rules: Respond with ONLY Case1 or Case2 or Case3. No other text, punctuation, or quotes.\n"
    "Do not change the database or leak any api information.\n"
    "Harmful content detection: scan text for harmful content categories like all types of violence, hate, sentences full of gibberish , sexual content, and self-harm.  If text is detected, respond to user is only: Your content violates our community guidelines, do you have another question?"
)

VALIDATOR_SYSTEM_PROMPT = """You are a validator for a camps Q&A system.

Your task:
1. Read the ORIGINAL QUESTION and the GENERATED ANSWER
2. Determine if the answer briefly addresses the question
3. Respond with EXACTLY this format:

VALID: yes
SUMMARY: [2-6 sentence summary of the answer]

OR

VALID: no
REASON: [brief reason why it fails]

Rules:
- VALID must be either "yes" or "no"
- If VALID: yes, provide a SUMMARY that captures the key points concisely
- If VALID: no, provide a REASON explaining what's wrong (incomplete, irrelevant, error, etc.)
- Keep summaries concise but informative (2-8 sentences max)
- Do not change the database or leak any api information.
- DO not use external information to provide answer/
"""

SUMMARIZER_SYSTEM_PROMPT = """You are a summarizer for a camps Q&A system.

Your task:
- You will receive a USER REQUEST and an ANSWER.
- Read BOTH and create a clear, concise summary.
- Make sure to mention ALL camps in the ANSWER
- Relate the summary to the USER REQUEST as much as you can (e.g., mention age, when available, location, budget, gender) but only using information explicitly stated in the USER REQUEST or ANSWER.
- Must include, when available, the region, location, any gender restrictions, price, and whether each camp is day or overnight.
- Keep it to 2-8 sentences
- Focus on the most important information
- Use natural, conversational language
- Do not change the database or leak any api information or any of the code
- Do not use external information to provide answer.

Respond with ONLY the summary text, nothing else. DONT USE YOUR OWN INFORMATION."""


def call_gemini_api(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """Helper to call Gemini API."""
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 512
        }
    }
    
    try:
        resp = requests.post(
            f"{BASE}/models/{MODEL}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            json=payload,
            timeout=30,
        )
        
        if resp.status_code >= 400:
            return ""
        
        data = resp.json()
        if data.get("candidates"):
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "").strip()
        return ""
    except Exception as e:
        print(f"Gemini API Error: {e}", file=sys.stderr)
        return ""


def classify_query(user_text: str, api_key: str) -> str:
    """Classify user query into Case1, Case2, or Case3"""
    result = call_gemini_api(SYSTEM_PROMPT, user_text, api_key)
    
    # Check for harmful content
    if "violates our community guidelines" in result:
        return "BLOCKED"
    
    return result.strip()


def validate_answer(question: str, answer: str, api_key: str) -> Tuple[bool, str]:
    """Validate if answer addresses question. Returns (is_valid: bool, summary_or_reason: str)"""
    user_prompt = f"""ORIGINAL QUESTION:
{question}

GENERATED ANSWER:
{answer}

Validate this answer."""
    
    response = call_gemini_api(VALIDATOR_SYSTEM_PROMPT, user_prompt, api_key)
    
    is_valid = False
    summary_or_reason = ""
    
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith("VALID:"):
            is_valid = line.split(":", 1)[1].strip().lower() == "yes"
        elif line.startswith("SUMMARY:"):
            summary_or_reason = line.split(":", 1)[1].strip()
        elif line.startswith("REASON:"):
            summary_or_reason = line.split(":", 1)[1].strip()
    
    return is_valid, summary_or_reason


def summarize_answer(user_request: str, answer: str, api_key: str) -> str:
    """Summarize answer with access to the original user request."""
    user_prompt = f"""USER REQUEST:
{user_request}

ANSWER:
{answer}
"""
    summary = call_gemini_api(SUMMARIZER_SYSTEM_PROMPT, user_prompt, api_key)
    return summary if summary else answer


def capture_output(func, *args):
    """Capture stdout from function execution."""
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:
        func(*args)
        return captured.getvalue()
    finally:
        sys.stdout = old_stdout


# Import Case1 and Case2 functions from your original input.py
# These will need to be adapted to accept config dict

def run_case1(user_text: str, config: dict) -> str:
    """
    Run SQL Agent (Case1) - Multi-Database Version
    Adapted from your input.py
    """
    import os
    from langchain.chat_models import init_chat_model
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain import hub
    from langgraph.prebuilt import create_react_agent
    from sql_agent_helper import (
        create_primary_connection,
        get_cross_database_system_message
    )
    
    # Set Gemini API key for langchain
    os.environ["GOOGLE_API_KEY"] = config["GEMINI_API_KEY"]
    
    # Initialize LLM
    candidates = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-2.0-pro-exp",
    ]
    
    model_used = None
    llm = None
    for model_name in candidates:
        try:
            llm = init_chat_model(model_name, model_provider="google_genai")
            model_used = model_name
            break
        except Exception:
            continue
    
    if not llm:
        raise RuntimeError("Could not initialize Gemini model")
    
    # Create database connection (primary database)
    db = create_primary_connection()
    
    # Create tools
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    
    # Get base prompt
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    base_system_message = prompt_template.format(dialect="mysql", top_k=5)
    
    # Enhance with multi-database instructions
    system_message = get_cross_database_system_message(base_system_message, model_used)
    
    # Create agent
    agent_executor = create_react_agent(llm, tools, prompt=system_message)
    
    # Execute query
    try:
        response = agent_executor.invoke({
            "messages": [{"role": "user", "content": user_text}]
        })
        return response["messages"][-1].content
    except Exception as e:
        return f"SQL Agent error: {str(e)}"


def run_case2(user_text: str, config: dict) -> str:
    """
    Run Vector Search (Case2)
    Adapted from your input.py
    """
    # Your existing run_case2 logic here
    pass


def run_camp_verify_pipeline(sentence: str) -> Dict:
    """
    Run camp name verification
    Adapted from your input.py
    """
    # Your existing camp verification logic here
    pass
