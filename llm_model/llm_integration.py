# llm/llm_integration.py
import openai
from typing import List
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()  # ✅ Loads .env variables

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def truncate_context(chunks: List[dict], max_tokens: int = 1500, model: str = "gpt-3.5-turbo") -> str:
    """
    Truncate the list of chunks so their total token count does not exceed max_tokens.
    """
    context = ""
    total_tokens = 0

    for chunk in chunks:
        chunk_text = chunk["content"]
        chunk_tokens = count_tokens(chunk_text, model)
        if total_tokens + chunk_tokens > max_tokens:
            break
        context += chunk_text + "\n"
        total_tokens += chunk_tokens

    return context.strip()

def generate_answer(query: str, context: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Generate final answer using OpenAI chat completion.
    """
    prompt = f"""
    Use the following context to answer the question as accurately as possible.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300,
    )
    return response["choices"][0]["message"]["content"].strip()
