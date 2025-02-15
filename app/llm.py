import openai
import os

API_KEY = os.getenv("AIPROXY_TOKEN")

def parse_task(task_description):
    """Uses LLM to interpret a plain-English task."""
    prompt = f"Parse this task: {task_description}. Return JSON output."
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        api_key=API_KEY
    )
    
    return response["choices"][0]["message"]["content"]

def call_llm(prompt: str) -> str:
    # Implement the function to call the LLM API
    return "LLM response based on: " + prompt
