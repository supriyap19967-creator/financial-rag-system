import os
import requests
from colorama import Fore, Style
import ollama
from dotenv import load_dotenv
load_dotenv()

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_API_KEY = os.getenv("HF_API_KEY")


def get_hybrid_llm(prompt: str):

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    try:
        print(Fore.CYAN + "⚡ Trying Cloud Model (HF Inference API)..." + Style.RESET_ALL)

        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": prompt},
            timeout=10
        )

        response.raise_for_status()

        result = response.json()[0]["generated_text"]

        print(Fore.GREEN + "✅ Cloud model used." + Style.RESET_ALL)
        return result

    except Exception as e:

        print(Fore.YELLOW + "☁️ Cloud failed. Switching to Local Ollama..." + Style.RESET_ALL)

        local_response = ollama.chat(
            model="gemma:2b",
            messages=[{"role": "user", "content": prompt}]
        )

        print(Fore.MAGENTA + "💻 Local model used." + Style.RESET_ALL)

        return local_response["message"]["content"]
# 🔍 Lightweight Self-RAG Validator
def validate_answer(context: str, answer: str):

    validation_prompt = f"""
Check if the answer is supported by the context.

Respond with ONLY one word:
YES
or
NO

Context:
{context}

Answer:
{answer}
"""

    result = get_hybrid_llm(validation_prompt)
    return result.strip()