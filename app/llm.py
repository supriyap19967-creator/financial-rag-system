import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_API_KEY = os.getenv("HF_API_KEY")


class HybridLLM:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    def invoke(self, prompt: str):

        try:
            response = requests.post(
                HF_API_URL,
                headers=self.headers,
                json={"inputs": prompt},
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            if isinstance(data, list):
                return data[0]["generated_text"]

            return str(data)

        except Exception as e:
            return f"LLM error: {str(e)}"


def get_hybrid_llm():
    return HybridLLM()


# Simple validator
def validate_answer(answer: str):
    if not answer or len(answer.strip()) < 5:
        return "Answer could not be generated."
    return answer
