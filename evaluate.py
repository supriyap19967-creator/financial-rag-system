import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from langchain_community.llms import Ollama

API_URL = "http://127.0.0.1:8000/query"

questions = [
    "Under which Rule should suspicious transactions be reported to FIU-IND?",
    "How must financial institutions report suspicious activity?",
    "What penalties apply for delayed reporting?"
]

records = []

for q in questions:
    response = requests.post(API_URL, json={"question": q})
    data = response.json()

    records.append({
        "question": q,
        "answer": data["answer"],
        "contexts": [data["answer"]],
        "ground_truth": ""
    })

dataset = Dataset.from_list(records)

evaluator_llm = LangchainLLMWrapper(
    Ollama(model="gemma:2b")
)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=evaluator_llm
)

print("\n=== RAGAS RESULTS ===")
print(result)