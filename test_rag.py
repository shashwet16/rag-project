# test_rag.py
from app.rag_pipeline import ask , format_response

questions = [
    "What is the investment multiplier?",
    "Q: What is the difference between GNP and NNP?",
]

for q in questions:
    print("=" * 60)
    result = ask(q)
    print(format_response(result))
    print()