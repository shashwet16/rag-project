from app.rag_pipeline import ask, format_response

# These should be IN your knowledge base
in_scope = [
    "What are the three types of machine learning?",
    "What is reinforcement learning?",
]

# These should NOT be in your knowledge base
out_of_scope = [
    "What is the GDP of France?",
    "Who won the 2024 Olympics 100m sprint?",
    "What is the recipe for chocolate cake?",
]

print("=== IN SCOPE (should answer) ===\n")
for q in in_scope:
    result = ask(q)
    print(format_response(result))
    print()

print("=== OUT OF SCOPE (should refuse) ===\n")
for q in out_of_scope:
    result = ask(q)
    print(format_response(result))
    print()