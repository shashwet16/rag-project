from openai import OpenAI  # Gemini via OpenAI-compatible endpoint

from app.config import GEMINI_API_KEY, LLM_MODEL

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[Source {i+1}]: {chunk['text']}"
        for i, chunk in enumerate(chunks)
    )
    return f"""Context:
{context}

Question: {query}"""


def generate_answer(query, chunks):
    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable research assistant. "
                    "You have been given context extracted from the user's documents.\n\n"

                    "How to answer:\n"
                    "1. Base your answer primarily on the provided context.\n"
                    "2. You may use your general knowledge to explain, clarify, or expand "
                    "   on what the context says — but make it clear when you do.\n"
                    "3. If the context covers the question well, stay close to it.\n"
                    "4. If the question has no relevant context at all, say so briefly, "
                    "   then answer from general knowledge if you can.\n"
                    "5. Answer thoroughly. Explain clearly. Don't be unnecessarily brief.\n"
                    "6. Do NOT make up facts or fabricate document content.\n"
                    # WHY these rules: The context comes first, but the model is
                    # allowed to fill gaps from general knowledge as long as it
                    # says so — rule 6 still forbids inventing document content.
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        # WHAT: Temperature controls randomness in generation.
        # WHY:  Low temperature (0.1) makes responses more deterministic and
        #       factual. High temperature (0.9+) makes responses creative but
        #       more likely to hallucinate. For RAG, low is always better.
        # ALTERNATIVES: 0 for fully deterministic, but some models don't support it.
    )

    return response.choices[0].message.content