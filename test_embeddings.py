from app.vectorstore.embedder import embed_text, embed_texts, embed_documents
from app.ingestion.document import Document

# --- Test 1: Single text ---
print("=== Single embedding ===")
vec = embed_text("machine learning is a subset of AI")
print(f"Type:       {type(vec)}")
print(f"Dimensions: {len(vec)}")
print(f"First 5:    {vec[:5]}")
print()

# --- Test 2: Semantic similarity ---
print("=== Semantic similarity check ===")
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    # Manual cosine similarity = dot product / (magnitude A * magnitude B)
    return dot(a, b) / (norm(a) * norm(b))

v1 = embed_text("I love dogs")
v2 = embed_text("I adore puppies")
v3 = embed_text("stock market crash")

sim_12 = cosine_similarity(v1, v2)
sim_13 = cosine_similarity(v1, v3)

print(f"'I love dogs' vs 'I adore puppies': {sim_12:.4f}")
print(f"'I love dogs' vs 'stock market':    {sim_13:.4f}")
print("sim_12 should be much higher than sim_13")
print()

# --- Test 3: Embed documents ---
print("=== Document embedding ===")
docs = [
    Document(text="Machine learning is a field of AI", metadata={"source": "test"}),
    Document(text="Python is a programming language", metadata={"source": "test"}),
]
embeddings = embed_documents(docs)
print(f"Got {len(embeddings)} vectors for {len(docs)} documents")
print(f"Each vector: {len(embeddings[0])} dimensions")