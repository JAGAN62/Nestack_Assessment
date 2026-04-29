import os
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')

index = faiss.read_index("data/faiss_index.bin")

# Load metadata (chunk text + page numbers)
with open("data/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

@app.route("/query", methods=["POST"])
def query():
    data = request.json

    # Get input
    query_text = data.get("query")
    top_k = data.get("top_k", 3)

    # Convert query to embedding
    query_vector = model.encode([query_text])

    # Search similar vectors
    distances, indices = index.search(np.array(query_vector), top_k)

    results = []
    seen = set()  # to remove duplicates

    for i, idx in enumerate(indices[0]):
        chunk_text = metadata[idx]["text"]
        page_number = metadata[idx]["page"]

        # Avoid duplicate chunks
        if chunk_text not in seen:
            seen.add(chunk_text)

            # Convert distance to similarity score (0 to 1)
            score = float(1 / (1 + distances[0][i]))

            results.append({
                "chunk_text": chunk_text,
                "page_number": page_number,
                "score": score
            })

    return jsonify(results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)