import fitz
import argparse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append((i+1, text))
    return pages

def chunk_text(pages, chunk_size=500, overlap=100):
    chunks = []
    for page_num, text in pages:
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append((chunk, page_num))
    return chunks

def create_embeddings(chunks):
    texts = [c[0] for c in chunks]
    embeddings = model.encode(texts)
    return embeddings

def store_faiss(embeddings, chunks):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    os.makedirs("data", exist_ok=True)

    faiss.write_index(index, "data/faiss_index.bin")

    metadata = [{"text": c[0], "page": c[1]} for c in chunks]
    with open("data/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

def main(pdf_path):
    pages = extract_text(pdf_path)
    chunks = chunk_text(pages)
    embeddings = create_embeddings(chunks)
    store_faiss(embeddings, chunks)
    print("Ingestion completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    main(args.file)