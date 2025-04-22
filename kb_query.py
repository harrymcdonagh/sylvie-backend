#knowledge base test

import json
from sentence_transformers import SentenceTransformer
import faiss

services = json.load(open("uea_services.json", encoding="utf-8"))
index     = faiss.read_index("uea_index.faiss")
model     = SentenceTransformer("all-MiniLM-L6-v2")

def get_top_services(question, k=3):
    qv = model.encode([question])
    _, idxs = index.search(qv, k)
    return [services[i] for i in idxs[0]]

if __name__=="__main__":
    while True:
        q = input("\nAsk me anything: ")
        top3 = get_top_services(q)
        for s in top3:
            print(f"• {s['name']} — {s['desc']} ({s['url']})")
