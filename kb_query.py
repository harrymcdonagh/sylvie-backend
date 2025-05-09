# kb_query.py
import json
from sentence_transformers import SentenceTransformer
import faiss

# Load services and the normalized index
with open("uea_services.json", "r", encoding="utf-8") as f:
    services = json.load(f)

index = faiss.read_index("uea_index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_relevant_services(
    question: str,
    top_k: int = 10,
    min_score: float = 0.5
):
    # 1) Embed & normalize your query
    qv = model.encode([question], normalize_embeddings=True)

    # 2) Retrieve more candidates than you need
    scores, idxs = index.search(qv, top_k)
    scores, idxs = scores[0], idxs[0]

    # 3) Filter by a cosine‐similarity threshold
    results = []
    for score, idx in zip(scores, idxs):
        if score >= min_score:
            svc = services[idx].copy()
            svc["score"] = float(score)
            results.append(svc)

    # 4) Sort by descending score and limit to top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

if __name__=="__main__":
    while True:
        q = input("\nAsk me anything: ")
        svcs = get_relevant_services(q, top_k=10, min_score=0.6)
        if svcs:
            for s in svcs:
                sc = f" ({s['score']:.2f})"
                print(f"• {s['name']}{sc} — {s['desc']} ({s['url']})")
        else:
            print("No matching service found.")
