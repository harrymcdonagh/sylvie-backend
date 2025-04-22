#knowledge base

import json
from sentence_transformers import SentenceTransformer
import faiss

with open("uea_services.json", "r", encoding="utf-8") as f:
    services = json.load(f)

texts = [s["name"] + " — " + s["desc"] for s in services]

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts)

dim = vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(vectors)

faiss.write_index(index, "uea_index.faiss")
with open("uea_services.json", "w", encoding="utf-8") as f:
    json.dump(services, f, ensure_ascii=False, indent=2)
