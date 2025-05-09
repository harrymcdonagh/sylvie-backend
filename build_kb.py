# build_kb.py
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1) Load your services file
with open("uea_services.json", "r", encoding="utf-8") as f:
    services = json.load(f)

# 2) Prepare the texts you want to embed
texts = [svc["name"] + " — " + svc["desc"] for svc in services]

# 3) Encode + normalize in one step
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, normalize_embeddings=True)
# embeddings.shape == (num_services, dim), each row L2==1

# 4) Build an Inner-Product index (cosine = dot on normalized vectors)
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)

# 5) Add vectors and write to disk
index.add(embeddings)
faiss.write_index(index, "uea_index.faiss")

