# kb.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SERVICES_PATH = Path("uea_services.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

class KnowledgeBase:
    def __init__(self, kb_path: Path = SERVICES_PATH, model_name: str = EMBED_MODEL_NAME):
        with open(kb_path, encoding="utf-8") as f:
            self.services = json.load(f)
        self.embed_model = SentenceTransformer(model_name)
        self.texts = [svc.get("description", svc.get("desc", "")) for svc in self.services]
        self.embeddings = self.embed_model.encode(self.texts, convert_to_numpy=True)

    def retrieve(self, query: str, top_k: int = 3) -> str:
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        ids = sims.argsort()[-top_k:][::-1]
        entries = []
        for i in ids:
            svc = self.services[i]
            name = svc.get("service", svc.get("name", "Service"))
            desc = svc.get("description", svc.get("desc", ""))
            url = svc.get("url", "")
            entry = f"- {name}: {desc}"
            if url:
                entry += f" ({url})"
            entries.append(entry)
        return "\n".join(entries)

# singleton instance for import
kb_instance = KnowledgeBase()
