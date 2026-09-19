from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss


class FAISSMemory:
    def __init__(self, embeddings, max_results=5):
        self.embeddings = embeddings
        self.max_results = max_results
        dim = len(self.embeddings.embed_query("dimension probe"))
        self.store = FAISS(
            embedding_function=self.embeddings,
            index=faiss.IndexFlatL2(dim),
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )
        self.memory_items = []
        self.reflections = []

    def add(self, observation, action, result=""):
        text = f"Observation: {observation} | Action: {action} | Result: {result}"
        self.memory_items.append(text)
        try:
            self.store.add_texts([text])
        except Exception:
            pass
        return text

    def add_reflection(self, reflection):
        self.reflections.append(reflection)
        self.memory_items.append(f"Reflection: {reflection}")
        try:
            self.store.add_texts([f"Reflection: {reflection}"])
        except Exception:
            pass

    def retrieve(self, query):
        if not self.memory_items:
            return []
        try:
            results = self.store.similarity_search(query, k=self.max_results)
        except Exception:
            return self.recent(self.max_results)
        return [r.page_content for r in results]

    def recent(self, n=5):
        return self.memory_items[-n:]

    def count(self):
        return len(self.memory_items)
