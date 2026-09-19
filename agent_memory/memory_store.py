from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss

class FAISSMemory:
    """
    FAISS-backed memory for an agent.
    Stores observations and actions as embeddings for semantic retrieval.
    """
    def __init__(self, embeddings, max_results=5):
        self.embeddings = embeddings
        self.max_results = max_results
        # Initialize empty FAISS index
        self.store = FAISS(
            embedding_function=self.embeddings,
            index=faiss.IndexFlatL2(self.embeddings.dim),  # L2 distance
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={}
        )
        self.memory_items = []  # Keep raw text for logging

    def add(self, observation, action):
        """
        Add a memory item (observation + action)
        """
        text = f"Observation: {observation} | Action: {action}"
        self.memory_items.append(text)
        self.store.add_texts([text])

    def retrieve(self, query):
        """
        Retrieve the most relevant memory items based on query
        """
        results = self.store.similarity_search(query, k=self.max_results)
        return [r.page_content for r in results]

    def recent(self, n=5):
        """
        Return the most recent n memory items
        """
        return self.memory_items[-n:]