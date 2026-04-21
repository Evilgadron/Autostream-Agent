import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


class GeminiAPIEmbeddings(Embeddings):
    def __init__(self, model: str = "gemini-embedding-2-preview"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _embed_one(self, text: str, task_type: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=EmbedContentConfig(task_type=task_type),
        )

        if not result.embeddings:
            raise RuntimeError("Gemini returned no embeddings.")

        first_embedding = result.embeddings[0]
        if first_embedding is None or first_embedding.values is None:
            raise RuntimeError("Gemini returned an empty embedding vector.")

        return list(first_embedding.values)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text, "RETRIEVAL_DOCUMENT") for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text, "RETRIEVAL_QUERY")


def setup_retriever():
    print("Initializing Knowledge Base...")

    kb_data = [
        "AutoStream Basic Plan Pricing & Features: $29/month. Includes 10 videos/month and 720p resolution.",
        "AutoStream Pro Plan Pricing & Features: $79/month. Includes Unlimited videos, 4K resolution, and AI captions.",
        "AutoStream Company Policy: No refunds after 7 days.",
        "AutoStream Company Policy: 24/7 support is available only on the Pro plan.",
    ]

    documents = [Document(page_content=text) for text in kb_data]

    embeddings = GeminiAPIEmbeddings(model="gemini-embedding-2-preview")
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store.as_retriever(search_kwargs={"k": 1})


if __name__ == "__main__":
    retriever = setup_retriever()

    query = "Do you offer refunds?"
    print(f"\nTesting Query: '{query}'")

    results = retriever.invoke(query)

    if results:
        print(f"Retrieved Fact: {results[0].page_content}")
    else:
        print("No result returned.")