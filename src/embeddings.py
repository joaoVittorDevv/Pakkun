from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from groq import Groq
from crewai.tools.base_tool import BaseTool
from typing import List, Optional, Any, Dict
from config import EMBEDDINGS_MODEL, DEVICE, LLM_MODEL


# Embedding model configuration
embeddings_model = SentenceTransformer(EMBEDDINGS_MODEL, device=DEVICE)

# ChromaDB persistent client
chroma_client = PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("code_collection")

# LLM configuration using Groq
llm = Groq(model=LLM_MODEL)


class ChromaRetriever:
    """Simple retriever compatible with CrewAI."""

    def __init__(
        self,
        collection_name: str = "code_collection",
        persist_directory: str = "./chroma_db",
        embedding_model: Any = embeddings_model,
        k: int = 5,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.k = k
        self.client = chroma_client
        self.collection = collection

    def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return documents matching the query."""
        if k is None:
            k = self.k

        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        documents: List[Dict[str, Any]] = []
        for i in range(len(results["documents"][0])):
            documents.append(
                {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                    "score": 1.0 - results["distances"][0][i] if results["distances"][0] else 0.0,
                }
            )
        return documents

    def invoke(self, query: str) -> str:
        docs = self.search(query)
        return "\n\n".join(
            [
                f"**Documento de {d['metadata'].get('full_path', 'desconhecido')}**\n"
                f"Relevância: {d['score']:.2f}\n\n{d['content']}"
                for d in docs
            ]
        )

    def as_tool(self) -> BaseTool:
        from crewai.tools import Tool

        return Tool(
            name="retriever_tool",
            description=(
                "Use essa ferramenta para buscar informações específicas dentro do "
                "contexto completo dos arquivos do projeto, incluindo Django, React, "
                "Docker, bancos de dados, testes e outras tecnologias associadas. "
                "Ideal para entender estruturas, padrões de projeto e esclarecer "
                "dúvidas técnicas sobre o código existente."
            ),
            func=self.invoke,
        )


retriever = ChromaRetriever()

