import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, List

import chardet
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

EMBEDDINGS_MODEL = "sentence-transformers/all-mpnet-base-v2"
PERSIST_DIR = "./chroma_db"
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300

ALLOWED_EXTS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".md",
    ".json",
    ".yml",
    ".yaml",
    ".txt",
    ".sh",
    ".dockerfile",
    ".html",
    ".css",
    ".vue",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".cs",
    ".go",
    ".php",
    ".rb",
    ".rust",
    ".swift",
}

EXCLUDED_DIRS = {
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    ".git",
    ".idea",
    "env",
    "dist",
    "build",
    ".pytest_cache",
    "__snapshots__",
    "chroma_db",
}

MAX_FILE_SIZE = 100_000_000  # 100 MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, any]


class SimpleTextSplitter:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        return chunks


class CodeIndexer:
    def __init__(self, embeddings_model: str = EMBEDDINGS_MODEL, persist_dir: str = PERSIST_DIR) -> None:
        self.persist_dir = persist_dir
        self.embeddings = SentenceTransformer(embeddings_model)
        self.splitter = SimpleTextSplitter()
        self.client = PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection("code_collection")
        self.stats = {"loaded": 0, "skipped_ext": 0, "skipped_size": 0, "error": 0}
        self.indexed_files_path = os.path.join(persist_dir, "indexed_files.txt")
        self.indexed_files: List[str] = []

    def load_documents_from_folder(self, folder_path: str) -> Dict[str, List[Document]]:
        logging.info(f"Carregando documentos do diretório: {folder_path}")
        docs_by_file: Dict[str, List[Document]] = {}

        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in ALLOWED_EXTS:
                    self.stats["skipped_ext"] += 1
                    continue

                full_path = os.path.join(root, file)
                if os.path.getsize(full_path) > MAX_FILE_SIZE:
                    logging.warning(f"Arquivo grande ignorado: {full_path}")
                    self.stats["skipped_size"] += 1
                    continue

                with open(full_path, "rb") as f:
                    raw_data = f.read()
                encoding = chardet.detect(raw_data)["encoding"] or "utf-8"
                text = raw_data.decode(encoding, errors="replace")

                chunks = self.splitter.split_text(text)
                documents: List[Document] = []
                for idx, chunk in enumerate(chunks):
                    documents.append(
                        Document(page_content=chunk, metadata={"full_path": full_path, "chunk_id": idx})
                    )

                docs_by_file[full_path] = documents
                self.stats["loaded"] += len(documents)

        return docs_by_file

    def index_documents(self, processed_docs: Dict[str, List[Document]]):
        all_docs: List[Document] = []
        for file_path, docs in processed_docs.items():
            if not docs:
                continue
            all_docs.extend(docs)
            self.indexed_files.append(os.path.relpath(file_path))

        if all_docs:
            documents = [d.page_content for d in all_docs]
            metadatas = [d.metadata for d in all_docs]
            ids = [f"{md['full_path']}_{md['chunk_id']}" for md in metadatas]
            embeddings = self.embeddings.encode(documents).tolist()
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            self.client.persist()

        self._save_indexed_files()

    def _save_indexed_files(self):
        with open(self.indexed_files_path, "w") as f:
            for file_name in self.indexed_files:
                f.write(file_name + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexador de código usando ChromaDB.")
    parser.add_argument("--folder", required=True, help="Pasta contendo os arquivos a serem indexados.")
    args = parser.parse_args()

    indexer = CodeIndexer()
    processed_docs = indexer.load_documents_from_folder(args.folder)
    indexer.index_documents(processed_docs)

    logging.info("Indexação concluída.")
