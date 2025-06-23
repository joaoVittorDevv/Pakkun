import argparse
import logging
import os
from typing import List, Dict, Optional, Tuple
import chardet
import torch
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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


class CodeIndexer:
    def __init__(
        self,
        embeddings_model: str = "sentence-transformers/all-mpnet-base-v2",
        persist_dir: str = "./chroma_db",
        device: Optional[str] = None,
    ):
        self.persist_dir = persist_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        self.stats = {"loaded": 0, "skipped_ext": 0, "skipped_size": 0, "error": 0}
        self.indexed_files_path = os.path.join(persist_dir, "indexed_files.txt")
        self.indexed_files = []

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
                    self.stats["skipped_size"] = self.stats.get("skipped_size", 0) + 1
                    continue

                with open(full_path, "rb") as f:
                    raw_data = f.read()
                encoding = chardet.detect(raw_data)["encoding"] or "utf-8"
                text = raw_data.decode(encoding, errors="replace")

                chunks = self.splitter.split_text(text)
                documents: List[Document] = []
                for idx, chunk in enumerate(chunks):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "full_path": full_path,
                                "chunk_id": idx,
                            },
                        )
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
            indexed_file_name = os.path.relpath(file_path)
            self.indexed_files.append(indexed_file_name)

        if all_docs:
            Chroma.from_documents(
                documents=all_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name="code_collection",
            )

        self._save_indexed_files()

    def _save_indexed_files(self):
        indexed_files_path = os.path.join(self.persist_dir, "indexed_files.txt")
        with open(indexed_files_path, "w") as f:
            for file_name in self.indexed_files:
                f.write(file_name + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Indexador de código usando ChromaDB.")
    parser.add_argument(
        "--folder", required=True, help="Pasta contendo os arquivos a serem indexados."
    )

    args = parser.parse_args()

    indexer = CodeIndexer()

    processed_docs = indexer.load_documents_from_folder(args.folder)
    indexer.index_documents(processed_docs)

    logging.info("Indexação concluída.")
