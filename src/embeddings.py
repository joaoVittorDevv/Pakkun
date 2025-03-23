from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_groq import ChatGroq
from config import EMBEDDINGS_MODEL, DEVICE, LLM_MODEL

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

chroma = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="code_collection",
)

llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

metadata_info = [
    AttributeInfo(
        name="source",
        description="Caminho relativo do arquivo a partir do diretório base",
        type="string",
    ),
    AttributeInfo(
        name="ext",
        description="Extensão do arquivo sem o ponto (ex: 'py', 'js', 'md')",
        type="string",
    ),
    AttributeInfo(
        name="full_path",
        description="Caminho completo do arquivo no sistema",
        type="string",
    ),
    AttributeInfo(
        name="file_size",
        description="Tamanho do conteúdo em número de caracteres",
        type="integer",
    ),
    AttributeInfo(
        name="file_type",
        description="Tipo do arquivo: 'code', 'document' ou 'data'",
        type="string",
    ),
    AttributeInfo(
        name="language",
        description="Linguagem do código (ex: 'python', 'javascript')",
        type="string",
    ),
    AttributeInfo(
        name="is_chunk",
        description="Indica se o documento é um chunk do original (True ou False)",
        type="boolean",
    ),
]

retriever = SelfQueryRetriever.from_llm(
    llm, chroma, "Django/React app codes", metadata_info, verbose=True
)
