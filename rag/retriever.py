from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


CHROMA_DIR = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
DOCS_DIR = "docs"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

def load_chunks():
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    return [chunk for chunk in chunks if len(chunk.page_content.strip()) > 150]


def load_retriever(k: int = 8):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 80}
        )

    chunks = load_chunks()
    bm25_retriever = BM25Retriever.from_documents(chunks, k=6)

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.4, 0.6]
    )


