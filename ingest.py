
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

def ingest(): 
    print("Loading Documents...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )

    docs = loader.load()

    print(f"Loaded {len(docs)} document(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 150]

    print(f"Split into {len(chunks)} chunks")
    print("Connecting to Ollama embeddings")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print("Storing chunks in Chroma DB")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"Vectors saved to '{CHROMA_DIR}/'")



if __name__ == "__main__":
    ingest()