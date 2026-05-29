from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag.retriever import load_retriever
from dotenv import load_dotenv

load_dotenv()


LLM_MODEL = "gemini-3.1-flash-lite"

PROMPT = ChatPromptTemplate.from_template("""
You are a document retrieval assistant. Your only job is to find and copy relevant information from the context below.
Do NOT paraphrase, elaborate, or use any knowledge outside the context.
Find the section in the context that matches the question and copy it out exactly as written.
If the context does not contain any information about the topic, respond only with: "Not found in knowledge base."

Context:
{context}

Question: {question}

Extracted information:
""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain():
    retriever = load_retriever(k=12)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

