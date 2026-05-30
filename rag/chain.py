from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag.retriever import load_retriever
from dotenv import load_dotenv

load_dotenv()


LLM_MODEL = "gemini-3.1-flash-lite"

PROMPT = ChatPromptTemplate.from_template("""
You are a helpful veterinary advisory assistant for farmers.
Answer the farmer's question in a clear, friendly tone that a non-expert can understand.
Use ONLY the information provided in the context below — do not use any outside knowledge.

Structure your answer as:
1. What the likely disease or issue is
2. Key signs to watch for
3. What the farmer should do (treatment, precautions, or who to contact)

If the context does not contain relevant information, respond only with: "I don't have information on that in my knowledge base."

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

