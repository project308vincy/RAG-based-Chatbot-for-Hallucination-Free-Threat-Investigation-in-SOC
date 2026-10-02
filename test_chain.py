from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

VECTOR_DB_PATH = "chroma_db_v_rag"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are a highly skilled Cybersecurity Threat Analyst.
Your mission is to synthesize the retrieved context to answer the user's query.

RULES:
1. Only use the provided 'Context' below. DO NOT use your general training knowledge.
2. Cite the specific ID (e.g., CVE-2024-XXXX, T1055.011) for every fact you present.
3. If the Context does not contain the answer, state, 'I cannot find relevant, actionable intelligence for this query in the knowledge base.'
4. Format mitigation steps as a clear, prioritized, bulleted list.

CONTEXT: {context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


print("Enter your Google API Key:")
api_key = input().strip()

question = "Describe technique T1055 and its mitigations."

print("\n=== STEP 1: Loading embeddings and vector store ===")
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

print("\n=== STEP 2: Retrieving documents for question ===")
print("Question:", question)
docs = retriever.invoke(question)
print(f"Number of documents retrieved: {len(docs)}")

print("\n=== STEP 3: Formatted context that will be sent to Gemini ===")
context_text = format_docs(docs)
print(context_text[:2000])
print("\n... (context truncated for display, full length was", len(context_text), "characters)")

print("\n=== STEP 4: Building the final prompt sent to Gemini ===")
formatted_prompt = prompt.invoke({"context": context_text, "question": question})
print(formatted_prompt)

print("\n=== STEP 5: Calling Gemini and printing raw response ===")
try:
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=0.0, google_api_key=api_key)
    response = llm.invoke(formatted_prompt)
    print("RAW RESPONSE OBJECT:")
    print(response)
    print("\nRESPONSE TEXT:")
    print(response.content)
except Exception as e:
    print("EXCEPTION OCCURRED:")
    print(repr(e))
    
