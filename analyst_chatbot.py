import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
VECTOR_DB_PATH = "chroma_db_v_rag"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-3.6-flash"

# --- 1. THE SECURITY ANALYST SYSTEM PROMPT ---
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
    """Joins retrieved Document objects into a single clean text block."""
    return "\n\n".join(doc.page_content for doc in docs)


# --- 2. RAG CHAIN SETUP ---

@st.cache_resource
def setup_rag_chain(api_key: str):
    """Loads the vector store and initializes the RAG chain components using Gemini."""

    if not os.path.isdir(VECTOR_DB_PATH):
        return None, f"FATAL: Vector database not found at path: {VECTOR_DB_PATH}. Run the indexing script first."

    try:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=0.0, google_api_key=api_key)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain, None

    except Exception as e:
        return None, f"Error setting up RAG Chain: {e}"


# --- 3. STREAMLIT INTERFACE ---

def main():
    st.set_page_config(page_title="CTI-RAG Analyst", layout="wide")
    st.title("CTI-RAG (Cyber Threat Intelligence Analyst)")

    with st.sidebar:
        st.header("API Key Setup")
        google_api_key = st.text_input(
            "Enter your Google API Key",
            type="password"
        )
        st.markdown("---")
        st.info("The key is used to connect to the Gemini API for generation.")

    if not google_api_key:
        st.warning("Please enter your Google API Key in the sidebar to start the CTI-RAG analysis.")
        return

    rag_chain, error_message = setup_rag_chain(google_api_key)

    if rag_chain is None:
        st.error(error_message)
        return

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to CTI-RAG. How can I assist with your threat intelligence query today? Please ask about a CVE ID or a MITRE ATT&CK technique."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_text := st.chat_input("Ask about a CVE, mitigation, or threat Tactic..."):
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing intelligence via Gemini..."):
                try:
                    response = rag_chain.invoke(prompt_text)
                    st.markdown(response)
                except Exception as e:
                    st.error(f"An error occurred during Generation: {e}")
                    response = "I encountered a processing error while attempting to generate the response."

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()

    