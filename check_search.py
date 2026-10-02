from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
e = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
v = Chroma(persist_directory='chroma_db_v_rag', embedding_function=e)
results = v.similarity_search('T1055 process injection', k=3)
for r in results:
 print("---")
 print(r.page_content[:300])
