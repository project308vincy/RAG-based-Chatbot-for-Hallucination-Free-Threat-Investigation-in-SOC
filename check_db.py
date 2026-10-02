from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from collections import Counter

e = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
v = Chroma(persist_directory='chroma_db_v_rag', embedding_function=e)

result = v._collection.get(include=['metadatas'])
metadatas = result['metadatas']

print(f"Total documents in DB: {len(metadatas)}\n")

source_counts = Counter(m.get('source', 'UNKNOWN') for m in metadatas)
print("Breakdown by source:")
for source, count in source_counts.items():
    print(f"  {source}: {count} documents")

type_counts = Counter(m.get('type', 'UNKNOWN') for m in metadatas)
print("\nBreakdown by type:")
for t, count in type_counts.items():
    print(f"  {t}: {count} documents")

    