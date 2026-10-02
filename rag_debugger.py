import pandas as pd
import json
import os
import shutil
from datasets import load_dataset
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
VECTOR_DB_PATH = "./chroma_db_v_rag"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_DATASET_ID = "Bouquets/Cybersecurity-LLM-CVE"
MITRE_JSON_PATH = 'dataset/enterprise-attack-19.2.json' # Ensure this file is in the root directory

# --- 1. DATA ACQUISITION AND UNIFICATION ---

def format_cve_df_to_corpus(cve_df: pd.DataFrame) -> list[Document]:
    """Converts the CVE DataFrame into a list of standardized Document objects."""
    documents = []
    
    # We must iterate over the rows and parse the raw output string from column 'outputs'
    for index, row in cve_df.iterrows():
        raw_output = row['outputs']
        
        # Simple string parsing based on the observed format (e.g., CVE:CVE-2020-13909\nDescription:...)
        try:
            # Safely extract CVE ID
            cve_id_match = raw_output.split('CVE:')
            if len(cve_id_match) > 1:
                cve_id = cve_id_match[1].split('\n')[0].strip()
            else:
                continue

            # Safely extract Description
            description_start = raw_output.find('Description:') + len('Description:')
            description_end = raw_output.find('published:')
            if description_start != -1 and description_end != -1:
                description = raw_output[description_start:description_end].strip()
            else:
                continue
            
            # Create a rich text body for embedding
            text_content = (
                f"VULNERABILITY ID: {cve_id}. "
                f"Description: {description}"
            )
            
            documents.append(Document(
                page_content=text_content,
                metadata={
                    'id': cve_id,
                    'source': 'CVE-HuggingFace',
                    'type': 'Vulnerability-Description'
                }
            ))
        except IndexError:
            continue
            
    return documents

def process_mitre_data(file_path: str) -> list[Document]:
    """Loads MITRE ATT&CK data and processes relationships using pure Python logic."""
    
    # Pure Python logic: This completely avoids ALL stix2 imports and methods.
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    objects = data.get('objects', [])
    mitre_documents = []
    
    # 1. Create a lookup table for all objects (by ID) and identify relationships
    object_lookup = {obj['id']: obj for obj in objects}
    relationships = [obj for obj in objects if obj.get('type') == 'relationship']
    
    # 2. Build a Mitigation lookup dictionary {technique_id: [mitigation_docs]}
    mitigation_map = {}
    for rel in relationships:
        if rel.get('relationship_type') == 'mitigates':
            source_id = rel['source_ref'] # Mitigation ID
            target_id = rel['target_ref'] # Technique ID (mitigated)
            
            mitigation_obj = object_lookup.get(source_id)
            technique_obj = object_lookup.get(target_id)
            
            if mitigation_obj and technique_obj and technique_obj.get('type') == 'attack-pattern':
                # Safely extract mitigation ID
                mitigation_id = next(
                    (ref['external_id'] for ref in mitigation_obj.get('external_references', []) if ref['source_name'] == 'mitre-attack'),
                    'N/A'
                )
                
                mitigation_detail = (
                    f" - Mitigation ({mitigation_id}): {mitigation_obj.get('description', 'No description provided.')}"
                )
                
                if target_id not in mitigation_map:
                    mitigation_map[target_id] = []
                mitigation_map[target_id].append(mitigation_detail)

    # 3. Iterate over all ATTACK PATTERNS (Techniques)
    techniques = [obj for obj in objects if obj.get('type') == 'attack-pattern']

    for tech in techniques:
        # Safely get the MITRE T-ID
        tech_ref = next((ref for ref in tech.get('external_references', []) if ref['source_name'] == 'mitre-attack'), None)
        if not tech_ref: continue
            
        tech_id = tech_ref['external_id']
        
        # Safely get tactics
        tactics_list = [
            t['phase_name'].replace('-', ' ').title() 
            for t in tech.get('kill_chain_phases', []) 
            if t.get('kill_chain_name') == 'mitre-attack'
        ]
        
        core_text = (
            f"MITRE ATT&CK Technique ID: {tech_id}. "
            f"Name: {tech.get('name')}. "
            f"Tactic(s): {', '.join(tactics_list)}. "
            f"Description: {tech.get('description')}"
        )
        
        # 4. Append Mitigations from the lookup map
        mitigation_details = mitigation_map.get(tech['id'])
        if mitigation_details:
            core_text += "\n\n**MITIGATION STRATEGIES:**\n" + "\n".join(mitigation_details)

        mitre_documents.append(Document(
            page_content=core_text,
            metadata={
                'id': tech_id,
                'name': tech.get('name'),
                'source': 'MITRE-ATTACK',
                'type': 'Technique-Mitigation',
            }
        ))
        
    return mitre_documents

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- V-RAG Phase 1: Knowledge Base Indexing Started ---")
    
    # 1. ACQUIRE AND FORMAT DATA
    print(f"1. Acquiring CVE data from {HF_DATASET_ID}...")
    try:
        # NOTE: This line requires the 'datasets' library and may take time to download.
        cve_df = load_dataset(HF_DATASET_ID, split='train').to_pandas()
        cve_df = cve_df.head(500)
    except Exception as e:
        print(f"ERROR: Could not load HuggingFace dataset. Check internet connection and dataset name. {e}")
        exit() 
        
    cve_corpus = format_cve_df_to_corpus(cve_df)
    print(f"   -> Formatted {len(cve_corpus)} CVE documents.")

    print(f"2. Processing MITRE ATT&CK data from {MITRE_JSON_PATH}...")
    if not os.path.exists(MITRE_JSON_PATH):
        print(f"FATAL ERROR: MITRE JSON file not found at {MITRE_JSON_PATH}. Please download it.")
        exit()
        
    # The pure Python processing is used here
    mitre_corpus = process_mitre_data(MITRE_JSON_PATH) 
    print(f"   -> Formatted {len(mitre_corpus)} MITRE documents.")
    
    final_document_corpus = cve_corpus + mitre_corpus
    print(f"3. Unified Corpus created with {len(final_document_corpus)} total documents.")

    # 4. CHUNKING
    print("4. Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        length_function=len
    )
    chunked_documents = text_splitter.split_documents(final_document_corpus)
    print(f"   -> Total Chunks created: {len(chunked_documents)}")

    # 5. EMBEDDING AND INDEXING
    print(f"5. Initializing Embedding Model: {EMBEDDING_MODEL_NAME} and Vector Store...")
    
    # Clean up previous index for a fresh start
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
        print("   -> Deleted previous index directory.")
        
    # NOTE: SentenceTransformerEmbeddings requires the model to be downloaded once
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # This step performs the vectorization and saving
    vectorstore = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    vectorstore.persist()
    print(f"   -> Successfully indexed and persisted {len(chunked_documents)} chunks to {VECTOR_DB_PATH}")
    
    print("--- V-RAG Phase 1: Indexing Complete. Ready for Chatbot. ---")
