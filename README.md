# CTI-RAG: Retrieval-Augmented Generation for Cyber Threat Intelligence

CTI-RAG is an open-source Retrieval-Augmented Generation (RAG) system for Cyber Threat Intelligence (CTI) analysis.  
It enables ingestion, semantic indexing, and analysis of CTI datasets with advanced semantic search and analyst chatbot capabilities.

---

## Features

- **Automatic ingestion & processing** of CVE and MITRE ATT&CK datasets.
- **Semantic chunking & embedding** using SentenceTransformer.
- **Vector database storage** (ChromaDB) for fast, context-rich retrieval.
- **RAG-powered chatbot** interface with Gemini LLM for threat analysis.
- **Cited intelligence**: Answers always reference specific threat IDs (CVE, TTP).
- **Mitigation extraction**: Get actionable, prioritized mitigation steps.
- **Extensible data support**: Easily add new CTI sources to the dataset folder.

---

## Project Structure

```
CTI_RAG/
├── chroma_db_v_rag/           # Vector database files (auto-generated, do not edit)
├── dataset/                   # Place your CTI datasets here (CSV, JSON, etc.)
│   ├── cve_data_raw.csv
│   ├── enterprise-attack-19.2.json
│   └── ... (other files you add)
├── analyst_chatbot.py         # Analyst chatbot (run second)
├── rag_debugger.py            # RAG pipeline setup & debugging (run first)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

- **chroma_db_v_rag/**: Stores vector database (auto-generated).
- **dataset/**: Add your CTI datasets here. This folder is ignored by git.
- **analyst_chatbot.py**: Streamlit-powered chatbot for CTI queries.
- **rag_debugger.py**: Script to ingest data and build the vector database.

---

## Tech Stack

- **Python 3.10+**
- **ChromaDB** (Vector Database)
- **SentenceTransformer** (`all-MiniLM-L6-v2` for embedding)
- **LangChain** (retriever, chaining, document processing)
- **Gemini LLM** (Google Generative AI, via API key)
- **Streamlit** (chatbot UI)
- **pandas, json** for data wrangling

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/VinishaSolanki/CTI-RAG.git
cd CTI-RAG
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your dataset

Place your CTI dataset(s) in the `dataset/` folder.  
Example files:
- `dataset/cve_data_raw.csv`
- `dataset/enterprise-attack-19.2.json`

> Note: The `dataset/` folder is **ignored by git**. Add your own files after cloning.

---
## Running the Project

### 1. Build the Vector Database

```bash
python rag_debugger.py
```
- Ingests and processes your datasets.
- Prompts for your API key (Google Gemini or other supported LLM).
- Builds and persists the vector database.

### 2. Launch the Chatbot

```bash
python analyst_chatbot.py
```
- Streamlit app for CTI analysis.
- Enter your API key when prompted.
- Ask about CVEs, MITRE ATT&CK techniques, or mitigation strategies.

---

## Example Usage

- **Query:** "What are the mitigations for CVE-2024-XXXX?"
- **Query:** "Describe technique T1055.011 and its mitigations."
- **Query:** "What threat tactics are associated with persistence?"

All answers are sourced directly from your indexed intelligence corpus, with referenced IDs and actionable mitigation steps.

---

## Troubleshooting

- **Missing Dataset:** Ensure your files are present in the `dataset/` folder.
- **API Key:** Use a valid API key for the Gemini or other supported LLM services.
- **Vector DB Missing:** Run `rag_debugger.py` before launching the chatbot.

---

## License

[Specify your license, e.g., MIT]

---

Created by Vinisha Solanki
