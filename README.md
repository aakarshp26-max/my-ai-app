# MUN Command Center — Phase 1 (RAG)

A Streamlit application that lets you upload MUN PDF, DOCX and TXT documents, index them with OpenAI embeddings, and ask questions grounded in the uploaded material.

## What this version does

- Upload multiple PDF, DOCX and TXT documents.
- Extract text and split it into overlapping chunks.
- Generate embeddings with `text-embedding-3-small`.
- Store vectors and source metadata locally in ChromaDB.
- Retrieve relevant passages for a question.
- Generate an answer using an OpenAI chat model.
- Display source document names, PDF page locations and retrieved excerpts.
- Skip exact duplicate uploads and remove individual documents from the knowledge base.
- Keep the vector database on your machine in `chroma_db/`.

## Requirements

- Python 3.10 or newer.
- An OpenAI API key with API billing/access enabled. A ChatGPT subscription does not automatically include API credits.
- Internet connection for OpenAI API calls.

## Setup — Windows

1. Extract the ZIP file.
2. Open the extracted `mun_command_center` folder.
3. In File Explorer, open the folder address bar, type `cmd`, and press Enter.
4. Create a virtual environment:

   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Copy `.env.example` to `.env`.
7. Open `.env` in Notepad and replace `your_openai_api_key_here` with your API key. Do not share this file or upload it to GitHub.
8. Start the app:

   ```bash
   streamlit run app.py
   ```

Streamlit will open the app in your browser, usually at `http://localhost:8501`.

## Setup — macOS

```bash
cd mun_command_center
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Add your API key to `.env`, then run:

```bash
streamlit run app.py
```

## First test

1. Upload one or more MUN PDFs/DOCX/TXT files in the sidebar.
2. Click **Process and index documents**.
3. Wait for the success message.
4. Ask a question whose answer you know is present in the uploaded document.
5. Open **Sources used** to inspect the retrieved passages.
6. Ask something that is definitely absent from the uploaded documents. The assistant should say it could not find the information.

## Important limitations in v1

- Scanned/image-only PDFs may not contain extractable text. OCR is not included yet.
- PDF page numbers are recorded; DOCX and TXT are currently identified by document name/type rather than exact page.
- Tables in DOCX are extracted as text; complex layouts may need cleanup.
- Retrieval uses a simple top-5 semantic search. It is a first version, not a production-grade evaluation pipeline.
- The app runs locally. It is not yet deployed for multiple users.
- The app does not execute agentic actions yet. Registration, finance and operations agents are Phase 2.

## Security

- Never hard-code your API key in Python.
- Keep `.env` private and out of version control.
- Do not upload confidential delegate data unless you have permission and an appropriate data-handling basis.
- This version stores extracted text and embeddings locally in ChromaDB. Text sent to OpenAI for embedding and answer generation is processed by the API.
- Uploaded documents can contain malicious or misleading instructions. The RAG prompt treats retrieved text as untrusted reference material.

## Architecture

```text
Uploaded PDF / DOCX / TXT
          |
          v
   Text extraction
          |
          v
 Chunking + metadata
          |
          v
 OpenAI embeddings
          |
          v
   ChromaDB (local)
          |
User question -> query embedding
          |
          v
  Top relevant chunks
          |
          v
 OpenAI chat completion
          |
          v
 Answer + source references
```

## Phase 2 roadmap

- Registration agent: spreadsheet validation, duplicate detection, committee capacity checks.
- Finance agent: budget calculations, break-even analysis and scenario planning.
- Operations agent: task lists, schedule conflict checks and report generation.
- Human approval before any consequential external action.
