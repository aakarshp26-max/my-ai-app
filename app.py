import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag import (
    get_collection, index_uploaded_files, answer_question,
    list_indexed_sources, delete_source, clear_knowledge_base
)

load_dotenv()

st.set_page_config(
    page_title="MUN Command Center",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.8rem; max-width: 1200px;}
.hero {padding: 1.25rem 1.4rem; border-radius: 16px; background: linear-gradient(110deg,#10243b,#234b69); color: white; margin-bottom: 1.2rem;}
.hero h1 {color:white; margin:0 0 .25rem 0;}
.hero p {color:#e4edf5; margin:0;}
.small-muted {color: #6b7280; font-size: .9rem;}
[data-testid="stChatMessage"] {border-radius: 12px;}
</style>
<div class="hero">
<h1>🌐 MUN Command Center</h1>
<p>Your document-grounded assistant for Model United Nations research and event operations.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Setup")
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API key",
        value=env_key,
        type="password",
        help="Alternatively set OPENAI_API_KEY in your .env file."
    )
    st.caption("Your API key is used by this local app to call OpenAI. Do not share it or commit it to GitHub.")
    st.divider()
    st.header("📚 Knowledge base")
    st.write("Upload MUN documents to create your searchable knowledge base.")
    uploads = st.file_uploader(
        "Add PDF, DOCX or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can upload multiple documents. Re-uploading the same file will not create duplicate chunks."
    )
    index_clicked = st.button("Process and index documents", type="primary", use_container_width=True)
    if index_clicked:
        if not api_key:
            st.error("Add your OpenAI API key first.")
        elif not uploads:
            st.warning("Choose at least one document.")
        else:
            try:
                client = OpenAI(api_key=api_key)
                with st.spinner("Reading documents, creating embeddings and indexing..."):
                    result = index_uploaded_files(client, uploads)
                st.success(
                    f"Indexed {result['chunks_added']} text chunks from "
                    f"{result['files_added']} new file(s)."
                )
                if result["skipped"]:
                    st.info("Already indexed: " + ", ".join(result["skipped"]))
                if result["warnings"]:
                    for warning in result["warnings"]:
                        st.warning(warning)
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.divider()
    st.subheader("Indexed documents")
    try:
        sources = list_indexed_sources()
        if not sources:
            st.caption("No documents indexed yet.")
        for source in sources:
            c1, c2 = st.columns([4, 1])
            c1.write("📄 " + source)
            if c2.button("✕", key=f"delete_{source}", help=f"Remove {source}"):
                delete_source(source)
                st.rerun()
        if sources and st.button("Clear entire knowledge base", use_container_width=True):
            clear_knowledge_base()
            st.rerun()
    except Exception as exc:
        st.caption(f"Knowledge base status unavailable: {exc}")

    st.divider()
    st.caption("Phase 1 · RAG document assistant")

# Main page
col1, col2, col3 = st.columns(3)
try:
    collection = get_collection()
    doc_count = len(list_indexed_sources())
    chunk_count = collection.count()
except Exception:
    doc_count, chunk_count = 0, 0

col1.metric("Documents indexed", doc_count)
col2.metric("Searchable passages", chunk_count)
col3.metric("AI mode", "RAG")

st.subheader("Ask your MUN documents")
st.markdown(
    '<p class="small-muted">Ask about rules of procedure, committee background guides, schedules, '
    'delegate instructions or organising documents. Answers are based on retrieved passages.</p>',
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources used"):
                for item in message["sources"]:
                    st.markdown(
                        f"**{item['source']}** · {item['location']}  \n"
                        f"> {item['excerpt']}"
                    )

question = st.chat_input("e.g. What is the procedure for submitting a draft resolution?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            answer = "Please add your OpenAI API key in the sidebar before asking a question."
            sources = []
            st.error(answer)
        elif chunk_count == 0:
            answer = "Your knowledge base is empty. Upload and index at least one MUN document using the sidebar, then ask your question."
            sources = []
            st.info(answer)
        else:
            try:
                client = OpenAI(api_key=api_key)
                with st.spinner("Searching your documents and preparing an answer..."):
                    answer, sources = answer_question(client, question)
                st.markdown(answer)
                if sources:
                    with st.expander("Sources used", expanded=True):
                        for item in sources:
                            st.markdown(
                                f"**{item['source']}** · {item['location']}  \n"
                                f"> {item['excerpt']}"
                            )
            except Exception as exc:
                answer = f"I couldn't complete that request. Check your API key, account API access and connection. Technical detail: {exc}"
                sources = []
                st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

if st.button("Clear chat history"):
    st.session_state.messages = []
    st.rerun()

with st.expander("💡 Try these test questions"):
    st.markdown("""
    - What are the rules for raising a point of personal privilege?
    - What is the deadline for submitting a draft resolution?
    - Which documents should a delegate prepare before committee?
    - What does the handbook say about attendance?
    - **Out-of-scope test:** What was the closing ceremony venue? (Ask this if your documents do not mention it. The assistant should say it cannot find the answer.)
    """)
