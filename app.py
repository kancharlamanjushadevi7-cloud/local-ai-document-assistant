
import hashlib
import io
import json
import re

import chromadb
import ollama
import streamlit as st
from pypdf import PdfReader


# =====================================
# 1. CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Local AI Document Assistant",
    page_icon="📄"
)

st.title("📄 Local AI Document Assistant")
st.write("Upload documents and ask questions about them.")

MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "nomic-embed-text"

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="uploaded_documents"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_document" not in st.session_state:
    st.session_state.active_document = None


# =====================================
# 2. EXTRACT TEXT
# =====================================

def extract_text(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))

        return "\n\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8-sig")

    return ""


# =====================================
# 3. SPLIT TEXT
# =====================================


def split_text(text, chunk_size=800, overlap=100):
    # Separate text into paragraphs first
    paragraphs = re.split(r"\n\s*\n", text.strip())

    chunks = []

    for paragraph in paragraphs:
        paragraph = " ".join(paragraph.split())

        if not paragraph:
            continue

        # Keep short paragraphs as individual chunks
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue

        # Split longer paragraphs into smaller chunks
        start = 0

        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))

            chunks.append(paragraph[start:end])

            if end == len(paragraph):
                break

            start = end - overlap

    return chunks


# =====================================
# 4. PROCESS DOCUMENT
# =====================================

def process_document(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    document_id = hashlib.sha256(file_bytes).hexdigest()

    text = extract_text(uploaded_file)
    chunks = split_text(text)

    if not chunks:
        raise ValueError("No readable text found in the document.")

    embeddings = []

    for chunk in chunks:
        result = ollama.embed(
            model=EMBED_MODEL,
            input=chunk
        )
        embeddings.append(result["embeddings"][0])

    collection.upsert(
        ids=[
            f"{document_id}_{i}"
            for i in range(len(chunks))
        ],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "document_id": document_id,
                "filename": uploaded_file.name,
                "chunk_number": i + 1
            }
            for i in range(len(chunks))
        ]
    )

    return document_id, len(chunks)


# =====================================
# 5. REWRITE FOLLOW-UP QUESTIONS
# =====================================

def rewrite_question(question, previous_messages):

    reference_words = {
        "it", "them", "they", "that",
        "those", "this", "their", "its"
    }

    words = set(re.findall(r"\b\w+\b", question.lower()))

    needs_rewrite = (
        bool(previous_messages)
        and len(question.split()) <= 15
        and bool(words.intersection(reference_words))
    )

    if not needs_rewrite:
        return question

    previous_questions = [
        message["content"]
        for message in previous_messages
        if message["role"] == "user"
    ]

    if not previous_questions:
        return question

    previous_question = previous_questions[-1]

    conversation = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in previous_messages
    )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite a follow-up question into a "
                    "standalone question. "
                    "Replace unclear references using "
                    "the previous conversation. "
                    "Preserve the original intent. "
                    "A WHY question must remain a WHY question. "
                    "Do not answer the question. "
                    "Return only the rewritten question. "
                    "Example: "
                    "Previous question: What is required "
                    "before a production network change? "
                    "Previous answer: Save a configuration backup. "
                    "Follow-up: Why is that required? "
                    "Rewrite: Why is a configuration backup "
                    "required before a production network change?"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Conversation:\n{conversation}\n\n"
                    f"Follow-up question: {question}"
                )
            }
        ],
        options={"temperature": 0}
    )

    rewritten = response["message"]["content"].strip()

    # Keep the original question type
    first_word = question.split()[0].lower()
    rewritten_first = (
        rewritten.split()[0].lower()
        if rewritten else ""
    )

    if rewritten_first != first_word:
        rewritten = ""

    # If rewriting failed, include previous context
    if not rewritten or rewritten.lower() == question.lower():
        return (
            f"{question}\n"
            f"Context: {previous_question}"
        )

    return rewritten


# =====================================
# 6. RETRIEVE RELEVANT CHUNKS
# =====================================


def retrieve_chunks(question, document_id):
    result = ollama.embed(
        model=EMBED_MODEL,
        input=question
    )

    embedding = result["embeddings"][0]

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3,
        where={"document_id": document_id}
    )

    return results["documents"][0]


# =====================================
# 7. GENERATE ANSWER WITH SOURCE QUOTE
# =====================================

def answer_question(question, chunks):
    context = "\n\n".join(chunks)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using only the "
                    "provided document. "
                    "Return a JSON object with exactly two "
                    "fields: answer and quote. "
                    "The quote must be an exact substring "
                    "copied from the document that directly "
                    "supports the answer. "
                    "If the document does not answer the "
                    "question, set answer to "
                    "'I couldn't find that information "
                    "in the document.' and quote to ''. "
                    "A requirement does not explain the "
                    "consequences of failing to follow it. "
                    "Do not invent procedures, fees, or facts. "
                    "Keep answers short."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Document:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        format="json",
        options={"temperature": 0}
    )

    try:
        data = json.loads(
            response["message"]["content"]
        )

        answer = data.get("answer", "")
        quote = data.get("quote", "")

        if not isinstance(answer, str):
            answer = ""

        if not isinstance(quote, str):
            quote = ""

        answer = answer.strip()
        quote = quote.strip()

        # Check that the quote exists in a retrieved chunk
        valid_quote = bool(quote) and any(
            quote in chunk
            for chunk in chunks
        )

        if not valid_quote:
            return (
                "I couldn't find that information "
                "in the document.",
                ""
            )

        return answer, quote

    except (json.JSONDecodeError, TypeError, ValueError):
        return (
            "I couldn't find that information "
            "in the document.",
            ""
        )


# =====================================
# 8. DOCUMENT UPLOAD
# =====================================

st.sidebar.header("Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file is not None:
    if st.sidebar.button("Process Document"):
        try:
            with st.spinner("Processing document..."):
                document_id, chunk_count = process_document(
                    uploaded_file
                )

            st.session_state.active_document = {
                "id": document_id,
                "name": uploaded_file.name
            }

            st.session_state.messages = []

            st.sidebar.success(
                f"Document ready! {chunk_count} chunks created."
            )

        except Exception as error:
            st.sidebar.error(str(error))


# =====================================
# 9. ACTIVE DOCUMENT
# =====================================

active_document = st.session_state.active_document

if active_document:
    st.success(
        f"Current document: {active_document['name']}"
    )
else:
    st.info("Upload and process a document to begin.")


# =====================================
# 10. CLEAR CHAT
# =====================================

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# =====================================
# 11. DISPLAY CHAT HISTORY
# =====================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# =====================================
# 12. CHAT
# =====================================

question = st.chat_input(
    "Ask about your document...",
    disabled=active_document is None
)

if question and active_document:
    previous_messages = st.session_state.messages[-4:]

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    try:
        with st.spinner("Searching document..."):
            search_question = rewrite_question(
                question,
                previous_messages
            )

            chunks = retrieve_chunks(
                search_question,
                active_document["id"]
            )

            answer, quote = answer_question(
                search_question,
                chunks
            )

        with st.chat_message("assistant"):
            st.write(answer)

            with st.expander("View sources"):
                st.write("**Original question:**")
                st.write(question)

                st.write("**Search question:**")
                st.write(search_question)

                st.write("**Supporting quote:**")
                st.write(quote or "No supporting quote found")

                st.write("**Retrieved document excerpts:**")

                for i, chunk in enumerate(chunks, start=1):
                    st.write(f"**Source {i}:**")
                    st.write(chunk)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as error:
        st.error(f"Something went wrong: {error}")
