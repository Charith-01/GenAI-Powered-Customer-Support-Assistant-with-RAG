from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    validate_config,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

validate_config()

client = genai.Client(
    api_key=GEMINI_API_KEY
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT / "knowledge_base"
)


# ---------------------------------------------------------
# Load knowledge-base documents
# ---------------------------------------------------------

def load_knowledge_base():
    """
    Load all .txt policy documents from the
    knowledge_base directory.
    """

    documents = []

    for file_path in KNOWLEDGE_BASE_PATH.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if text:
            documents.append({
                "source": file_path.name,
                "text": text
            })

    if not documents:
        raise RuntimeError(
            "No knowledge-base documents were found."
        )

    return documents


# ---------------------------------------------------------
# Split documents into chunks
# ---------------------------------------------------------

def chunk_documents(documents):
    """
    Split policy documents into paragraph-sized chunks.
    """

    chunks = []

    for document in documents:

        paragraphs = [
            paragraph.strip()
            for paragraph in document["text"].split("\n\n")
            if paragraph.strip()
        ]

        for index, paragraph in enumerate(paragraphs):

            chunks.append({
                "source": document["source"],
                "chunk_id": index,
                "text": paragraph
            })

    return chunks


# ---------------------------------------------------------
# Build TF-IDF retriever
# ---------------------------------------------------------

documents = load_knowledge_base()

chunks = chunk_documents(
    documents
)

chunk_texts = [
    chunk["text"]
    for chunk in chunks
]


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    stop_words="english"
)


chunk_vectors = vectorizer.fit_transform(
    chunk_texts
)


# ---------------------------------------------------------
# Retrieve relevant knowledge
# ---------------------------------------------------------

def retrieve_context(
    query,
    top_k=3
):
    """
    Retrieve the top-k policy chunks most relevant
    to a customer query.
    """

    query = str(query).strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        chunk_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append({
            "source": chunks[index]["source"],
            "text": chunks[index]["text"],
            "score": float(
                similarities[index]
            )
        })

    return results


# ---------------------------------------------------------
# Build RAG context
# ---------------------------------------------------------

def format_context(retrieved_chunks):
    """
    Convert retrieved chunks into context
    that can be sent to Gemini.
    """

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"Source: {item['source']}\n"
            f"{item['text']}"
        )

    return "\n\n".join(
        context_parts
    )


# ---------------------------------------------------------
# Generate grounded Gemini response
# ---------------------------------------------------------

def generate_rag_response(
    customer_message,
    top_k=3
):
    """
    Generate a customer-support response grounded
    in retrieved company policy.
    """

    customer_message = str(
        customer_message
    ).strip()

    if not customer_message:
        raise ValueError(
            "Customer message cannot be empty."
        )

    retrieved_chunks = retrieve_context(
        customer_message,
        top_k=top_k
    )

    context = format_context(
        retrieved_chunks
    )

    prompt = f"""
You are a professional customer support assistant.

Answer the customer's request using the company policy
information provided below.

IMPORTANT RULES:

1. Use the provided context as the source of
   company-specific facts.
2. Do not invent company policies.
3. Do not invent refund periods, delivery times,
   payment rules, or cancellation rules.
4. If the provided context does not contain enough
   information, say that the issue requires verification.
5. Keep the response concise and professional.
6. Do not claim that an action has already been completed.
7. Do not expose internal technical details.

COMPANY POLICY CONTEXT:

{context}

CUSTOMER MESSAGE:

{customer_message}

Generate a helpful customer-support response.
"""

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=250,
            ),
        )

    except Exception as error:

        raise RuntimeError(
            f"RAG generation failed: {error}"
        ) from error

    if response is None or not response.text:

        raise RuntimeError(
            "Gemini returned an empty RAG response."
        )

    return {
        "response": response.text.strip(),
        "retrieved_context": retrieved_chunks
    }