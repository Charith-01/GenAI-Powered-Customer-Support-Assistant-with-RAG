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


# =========================================================
# Configuration
# =========================================================

validate_config()

client = genai.Client(
    api_key=GEMINI_API_KEY
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT / "knowledge_base"
)


# =========================================================
# Load knowledge base
# =========================================================

def load_knowledge_base():
    """
    Load all text documents from the knowledge base.
    """

    documents = []

    for file_path in sorted(
        KNOWLEDGE_BASE_PATH.glob("*.txt")
    ):

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


# =========================================================
# Chunk documents
# =========================================================

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


# =========================================================
# Build local TF-IDF index
# =========================================================

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


# =========================================================
# Business keyword routing
# =========================================================

POLICY_KEYWORDS = {

    "payment_policy.txt": [
        "charged twice",
        "duplicate charge",
        "charged",
        "charge",
        "payment",
        "paid",
        "billing",
        "transaction",
        "card",
    ],

    "refund_policy.txt": [
        "refund",
        "money back",
        "returned",
        "return",
        "reimbursement",
    ],

    "delivery_policy.txt": [
        "delivery",
        "delivered",
        "package",
        "parcel",
        "shipment",
        "shipping",
        "tracking",
        "arrive",
        "arrived",
        "late delivery",
    ],

    "cancellation_policy.txt": [
        "cancel order",
        "cancel",
        "cancellation",
        "stop my order",
        "stop order",
    ],

    "account_policy.txt": [
        "account",
        "login",
        "log in",
        "sign in",
        "password",
        "access my account",
        "cannot access",
        "hacked",
        "unauthorized access",
    ],
}


def detect_policy_from_keywords(query):
    """
    Identify the most likely policy using business keywords.

    Returns
    -------
    str or None
        Best matching policy filename.
    """

    query_lower = query.lower()

    policy_scores = {}

    for policy, keywords in POLICY_KEYWORDS.items():

        score = 0

        for keyword in keywords:

            if keyword in query_lower:
                score += 1

        policy_scores[policy] = score

    best_policy = max(
        policy_scores,
        key=policy_scores.get
    )

    if policy_scores[best_policy] == 0:
        return None

    return best_policy


# =========================================================
# Retrieve relevant context
# =========================================================

def retrieve_context(
    query,
    top_k=3
):
    """
    Retrieve relevant policy chunks.

    Uses:
    1. Business keyword routing
    2. TF-IDF similarity

    If keyword routing identifies a policy, chunks from
    that policy are prioritized. Otherwise retrieval falls
    back to global TF-IDF similarity.
    """

    query = str(query).strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    # -----------------------------------------------------
    # TF-IDF similarity
    # -----------------------------------------------------

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        chunk_vectors
    )[0]

    # -----------------------------------------------------
    # Keyword-based policy selection
    # -----------------------------------------------------

    preferred_policy = detect_policy_from_keywords(
        query
    )

    scored_results = []

    for index, similarity in enumerate(
        similarities
    ):

        source = chunks[index]["source"]

        # Strong routing bonus if the chunk belongs
        # to the business policy selected by keywords.
        routing_bonus = 0.0

        if (
            preferred_policy is not None
            and source == preferred_policy
        ):
            routing_bonus = 1.0

        selection_score = (
            float(similarity)
            + routing_bonus
        )

        scored_results.append({
            "source": source,
            "text": chunks[index]["text"],
            "similarity_score": float(similarity),
            "selection_score": selection_score
        })

    scored_results.sort(
        key=lambda item: item["selection_score"],
        reverse=True
    )

    results = scored_results[:top_k]

    return results


# =========================================================
# Format retrieved context
# =========================================================

def format_context(retrieved_chunks):
    """
    Format retrieved chunks for Gemini.
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


# =========================================================
# Generate RAG response
# =========================================================

def generate_rag_response(
    customer_message,
    top_k=3
):
    """
    Generate a Gemini response grounded in
    retrieved company policies.
    """

    customer_message = str(
        customer_message
    ).strip()

    if not customer_message:
        raise ValueError(
            "Customer message cannot be empty."
        )

    # -----------------------------------------------------
    # Retrieve evidence
    # -----------------------------------------------------

    retrieved_chunks = retrieve_context(
        customer_message,
        top_k=top_k
    )

    context = format_context(
        retrieved_chunks
    )

    # -----------------------------------------------------
    # RAG prompt
    # -----------------------------------------------------

    prompt = f"""
You are a professional customer support assistant.

Answer the customer's request using ONLY the relevant
company policy information provided below.

IMPORTANT RULES:

1. Use the supplied company policy as the source of
   company-specific information.
2. Do not invent company policies.
3. Do not invent refund periods, delivery times,
   payment rules, or cancellation rules.
4. Do not claim an action has already been completed.
5. If the supplied information is insufficient,
   say that additional verification is required.
6. Keep the response concise, helpful, and professional.
7. Never request passwords, full card numbers,
   or security codes.

COMPANY POLICY CONTEXT:

{context}

CUSTOMER MESSAGE:

{customer_message}

Generate a professional customer support response.
"""

    # -----------------------------------------------------
    # Gemini generation
    # -----------------------------------------------------

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

    if response is None:

        raise RuntimeError(
            "Gemini returned no response."
        )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return {
        "response": response.text.strip(),
        "retrieved_context": retrieved_chunks
    }