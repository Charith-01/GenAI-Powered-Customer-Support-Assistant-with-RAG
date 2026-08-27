from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.rag import (
    retrieve_context,
    generate_rag_response,
)


# =========================================================
# Streamlit page configuration
# =========================================================

st.set_page_config(
    page_title="GenAI Customer Support Assistant",
    page_icon="💬",
    layout="wide",
)


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tfidf_logistic_regression_intent.joblib"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_support_tickets.csv"
)


# =========================================================
# Load baseline model
# =========================================================

@st.cache_resource
def load_classifier():
    """
    Load the trained TF-IDF + Logistic Regression model.
    """

    if not MODEL_PATH.exists():
        return None

    return joblib.load(
        MODEL_PATH
    )


# =========================================================
# Intent -> category mapping
# =========================================================

@st.cache_data
def load_intent_category_mapping():
    """
    Create mapping from intent labels to dataset categories.
    """

    if not DATA_PATH.exists():
        return {}

    df = pd.read_csv(
        DATA_PATH
    )

    mapping = (
        df[
            [
                "intent",
                "category"
            ]
        ]
        .drop_duplicates()
        .set_index(
            "intent"
        )["category"]
        .to_dict()
    )

    return mapping


# =========================================================
# Policy -> category mapping
# =========================================================

def policy_to_category(policy_source):
    """
    Convert retrieved policy filename into
    a business category.
    """

    mapping = {
        "payment_policy.txt": "PAYMENT",
        "refund_policy.txt": "REFUND",
        "delivery_policy.txt": "DELIVERY",
        "cancellation_policy.txt": "CANCELLATION",
        "account_policy.txt": "ACCOUNT",
    }

    return mapping.get(
        policy_source,
        "UNKNOWN"
    )


# =========================================================
# Load resources
# =========================================================

classifier = load_classifier()

intent_category_mapping = (
    load_intent_category_mapping()
)


# =========================================================
# Header
# =========================================================

st.title(
    "💬 GenAI Customer Support Assistant"
)

st.write(
    """
    This application analyzes customer support messages
    using a traditional machine-learning baseline,
    retrieves relevant company policy information,
    and uses Gemini to generate a grounded customer
    support response.
    """
)

st.divider()


# =========================================================
# Customer input
# =========================================================

st.subheader(
    "Customer Message"
)

customer_message = st.text_area(
    "Enter a customer support request:",
    height=140,
    placeholder=(
        "Example: I was charged twice "
        "for the same order."
    ),
)


# =========================================================
# Analyze ticket
# =========================================================

if st.button(
    "Analyze Ticket",
    type="primary",
    use_container_width=True,
):

    # Clear previous Gemini response when
    # a new ticket is analyzed.
    st.session_state.pop(
        "rag_response",
        None
    )

    if not customer_message.strip():

        st.warning(
            "Please enter a customer message."
        )

    else:

        st.session_state[
            "customer_message"
        ] = customer_message

        # -------------------------------------------------
        # Traditional ML baseline
        # -------------------------------------------------

        baseline_intent = "Unavailable"
        baseline_category = "Unavailable"

        if classifier is not None:

            try:

                baseline_intent = (
                    classifier.predict(
                        [customer_message]
                    )[0]
                )

                baseline_category = (
                    intent_category_mapping.get(
                        baseline_intent,
                        "Unknown"
                    )
                )

            except Exception as error:

                st.warning(
                    f"Baseline classifier failed: {error}"
                )

        # -------------------------------------------------
        # RAG retrieval
        # -------------------------------------------------

        try:

            retrieved_chunks = (
                retrieve_context(
                    customer_message,
                    top_k=3,
                )
            )

            top_policy = (
                retrieved_chunks[0]["source"]
            )

            detected_category = (
                policy_to_category(
                    top_policy
                )
            )

            # Save results in session state

            st.session_state[
                "retrieved_chunks"
            ] = retrieved_chunks

            st.session_state[
                "baseline_intent"
            ] = baseline_intent

            st.session_state[
                "baseline_category"
            ] = baseline_category

            st.session_state[
                "detected_category"
            ] = detected_category

        except Exception as error:

            st.error(
                f"Policy retrieval failed: {error}"
            )


# =========================================================
# Display ticket analysis
# =========================================================

if "retrieved_chunks" in st.session_state:

    st.divider()

    st.subheader(
        "Ticket Analysis"
    )

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # RAG/business detected category
    # -----------------------------------------------------

    with col1:

        st.metric(
            "Detected Category",
            st.session_state.get(
                "detected_category",
                "Unavailable",
            ),
        )

    # -----------------------------------------------------
    # Traditional ML baseline
    # -----------------------------------------------------

    with col2:

        st.metric(
            "Baseline ML Intent",
            st.session_state.get(
                "baseline_intent",
                "Unavailable",
            ),
        )


    # -----------------------------------------------------
    # Optional baseline details
    # -----------------------------------------------------

    with st.expander(
        "View baseline model details"
    ):

        st.write(
            "**Baseline Category:**",
            st.session_state.get(
                "baseline_category",
                "Unavailable",
            )
        )

        st.write(
            "**Baseline Intent:**",
            st.session_state.get(
                "baseline_intent",
                "Unavailable",
            )
        )

        st.caption(
            "The baseline uses TF-IDF + "
            "Logistic Regression and may occasionally "
            "misclassify semantically similar requests."
        )


    # =====================================================
    # Retrieved policy
    # =====================================================

    st.subheader(
        "Retrieved Company Policy"
    )

    retrieved_chunks = (
        st.session_state[
            "retrieved_chunks"
        ]
    )

    top_result = (
        retrieved_chunks[0]
    )

    st.success(
        f"Top Policy: "
        f"{top_result['source']}"
    )

    st.write(
        top_result["text"]
    )

    st.caption(
        "TF-IDF similarity score: "
        f"{top_result['similarity_score']:.4f}"
    )


    # =====================================================
    # Additional evidence
    # =====================================================

    with st.expander(
        "View additional retrieved context"
    ):

        for number, item in enumerate(
            retrieved_chunks,
            start=1,
        ):

            st.markdown(
                f"### Result {number}"
            )

            st.write(
                f"**Source:** "
                f"{item['source']}"
            )

            st.write(
                item["text"]
            )

            st.caption(
                "Similarity score: "
                f"{item['similarity_score']:.4f}"
            )

            st.divider()


    # =====================================================
    # Gemini response generation
    # =====================================================

    st.subheader(
        "AI Suggested Response"
    )

    st.info(
        "The response is generated using Gemini "
        "and the retrieved company policy."
    )

    if st.button(
        "Generate Grounded Response",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Generating grounded support response..."
            ):

                result = (
                    generate_rag_response(
                        st.session_state[
                            "customer_message"
                        ],
                        top_k=3,
                    )
                )

            st.session_state[
                "rag_response"
            ] = result["response"]

        except Exception as error:

            st.error(
                f"Gemini generation failed: {error}"
            )


# =========================================================
# Display Gemini response
# =========================================================

if "rag_response" in st.session_state:

    st.success(
        st.session_state[
            "rag_response"
        ]
    )


# =========================================================
# System architecture
# =========================================================

with st.expander(
    "How this prototype works"
):

    st.markdown(
        """
        **1. Customer Message**

        The user enters a customer support request.

        **2. Baseline Machine Learning**

        A TF-IDF + Logistic Regression model predicts
        the customer intent.

        **3. Policy Retrieval**

        The RAG retriever combines business keyword
        routing with TF-IDF similarity to locate the
        most relevant company policy.

        **4. Gemini Generation**

        The retrieved policy is supplied to Gemini as
        context so the generated response is grounded
        in available company information.
        """
    )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "Prototype developed for a Data Science Internship "
    "Generative AI Application task."
)