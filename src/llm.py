from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    validate_config,
)

from src.prompts import (
    CUSTOMER_SUPPORT_INSTRUCTIONS,
)


# ---------------------------------------------------------
# Validate configuration
# ---------------------------------------------------------

validate_config()


# ---------------------------------------------------------
# Create Gemini client
# ---------------------------------------------------------

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ---------------------------------------------------------
# Customer support response generation
# ---------------------------------------------------------

def generate_support_response(customer_message):
    """
    Generate a professional customer support response
    using the Gemini API.

    Parameters
    ----------
    customer_message : str
        Customer's support request.

    Returns
    -------
    str
        Generated customer support response.
    """

    # -----------------------------------------------------
    # Input validation
    # -----------------------------------------------------

    if customer_message is None:
        raise ValueError(
            "Customer message cannot be empty."
        )

    customer_message = str(
        customer_message
    ).strip()

    if not customer_message:
        raise ValueError(
            "Customer message cannot be empty."
        )

    # -----------------------------------------------------
    # Gemini API request
    # -----------------------------------------------------

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=customer_message,
            config=types.GenerateContentConfig(
                system_instruction=(
                    CUSTOMER_SUPPORT_INSTRUCTIONS
                ),
                max_output_tokens=200,
            ),
        )

    except Exception as error:
        raise RuntimeError(
            f"Gemini API request failed: {error}"
        ) from error

    # -----------------------------------------------------
    # Validate Gemini response
    # -----------------------------------------------------

    if response is None:
        raise RuntimeError(
            "Gemini returned no response."
        )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty text response."
        )

    return response.text.strip()