"""
API client module.

Provides HTTP client for sending predictions to FastAPI backend.
"""

import logging
from typing import Optional

import requests

from src.config import settings

logger = logging.getLogger(__name__)


def send_to_api(input_data: dict) -> dict:
    """
    Send prediction request to FastAPI backend.

    Never raises exceptions. Always returns a dict (either with prediction or error).

    Args:
        input_data: Dictionary with customer features.

    Returns:
        Response dict. On success: {'prediction': ..., 'churn_probability': ..., 'label': ...}
        On error: {'error': error_message}

    Example:
        >>> data = {'Age': 30, 'Tenure': 12, ...}
        >>> response = send_to_api(data)
        >>> if 'error' in response:
        ...     print(f"API Error: {response['error']}")
        ... else:
        ...     print(f"Churn: {response['label']}")
    """
    url = f"{settings.API_URL}{settings.PREDICT_ENDPOINT}"

    try:
        logger.debug(f"Sending POST request to {url}")

        response = requests.post(
            url,
            json=input_data,
            timeout=10,
            headers={"Content-Type": "application/json"},
        )

        # Check for HTTP errors
        response.raise_for_status()

        # Parse JSON response
        result = response.json()
        logger.debug(f"API response: {result}")

        return result

    except requests.exceptions.ConnectionError as e:
        error_msg = f"API unreachable at {settings.API_URL}: {str(e)}"
        logger.warning(error_msg)
        return {"error": error_msg}

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)

        error_msg = f"API error {status_code}: {detail}"
        logger.warning(error_msg)
        return {"error": error_msg}

    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


if __name__ == "__main__":
    # Smoke test
    test_input = {
        "Age": 30,
        "Tenure": 12,
        "Usage Frequency": 14,
        "Support Calls": 5,
        "Payment Delay": 18,
        "Subscription Type": "Standard",
        "Contract Length": "Annual",
        "Total Spend": 932,
        "Last Interaction": 17,
        "Gender": "Female",
    }

    result = send_to_api(test_input)
    if "error" in result:
        print(f"⚠ {result['error']}")
    else:
        print(f"✓ API call successful: {result['label']}")
