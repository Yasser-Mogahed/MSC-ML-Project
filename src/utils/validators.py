"""
Input validation module.

Pydantic models for validating customer input before prediction.
"""

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class CustomerInput(BaseModel):
    """
    Pydantic model for validating customer input.

    All 10 features with type hints and constraints.
    """

    # Demographics
    Age: int = Field(
        ..., ge=0, le=120, description="Customer age in years"
    )
    Gender: Literal["Male", "Female"] = Field(
        ..., description="Customer gender"
    )

    # Service & Usage
    Subscription_Type: Literal["Basic", "Standard", "Premium"] = Field(
        ..., alias="Subscription Type", description="Subscription tier"
    )
    Contract_Length: Literal["Monthly", "Quarterly", "Annual"] = Field(
        ..., alias="Contract Length", description="Contract duration"
    )
    Usage_Frequency: int = Field(
        ..., alias="Usage Frequency", ge=0, le=100, description="Usage frequency (0-100)"
    )
    Support_Calls: int = Field(
        ..., alias="Support Calls", ge=0, le=50, description="Number of support calls"
    )

    # Account & Billing
    Tenure: int = Field(
        ..., ge=0, le=100, description="Months as customer"
    )
    Total_Spend: float = Field(
        ..., alias="Total Spend", ge=0.0, description="Total amount spent"
    )
    Last_Interaction: int = Field(
        ..., alias="Last Interaction", ge=0, le=365, description="Days since last interaction"
    )
    Payment_Delay: int = Field(
        ..., alias="Payment Delay", ge=0, le=100, description="Days of payment delay"
    )

    class Config:
        """Pydantic configuration."""

        # Allow both snake_case (Python) and space-separated (CSV) field names
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Age": 30,
                "Tenure": 12,
                "Usage Frequency": 14,
                "Support Calls": 5,
                "Payment Delay": 18,
                "Subscription Type": "Standard",
                "Contract Length": "Annual",
                "Total Spend": 932.0,
                "Last Interaction": 17,
                "Gender": "Female",
            }
        }

    def to_prediction_input(self) -> dict:
        """
        Convert to format expected by inference.predict().

        Returns:
            Dictionary with original feature names (space-separated).
        """
        return {
            "Age": self.Age,
            "Tenure": self.Tenure,
            "Usage Frequency": self.Usage_Frequency,
            "Support Calls": self.Support_Calls,
            "Payment Delay": self.Payment_Delay,
            "Subscription Type": self.Subscription_Type,
            "Contract Length": self.Contract_Length,
            "Total Spend": self.Total_Spend,
            "Last Interaction": self.Last_Interaction,
            "Gender": self.Gender,
        }


def validate_input(data: dict) -> tuple[bool, Optional[str]]:
    """
    Validate customer input against CustomerInput schema.

    Args:
        data: Dictionary of customer features.

    Returns:
        Tuple of (is_valid, error_message).
        - (True, None) if valid
        - (False, error_string) if invalid

    Example:
        >>> data = {'Age': 30, 'Tenure': 12, ...}
        >>> is_valid, error = validate_input(data)
        >>> if not is_valid:
        ...     print(f"Validation error: {error}")
    """
    try:
        CustomerInput(**data)
        logger.debug("Input validation successful")
        return True, None

    except ValidationError as e:
        # Extract field-level error messages
        errors = []
        for error in e.errors():
            field = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"{field}: {msg}")

        error_message = " | ".join(errors)
        logger.warning(f"Validation failed: {error_message}")
        return False, error_message


if __name__ == "__main__":
    # Smoke test - valid input
    valid_input = {
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

    is_valid, error = validate_input(valid_input)
    print(f"✓ Valid input: {is_valid}")

    # Smoke test - invalid input
    invalid_input = {
        "Age": 999,  # Too high
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

    is_valid, error = validate_input(invalid_input)
    print(f"✓ Invalid input caught: {not is_valid}")
    if error:
        print(f"  Error: {error}")
