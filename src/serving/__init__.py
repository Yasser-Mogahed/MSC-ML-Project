"""Model serving and inference module."""

from .api_client import send_to_api
from .inference import predict

__all__ = ["predict", "send_to_api"]
