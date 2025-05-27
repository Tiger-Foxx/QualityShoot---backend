from fastapi import HTTPException
from typing import Any, Dict, Optional

class QualityShootException(Exception):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class ModelNotFoundError(QualityShootException):
    pass

class InvalidFileFormatError(QualityShootException):
    pass

class ProcessingError(QualityShootException):
    pass

class InsufficientVRAMError(QualityShootException):
    pass

def create_http_exception(exc: QualityShootException, status_code: int = 500) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details
        }
    )