from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ProcessStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

class FileInfo(BaseModel):
    file_path: str
    file_name: str
    file_size: int
    resolution: Optional[tuple[int, int]] = None
    duration: Optional[float] = None  # Pour les vidéos
    frames_count: Optional[int] = None  # Pour les vidéos
    is_video: bool

class UpscaleResponse(BaseModel):
    process_id: str = Field(..., description="ID unique du processus")
    status: ProcessStatus = Field(..., description="Statut du processus")
    message: str = Field(..., description="Message de statut")
    files_info: List[FileInfo] = Field(default=[], description="Informations sur les fichiers")
    estimated_time: Optional[float] = Field(None, description="Temps estimé en secondes")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progression en pourcentage")

class ProcessStatusResponse(BaseModel):
    process_id: str
    status: ProcessStatus
    progress: float = Field(ge=0.0, le=100.0)
    current_file: Optional[str] = None
    current_step: Optional[str] = None
    estimated_time_remaining: Optional[float] = None
    error_message: Optional[str] = None
    completed_files: List[str] = Field(default=[])
    failed_files: List[str] = Field(default=[])

class FileValidationResponse(BaseModel):
    valid_files: List[FileInfo]
    invalid_files: List[Dict[str, str]]  # {"file": "path", "reason": "error"}
    total_valid: int
    total_invalid: int

class ModelInfoResponse(BaseModel):
    available_models: Dict[str, Dict[str, Any]]
    default_model: str
    supported_formats: Dict[str, List[str]]

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    models_loaded: bool = False
    gpu_available: bool = False