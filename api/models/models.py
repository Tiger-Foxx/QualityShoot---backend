from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime

# Import des enums
from .enums import AIModel, GPU, ProcessStatus, BlendingMode, VideoCodec

class UpscaleRequest(BaseModel):
    """Requête d'upscaling"""
    file_paths: List[str] = Field(..., description="Chemins vers les fichiers à traiter")
    ai_model: AIModel = Field(default=AIModel.REALESR_GX4, description="Modèle d'IA à utiliser")
    input_resize_factor: float = Field(default=1.0, ge=0.1, le=2.0, description="Facteur de redimensionnement en entrée")
    output_resize_factor: float = Field(default=1.0, ge=0.1, le=2.0, description="Facteur de redimensionnement en sortie")
    gpu: GPU = Field(default=GPU.AUTO, description="GPU à utiliser")
    vram_limit: float = Field(default=4.0, ge=1.0, le=24.0, description="Limite VRAM en GB")
    blending: BlendingMode = Field(default=BlendingMode.OFF, description="Mode de blending")
    multithreading: int = Field(default=1, ge=1, le=8, description="Nombre de threads")
    keep_frames: bool = Field(default=False, description="Conserver les frames intermédiaires")
    image_extension: str = Field(default=".png", description="Extension pour les images de sortie")
    video_extension: str = Field(default=".mp4", description="Extension pour les vidéos de sortie")
    video_codec: VideoCodec = Field(default=VideoCodec.X264, description="Codec vidéo")

    @validator('file_paths')
    def validate_file_paths(cls, v):
        if not v:
            raise ValueError("Au moins un fichier doit être spécifié")
        return v

    @validator('image_extension', 'video_extension')
    def validate_extensions(cls, v):
        if not v.startswith('.'):
            return f'.{v}'
        return v

class FileInfo(BaseModel):
    """Informations sur un fichier"""
    file_path: str
    file_name: str
    file_size: int
    is_video: bool
    resolution: Optional[tuple] = None
    duration: Optional[float] = None
    frames_count: Optional[int] = None

class UpscaleResponse(BaseModel):
    """Réponse d'upscaling"""
    process_id: str
    status: ProcessStatus
    message: str
    files_info: Optional[List[FileInfo]] = None
    progress: float = 0.0

class ProcessStatusResponse(BaseModel):
    """Réponse de statut de processus"""
    process_id: str
    status: ProcessStatus
    progress: float
    current_file: Optional[str] = None
    current_step: Optional[str] = None
    estimated_time_remaining: Optional[float] = None
    error_message: Optional[str] = None
    completed_files: List[str] = []
    failed_files: List[str] = []

class ModelInfoResponse(BaseModel):
    """Informations sur les modèles disponibles"""
    available_models: Dict[str, Dict[str, Any]]
    default_model: str
    supported_formats: Dict[str, List[str]]

class HealthResponse(BaseModel):
    """Réponse de santé du système"""
    status: str
    version: str
    uptime: float
    models_loaded: int
    system_info: Dict[str, Any]