from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum

class AIModel(str, Enum):
    REALESR_GX4 = "RealESR_Gx4"
    REALESR_ANIMEX4 = "RealESR_Animex4"
    BSRGANX4 = "BSRGANx4"
    REALESRGANX4 = "RealESRGANx4"
    BSRGANX2 = "BSRGANx2"
    IRCNN_MX1 = "IRCNN_Mx1"
    IRCNN_LX1 = "IRCNN_Lx1"

class BlendingLevel(str, Enum):
    OFF = "OFF"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class VideoCodec(str, Enum):
    X264 = "x264"
    X265 = "x265"
    H264_NVENC = "h264_nvenc"
    HEVC_NVENC = "hevc_nvenc"
    H264_AMF = "h264_amf"
    HEVC_AMF = "hevc_amf"
    H264_QSV = "h264_qsv"
    HEVC_QSV = "hevc_qsv"

class GPU(str, Enum):
    AUTO = "Auto"
    GPU1 = "GPU 1"
    GPU2 = "GPU 2"
    GPU3 = "GPU 3"
    GPU4 = "GPU 4"

class UpscaleRequest(BaseModel):
    file_paths: List[str] = Field(..., description="Chemins des fichiers à traiter")
    ai_model: AIModel = Field(default=AIModel.REALESR_GX4, description="Modèle IA à utiliser")
    input_resize_factor: float = Field(default=1.0, ge=0.1, le=2.0, description="Facteur de redimensionnement d'entrée")
    output_resize_factor: float = Field(default=1.0, ge=0.1, le=2.0, description="Facteur de redimensionnement de sortie")
    gpu: GPU = Field(default=GPU.AUTO, description="GPU à utiliser")
    vram_limit: float = Field(default=4.0, ge=1.0, le=24.0, description="Limite VRAM en GB")
    blending: BlendingLevel = Field(default=BlendingLevel.OFF, description="Niveau de mélange")
    multithreading: int = Field(default=1, ge=1, le=8, description="Nombre de threads")
    output_path: Optional[str] = Field(None, description="Chemin de sortie personnalisé")
    keep_frames: bool = Field(default=False, description="Conserver les frames vidéo")
    image_extension: str = Field(default=".png", description="Extension des images de sortie")
    video_extension: str = Field(default=".mp4", description="Extension des vidéos de sortie")
    video_codec: VideoCodec = Field(default=VideoCodec.X264, description="Codec vidéo")

    @validator('input_resize_factor', 'output_resize_factor')
    def validate_resize_factors(cls, v):
        if v <= 0:
            raise ValueError("Les facteurs de redimensionnement doivent être positifs")
        return v

class ProcessStatusRequest(BaseModel):
    process_id: str = Field(..., description="ID du processus à surveiller")