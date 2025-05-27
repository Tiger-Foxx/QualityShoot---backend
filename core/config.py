import os
from pathlib import Path
from typing import Dict, List

class Settings:
    # Chemins de base
    BASE_DIR = Path(__file__).parent.parent
    ASSETS_DIR = BASE_DIR / "assets"
    AI_MODELS_DIR = ASSETS_DIR / "AI-onnx"
    FFMPEG_DIR = ASSETS_DIR / "ffmpeg"
    TEMP_DIR = BASE_DIR / "temp"
    
    # Configuration API
    API_HOST = "127.0.0.1"
    API_PORT = 8000
    
    # Extensions supportées
    SUPPORTED_IMAGE_EXTENSIONS = [
        '.heic', '.jpg', '.jpeg', '.JPG', '.JPEG', '.png',
        '.PNG', '.webp', '.WEBP', '.bmp', '.BMP', '.tif',
        '.tiff', '.TIF', '.TIFF'
    ]
    
    SUPPORTED_VIDEO_EXTENSIONS = [
        '.mp4', '.MP4', '.webm', '.WEBM', '.mkv', '.MKV',
        '.flv', '.FLV', '.gif', '.GIF', '.m4v', '.M4V',
        '.avi', '.AVI', '.mov', '.MOV', '.qt', '.3gp',
        '.mpg', '.mpeg', '.vob'
    ]
    
    # Modèles IA disponibles
    AI_MODELS = {
        'RealESR_Gx4': {'file': 'RealESR_Gx4_fp16.onnx', 'scale': 4, 'vram_usage': 2.5},
        'RealESR_Animex4': {'file': 'RealESR_Animex4_fp16.onnx', 'scale': 4, 'vram_usage': 2.5},
        'BSRGANx4': {'file': 'BSRGANx4_fp16.onnx', 'scale': 4, 'vram_usage': 0.75},
        'RealESRGANx4': {'file': 'RealESRGANx4_fp16.onnx', 'scale': 4, 'vram_usage': 0.75},
        'BSRGANx2': {'file': 'BSRGANx2_fp16.onnx', 'scale': 2, 'vram_usage': 0.8},
        'IRCNN_Mx1': {'file': 'IRCNN_Mx1_fp16.onnx', 'scale': 1, 'vram_usage': 4},
        'IRCNN_Lx1': {'file': 'IRCNN_Lx1_fp16.onnx', 'scale': 1, 'vram_usage': 4},
    }
    
    # Codecs vidéo
    VIDEO_CODECS = [
        "x264", "x265", "h264_nvenc", "hevc_nvenc", 
        "h264_amf", "hevc_amf", "h264_qsv", "hevc_qsv"
    ]

settings = Settings()