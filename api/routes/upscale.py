from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from api.models.requests import UpscaleRequest, ProcessStatusRequest
from api.models.responses import (
    UpscaleResponse, ProcessStatusResponse, ProcessStatus,
    FileInfo, ModelInfoResponse
)
from services.image_processor import ImageProcessor
from services.video_processor import VideoProcessor
from utils.file_utils import validate_file_paths, is_video_file, is_image_file, get_file_size
from utils.image_utils import get_image_resolution, get_video_info, image_read
from core.config import settings
from core.exceptions import create_http_exception, QualityShootException
from core.process_manager import process_manager

router = APIRouter()
logger = logging.getLogger(__name__)

# Instances globales des processeurs
image_processor = ImageProcessor()
video_processor = VideoProcessor()

@router.post("/start", response_model=UpscaleResponse)
async def start_upscaling(
    request: UpscaleRequest,
    background_tasks: BackgroundTasks
):
    """Démarre un processus d'upscaling"""
    try:
        # Valider les fichiers
        valid_files, invalid_files = validate_file_paths(request.file_paths)
        
        if not valid_files:
            raise HTTPException(
                status_code=400,
                detail="Aucun fichier valide trouvé"
            )
        
        if invalid_files:
            logger.warning(f"Fichiers invalides ignorés: {invalid_files}")
        
        # Créer les infos des fichiers
        files_info = []
        for file_path in valid_files:
            try:
                file_info = FileInfo(
                    file_path=file_path,
                    file_name=Path(file_path).name,
                    file_size=get_file_size(file_path),
                    is_video=is_video_file(file_path)
                )
                
                if is_video_file(file_path):
                    video_info = get_video_info(file_path)
                    file_info.resolution = (video_info['width'], video_info['height'])
                    file_info.duration = video_info['duration']
                    file_info.frames_count = video_info['frame_count']
                elif is_image_file(file_path):
                    image = image_read(file_path)
                    height, width = get_image_resolution(image)
                    file_info.resolution = (width, height)
                
                files_info.append(file_info)
                
            except Exception as e:
                logger.warning(f"Erreur lecture infos {file_path}: {e}")
        
        # Séparer images et vidéos
        image_paths = [f.file_path for f in files_info if not f.is_video]
        video_paths = [f.file_path for f in files_info if f.is_video]
        
        # Créer un processus principal
        main_process_id = process_manager.create_process(
            process_type="upscale_batch",
            total_files=len(valid_files),
            files_info=files_info
        )
        
        # Démarrer le traitement en arrière-plan
        if image_paths:
            image_request = UpscaleRequest(**request.dict())
            image_request.file_paths = image_paths
            
            background_tasks.add_task(
                image_processor.process_images,
                image_request,
                main_process_id
            )
        
        if video_paths:
            video_request = UpscaleRequest(**request.dict())
            video_request.file_paths = video_paths
            
            background_tasks.add_task(
                video_processor.process_videos,
                video_request,
                main_process_id
            )
        
        logger.info(f"🚀 Processus d'upscaling démarré: {main_process_id}")
        
        return UpscaleResponse(
            process_id=main_process_id,
            status=ProcessStatus.PROCESSING,
            message="Traitement démarré",
            files_info=files_info,
            progress=0.0
        )
        
    except QualityShootException as e:
        raise create_http_exception(e, 400)
    except Exception as e:
        logger.error(f"Erreur démarrage upscaling: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{process_id}", response_model=ProcessStatusResponse)
async def get_process_status(process_id: str):
    """Récupère le statut d'un processus"""
    try:
        # Chercher le processus dans le gestionnaire global
        status = process_manager.get_process_status(process_id)
        
        if not status:
            logger.warning(f"❌ Processus non trouvé: {process_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Processus {process_id} non trouvé"
            )
        
        logger.debug(f"📊 Statut processus {process_id}: {status['status']} - {status['progress']:.1f}%")
        
        return ProcessStatusResponse(
            process_id=process_id,
            status=status['status'],
            progress=status['progress'],
            current_file=status.get('current_file'),
            current_step=status.get('current_step'),
            estimated_time_remaining=status.get('estimated_time_remaining'),
            error_message=status.get('error_message'),
            completed_files=status.get('completed_files', []),
            failed_files=status.get('failed_files', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur récupération statut: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cancel/{process_id}")
async def cancel_process(process_id: str):
    """Annule un processus"""
    try:
        # Utiliser le gestionnaire global
        cancelled = process_manager.cancel_process(process_id)
        
        if not cancelled:
            raise HTTPException(
                status_code=404,
                detail="Processus non trouvé"
            )
        
        logger.info(f"🛑 Processus annulé: {process_id}")
        return {"message": "Processus annulé", "process_id": process_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur annulation processus: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup/{process_id}")
async def cleanup_process(process_id: str):
    """Nettoie un processus terminé"""
    try:
        # Utiliser le gestionnaire global
        cleaned = process_manager.cleanup_process(process_id)
        
        logger.info(f"🗑️ Processus nettoyé: {process_id}")
        return {"message": "Processus nettoyé", "process_id": process_id}
        
    except Exception as e:
        logger.error(f"Erreur nettoyage processus: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=ModelInfoResponse)
async def get_available_models():
    """Récupère la liste des modèles disponibles"""
    try:
        # Vérifier quels modèles sont disponibles
        available_models = {}
        for model_name, model_info in settings.AI_MODELS.items():
            model_path = settings.AI_MODELS_DIR / model_info['file']
            if model_path.exists():
                available_models[model_name] = {
                    "scale_factor": model_info['scale'],
                    "vram_usage_gb": model_info['vram_usage'],
                    "file_path": str(model_path),
                    "available": True
                }
            else:
                available_models[model_name] = {
                    "scale_factor": model_info['scale'],
                    "vram_usage_gb": model_info['vram_usage'],
                    "file_path": str(model_path),
                    "available": False
                }
        
        return ModelInfoResponse(
            available_models=available_models,
            default_model="RealESR_Gx4",
            supported_formats={
                "images": settings.SUPPORTED_IMAGE_EXTENSIONS,
                "videos": settings.SUPPORTED_VIDEO_EXTENSIONS
            }
        )
        
    except Exception as e:
        logger.error(f"Erreur récupération modèles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/preview/{process_id}")
async def get_process_preview(process_id: str):
    """Récupère un aperçu du processus en cours"""
    try:
        # Récupérer le statut détaillé
        status = process_manager.get_process_status(process_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Processus non trouvé")
        
        # Ajouter des informations d'aperçu
        preview_info = {
            "process_id": process_id,
            "status": status['status'],
            "progress": status['progress'],
            "current_file": status.get('current_file'),
            "current_step": status.get('current_step'),
            "files_completed": len(status.get('completed_files', [])),
            "files_failed": len(status.get('failed_files', [])),
            "files_total": status.get('total_files', 0),
            "estimated_time": status.get('estimated_time_remaining')
        }
        
        return preview_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur aperçu processus: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Route pour nettoyer les anciens processus
@router.post("/cleanup-old")
async def cleanup_old_processes():
    """Nettoie les anciens processus terminés"""
    try:
        cleaned_count = process_manager.cleanup_old_processes(max_age_hours=24)
        return {
            "message": f"{cleaned_count} anciens processus nettoyés",
            "cleaned_count": cleaned_count
        }
    except Exception as e:
        logger.error(f"Erreur nettoyage anciens processus: {e}")
        raise HTTPException(status_code=500, detail=str(e))