from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import List
import os
from pathlib import Path
import tempfile
import shutil
import logging

from api.models.responses import FileValidationResponse, FileInfo
from utils.file_utils import (
    validate_file_paths, is_video_file, is_image_file, 
    get_file_size, ensure_directory_exists
)
from utils.image_utils import get_image_resolution, get_video_info, image_read
from core.config import settings
from core.exceptions import create_http_exception, InvalidFileFormatError

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/validate", response_model=FileValidationResponse)
async def validate_files(file_paths: List[str]):
    """Valide une liste de chemins de fichiers"""
    try:
        valid_files, invalid_files = validate_file_paths(file_paths)
        
        # Créer les infos détaillées pour les fichiers valides
        valid_files_info = []
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
                
                valid_files_info.append(file_info)
                
            except Exception as e:
                logger.warning(f"Erreur lecture infos {file_path}: {e}")
                # Ajouter aux fichiers invalides si erreur de lecture
                invalid_files.append({
                    "file": file_path,
                    "reason": f"Erreur lecture: {str(e)}"
                })
        
        return FileValidationResponse(
            valid_files=valid_files_info,
            invalid_files=invalid_files,
            total_valid=len(valid_files_info),
            total_invalid=len(invalid_files)
        )
        
    except Exception as e:
        logger.error(f"Erreur validation fichiers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload des fichiers vers le serveur"""
    try:
        uploaded_files = []
        temp_dir = settings.TEMP_DIR / "uploads"
        ensure_directory_exists(str(temp_dir))
        
        for file in files:
            # Vérifier l'extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in (settings.SUPPORTED_IMAGE_EXTENSIONS + settings.SUPPORTED_VIDEO_EXTENSIONS):
                raise InvalidFileFormatError(f"Format {file_ext} non supporté")
            
            # Sauvegarder le fichier
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append({
                "filename": file.filename,
                "path": str(file_path),
                "size": file_path.stat().st_size
            })
        
        return {
            "message": f"{len(uploaded_files)} fichiers uploadés",
            "files": uploaded_files
        }
        
    except InvalidFileFormatError as e:
        raise create_http_exception(e, 400)
    except Exception as e:
        logger.error(f"Erreur upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{filename}")
async def download_file(filename: str):
    """Télécharge un fichier traité"""
    try:
        # Chercher le fichier dans différents répertoires
        search_dirs = [
            settings.TEMP_DIR,
            Path.home() / "Downloads",
            Path.cwd()
        ]
        
        file_path = None
        for search_dir in search_dirs:
            potential_path = search_dir / filename
            if potential_path.exists():
                file_path = potential_path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur téléchargement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info/{filename}")
async def get_file_info(filename: str):
    """Récupère les informations détaillées d'un fichier"""
    try:
        # Chercher le fichier
        search_dirs = [settings.TEMP_DIR, Path.home() / "Downloads"]
        
        file_path = None
        for search_dir in search_dirs:
            potential_path = search_dir / filename
            if potential_path.exists():
                file_path = potential_path
                break
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        # Créer les infos du fichier
        file_info = {
            "filename": filename,
            "path": str(file_path),
            "size": file_path.stat().st_size,
            "is_video": is_video_file(str(file_path)),
            "is_image": is_image_file(str(file_path))
        }
        
        if is_video_file(str(file_path)):
            video_info = get_video_info(str(file_path))
            file_info.update({
                "width": video_info['width'],
                "height": video_info['height'],
                "duration": video_info['duration'],
                "fps": video_info['fps'],
                "frame_count": video_info['frame_count']
            })
        elif is_image_file(str(file_path)):
            image = image_read(str(file_path))
            height, width = get_image_resolution(image)
            file_info.update({
                "width": width,
                "height": height
            })
        
        return file_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur info fichier: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup")
async def cleanup_temp_files():
    """Nettoie les fichiers temporaires"""
    try:
        temp_dir = settings.TEMP_DIR
        cleaned_count = 0
        
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                if item.is_file():
                    try:
                        item.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Impossible de supprimer {item}: {e}")
                elif item.is_dir():
                    try:
                        shutil.rmtree(item)
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Impossible de supprimer {item}: {e}")
        
        return {
            "message": f"{cleaned_count} éléments nettoyés",
            "temp_dir": str(temp_dir)
        }
        
    except Exception as e:
        logger.error(f"Erreur nettoyage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-formats")
async def get_supported_formats():
    """Récupère la liste des formats supportés"""
    return {
        "images": settings.SUPPORTED_IMAGE_EXTENSIONS,
        "videos": settings.SUPPORTED_VIDEO_EXTENSIONS,
        "video_codecs": settings.VIDEO_CODECS
    }