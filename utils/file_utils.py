# Utilitaires pour les fichiers
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile
from core.config import settings
from core.exceptions import InvalidFileFormatError
from pathlib import Path

def is_supported_file(file_path: str) -> bool:
    """Vérifie si le fichier est supporté"""
    suffix = Path(file_path).suffix.lower()
    return suffix in (settings.SUPPORTED_IMAGE_EXTENSIONS + settings.SUPPORTED_VIDEO_EXTENSIONS)

def is_video_file(file_path: str) -> bool:
    """Vérifie si le fichier est une vidéo"""
    suffix = Path(file_path).suffix.lower()
    return suffix in settings.SUPPORTED_VIDEO_EXTENSIONS

def is_image_file(file_path: str) -> bool:
    """Vérifie si le fichier est une image"""
    suffix = Path(file_path).suffix.lower()
    return suffix in settings.SUPPORTED_IMAGE_EXTENSIONS

def validate_file_paths(file_paths: List[str]) -> Tuple[List[str], List[dict]]:
    """Valide une liste de chemins de fichiers"""
    valid_files = []
    invalid_files = []
    
    for file_path in file_paths:
        try:
            path = Path(file_path)
            
            if not path.exists():
                invalid_files.append({
                    "file": file_path,
                    "reason": "Le fichier n'existe pas"
                })
                continue
                
            if not path.is_file():
                invalid_files.append({
                    "file": file_path,
                    "reason": "N'est pas un fichier"
                })
                continue
                
            if not is_supported_file(file_path):
                invalid_files.append({
                    "file": file_path,
                    "reason": "Format de fichier non supporté"
                })
                continue
                
            valid_files.append(file_path)
            
        except Exception as e:
            invalid_files.append({
                "file": file_path,
                "reason": f"Erreur de validation: {str(e)}"
            })
    
    return valid_files, invalid_files

def create_temp_directory(prefix: str = "qualityshoot_") -> str:
    """Crée un répertoire temporaire"""
    temp_dir = tempfile.mkdtemp(prefix=prefix, dir=settings.TEMP_DIR)
    return temp_dir

def cleanup_temp_directory(temp_dir: str) -> None:
    """Nettoie un répertoire temporaire"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Erreur lors du nettoyage de {temp_dir}: {e}")

def prepare_output_path(
    input_path: str,
    output_dir: Optional[str],
    ai_model: str,
    input_resize: float,
    output_resize: float,
    blending: str,
    extension: str
) -> str:
    """Prépare le chemin de sortie pour un fichier traité"""
    input_file = Path(input_path)
    
    if output_dir:
        output_base = Path(output_dir) / input_file.stem
    else:
        output_base = input_file.parent / input_file.stem
    
    # Suffixes pour identifier le traitement
    suffix = f"_{ai_model}"
    suffix += f"_InputR-{int(input_resize * 100)}"
    suffix += f"_OutputR-{int(output_resize * 100)}"
    
    if blending != "OFF":
        suffix += f"_Blending-{blending}"
    
    return str(output_base) + suffix + extension

# AJOUTER LA FONCTION MANQUANTE
def create_output_filename(input_path: str, suffix: str = "_upscaled", extension: str = None) -> str:
    """Crée un nom de fichier de sortie basé sur le fichier d'entrée"""
    try:
        input_path_obj = Path(input_path)
        
        # Utiliser l'extension d'origine si pas spécifiée
        if extension is None:
            extension = input_path_obj.suffix
        
        # Construire le nouveau nom
        output_name = f"{input_path_obj.stem}{suffix}{extension}"
        output_path = input_path_obj.parent / output_name
        
        return str(output_path)
        
    except Exception as e:
        # Fallback simple
        return f"{input_path}{suffix}{extension or '.png'}"

def get_file_size(file_path: str) -> int:
    """Retourne la taille du fichier en octets"""
    return os.path.getsize(file_path)

def ensure_directory_exists(directory: str) -> None:
    """S'assure qu'un répertoire existe"""
    Path(directory).mkdir(parents=True, exist_ok=True)