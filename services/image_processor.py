import asyncio
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import uuid
from pathlib import Path

from services.ai_upscale import AIUpscaler
from utils.image_utils import image_read, image_write, blend_images, get_image_resolution
from utils.file_utils import prepare_output_path
from core.exceptions import ProcessingError
from core.process_manager import process_manager
from api.models.requests import UpscaleRequest
from api.models.responses import ProcessStatus

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        pass  # Plus besoin de active_processes, on utilise process_manager
    
    async def process_images(
        self, 
        request: UpscaleRequest,
        process_id: str = None,
        progress_callback: Optional[callable] = None
    ) -> str:
        """Traite une liste d'images"""
        
        # Créer un processus si pas fourni
        if not process_id:
            process_id = process_manager.create_process(
                process_type="image_upscale",
                total_files=len(request.file_paths)
            )
        
        # Filtrer les images seulement
        from utils.file_utils import is_image_file
        image_paths = [path for path in request.file_paths if is_image_file(path)]
        
        if not image_paths:
            process_manager.update_process(
                process_id,
                status=ProcessStatus.ERROR,
                error_message="Aucun fichier image valide trouvé"
            )
            raise ProcessingError("Aucun fichier image valide trouvé")
        
        # Mettre à jour le processus
        process_manager.update_process(
            process_id,
            status=ProcessStatus.PROCESSING,
            total_files=len(image_paths),
            current_step="Initialisation de l'upscaler..."
        )
        
        try:
            # Créer l'upscaler
            process_manager.update_process(
                process_id,
                current_step="Chargement du modèle IA..."
            )
            
            upscaler = AIUpscaler(
                model_name=request.ai_model.value if hasattr(request.ai_model, 'value') else request.ai_model,
                gpu_device=request.gpu.value if hasattr(request.gpu, 'value') else request.gpu,
                input_resize_factor=request.input_resize_factor,
                output_resize_factor=request.output_resize_factor,
                max_resolution=int(request.vram_limit * 100)
            )
            
            # Traiter chaque image
            for i, image_path in enumerate(image_paths):
                try:
                    # Mettre à jour le statut
                    progress = (i / len(image_paths)) * 100
                    process_manager.update_process(
                        process_id,
                        current_file=Path(image_path).name,
                        progress=progress,
                        current_step=f"Traitement image {i+1}/{len(image_paths)}"
                    )
                    
                    if progress_callback:
                        await progress_callback(process_id, process_manager.get_process_status(process_id))
                    
                    # Traitement de l'image
                    await self._process_single_image(
                        image_path, 
                        upscaler, 
                        request,
                        process_id
                    )
                    
                    process_manager.add_completed_file(process_id, image_path)
                    
                except Exception as e:
                    logger.error(f"Erreur traitement {image_path}: {e}")
                    process_manager.add_failed_file(process_id, image_path)
                    process_manager.update_process(
                        process_id,
                        current_step=f"Erreur sur {Path(image_path).name}: {str(e)}"
                    )
            
            # Finaliser
            completed_files = process_manager.get_process(process_id).get('completed_files', [])
            failed_files = process_manager.get_process(process_id).get('failed_files', [])
            
            if len(completed_files) == len(image_paths):
                final_status = ProcessStatus.COMPLETED
                final_message = f"Toutes les images traitées avec succès ({len(completed_files)})"
            elif len(completed_files) > 0:
                final_status = ProcessStatus.COMPLETED
                final_message = f"{len(completed_files)} images traitées, {len(failed_files)} échouées"
            else:
                final_status = ProcessStatus.ERROR
                final_message = "Aucune image n'a pu être traitée"
            
            process_manager.update_process(
                process_id,
                status=final_status,
                progress=100.0,
                current_file=None,
                current_step=final_message
            )
            
            logger.info(f"✅ Processus d'upscaling terminé: {process_id}")
            return process_id
            
        except Exception as e:
            process_manager.update_process(
                process_id,
                status=ProcessStatus.ERROR,
                error_message=str(e),
                current_step="Erreur fatale lors du traitement"
            )
            logger.error(f"❌ Erreur fatale processus {process_id}: {e}")
            raise ProcessingError(f"Erreur lors du traitement des images: {e}")
    
    async def _process_single_image(
        self,
        image_path: str,
        upscaler: AIUpscaler,
        request: UpscaleRequest,
        process_id: str
    ) -> None:
        """Traite une seule image"""
        try:
            # Charger l'image
            process_manager.update_process(
                process_id,
                current_step=f"Chargement de {Path(image_path).name}..."
            )
            
            original_image = image_read(image_path)
            
            # Upscaling
            process_manager.update_process(
                process_id,
                current_step=f"Upscaling de {Path(image_path).name}..."
            )
            
            upscaled_image = upscaler.process_image(original_image)
            
            # Blending si nécessaire
            blending_value = request.blending.value if hasattr(request.blending, 'value') else request.blending
            if blending_value != "OFF":
                process_manager.update_process(
                    process_id,
                    current_step=f"Application du blending sur {Path(image_path).name}..."
                )
                
                blend_factors = {
                    "Low": 0.3,
                    "Medium": 0.5,
                    "High": 0.7
                }
                blend_factor = blend_factors.get(blending_value, 0.0)
                final_image = blend_images(original_image, upscaled_image, blend_factor)
            else:
                final_image = upscaled_image
            
            # Préparer le chemin de sortie
            ai_model_value = request.ai_model.value if hasattr(request.ai_model, 'value') else request.ai_model
            blending_value = request.blending.value if hasattr(request.blending, 'value') else request.blending
            
            output_path = prepare_output_path(
                input_path=image_path,
                output_dir=request.output_path,
                ai_model=ai_model_value,
                input_resize=request.input_resize_factor,
                output_resize=request.output_resize_factor,
                blending=blending_value,
                extension=request.image_extension
            )
            
            # Sauvegarder
            process_manager.update_process(
                process_id,
                current_step=f"Sauvegarde de {Path(image_path).name}..."
            )
            
            image_write(output_path, final_image, request.image_extension)
            
            # Copier les métadonnées (si possible)
            await self._copy_metadata(image_path, output_path)
            
            logger.info(f"✅ Image traitée avec succès: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement image {image_path}: {e}")
            raise
    
    async def _copy_metadata(self, source_path: str, target_path: str) -> None:
        """Copie les métadonnées EXIF (si exiftool est disponible)"""
        try:
            import subprocess
            from core.config import settings
            
            exiftool_path = settings.ASSETS_DIR / "exiftool.exe"
            if exiftool_path.exists():
                cmd = [
                    str(exiftool_path),
                    '-fast',
                    '-TagsFromFile',
                    source_path,
                    '-overwrite_original',
                    '-all:all',
                    '-unsafe',
                    '-largetags',
                    target_path
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                
        except Exception as e:
            logger.warning(f"Impossible de copier les métadonnées: {e}")
    
    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Récupère le statut d'un processus"""
        return process_manager.get_process_status(process_id)
    
    def cancel_process(self, process_id: str) -> bool:
        """Annule un processus"""
        return process_manager.cancel_process(process_id)
    
    def cleanup_process(self, process_id: str) -> None:
        """Nettoie un processus terminé"""
        process_manager.cleanup_process(process_id)