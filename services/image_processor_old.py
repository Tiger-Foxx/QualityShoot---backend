import os
import cv2
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

from services.ai_upscale import AIUpscaleService
from utils.image_utils import image_read, image_write, get_image_resolution
from utils.file_utils import create_output_filename
from api.models.requests import UpscaleRequest
from core.config import settings

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Service de traitement d'images"""
    
    def __init__(self):
        self.ai_service = AIUpscaleService()
        self.processes: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
    def process_images(self, request: UpscaleRequest, main_process_id: str):
        """Traite une liste d'images avec l'ID fourni"""
        try:
            logger.info(f"🎨 Début traitement {len(request.file_paths)} images avec ID: {main_process_id}")
            
            # UTILISER l'ID fourni au lieu d'en créer un nouveau
            process_data = {
                'id': main_process_id,
                'status': 'processing',
                'progress': 0.0,
                'total_files': len(request.file_paths),
                'completed_files': [],
                'failed_files': [],
                'current_file': None,
                'current_step': 'Initializing...',
                'created_at': datetime.now(),
                'error_message': None
            }
            
            with self.lock:
                self.processes[main_process_id] = process_data
            
            # Traiter chaque image
            for i, file_path in enumerate(request.file_paths):
                try:
                    # Mettre à jour le statut
                    file_name = Path(file_path).name
                    progress = (i / len(request.file_paths)) * 90  # Laisser 10% pour finalisation
                    
                    self._update_process(main_process_id, {
                        'current_file': file_name,
                        'current_step': f'Processing image {i+1}/{len(request.file_paths)}...',
                        'progress': progress
                    })
                    
                    logger.info(f"🖼️ Traitement image {i+1}/{len(request.file_paths)}: {file_name}")
                    
                    # Traiter l'image
                    success = self._process_single_image(file_path, request, main_process_id)
                    
                    # Mettre à jour les résultats
                    current_data = self.processes[main_process_id]
                    if success:
                        current_data['completed_files'].append(file_path)
                        logger.info(f"✅ Image {file_name} traitée avec succès")
                    else:
                        current_data['failed_files'].append(file_path)
                        logger.error(f"❌ Échec traitement {file_name}")
                        
                except Exception as e:
                    logger.error(f"❌ Erreur traitement {file_path}: {e}")
                    current_data = self.processes[main_process_id]
                    current_data['failed_files'].append(file_path)
                    current_data['error_message'] = str(e)
            
            # Finaliser
            final_status = 'completed'
            current_data = self.processes[main_process_id]
            if current_data['failed_files']:
                final_status = 'completed_with_errors'
            
            self._update_process(main_process_id, {
                'status': final_status,
                'progress': 100.0,
                'current_file': None,
                'current_step': 'Completed',
                'completed_at': datetime.now()
            })
            
            logger.info(f"🎉 Traitement terminé: {len(current_data['completed_files'])} succès, {len(current_data['failed_files'])} échecs")
            
        except Exception as e:
            logger.error(f"❌ Erreur générale traitement images: {e}")
            self._update_process(main_process_id, {
                'status': 'error',
                'error_message': str(e),
                'current_step': 'Error occurred'
            })
    
    def _process_single_image(self, file_path: str, request: UpscaleRequest, process_id: str) -> bool:
        """Traite une seule image"""
        try:
            file_path_obj = Path(file_path)
            
            # Vérifier que le fichier existe
            if not file_path_obj.exists():
                logger.error(f"❌ Fichier non trouvé: {file_path}")
                return False
            
            # Mettre à jour le statut
            self._update_process(process_id, {
                'current_step': 'Reading image...'
            })
            
            # Lire l'image
            logger.debug(f"📖 Lecture image: {file_path}")
            image = image_read(file_path)
            if image is None:
                logger.error(f"❌ Impossible de lire l'image: {file_path}")
                return False
            
            # Obtenir les dimensions
            original_height, original_width = image.shape[:2]
            logger.info(f"📐 Dimensions originales: {original_width}x{original_height}")
            
            # Redimensionner en entrée si nécessaire
            if request.input_resize_factor != 1.0:
                self._update_process(process_id, {
                    'current_step': f'Resizing input ({request.input_resize_factor}x)...'
                })
                
                new_width = int(original_width * request.input_resize_factor)
                new_height = int(original_height * request.input_resize_factor)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                logger.info(f"📏 Redimensionné en entrée: {new_width}x{new_height}")
            
            # Upscaling IA
            self._update_process(process_id, {
                'current_step': f'AI upscaling with {request.ai_model.value}...'
            })
            
            logger.info(f"🤖 Upscaling IA avec {request.ai_model.value}")
            upscaled_image = self.ai_service.upscale_image(
                image=image,
                model=request.ai_model,
                gpu=request.gpu,
                vram_limit=request.vram_limit
            )
            
            if upscaled_image is None:
                logger.error(f"❌ Échec upscaling IA")
                return False
            
            upscaled_height, upscaled_width = upscaled_image.shape[:2]
            logger.info(f"🚀 Upscaling terminé: {upscaled_width}x{upscaled_height}")
            
            # Redimensionner en sortie si nécessaire
            if request.output_resize_factor != 1.0:
                self._update_process(process_id, {
                    'current_step': f'Resizing output ({request.output_resize_factor}x)...'
                })
                
                final_width = int(upscaled_width * request.output_resize_factor)
                final_height = int(upscaled_height * request.output_resize_factor)
                upscaled_image = cv2.resize(upscaled_image, (final_width, final_height), interpolation=cv2.INTER_LANCZOS4)
                logger.info(f"📏 Redimensionné en sortie: {final_width}x{final_height}")
            
            # Créer le nom de fichier de sortie
            output_path = create_output_filename(
                input_path=file_path,
                suffix="_upscaled",
                extension=request.image_extension
            )
            
            # Sauvegarder
            self._update_process(process_id, {
                'current_step': 'Saving result...'
            })
            
            logger.info(f"💾 Sauvegarde: {output_path}")
            success = image_write(upscaled_image, output_path)
            
            if success:
                output_size = Path(output_path).stat().st_size
                logger.info(f"✅ Image sauvegardée: {output_path} ({output_size/1024/1024:.1f}MB)")
                return True
            else:
                logger.error(f"❌ Échec sauvegarde: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur traitement image {file_path}: {e}")
            return False
    
    def _update_process(self, process_id: str, updates: Dict[str, Any]):
        """Met à jour un processus"""
        with self.lock:
            if process_id in self.processes:
                self.processes[process_id].update(updates)
                self.processes[process_id]['updated_at'] = datetime.now()
                logger.debug(f"📊 Process {process_id}: {updates}")
    
    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Récupère le statut d'un processus"""
        with self.lock:
            return self.processes.get(process_id)
    
    def cancel_process(self, process_id: str) -> bool:
        """Annule un processus"""
        with self.lock:
            if process_id in self.processes:
                self.processes[process_id].update({
                    'status': 'cancelled',
                    'cancelled_at': datetime.now()
                })
                logger.info(f"⏹️ Processus image annulé: {process_id}")
                return True
            return False
    
    def cleanup_process(self, process_id: str):
        """Nettoie un processus"""
        with self.lock:
            if process_id in self.processes:
                del self.processes[process_id]
                logger.info(f"🧹 Processus image nettoyé: {process_id}")
    
    def pause_process(self, process_id: str) -> bool:
        """Met en pause un processus"""
        with self.lock:
            if process_id in self.processes:
                self.processes[process_id]['status'] = 'paused'
                return True
            return False
    
    def resume_process(self, process_id: str) -> bool:
        """Reprend un processus"""
        with self.lock:
            if process_id in self.processes:
                self.processes[process_id]['status'] = 'processing'
                return True
            return False
    
    def get_queue_status(self) -> List[Dict[str, Any]]:
        """Récupère le statut de la file"""
        with self.lock:
            return list(self.processes.values())