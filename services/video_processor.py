import asyncio
import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from services.ai_upscale import MultiThreadAIUpscaler
from utils.image_utils import image_read, image_write, blend_images, get_video_info
from utils.file_utils import prepare_output_path, create_temp_directory, cleanup_temp_directory
from core.exceptions import ProcessingError
from api.models.requests import UpscaleRequest
from api.models.responses import ProcessStatus
from core.config import settings
import traceback

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.active_processes: Dict[str, Dict[str, Any]] = {}
        self.ffmpeg_path = settings.ASSETS_DIR / "ffmpeg" / "ffmpeg.exe"
    
    async def process_videos(
        self,
        request: UpscaleRequest,
        process_id: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> str:
        """Traite une liste de vidéos"""
        if not process_id:
            process_id = str(uuid.uuid4())

        # Filtrer les vidéos seulement
        from utils.file_utils import is_video_file
        video_paths = [path for path in request.file_paths if is_video_file(path)]
        
        if not video_paths:
            raise ProcessingError("Aucun fichier vidéo valide trouvé")
        
        # Initialiser le processus
        self.active_processes[process_id] = {
            'status': ProcessStatus.PROCESSING,
            'progress': 0.0,
            'current_file': None,
            'current_step': 'Initialisation',
            'total_files': len(video_paths),
            'completed_files': [],
            'failed_files': [],
            'error_message': None,
            'estimated_time_remaining': None
        }
        
        try:
            # Créer l'upscaler multi-thread
            upscaler = MultiThreadAIUpscaler(
                num_threads=request.multithreading,
                model_name=request.ai_model.value,
                gpu_device=request.gpu.value,
                input_resize_factor=request.input_resize_factor,
                output_resize_factor=request.output_resize_factor,
                max_resolution=int(request.vram_limit * 100)
            )
            
            # Traiter chaque vidéo
            for i, video_path in enumerate(video_paths):
                try:
                    # Mettre à jour le statut
                    self.active_processes[process_id].update({
                        'current_file': Path(video_path).name,
                        'progress': (i / len(video_paths)) * 100,
                        'current_step': 'Traitement vidéo'
                    })
                    
                    if progress_callback and callable(progress_callback):
                        await progress_callback(process_id, self.active_processes[process_id])
                    
                    # Traitement de la vidéo
                    await self._process_single_video(
                        video_path,
                        upscaler,
                        request,
                        process_id,
                        progress_callback
                    )
                    
                    self.active_processes[process_id]['completed_files'].append(video_path)
                    
                except Exception as e:
                    logger.error(f"Erreur traitement {video_path}: {e}")
                    traceback.print_exc()
                    self.active_processes[process_id]['failed_files'].append(video_path)
            
            # Finaliser
            self.active_processes[process_id].update({
                'status': ProcessStatus.COMPLETED,
                'progress': 100.0,
                'current_file': None,
                'current_step': 'Terminé'
            })
            
            return process_id
            
        except Exception as e:
            self.active_processes[process_id].update({
                'status': ProcessStatus.ERROR,
                'error_message': str(e)
            })
            raise ProcessingError(f"Erreur lors du traitement des vidéos: {e}")
    
    async def _process_single_video(
        self,
        video_path: str,
        upscaler: MultiThreadAIUpscaler,
        request: UpscaleRequest,
        process_id: str,
        progress_callback: Optional[callable] = None
    ) -> None:
        """Traite une seule vidéo"""
        temp_dir = None
        try:
            # Créer répertoire temporaire
            temp_dir = create_temp_directory(f"video_{Path(video_path).stem}_")
            
            # Obtenir infos vidéo
            video_info = get_video_info(video_path)
            
            # Étape 1: Extraction des frames
            self.active_processes[process_id]['current_step'] = 'Extraction des frames'
            frame_paths = await self._extract_frames(video_path, temp_dir, progress_callback, process_id)
            if not frame_paths:
                raise ProcessingError(f"Aucune frame extraite de la vidéo {video_path}. Le fichier est peut-être corrompu ou non pris en charge.")
            
            # Étape 2: Upscaling des frames
            self.active_processes[process_id]['current_step'] = 'Upscaling des frames'
            upscaled_frame_paths = await self._upscale_frames(
                frame_paths, 
                upscaler, 
                request, 
                temp_dir,
                progress_callback,
                process_id
            )
            if not upscaled_frame_paths:
                raise ProcessingError(f"Aucune frame upscalée générée pour la vidéo {video_path}.")
            
            # Étape 3: Reconstruction vidéo
            self.active_processes[process_id]['current_step'] = 'Reconstruction vidéo'
            output_path = prepare_output_path(
                input_path=video_path,
                output_dir=request.output_path,
                ai_model=request.ai_model.value,
                input_resize=request.input_resize_factor,
                output_resize=request.output_resize_factor,
                blending=request.blending.value,
                extension=request.video_extension
            )
            
            await self._encode_video(
                video_path,
                output_path,
                upscaled_frame_paths,
                video_info,
                request.video_codec.value,
                progress_callback,
                process_id
            )
            
            # Nettoyer les frames si demandé
            if not request.keep_frames and temp_dir:
                cleanup_temp_directory(temp_dir)
            
            logger.info(f"Vidéo traitée avec succès: {output_path}")
            
        except Exception as e:
            if temp_dir:
                cleanup_temp_directory(temp_dir)
            logger.error(f"Erreur traitement vidéo {video_path}: {e}")
            raise
    
    async def _extract_frames(
        self, 
        video_path: str, 
        temp_dir: str,
        progress_callback: Optional[callable],
        process_id: str
    ) -> List[str]:
        """Extrait les frames d'une vidéo"""
        frame_paths = []
        
        def extract_sync():
            # SÉCURITÉ: Création du dossier si pas existant
            os.makedirs(temp_dir, exist_ok=True)

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"[OpenCV] Impossible d'ouvrir la vidéo source : {video_path}")
                raise ProcessingError(f"Impossible d'ouvrir la vidéo source : {video_path}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cpu_cores = cpu_count()
            frames_to_extract = cpu_cores * 30  # 30 frames par core

            frames_extracted = []
            frame_index = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"[OpenCV] Fin de lecture ou lecture frame échouée à l'index {frame_index} sur {frame_count} frames.")
                    break

                frame_path = os.path.normpath(os.path.join(temp_dir, f"frame_{frame_index:06d}.jpg"))
                # LOG: Vérifie l'existence du dossier
                if not os.path.isdir(os.path.dirname(frame_path)):
                    logger.error(f"Le dossier pour frame {frame_index} n'existe pas: {os.path.dirname(frame_path)}")
                write_ok = cv2.imwrite(frame_path, frame)
                if write_ok:
                    frames_extracted.append(frame_path)
                else:
                    logger.error(f"[OpenCV] Échec écriture frame {frame_index} dans {frame_path} (chemin existant ? permissions ?)")
                    logger.error(f"Chemin absolu: {os.path.abspath(frame_path)} | Dossier existe: {os.path.isdir(os.path.dirname(frame_path))}")
                try:
                    write_ok = cv2.imwrite(frame_path, frame)
                    if write_ok:
                        frames_extracted.append(frame_path)
                    else:
                        logger.error(f"Erreur lors de la sauvegarde de la frame {frame_index} dans {frame_path}")
                except Exception as ex:
                    logger.error(f"Erreur exception lors de la sauvegarde de la frame {frame_index}: {ex}")

                # Sauvegarder par batch
                if len(frames_extracted) >= frames_to_extract:
                    frame_paths.extend(frames_extracted)
                    frames_extracted = []
                    
                    # Mettre à jour progression
                    progress = (frame_index / frame_count) * 100 if frame_count else 0
                    if process_id in self.active_processes:
                        self.active_processes[process_id]['progress'] = progress * 0.3  # 30% pour extraction
                
                frame_index += 1
            
            # Ajouter les frames restantes
            frame_paths.extend(frames_extracted)
            cap.release()
            
            return frame_paths
        
        # Exécuter dans un thread séparé
        loop = asyncio.get_event_loop()
        frame_paths = await loop.run_in_executor(None, extract_sync)
        if not frame_paths:
            logger.error(f"Aucune frame extraite pour {video_path} (chemin: {temp_dir})")
            raise ProcessingError(f"Aucune frame extraite de la vidéo {video_path}. Le fichier est peut-être corrompu ou non pris en charge.")
        # Loguer les chemins pour debug
        for p in frame_paths[:3]:
            logger.debug(f"Frame extraite: {p}")
        return frame_paths

    async def _upscale_frames(
        self,
        frame_paths: List[str],
        upscaler: MultiThreadAIUpscaler,
        request: UpscaleRequest,
        temp_dir: str,
        progress_callback: Optional[callable],
        process_id: str
    ) -> List[str]:
        """Upscale les frames de la vidéo"""
        upscaled_paths = []
        
        # Calculer la capacité GPU
        if frame_paths:
            try:
                test_img = image_read(frame_paths[0])
                if test_img is None:
                    logger.error(f"Erreur calcul capacité: Impossible de lire {frame_paths[0]}")
                    capacity = 1
                else:
                    capacity = upscaler.calculate_frames_capacity(frame_paths[0])
            except Exception as e:
                logger.warning(f"Erreur calcul capacité: {e}")
                capacity = 1
        else:
            capacity = 1
        
        effective_threads = min(capacity, request.multithreading)

        def process_frame_batch(frame_batch, thread_id):
            upscaler_instance = upscaler.get_upscaler(thread_id)
            batch_results = []
            
            for frame_path in frame_batch:
                try:
                    frame = image_read(frame_path)
                    if frame is None:
                        logger.error(f"Frame non lisible: {frame_path}")
                        batch_results.append(None)
                        continue

                    upscaled_frame = upscaler_instance.process_image(frame)
                    
                    # Blending si nécessaire
                    if request.blending.value != "OFF":
                        blend_factors = {"Low": 0.3, "Medium": 0.5, "High": 0.7}
                        blend_factor = blend_factors.get(request.blending.value, 0.0)
                        final_frame = blend_images(frame, upscaled_frame, blend_factor)
                    else:
                        final_frame = upscaled_frame
                    
                    # Sauvegarder
                    frame_name = Path(frame_path).stem
                    output_path = os.path.join(temp_dir, f"{frame_name}_upscaled.jpg")
                    image_write(output_path, final_frame, ".jpg")
                    batch_results.append(output_path)
                    
                except Exception as e:
                    logger.error(f"Erreur frame {frame_path}: {e}")
                    batch_results.append(None)
            
            return batch_results
        
        # Diviser les frames en batches
        batch_size = len(frame_paths) // effective_threads if effective_threads else len(frame_paths)
        if batch_size == 0:
            batch_size = 1
        
        batches = [frame_paths[i:i + batch_size] for i in range(0, len(frame_paths), batch_size)]
        
        # Traitement parallèle
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=effective_threads) as executor:
            tasks = [
                loop.run_in_executor(executor, process_frame_batch, batch, i)
                for i, batch in enumerate(batches)
            ]
            
            # Traiter avec mise à jour de progression
            for i, task in enumerate(asyncio.as_completed(tasks)):
                batch_results = await task
                upscaled_paths.extend([p for p in batch_results if p])
                
                # Mettre à jour progression (30-80% pour upscaling)
                progress = 30 + ((i + 1) / len(tasks)) * 50
                if process_id in self.active_processes:
                    self.active_processes[process_id]['progress'] = progress

        if not upscaled_paths:
            logger.error("Aucun frame upscalé généré.")
        return sorted(upscaled_paths)
    
    async def _encode_video(
        self,
        original_video_path: str,
        output_path: str,
        frame_paths: List[str],
        video_info: dict,
        codec: str,
        progress_callback: Optional[callable],
        process_id: str
    ) -> None:
        """Encode la vidéo finale"""
        if not self.ffmpeg_path.exists():
            raise ProcessingError("FFmpeg non trouvé")
        
        if not frame_paths:
            raise ProcessingError("Aucune frame upscalée à encoder pour la vidéo.")
        
        # Préparer les chemins
        frames_list_path = os.path.join(Path(frame_paths[0]).parent, "frames_list.txt")
        temp_video_path = output_path.replace(Path(output_path).suffix, "_temp.mp4")
        
        try:
            # Créer liste des frames
            with open(frames_list_path, 'w', encoding='utf-8') as f:
                for frame_path in frame_paths:
                    f.write(f"file '{frame_path}'\n")
            
            # Mapper les codecs
            codec_map = {
                "x264": "libx264",
                "x265": "libx265"
            }
            actual_codec = codec_map.get(codec, codec)
            
            # Commande FFmpeg pour créer vidéo sans audio
            cmd_video = [
                str(self.ffmpeg_path),
                "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-r", str(video_info['fps']),
                "-i", frames_list_path,
                "-c:v", actual_codec,
                "-vf", "scale=in_range=full:out_range=limited,format=yuv420p",
                "-color_range", "tv",
                "-b:v", "12000k",
                temp_video_path
            ]
            
            # Exécuter encodage vidéo
            process = await asyncio.create_subprocess_exec(
                *cmd_video,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if process.returncode != 0:
                raise ProcessingError("Erreur lors de l'encodage vidéo")
            
            # Ajouter l'audio de la vidéo originale
            cmd_audio = [
                str(self.ffmpeg_path),
                "-y", "-loglevel", "error",
                "-i", original_video_path,
                "-i", temp_video_path,
                "-c:v", "copy",
                "-map", "1:v:0",
                "-map", "0:a?",
                "-c:a", "copy",
                output_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd_audio,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            # Nettoyer fichier temporaire
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(frames_list_path):
                os.remove(frames_list_path)
            
            # Mettre à jour progression finale
            if process_id in self.active_processes:
                self.active_processes[process_id]['progress'] = 100.0
            
        except Exception as e:
            # Nettoyer en cas d'erreur
            for temp_file in [temp_video_path, frames_list_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            raise ProcessingError(f"Erreur encodage: {e}")
    
    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Récupère le statut d'un processus"""
        return self.active_processes.get(process_id)
    
    def cancel_process(self, process_id: str) -> bool:
        """Annule un processus"""
        if process_id in self.active_processes:
            self.active_processes[process_id]['status'] = ProcessStatus.CANCELLED
            return True
        return False
    
    def cleanup_process(self, process_id: str) -> None:
        """Nettoie un processus terminé"""
        if process_id in self.active_processes:
            del self.active_processes[process_id]