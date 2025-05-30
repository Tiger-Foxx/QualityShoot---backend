﻿import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import unicodedata
import traceback

from services.ai_upscale import MultiThreadAIUpscaler
from utils.image_utils import image_read, image_write, blend_images, get_video_info
from utils.file_utils import prepare_output_path, create_temp_directory, cleanup_temp_directory
from core.exceptions import ProcessingError
from core.process_manager import process_manager
from api.models.requests import UpscaleRequest
from api.models.responses import ProcessStatus
from core.config import settings

logger = logging.getLogger(__name__)

def slugify(value: str) -> str:
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    return ''.join(c if c.isalnum() else '_' for c in value)

def safe_image_write(file_path, image, ext=".jpg"):
    """Robust image write, encoding first then writing to file (better for Windows/accents)."""
    try:
        result, encoded_img = cv2.imencode(ext, image)
        if result:
            encoded_img.tofile(file_path)
            return True
        logger.error(f"[safe_image_write] cv2.imencode a échoué pour {file_path}")
        return False
    except Exception as e:
        logger.error(f"[safe_image_write] Exception lors de imencode/tofile: {e} (chemin: {file_path})")
        return False

class VideoProcessor:
    def __init__(self):
        pass  # Utilisation exclusive du process_manager

    async def process_videos(
        self,
        request: UpscaleRequest,
        process_id: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Traite une liste de vidéos avec gestion de processus via process_manager.
        """
        from utils.file_utils import is_video_file
        video_paths = [path for path in request.file_paths if is_video_file(path)]
        if not video_paths:
            raise ProcessingError("Aucun fichier vidéo valide trouvé")

        # Créer le process dans le process_manager (statut partagé avec le frontend)
        if not process_id:
            process_id = process_manager.create_process(
                process_type="video_upscale",
                total_files=len(video_paths)
            )

        process_manager.update_process(
            process_id,
            status=ProcessStatus.PROCESSING,
            total_files=len(video_paths),
            progress=0.0,
            current_file=None,
            current_step="Initialisation de l'upscaler..."
        )

        try:
            process_manager.update_process(
                process_id,
                current_step="Chargement du modèle IA..."
            )
            upscaler = MultiThreadAIUpscaler(
                num_threads=request.multithreading,
                model_name=request.ai_model.value if hasattr(request.ai_model, 'value') else request.ai_model,
                gpu_device=request.gpu.value if hasattr(request.gpu, 'value') else request.gpu,
                input_resize_factor=request.input_resize_factor,
                output_resize_factor=request.output_resize_factor,
                max_resolution=int(request.vram_limit * 100)
            )
            logger.info(f"[process_videos] Upscaler initialisé avec modèle={request.ai_model.value}")

            for i, video_path in enumerate(video_paths):
                try:
                    progress = (i / len(video_paths)) * 100
                    process_manager.update_process(
                        process_id,
                        current_file=Path(video_path).name,
                        progress=progress,
                        current_step=f"Traitement vidéo {i+1}/{len(video_paths)}"
                    )
                    if progress_callback and callable(progress_callback):
                        await progress_callback(process_id, process_manager.get_process_status(process_id))
                    await self._process_single_video(
                        video_path,
                        upscaler,
                        request,
                        process_id,
                        progress_callback
                    )
                    process_manager.add_completed_file(process_id, video_path)
                    logger.info(f"[process_videos] Vidéo traitée (succès) : {video_path}")
                except Exception as e:
                    logger.error(f"[process_videos] Erreur traitement {video_path}: {e}")
                    traceback.print_exc()
                    process_manager.add_failed_file(process_id, video_path)
                    process_manager.update_process(
                        process_id,
                        current_step=f"Erreur sur {Path(video_path).name}: {str(e)}"
                    )

            completed_files = process_manager.get_process(process_id).get('completed_files', [])
            failed_files = process_manager.get_process(process_id).get('failed_files', [])
            if len(completed_files) == len(video_paths):
                final_status = ProcessStatus.COMPLETED
                final_message = f"Toutes les vidéos traitées avec succès ({len(completed_files)})"
            elif len(completed_files) > 0:
                final_status = ProcessStatus.COMPLETED
                final_message = f"{len(completed_files)} vidéos traitées, {len(failed_files)} échouées"
            else:
                final_status = ProcessStatus.ERROR
                final_message = "Aucune vidéo n'a pu être traitée"

            process_manager.update_process(
                process_id,
                status=final_status,
                progress=100.0,
                current_file=None,
                current_step=final_message
            )

            logger.info(f"✅ Processus d'upscaling vidéo terminé: {process_id}")
            return process_id

        except Exception as e:
            process_manager.update_process(
                process_id,
                status=ProcessStatus.ERROR,
                error_message=str(e),
                current_step="Erreur fatale lors du traitement"
            )
            logger.error(f"❌ Erreur fatale processus {process_id}: {e}")
            raise ProcessingError(f"Erreur lors du traitement des vidéos: {e}")

    async def _process_single_video(
        self,
        video_path: str,
        upscaler: MultiThreadAIUpscaler,
        request: UpscaleRequest,
        process_id: str,
        progress_callback: Optional[callable] = None
    ) -> None:
        temp_dir = None
        try:
            stem = slugify(Path(video_path).stem)
            temp_dir = create_temp_directory(f"video_{stem}_")
            logger.info(f"[single_video] Temp dir créé: {temp_dir}")
            video_info = get_video_info(video_path)
            logger.info(f"[single_video] Infos vidéo: {video_info}")

            process_manager.update_process(
                process_id,
                current_step="Extraction des frames"
            )
            frame_paths = await self._extract_frames(video_path, temp_dir, progress_callback, process_id)
            if not frame_paths:
                logger.error(f"[single_video] Extraction: aucune frame extraite pour {video_path}")
                raise ProcessingError(f"Aucune frame extraite de la vidéo {video_path}. Le fichier est peut-être corrompu ou non pris en charge.")

            process_manager.update_process(
                process_id,
                current_step="Upscaling des frames"
            )
            upscaled_frame_paths = await self._upscale_frames(
                frame_paths, upscaler, request, temp_dir, progress_callback, process_id
            )
            if not upscaled_frame_paths:
                logger.error(f"[single_video] Upscaling: aucune frame upscalée générée pour {video_path}")
                raise ProcessingError(f"Aucune frame upscalée générée pour la vidéo {video_path}.")

            process_manager.update_process(
                process_id,
                current_step="Reconstruction vidéo"
            )
            output_path = prepare_output_path(
                input_path=video_path,
                output_dir=request.output_path,
                ai_model=request.ai_model.value if hasattr(request.ai_model, 'value') else request.ai_model,
                input_resize=request.input_resize_factor,
                output_resize=request.output_resize_factor,
                blending=request.blending.value if hasattr(request.blending, 'value') else request.blending,
                extension=request.video_extension
            )
            logger.info(f"[single_video] Reconstruction vidéo: output_path={output_path}")

            await self._encode_video(
                video_path,
                output_path,
                upscaled_frame_paths,
                video_info,
                request.video_codec.value if hasattr(request.video_codec, 'value') else request.video_codec,
                progress_callback,
                process_id
            )
            if not request.keep_frames and temp_dir:
                cleanup_temp_directory(temp_dir)
                logger.info(f"[single_video] Temp dir nettoyé: {temp_dir}")
            logger.info(f"[single_video] Vidéo traitée avec succès: {output_path}")
        except Exception as e:
            if temp_dir:
                cleanup_temp_directory(temp_dir)
            logger.error(f"[single_video] Erreur traitement vidéo {video_path}: {e}")
            raise

    async def _extract_frames(
        self,
        video_path: str,
        temp_dir: str,
        progress_callback: Optional[callable],
        process_id: str
    ) -> List[str]:
        frame_paths = []
        def extract_sync():
            os.makedirs(temp_dir, exist_ok=True)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"[extract_sync] Impossible d'ouvrir la vidéo source: {video_path}")
                raise ProcessingError(f"Impossible d'ouvrir la vidéo source: {video_path}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"[extract_sync] Nombre de frames détectées: {frame_count}")
            cpu_cores = cpu_count()
            frames_to_extract = max(cpu_cores * 30, 1)
            frames_extracted = []
            frame_index = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    if frame_index < frame_count:
                        logger.warning(f"[extract_sync] Fin prématurée de lecture à la frame {frame_index}/{frame_count}")
                    break
                frame_path = os.path.normpath(os.path.join(temp_dir, f"frame_{frame_index:06d}.jpg"))
                try:
                    write_ok = safe_image_write(frame_path, frame, ".jpg")
                    if write_ok:
                        frames_extracted.append(frame_path)
                    else:
                        logger.error(f"[extract_sync] Erreur sauvegarde frame {frame_index} dans {frame_path}")
                except Exception as ex:
                    logger.error(f"[extract_sync] Exception sauvegarde frame {frame_index}: {ex}")
                if len(frames_extracted) >= frames_to_extract:
                    frame_paths.extend(frames_extracted)
                    frames_extracted = []
                    progress = (frame_index / frame_count) * 100 if frame_count else 0
                    process_manager.update_process(
                        process_id,
                        progress=progress*0.3,
                        current_step=f"Extraction de la frame {frame_index}/{frame_count}"
                    )
                frame_index += 1
            frame_paths.extend(frames_extracted)
            cap.release()
            logger.info(f"[extract_sync] Nombre total de frames extraites: {len(frame_paths)}")
            return frame_paths
        loop = asyncio.get_event_loop()
        frame_paths = await loop.run_in_executor(None, extract_sync)
        if not frame_paths:
            logger.error(f"[extract_frames] Extraction: aucune frame extraite (chemin: {temp_dir})")
            raise ProcessingError(f"Aucune frame extraite de la vidéo {video_path}. Le fichier est peut-être corrompu ou non pris en charge.")
        for p in frame_paths[:3]:
            logger.debug(f"[extract_frames] Frame extraite: {p}")
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
        upscaled_paths = []
        if frame_paths:
            try:
                test_img = image_read(frame_paths[0])
                if test_img is None:
                    logger.error(f"[upscale_frames] Impossible de lire {frame_paths[0]} pour calcul capacité")
                    capacity = 1
                else:
                    capacity = upscaler.calculate_frames_capacity(frame_paths[0])
            except Exception as e:
                logger.warning(f"[upscale_frames] Erreur calcul capacité: {e}")
                capacity = 1
        else:
            capacity = 1
        effective_threads = min(capacity, request.multithreading)
        logger.info(f"[upscale_frames] Traitement en {effective_threads} threads")

        def process_frame_batch(frame_batch, thread_id):
            upscaler_instance = upscaler.get_upscaler(thread_id)
            batch_results = []
            for idx, frame_path in enumerate(frame_batch):
                try:
                    frame = image_read(frame_path)
                    if frame is None:
                        logger.error(f"[upscale_frames] Frame non lisible: {frame_path}")
                        batch_results.append(None)
                        continue
                    upscaled_frame = upscaler_instance.process_image(frame)
                    blending_value = request.blending.value if hasattr(request.blending, 'value') else request.blending
                    if blending_value != "OFF":
                        blend_factors = {"Low": 0.3, "Medium": 0.5, "High": 0.7}
                        blend_factor = blend_factors.get(blending_value, 0.0)
                        final_frame = blend_images(frame, upscaled_frame, blend_factor)
                    else:
                        final_frame = upscaled_frame
                    frame_name = Path(frame_path).stem
                    output_path = os.path.join(temp_dir, f"{frame_name}_upscaled.jpg")
                    write_ok = safe_image_write(output_path, final_frame, ".jpg")
                    if write_ok:
                        batch_results.append(output_path)
                    else:
                        logger.error(f"[upscale_frames] Erreur sauvegarde frame upscalée: {output_path}")
                        batch_results.append(None)
                except Exception as e:
                    logger.error(f"[upscale_frames] Exception sur {frame_path}: {e}")
                    batch_results.append(None)
            return batch_results

        batch_size = len(frame_paths) // effective_threads if effective_threads else len(frame_paths)
        if batch_size == 0:
            batch_size = 1
        batches = [frame_paths[i:i + batch_size] for i in range(0, len(frame_paths), batch_size)]
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=effective_threads) as executor:
            tasks = [
                loop.run_in_executor(executor, process_frame_batch, batch, i)
                for i, batch in enumerate(batches)
            ]
            total_batches = len(tasks)
            processed_batches = 0
            for i, task in enumerate(asyncio.as_completed(tasks)):
                batch_results = await task
                upscaled_paths.extend([p for p in batch_results if p])
                processed_batches += 1
                progress = 30 + (processed_batches / total_batches) * 50
                process_manager.update_process(
                    process_id,
                    progress=progress,
                    current_step=f"Upscaling batch {processed_batches}/{total_batches}"
                )
        logger.info(f"[upscale_frames] Nombre total de frames upscalées: {len(upscaled_paths)}")
        if not upscaled_paths:
            logger.error("[upscale_frames] Aucun frame upscalé généré.")
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
        ffmpeg_path = settings.ASSETS_DIR / "ffmpeg" / "ffmpeg.exe"
        if not ffmpeg_path.exists():
            logger.error("[encode_video] FFmpeg non trouvé")
            raise ProcessingError("FFmpeg non trouvé")
        if not frame_paths:
            logger.error("[encode_video] Aucun frame à encoder pour la vidéo")
            raise ProcessingError("Aucune frame upscalée à encoder pour la vidéo.")

        frames_list_path = os.path.join(Path(frame_paths[0]).parent, "frames_list.txt")
        temp_video_path = output_path.replace(Path(output_path).suffix, "_temp.mp4")
        logger.info(f"[encode_video] frames_list_path={frames_list_path}")
        try:
            with open(frames_list_path, 'w', encoding='utf-8') as f:
                for frame_path in frame_paths:
                    abs_path = os.path.abspath(frame_path).replace("\\", "/")
                    f.write(f"file '{abs_path}'\n")
            logger.info(f"[encode_video] Premieres lignes frames_list.txt: " +
                "".join([f"file '{os.path.abspath(frame_paths[i]).replace(chr(92), '/')}'\n" for i in range(min(5, len(frame_paths)))]))

            codec_map = {"x264": "libx264", "x265": "libx265"}
            actual_codec = codec_map.get(codec, codec)
            cmd_video = [
                str(ffmpeg_path),
                "-y", "-loglevel", "info",
                "-f", "concat", "-safe", "0",
                "-r", str(video_info['fps']),
                "-i", frames_list_path,
                "-c:v", actual_codec,
                "-vf", "scale=in_range=full:out_range=limited,format=yuv420p",
                "-color_range", "tv",
                "-b:v", "12000k",
                temp_video_path
            ]
            logger.info(f"[encode_video] Commande FFmpeg (vidéo): {' '.join(cmd_video)}")

            process = await asyncio.create_subprocess_exec(
                *cmd_video,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"[encode_video] FFmpeg erreur vidéo: {stderr.decode(errors='replace')}")
                raise ProcessingError("Erreur lors de l'encodage vidéo")
            logger.info(f"[encode_video] Vidéo sans audio encodée avec succès: {temp_video_path}")

            cmd_audio = [
                str(ffmpeg_path),
                "-y", "-loglevel", "info",
                "-i", original_video_path,
                "-i", temp_video_path,
                "-c:v", "copy",
                "-map", "1:v:0",
                "-map", "0:a?",
                "-c:a", "copy",
                output_path
            ]
            logger.info(f"[encode_video] Commande FFmpeg (audio): {' '.join(cmd_audio)}")
            process = await asyncio.create_subprocess_exec(
                *cmd_audio,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"[encode_video] FFmpeg erreur audio: {stderr.decode(errors='replace')}")
                raise ProcessingError("Erreur lors de l'encodage vidéo (audio)")
            logger.info(f"[encode_video] Vidéo finale encodée avec succès: {output_path}")

            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(frames_list_path):
                os.remove(frames_list_path)

            process_manager.update_process(
                process_id,
                progress=100.0,
                current_step="Vidéo encodée et finalisée"
            )

        except Exception as e:
            logger.error(f"[encode_video] Exception: {e}")
            for temp_file in [temp_video_path, frames_list_path]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception as ex:
                        logger.warning(f"[encode_video] Impossible de supprimer {temp_file} : {ex}")
            raise ProcessingError(f"Erreur encodage: {e}")

    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        return process_manager.get_process_status(process_id)

    def cancel_process(self, process_id: str) -> bool:
        return process_manager.cancel_process(process_id)

    def cleanup_process(self, process_id: str) -> None:
        process_manager.cleanup_process(process_id)