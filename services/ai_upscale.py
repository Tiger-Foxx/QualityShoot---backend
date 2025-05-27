import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional
from onnxruntime import InferenceSession
import logging

from pathlib import Path

from core.config import settings
from core.exceptions import ModelNotFoundError, ProcessingError, InsufficientVRAMError
from utils.image_utils import (
    normalize_image, denormalize_image, preprocess_image, 
    postprocess_image, get_image_resolution, get_image_mode
)

logger = logging.getLogger(__name__)

class AIUpscaler:
    def __init__(
        self,
        model_name: str,
        gpu_device: str = "Auto",
        input_resize_factor: float = 1.0,
        output_resize_factor: float = 1.0,
        max_resolution: int = 2048
    ):
        self.model_name = model_name
        self.gpu_device = gpu_device
        self.input_resize_factor = input_resize_factor
        self.output_resize_factor = output_resize_factor
        self.max_resolution = max_resolution
        
        # Charger les informations du modèle
        if model_name not in settings.AI_MODELS:
            raise ModelNotFoundError(f"Modèle {model_name} non trouvé")
        
        self.model_info = settings.AI_MODELS[model_name]
        self.upscale_factor = self.model_info['scale']
        self.model_path = settings.AI_MODELS_DIR / self.model_info['file']
        
        if not self.model_path.exists():
            raise ModelNotFoundError(f"Fichier modèle {self.model_path} non trouvé")
        
        # Charger le modèle ONNX
        self.inference_session = self._load_inference_session()
        
    def _load_inference_session(self) -> InferenceSession:
        """Charge la session d'inférence ONNX"""
        try:
            providers = ['DmlExecutionProvider']
            
            # Configuration GPU
            provider_options = self._get_provider_options()
            
            session = InferenceSession(
                str(self.model_path),
                providers=providers,
                provider_options=provider_options
            )
            
            logger.info(f"Modèle {self.model_name} chargé avec succès")
            return session
            
        except Exception as e:
            raise ProcessingError(f"Erreur lors du chargement du modèle: {e}")
    
    def _get_provider_options(self) -> List[dict]:
        """Configure les options du provider GPU"""
        if self.gpu_device == 'Auto':
            return [{"performance_preference": "high_performance"}]
        elif self.gpu_device.startswith('GPU'):
            gpu_id = int(self.gpu_device.split()[-1]) - 1
            return [{"device_id": str(gpu_id)}]
        else:
            return [{"performance_preference": "high_performance"}]
    
    def resize_with_factor(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Redimensionne une image avec un facteur donné"""
        if factor == 1.0:
            return image
            
        height, width = get_image_resolution(image)
        new_width = int(width * factor)
        new_height = int(height * factor)
        
        # S'assurer que les dimensions sont paires
        new_width = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1
        
        if factor > 1:
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def needs_tiling(self, image: np.ndarray) -> bool:
        """Vérifie si l'image nécessite un découpage en tuiles"""
        height, width = get_image_resolution(image)
        image_pixels = height * width
        max_pixels = self.max_resolution * self.max_resolution
        return image_pixels > max_pixels
    
    def calculate_tiles(self, image: np.ndarray) -> Tuple[int, int]:
        """Calcule le nombre de tuiles nécessaires"""
        height, width = get_image_resolution(image)
        tiles_x = (width + self.max_resolution - 1) // self.max_resolution
        tiles_y = (height + self.max_resolution - 1) // self.max_resolution
        return tiles_x, tiles_y
    
    def split_into_tiles(self, image: np.ndarray, tiles_x: int, tiles_y: int) -> List[np.ndarray]:
        """Découpe l'image en tuiles"""
        height, width = get_image_resolution(image)
        tile_width = width // tiles_x
        tile_height = height // tiles_y
        
        tiles = []
        for y in range(tiles_y):
            y_start = y * tile_height
            y_end = (y + 1) * tile_height
            
            for x in range(tiles_x):
                x_start = x * tile_width
                x_end = (x + 1) * tile_width
                tile = image[y_start:y_end, x_start:x_end]
                tiles.append(tile)
        
        return tiles
    
    def combine_tiles(self, tiles: List[np.ndarray], tiles_x: int, original_shape: tuple) -> np.ndarray:
        """Recombine les tuiles en image complète"""
        # Calculer les dimensions de l'image finale
        tile_height, tile_width = get_image_resolution(tiles[0])
        total_height = tile_height * (len(tiles) // tiles_x)
        total_width = tile_width * tiles_x
        
        # Déterminer le nombre de canaux
        if len(original_shape) == 3:
            channels = original_shape[2]
            combined = np.zeros((total_height, total_width, channels), dtype=np.uint8)
        else:
            combined = np.zeros((total_height, total_width), dtype=np.uint8)
        
        for i, tile in enumerate(tiles):
            row = i // tiles_x
            col = i % tiles_x
            y_start = row * tile_height
            y_end = y_start + tile_height
            x_start = col * tile_width
            x_end = x_start + tile_width
            
            if len(original_shape) == 3:
                combined[y_start:y_end, x_start:x_end] = tile
            else:
                combined[y_start:y_end, x_start:x_end] = tile
        
        return combined
    
    def upscale_single_image(self, image: np.ndarray) -> np.ndarray:
        """Upscale une seule image sans découpage"""
        try:
            # Normalisation et préparation
            image_mode = get_image_mode(image)
            image_float = image.astype(np.float32)
            normalized_image, max_range = normalize_image(image_float)
            
            if image_mode == "RGB":
                # Traitement RGB standard
                preprocessed = preprocess_image(normalized_image)
                onnx_input = {self.inference_session.get_inputs()[0].name: preprocessed}
                onnx_output = self.inference_session.run(None, onnx_input)[0]
                postprocessed = postprocess_image(onnx_output)
                result = denormalize_image(postprocessed, max_range)
                
            elif image_mode == "RGBA":
                # Traitement RGBA (séparer alpha)
                alpha_channel = normalized_image[:, :, 3]
                rgb_image = normalized_image[:, :, :3]
                
                # Traiter RGB
                preprocessed_rgb = preprocess_image(rgb_image)
                onnx_input = {self.inference_session.get_inputs()[0].name: preprocessed_rgb}
                onnx_output_rgb = self.inference_session.run(None, onnx_input)[0]
                postprocessed_rgb = postprocess_image(onnx_output_rgb)
                
                # Traiter Alpha
                alpha_3ch = np.repeat(alpha_channel[:, :, np.newaxis], 3, axis=2)
                preprocessed_alpha = preprocess_image(alpha_3ch)
                onnx_input_alpha = {self.inference_session.get_inputs()[0].name: preprocessed_alpha}
                onnx_output_alpha = self.inference_session.run(None, onnx_input_alpha)[0]
                postprocessed_alpha = postprocess_image(onnx_output_alpha)
                alpha_result = cv2.cvtColor(postprocessed_alpha, cv2.COLOR_RGB2GRAY)
                
                # Combiner RGB + Alpha
                result_rgba = np.zeros((*postprocessed_rgb.shape[:2], 4), dtype=np.float32)
                result_rgba[:, :, :3] = postprocessed_rgb
                result_rgba[:, :, 3] = alpha_result
                result = denormalize_image(result_rgba, max_range)
                
            elif image_mode == "Grayscale":
                # Convertir en RGB pour le traitement
                rgb_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)
                preprocessed = preprocess_image(rgb_image)
                onnx_input = {self.inference_session.get_inputs()[0].name: preprocessed}
                onnx_output = self.inference_session.run(None, onnx_input)[0]
                postprocessed = postprocess_image(onnx_output)
                # Reconvertir en grayscale
                result = cv2.cvtColor(postprocessed, cv2.COLOR_RGB2GRAY)
                result = denormalize_image(result, max_range)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            raise ProcessingError(f"Erreur lors de l'upscaling: {e}")
    
    def upscale_with_tiling(self, image: np.ndarray) -> np.ndarray:
        """Upscale une image avec découpage en tuiles"""
        try:
            tiles_x, tiles_y = self.calculate_tiles(image)
            tiles = self.split_into_tiles(image, tiles_x, tiles_y)
            
            # Upscaler chaque tuile
            upscaled_tiles = []
            for tile in tiles:
                upscaled_tile = self.upscale_single_image(tile)
                upscaled_tiles.append(upscaled_tile)
            
            # Recombiner les tuiles
            target_height = image.shape[0] * self.upscale_factor
            target_width = image.shape[1] * self.upscale_factor
            
            return self.combine_tiles(upscaled_tiles, tiles_x, image.shape)
            
        except Exception as e:
            raise ProcessingError(f"Erreur lors de l'upscaling avec tuiles: {e}")
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Point d'entrée principal pour traiter une image"""
        try:
            # Redimensionnement d'entrée
            resized_input = self.resize_with_factor(image, self.input_resize_factor)
            
            # Upscaling avec ou sans tuiles
            if self.needs_tiling(resized_input):
                logger.info("Utilisation du mode tuiles pour l'upscaling")
                upscaled = self.upscale_with_tiling(resized_input)
            else:
                upscaled = self.upscale_single_image(resized_input)
            
            # Redimensionnement de sortie
            final_result = self.resize_with_factor(upscaled, self.output_resize_factor)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'image: {e}")
            raise ProcessingError(f"Erreur de traitement: {e}")

class MultiThreadAIUpscaler:
    """Gestionnaire multi-thread pour l'upscaling"""
    
    def __init__(self, num_threads: int = 1, **upscaler_kwargs):
        self.num_threads = num_threads
        self.upscalers = [
            AIUpscaler(**upscaler_kwargs) 
            for _ in range(num_threads)
        ]
    
    def get_upscaler(self, thread_id: int = 0) -> AIUpscaler:
        """Retourne un upscaler pour un thread donné"""
        return self.upscalers[thread_id % len(self.upscalers)]
    
    def calculate_frames_capacity(self, sample_frame_path: str) -> int:
        """Calcule combien de frames peuvent être traitées simultanément"""
        try:
            sample_image = cv2.imread(sample_frame_path, cv2.IMREAD_UNCHANGED)
            upscaler = self.upscalers[0]
            resized_sample = upscaler.resize_with_factor(sample_image, upscaler.input_resize_factor)
            
            height, width = get_image_resolution(resized_sample)
            frame_pixels = height * width
            max_pixels = upscaler.max_resolution * upscaler.max_resolution
            
            capacity = max_pixels // frame_pixels
            return max(1, min(capacity, self.num_threads))
            
        except Exception as e:
            logger.warning(f"Erreur calcul capacité: {e}")
            return 1