import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

def image_read(file_path: str) -> np.ndarray:
    """Lit une image depuis un fichier"""
    with open(file_path, 'rb') as file:
        file_data = np.frombuffer(file.read(), np.uint8)
        return cv2.imdecode(file_data, cv2.IMREAD_UNCHANGED)

def image_write(file_path: str, image: np.ndarray, extension: str = ".jpg") -> None:
    """Écrit une image dans un fichier"""
    success, encoded_img = cv2.imencode(extension, image)
    if success:
        with open(file_path, 'wb') as file:
            file.write(encoded_img.tobytes())
    else:
        raise Exception(f"Erreur lors de l'encodage de l'image: {file_path}")

def get_image_resolution(image: np.ndarray) -> Tuple[int, int]:
    """Retourne la résolution de l'image (hauteur, largeur)"""
    return image.shape[0], image.shape[1]

def get_image_mode(image: np.ndarray) -> str:
    """Détermine le mode de l'image"""
    if len(image.shape) == 2:
        return "Grayscale"
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            return "RGB"
        elif image.shape[2] == 4:
            return "RGBA"
    return "Unknown"

def normalize_image(image: np.ndarray) -> Tuple[np.ndarray, int]:
    """Normalise une image entre 0 et 1"""
    max_val = np.max(image)
    if max_val > 256:
        range_val = 65535
    else:
        range_val = 255
    
    normalized = image / range_val
    return normalized, range_val

def denormalize_image(image: np.ndarray, max_range: int) -> np.ndarray:
    """Dénormalise une image"""
    if max_range == 255:
        return (image * max_range).astype(np.uint8)
    else:
        return (image * max_range).round().astype(np.float32)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Prétraite une image pour l'inférence ONNX"""
    # Transposer (H, W, C) -> (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    # Ajouter dimension batch (C, H, W) -> (1, C, H, W)
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_image(output: np.ndarray) -> np.ndarray:
    """Post-traite la sortie ONNX"""
    # Supprimer dimension batch (1, C, H, W) -> (C, H, W)
    output = np.squeeze(output, axis=0)
    # Clipper les valeurs
    output = np.clip(output, 0, 1)
    # Transposer (C, H, W) -> (H, W, C)
    output = np.transpose(output, (1, 2, 0))
    return output

def blend_images(
    original: np.ndarray,
    upscaled: np.ndarray,
    blend_factor: float
) -> np.ndarray:
    """Mélange l'image originale avec l'image upscalée"""
    if blend_factor <= 0:
        return upscaled
    
    # S'assurer que les images ont la même taille
    if original.shape != upscaled.shape:
        original_height, original_width = get_image_resolution(original)
        upscaled_height, upscaled_width = get_image_resolution(upscaled)
        
        if original_height * original_width > upscaled_height * upscaled_width:
            # Redimensionner l'original vers la taille upscalée
            original = cv2.resize(original, (upscaled_width, upscaled_height), interpolation=cv2.INTER_AREA)
        else:
            # Redimensionner l'original vers la taille upscalée
            original = cv2.resize(original, (upscaled_width, upscaled_height))
    
    # Gérer les canaux alpha si nécessaire
    if len(original.shape) == 3 and original.shape[2] == 4:
        if len(upscaled.shape) == 3 and upscaled.shape[2] == 3:
            # Ajouter canal alpha à l'image upscalée
            alpha = np.full((upscaled.shape[0], upscaled.shape[1], 1), 255, dtype=np.uint8)
            upscaled = np.concatenate((upscaled, alpha), axis=2)
    
    try:
        # Mélange pondéré
        upscale_weight = 1 - blend_factor
        blended = cv2.addWeighted(original, blend_factor, upscaled, upscale_weight, 0)
        return blended
    except:
        # En cas d'erreur, retourner l'image upscalée
        return upscaled

def get_video_info(video_path: str) -> dict:
    """Extrait les informations d'une vidéo"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Impossible d'ouvrir la vidéo: {video_path}")
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': 0
    }
    
    if info['fps'] > 0:
        info['duration'] = info['frame_count'] / info['fps']
    
    cap.release()
    return info

def extract_video_frame(video_path: str, frame_number: int = 0) -> Optional[np.ndarray]:
    """Extrait une frame spécifique d'une vidéo"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    return None