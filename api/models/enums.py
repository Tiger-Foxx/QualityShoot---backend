from enum import Enum

class AIModel(Enum):
    """Modèles d'IA disponibles pour l'upscaling"""
    REALESR_GX4 = "RealESR_Gx4"
    REALESR_ANIMEX4 = "RealESR_Animex4"
    REALESR_GX2 = "RealESR_Gx2"
    REALESR_ANIME6B = "RealESR_Anime6B"
    ESRGAN_X4 = "ESRGAN_x4"
    
    @property
    def scale_factor(self) -> int:
        """Facteur d'agrandissement du modèle"""
        scale_mapping = {
            self.REALESR_GX4: 4,
            self.REALESR_ANIMEX4: 4,
            self.REALESR_GX2: 2,
            self.REALESR_ANIME6B: 4,
            self.ESRGAN_X4: 4
        }
        return scale_mapping.get(self, 4)
    
    @property
    def description(self) -> str:
        """Description du modèle"""
        descriptions = {
            self.REALESR_GX4: "Modèle généraliste haute qualité pour photos réelles",
            self.REALESR_ANIMEX4: "Optimisé pour les animations et dessins",
            self.REALESR_GX2: "Agrandissement x2 pour traitement rapide",
            self.REALESR_ANIME6B: "Modèle anime avancé avec plus de paramètres",
            self.ESRGAN_X4: "Modèle ESRGAN classique"
        }
        return descriptions.get(self, "Modèle d'upscaling IA")

class GPU(Enum):
    """Options de GPU"""
    AUTO = "Auto"
    CPU = "CPU"
    CUDA_0 = "cuda:0"
    CUDA_1 = "cuda:1"
    
    @property
    def description(self) -> str:
        descriptions = {
            self.AUTO: "Détection automatique du meilleur dispositif",
            self.CPU: "Utiliser uniquement le processeur",
            self.CUDA_0: "GPU NVIDIA #1",
            self.CUDA_1: "GPU NVIDIA #2"
        }
        return descriptions.get(self, "Dispositif de calcul")

class ProcessStatus(Enum):
    """Statuts de processus"""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    ERROR = "error"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    
    @property
    def description(self) -> str:
        descriptions = {
            self.PENDING: "En attente de traitement",
            self.PROCESSING: "Traitement en cours",
            self.COMPLETED: "Terminé avec succès",
            self.COMPLETED_WITH_ERRORS: "Terminé avec des erreurs",
            self.ERROR: "Erreur lors du traitement",
            self.CANCELLED: "Annulé par l'utilisateur",
            self.PAUSED: "Mis en pause"
        }
        return descriptions.get(self, "Statut inconnu")

class BlendingMode(Enum):
    """Modes de blending pour le post-traitement"""
    OFF = "OFF"
    LINEAR = "LINEAR"
    ADAPTIVE = "ADAPTIVE"
    
    @property
    def description(self) -> str:
        descriptions = {
            self.OFF: "Pas de blending",
            self.LINEAR: "Blending linéaire avec l'image originale",
            self.ADAPTIVE: "Blending adaptatif basé sur le contenu"
        }
        return descriptions.get(self, "Mode de blending")

class VideoCodec(Enum):
    """Codecs vidéo supportés"""
    X264 = "x264"
    X265 = "x265"
    VP9 = "vp9"
    AV1 = "av1"
    
    @property
    def description(self) -> str:
        descriptions = {
            self.X264: "H.264/AVC - Compatible universellement",
            self.X265: "H.265/HEVC - Meilleure compression",
            self.VP9: "VP9 - Open source, bonne qualité",
            self.AV1: "AV1 - Dernière génération, excellente compression"
        }
        return descriptions.get(self, "Codec vidéo")

class FileType(Enum):
    """Types de fichiers supportés"""
    IMAGE = "image"
    VIDEO = "video"
    
    @property
    def extensions(self) -> list:
        """Extensions de fichiers pour ce type"""
        if self == self.IMAGE:
            return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        elif self == self.VIDEO:
            return ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.gif']
        return []

class QualityPreset(Enum):
    """Presets de qualité prédéfinis"""
    FAST = "fast"
    BALANCED = "balanced" 
    QUALITY = "quality"
    MAXIMUM = "maximum"
    
    @property
    def settings(self) -> dict:
        """Paramètres associés au preset"""
        settings_map = {
            self.FAST: {
                "input_resize": 0.8,
                "output_resize": 1.0,
                "vram_limit": 2.0,
                "tile_size": 256
            },
            self.BALANCED: {
                "input_resize": 1.0,
                "output_resize": 1.0, 
                "vram_limit": 4.0,
                "tile_size": 512
            },
            self.QUALITY: {
                "input_resize": 1.0,
                "output_resize": 1.2,
                "vram_limit": 6.0,
                "tile_size": 768
            },
            self.MAXIMUM: {
                "input_resize": 1.0,
                "output_resize": 1.5,
                "vram_limit": 8.0,
                "tile_size": 1024
            }
        }
        return settings_map.get(self, settings_map[self.BALANCED])

class LogLevel(Enum):
    """Niveaux de logging"""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class SystemResource(Enum):
    """Ressources système à monitorer"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    VRAM = "vram"
    DISK = "disk"
    
    @property
    def unit(self) -> str:
        """Unité de mesure"""
        units = {
            self.CPU: "%",
            self.MEMORY: "GB",
            self.GPU: "%", 
            self.VRAM: "GB",
            self.DISK: "GB"
        }
        return units.get(self, "")