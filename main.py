from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
from pathlib import Path

from api.routes import upscale, files
from core.config import settings
from core.exceptions import QualityShootException, create_http_exception
import sys
from api.routes import shutdown


# Patch pour PyInstaller --noconsole (empêche le crash logging)
if sys.stdout is None:
    import io
    sys.stdout = io.StringIO()
if sys.stderr is None:
    import io
    sys.stderr = io.StringIO()

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Créer l'application FastAPI
app = FastAPI(
    title="QualityShoot API",
    version="1.0.0",
    description="API pour l'upscaling d'images et vidéos avec IA"
)

# Configuration CORS pour Electron
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, limiter aux origines Electron
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir les fichiers statiques (pour les aperçus)
app.mount("/static", StaticFiles(directory=str(settings.TEMP_DIR)), name="static")

# Inclusion des routes
app.include_router(upscale.router, prefix="/api/upscale", tags=["upscale"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(shutdown.router)

# Gestionnaire d'exceptions global
@app.exception_handler(QualityShootException)
async def qualityshoot_exception_handler(request, exc: QualityShootException):
    return create_http_exception(exc, 400)

@app.get("/")
async def root():
    return {
        "message": "QualityShoot API is running!",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Vérification de santé de l'API"""
    try:
        # Vérifier que les répertoires existent
        directories_status = {
            "ai_models": settings.AI_MODELS_DIR.exists(),
            "temp": settings.TEMP_DIR.exists(),
            "assets": settings.ASSETS_DIR.exists()
        }
        
        # Vérifier les modèles disponibles
        available_models = []
        for model_name, model_info in settings.AI_MODELS.items():
            model_path = settings.AI_MODELS_DIR / model_info['file']
            if model_path.exists():
                available_models.append(model_name)
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "directories": directories_status,
            "available_models": available_models,
            "models_count": len(available_models)
        }
        
    except Exception as e:
        logger.error(f"Erreur health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Initialisation au démarrage
@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage de l'application"""
    logger.info("🚀 Démarrage de QualityShoot API")
    
    # Créer les répertoires nécessaires
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    settings.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    settings.AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Vérifier les modèles
    models_found = 0
    for model_name, model_info in settings.AI_MODELS.items():
        model_path = settings.AI_MODELS_DIR / model_info['file']
        if model_path.exists():
            models_found += 1
            logger.info(f"✅ Modèle trouvé: {model_name}")
        else:
            logger.warning(f"❌ Modèle manquant: {model_name} -> {model_path}")
    
    logger.info(f"📊 {models_found}/{len(settings.AI_MODELS)} modèles disponibles")
    logger.info("🎯 API prête!")

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt de l'application"""
    logger.info("🛑 Arrêt de QualityShoot API")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=settings.API_HOST, 
        port=settings.API_PORT, 
        log_level="info"
    )