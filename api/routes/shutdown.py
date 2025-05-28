from fastapi import APIRouter, Response
import os
import shutil
import sys

router = APIRouter()

@router.post("/shutdown-instant")
def shutdown_and_cleanup():
    # 1. Vider le dossier temp (attention à ne pas supprimer des fichiers critiques)
    temp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "temp")
    temp_dir = os.path.abspath(temp_dir)
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    # 2. Stopper le serveur immédiatement
    print("Shutdown requested. Exiting process.")
    sys.stdout.flush()
    os._exit(0)  # Quitte le process Python, sans attendre

    return Response(content="Shutdown initiated and temp cleaned.", status_code=200)