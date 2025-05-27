import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
from threading import Lock
from api.models.responses import ProcessStatus

logger = logging.getLogger(__name__)

class ProcessManager:
    """Gestionnaire global des processus d'upscaling"""
    
    def __init__(self):
        self.processes: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()
        
    def create_process(
        self, 
        process_type: str = "upscale",
        total_files: int = 0,
        **kwargs
    ) -> str:
        """Crée un nouveau processus et retourne son ID"""
        process_id = str(uuid.uuid4())
        
        with self.lock:
            self.processes[process_id] = {
                'process_id': process_id,
                'process_type': process_type,
                'status': ProcessStatus.PENDING,
                'progress': 0.0,
                'current_file': None,
                'current_step': 'Initializing...',
                'total_files': total_files,
                'completed_files': [],
                'failed_files': [],
                'error_message': None,
                'estimated_time_remaining': None,
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'completed_at': None,
                **kwargs
            }
        
        logger.info(f"✅ Processus créé: {process_id}")
        return process_id
    
    def update_process(
        self, 
        process_id: str, 
        **updates
    ) -> bool:
        """Met à jour un processus"""
        with self.lock:
            if process_id not in self.processes:
                logger.warning(f"❌ Processus introuvable: {process_id}")
                return False
            
            self.processes[process_id].update(updates)
            self.processes[process_id]['updated_at'] = datetime.now()
            
            # Marquer comme terminé si status final
            if updates.get('status') in [
                ProcessStatus.COMPLETED, 
                ProcessStatus.ERROR, 
                ProcessStatus.CANCELLED
            ]:
                self.processes[process_id]['completed_at'] = datetime.now()
            
            return True
    
    def get_process(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un processus par son ID"""
        with self.lock:
            return self.processes.get(process_id)
    
    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Récupère le statut d'un processus"""
        process = self.get_process(process_id)
        if not process:
            return None
            
        return {
            'process_id': process['process_id'],
            'status': process['status'],
            'progress': process['progress'],
            'current_file': process.get('current_file'),
            'current_step': process.get('current_step'),
            'estimated_time_remaining': process.get('estimated_time_remaining'),
            'error_message': process.get('error_message'),
            'completed_files': process.get('completed_files', []),
            'failed_files': process.get('failed_files', []),
            'total_files': process.get('total_files', 0)
        }
    
    def cancel_process(self, process_id: str) -> bool:
        """Annule un processus"""
        return self.update_process(
            process_id, 
            status=ProcessStatus.CANCELLED,
            current_step='Cancelled by user'
        )
    
    def cleanup_process(self, process_id: str) -> bool:
        """Supprime un processus (après traitement)"""
        with self.lock:
            if process_id in self.processes:
                del self.processes[process_id]
                logger.info(f"🗑️ Processus nettoyé: {process_id}")
                return True
            return False
    
    def cleanup_old_processes(self, max_age_hours: int = 24) -> int:
        """Nettoie les anciens processus"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        with self.lock:
            to_remove = [
                pid for pid, process in self.processes.items()
                if process.get('completed_at') and process['completed_at'] < cutoff_time
            ]
            
            for pid in to_remove:
                del self.processes[pid]
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"🗑️ {cleaned_count} anciens processus nettoyés")
        
        return cleaned_count
    
    def get_all_processes(self) -> List[Dict[str, Any]]:
        """Récupère tous les processus actifs"""
        with self.lock:
            return list(self.processes.values())
    
    def add_completed_file(self, process_id: str, file_path: str) -> bool:
        """Ajoute un fichier à la liste des fichiers terminés"""
        with self.lock:
            if process_id not in self.processes:
                return False
            
            if 'completed_files' not in self.processes[process_id]:
                self.processes[process_id]['completed_files'] = []
            
            if file_path not in self.processes[process_id]['completed_files']:
                self.processes[process_id]['completed_files'].append(file_path)
            
            return True
    
    def add_failed_file(self, process_id: str, file_path: str) -> bool:
        """Ajoute un fichier à la liste des fichiers échoués"""
        with self.lock:
            if process_id not in self.processes:
                return False
            
            if 'failed_files' not in self.processes[process_id]:
                self.processes[process_id]['failed_files'] = []
            
            if file_path not in self.processes[process_id]['failed_files']:
                self.processes[process_id]['failed_files'].append(file_path)
            
            return True

# Instance globale
process_manager = ProcessManager()