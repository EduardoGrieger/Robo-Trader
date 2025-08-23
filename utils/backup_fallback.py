import os
import shutil
from datetime import datetime
from utils.debug_logger import log_event

def backup_modelo(modelo_path, backup_dir="backup_modelo"):
    """
    Salva backup do modelo IA com timestamp.
    """
    try:
        os.makedirs(backup_dir, exist_ok=True)
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        nome_backup = f"{os.path.basename(modelo_path)}_{now}.bak"
        destino = os.path.join(backup_dir, nome_backup)
        shutil.copy2(modelo_path, destino)
        log_event(f"[BACKUP] Backup do modelo IA salvo em: {destino}", level="info")
        return destino
    except Exception as e:
        log_event(f"[BACKUP] Falha ao criar backup do modelo: {e}", level="error")
        return None

def backup_banco(banco_path, backup_dir="backup_banco"):
    """
    Salva backup do banco de dados com timestamp.
    """
    try:
        os.makedirs(backup_dir, exist_ok=True)
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        nome_backup = f"{os.path.basename(banco_path)}_{now}.bak"
        destino = os.path.join(backup_dir, nome_backup)
        shutil.copy2(banco_path, destino)
        log_event(f"[BACKUP] Backup do banco de dados salvo em: {destino}", level="info")
        return destino
    except Exception as e:
        log_event(f"[BACKUP] Falha ao criar backup do banco: {e}", level="error")
        return None

def restaurar_backup(origem, destino):
    """
    Restaura backup (modelo/banco) em caso de erro crítico.
    """
    try:
        shutil.copy2(origem, destino)
        log_event(f"[BACKUP] Backup restaurado de {origem} para {destino}", level="warning")
    except Exception as e:
        log_event(f"[BACKUP] Falha ao restaurar backup: {e}", level="error")
