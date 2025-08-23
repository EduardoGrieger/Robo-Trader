import os
import time
from datetime import datetime, timedelta
from utils.debug_logger import log_event
from gestao.fechar_todas_ordens import fechar_todas_ordens
from comunicacao.telegram_alertas import enviar_telegram

ARQUIVO_STATUS = "dados/ultimo_ciclo.txt"
TEMPO_MAXIMO_MIN = 10   # Tempo máximo sem rodar ciclo (em minutos)

def registrar_ciclo():
    """
    Salve este método no main_loop ao final de cada ciclo!
    """
    with open(ARQUIVO_STATUS, "w") as f:
        f.write(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))

def monitorar_inatividade():
    """
    Deve rodar em processo/thread separada, monitora se o ciclo travou.
    Se sim, fecha todas ordens e alerta.
    """
    while True:
        try:
            if not os.path.exists(ARQUIVO_STATUS):
                time.sleep(60)
                continue
            with open(ARQUIVO_STATUS, "r") as f:
                ultima = f.read().strip()
            ultima_dt = datetime.strptime(ultima, "%Y-%m-%d %H:%M:%S")
            agora = datetime.utcnow()
            minutos = (agora - ultima_dt).total_seconds() / 60
            if minutos > TEMPO_MAXIMO_MIN:
                msg = f"⚠️ Robô parado há {minutos:.1f} min. Fechando todas as ordens como proteção!"
                log_event(msg, level="error")
                try:
                    enviar_telegram(msg)
                except Exception as e:
                    log_event(f"[INATIVIDADE] Falha ao enviar alerta: {e}", level="warning")
                fechar_todas_ordens(motivo="inatividade")
                # Atualize o timestamp para não repetir ação!
                with open(ARQUIVO_STATUS, "w") as f:
                    f.write(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
            time.sleep(60)
        except Exception as e:
            log_event(f"[INATIVIDADE] Monitor falhou: {e}", level="error")
            time.sleep(60)
