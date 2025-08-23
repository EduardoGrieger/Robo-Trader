# utils/monitor_delay_execucao.py
import os
from datetime import datetime
import pandas as pd
from utils.debug_logger import log_event

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "monitor_delay_execucao.csv")

def to_datetime_naive(val):
    """
    Converte qualquer valor para datetime tz-naive (sem timezone) de modo robusto.
    """
    # Pandas Timestamp?
    if isinstance(val, pd.Timestamp):
        return val.tz_convert(None).to_pydatetime() if val.tzinfo else val.to_pydatetime()
    if isinstance(val, datetime):
        return val.replace(tzinfo=None)  # Remove qualquer timezone
    if isinstance(val, (int, float)):
        if val > 1e12:
            val = val / 1000
        return datetime.utcfromtimestamp(val)
    if isinstance(val, str):
        try:
            return pd.to_datetime(val, utc=True).tz_convert(None).to_pydatetime()
        except Exception:
            log_event(f"[MONITOR DELAY] Erro conversão de timestamp: {val}", level="error")
            return None
    return None

def registrar_delay_execucao(timestamp_inicio_candle, timestamp_execucao_ordem, ativo, ciclo, info_extra=""):
    """
    Salva no CSV o delay entre o início do candle e a execução da ordem (sempre UTC).
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    t1 = to_datetime_naive(timestamp_inicio_candle)
    t2 = to_datetime_naive(timestamp_execucao_ordem)

    log_event(f"[MONITOR DELAY] RAW t1={timestamp_inicio_candle} ({t1}), t2={timestamp_execucao_ordem} ({t2})", level="debug")

    delay = float('nan')
    if t1 is not None and t2 is not None:
        delay = (t2 - t1).total_seconds()
        # Corrige se delay negativo (entrada invertida ou erro de timezone)
        if delay < 0 or delay > 1200:
            log_event(f"[MONITOR DELAY] Delay estranho detectado: {delay} seg. Ajustando para nan.", level="warning")
            delay = float('nan')
    else:
        log_event(f"[MONITOR DELAY] Falha conversão dos timestamps para delay (t1={t1}, t2={t2})", level="error")

    linha = f"{t1.isoformat() if t1 else 'nan'},{t2.isoformat() if t2 else 'nan'},{delay if not (delay != delay) else 'nan'},{ativo},{ciclo},{info_extra}\n"
    arquivo_novo = not os.path.exists(LOG_PATH)

    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            if arquivo_novo:
                f.write("inicio_candle,execucao_ordem,delay_segundos,ativo,ciclo,info\n")
            f.write(linha)
        log_event(f"[MONITOR DELAY] Delay de execução registrado para {ativo} (ciclo {ciclo}): {delay} seg (início={t1}, execução={t2})", level="info")
    except Exception as e:
        log_event(f"[MONITOR DELAY] Erro ao registrar delay: {e}", level="error")
        return None

    return delay

# Alias para uso no main_loop
log_delay_execucao = registrar_delay_execucao
