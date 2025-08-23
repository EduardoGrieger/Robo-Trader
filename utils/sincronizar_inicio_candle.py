import time
from datetime import datetime, timedelta, timezone

def proximo_horario_candle(timeframe="M1"):
    """
    Retorna o datetime do início do próximo candle, em UTC.
    Ex: timeframe="M5" retorna o horário do próximo candle de 5 minutos.
    """
    tf = timeframe.upper()
    agora = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    if tf == "M1":
        return agora + timedelta(minutes=1)
    elif tf == "M5":
        minuto = ((agora.minute // 5) + 1) * 5
        return agora.replace(minute=0) + timedelta(minutes=minuto)
    elif tf == "M15":
        minuto = ((agora.minute // 15) + 1) * 15
        return agora.replace(minute=0) + timedelta(minutes=minuto)
    elif tf == "M30":
        minuto = ((agora.minute // 30) + 1) * 30
        return agora.replace(minute=0) + timedelta(minutes=minuto)
    elif tf == "H1":
        return agora.replace(minute=0) + timedelta(hours=1)
    elif tf == "H4":
        hora = ((agora.hour // 4) + 1) * 4
        return agora.replace(hour=0, minute=0) + timedelta(hours=hora)
    elif tf == "D1":
        return agora.replace(hour=0, minute=0) + timedelta(days=1)
    else:
        # Default para 1 minuto
        return agora + timedelta(minutes=1)

def aguardar_inicio_novo_candle(timeframe="M1"):
    """
    Aguarda até o início do próximo candle do timeframe especificado (em UTC).
    Loga o tempo restante para o próximo candle.
    """
    from utils.debug_logger import log_event

    prox = proximo_horario_candle(timeframe)
    agora = datetime.now(timezone.utc)
    segundos = max(0, (prox - agora).total_seconds())
    log_event(f"[SYNC CANDLE] Aguardando {segundos:.1f} seg para novo candle {timeframe} (UTC={prox})")
    if segundos > 0:
        time.sleep(segundos)
    else:
        log_event(f"[SYNC CANDLE] Executando imediatamente novo candle {timeframe} (UTC={prox})")

# Teste isolado
if __name__ == "__main__":
    print("Aguardando novo candle...")
    aguardar_inicio_novo_candle("M5")
    print("Novo candle iniciado!")
