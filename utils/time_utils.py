
from __future__ import annotations
from typing import Tuple, Optional
import time as _time
from datetime import datetime, timedelta, timezone

def _as_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def agora_utc_ts() -> float:
    return _time.time()

def _tz_from_offset_horas(offset_horas: float) -> timezone:
    # offset em horas (pode ser fracionário, ex.: -3.0)
    segundos = int(round(offset_horas * 3600))
    return timezone(timedelta(seconds=segundos))

def inicio_fim_dia_servidor_utc(ts_utc: Optional[float], offset_horas: float) -> Tuple[datetime, datetime]:
    """
    Dado um timestamp UTC e um offset de timezone do broker (em horas),
    retorna (start_utc, end_utc) do dia do broker em UTC.
    - Dia é de 00:00:00 a 24:00:00 no timezone do broker.
    """
    if ts_utc is None:
        ts_utc = agora_utc_ts()
    tz_srv = _tz_from_offset_horas(offset_horas)
    dt_utc = datetime.fromtimestamp(ts_utc, timezone.utc)
    # Converter para hora do servidor
    dt_srv = dt_utc.astimezone(tz_srv)
    d0 = dt_srv.replace(hour=0, minute=0, second=0, microsecond=0)
    d1 = d0 + timedelta(days=1)
    # Voltar para UTC
    start_utc = d0.astimezone(timezone.utc)
    end_utc = d1.astimezone(timezone.utc)
    return start_utc, end_utc

def to_datetime_utc(ts: float) -> datetime:
    return datetime.fromtimestamp(float(ts), timezone.utc)

def seconds_until_end_of_broker_day(ts_utc: Optional[float], offset_horas: float) -> int:
    start_utc, end_utc = inicio_fim_dia_servidor_utc(ts_utc, offset_horas)
    now = datetime.now(timezone.utc)
    delta = end_utc - now
    return max(0, int(delta.total_seconds()))
