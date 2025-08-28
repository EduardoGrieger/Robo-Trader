# utils/time_utils.py
from __future__ import annotations
from typing import Tuple, Optional, Any
import time as _time
from datetime import datetime, timedelta, timezone

# =========================
# EXISTENTE (mantido)
# =========================
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

# =========================
# NOVO (aditivo)
# =========================
def _to_utc_dt(val: Any) -> Optional[datetime]:
    """
    Converte 'val' em datetime **aware UTC** de forma robusta.
    Aceita: pandas.Timestamp, datetime (naive/aware), int/float epoch (s ou ms), str ISO.
    """
    # Importa pandas apenas se existir (não quebra se faltar)
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore

    # pandas.Timestamp
    if pd is not None and isinstance(val, pd.Timestamp):  # type: ignore
        # Se vier tz-naive, assume UTC; se vier com tz, converte para UTC
        return (val.tz_localize("UTC") if val.tzinfo is None else val.tz_convert("UTC")).to_pydatetime()

    # datetime
    if isinstance(val, datetime):
        return val.replace(tzinfo=timezone.utc) if val.tzinfo is None else val.astimezone(timezone.utc)

    # numérico (epoch)
    if isinstance(val, (int, float)):
        x = float(val)
        if x > 1e12:  # epoch em ms
            x = x / 1000.0
        try:
            return datetime.fromtimestamp(x, tz=timezone.utc)
        except Exception:
            return None

    # string
    if isinstance(val, str):
        if pd is not None:
            try:
                ts = pd.to_datetime(val, utc=True, errors="coerce")  # type: ignore
                if ts is not None and not (getattr(ts, "tzinfo", None) is None and getattr(ts, "tz", None) is None):
                    return ts.to_pydatetime()
                # se não reconheceu, tenta como float
            except Exception:
                pass
        # fallback: tenta como epoch em string
        try:
            x = float(val)
            if x > 1e12:
                x = x / 1000.0
            return datetime.fromtimestamp(x, tz=timezone.utc)
        except Exception:
            return None

    return None

def delay_execucao_seg(candle_close: Any, exec_time: Any = None) -> float:
    """
    Retorna (exec_time - candle_close) em segundos, **sempre >= 0**.
    - Se 'exec_time' não for informado, usa agora_utc.
    - Valores aceitos: Timestamp/datetime/epoch(s|ms)/string ISO.
    """
    inicio_utc = _to_utc_dt(candle_close)
    fim_utc = _to_utc_dt(exec_time) if exec_time is not None else datetime.now(timezone.utc)
    if inicio_utc is None or fim_utc is None:
        return float("nan")
    diff = (fim_utc - inicio_utc).total_seconds()
    # clamp não-negativo (evita delay negativo por timezone/inversão de args)
    return diff if diff >= 0 else 0.0
