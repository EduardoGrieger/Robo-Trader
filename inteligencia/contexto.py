# inteligencia/contexto.py
import pandas as pd
import numpy as np
from utils.debug_logger import log_event
from utils.utils import carregar_config

def _pick_timestamp_column(df: pd.DataFrame):
    for c in ("timestamp", "datahora", "time"):
        if c in df.columns:
            return c
    return None

def _to_utc_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def _cfg_get(cfg: dict, keys, default):
    """Retorna o primeiro campo existente em 'keys'; caso nenhum exista, usa default."""
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default

def analisar_contexto(df: pd.DataFrame):
    """
    Analisa o contexto do mercado a partir do DataFrame de candles e retorna um dicionário
    com: sessão, volatilidade (alta/baixa), volume_atipico (bool), squeeze (bool) e métricas úteis.
    Sempre robusto: nunca quebra e sempre retorna um dicionário válido.
    """
    try:
        if df is None or not hasattr(df, "columns"):
            log_event("[CONTEXTO] DataFrame inválido ou None — usando contexto neutro.", level="warning")
            return {"sessao": "indefinida", "volatilidade": "indefinida", "volume_atipico": False, "squeeze": False}

        cfg = carregar_config()
        # Janelas com nomes canônicos + retrocompat
        MIN   = int(_cfg_get(cfg, ["contexto_min_candles"], 100))
        W     = int(_cfg_get(cfg, ["infer_volatility_window", "contexto_vol_window"], 20))
        WREF  = int(_cfg_get(cfg, ["infer_volatility_ref_window", "contexto_vol_ref_window"], 200))
        WVOL  = int(_cfg_get(cfg, ["infer_volume_window", "contexto_volume_window"], 20))
        # Multiplicador de volume (canônico + fallback antigo)
        VOL_MULT = float(_cfg_get(cfg, ["infer_volume_multiplier", "contexto_volume_multiplier"], 1.5))
        BB_BW_FACTOR = float(_cfg_get(cfg, ["squeeze_bandwidth_factor"], 0.005))

        if len(df) < max(MIN, WREF, W, WVOL):
            log_event(f"[CONTEXTO] DataFrame insuficiente (<{max(MIN, WREF, W, WVOL)} candles)", level="warning")
            return {"sessao": "indefinida", "volatilidade": "indefinida", "volume_atipico": False, "squeeze": False}

        # ---------- timestamps ----------
        ts_col = _pick_timestamp_column(df)
        if ts_col is None:
            log_event("[CONTEXTO] Coluna de tempo não encontrada (timestamp/datahora/time).", level="warning")
            hora_utc = -1
        else:
            ts = _to_utc_series(df[ts_col])
            hora_utc = int(ts.iloc[-1].hour) if pd.notna(ts.iloc[-1]) else -1

        # ---------- sessão ----------
        if 7 <= hora_utc < 15:
            sessao = "londres"
        elif 15 <= hora_utc < 22:
            sessao = "nova_york"
        elif 0 <= hora_utc < 7:
            sessao = "asia"
        else:
            sessao = "fora_horario"

        # ---------- preços ----------
        if "close" not in df.columns:
            log_event("[CONTEXTO] Coluna 'close' ausente.", level="warning")
            return {"sessao": sessao, "volatilidade": "indefinida", "volume_atipico": False, "squeeze": False}
        close = pd.to_numeric(df["close"], errors="coerce")
        if close.tail(WREF).isna().all():
            log_event("[CONTEXTO] Série de 'close' inválida (NaN).", level="warning")
            return {"sessao": sessao, "volatilidade": "indefinida", "volume_atipico": False, "squeeze": False}

        # ---------- volatilidade ----------
        std_20_series = close.rolling(window=W).std()
        std_20 = float(std_20_series.iloc[-1])
        std_ref_median = float(close.rolling(window=WREF).std().median())
        volatilidade = "alta" if (not np.isnan(std_20) and not np.isnan(std_ref_median) and std_20 > std_ref_median) else "baixa"

        # ---------- volume atípico ----------
        volume_atipico = False
        vol_col = None
        for c in ("volume", "tick_volume"):
            if c in df.columns:
                vol_col = c
                break
        if vol_col is not None:
            vol = pd.to_numeric(df[vol_col], errors="coerce")
            media_vol = vol.rolling(window=WVOL).mean().iloc[-1]
            if pd.notna(media_vol) and pd.notna(vol.iloc[-1]):
                volume_atipico = bool(vol.iloc[-1] > VOL_MULT * media_vol)

        # ---------- squeeze (bandwidth de Bollinger) ----------
        sma = close.rolling(window=W).mean()
        std = std_20_series
        upper = sma + 2 * std
        lower = sma - 2 * std
        if pd.isna(upper.iloc[-1]) or pd.isna(lower.iloc[-1]) or pd.isna(close.iloc[-1]) or close.iloc[-1] == 0:
            squeeze = False
            bb_bw = None
        else:
            bb_bw = float((upper.iloc[-1] - lower.iloc[-1]) / abs(close.iloc[-1]))
            squeeze = bb_bw < BB_BW_FACTOR

        contexto = {
            "sessao": sessao,
            "volatilidade": volatilidade,
            "volume_atipico": volume_atipico,
            "squeeze": squeeze,
            "bb_bandwidth": bb_bw,
            "std20": std_20,
            "std_ref_median": std_ref_median
        }

        log_event(f"[CONTEXTO] {contexto}", level="info")
        return contexto

    except Exception as e:
        log_event(f"[CONTEXTO] Erro ao analisar contexto: {e}", level="error")
        return {"sessao": "erro", "volatilidade": "erro", "volume_atipico": False, "squeeze": False}
