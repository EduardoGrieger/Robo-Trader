# inteligencia/inferencia_contextual.py
import numpy as np
import pandas as pd
from datetime import time
from utils.utils import carregar_config
from utils.debug_logger import log_event

def _pick_volume_col(df: pd.DataFrame):
    for c in ("tick_volume", "volume"):
        if c in df.columns and df[c].notna().any():
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

def prever_contexto(candles, ativo: str = "EURUSD") -> dict:
    """
    Prevê o contexto de mercado a partir dos candles.
    Retorna: horario, volatilidade, volume, tendencia, squeeze.
    Lê chaves novas (infer_volatility_*, infer_volume_*) e mantém fallback p/ antigas (infer_vol_*).
    """
    contexto = {
        "horario": "indefinido",
        "volatilidade": "indefinida",
        "volume": "indefinido",
        "tendencia": "indefinida",
        "squeeze": False
    }

    try:
        if candles is None or not hasattr(candles, "columns"):
            return contexto

        cfg = carregar_config()
        # Janelas (canônico + retrocompat)
        MIN   = int(_cfg_get(cfg, ["contexto_min_candles"], 60))
        W     = int(_cfg_get(cfg, ["infer_volatility_window", "contexto_vol_window"], 20))
        WREF  = int(_cfg_get(cfg, ["infer_volatility_ref_window", "contexto_vol_ref_window"], 200))
        WVOL  = int(_cfg_get(cfg, ["infer_volume_window", "contexto_volume_window"], 20))
        USE_LOG = bool(_cfg_get(cfg, ["regime_use_logret"], False))

        # Volatilidade — nomes canônicos + fallback
        VOL_HI = float(_cfg_get(cfg, ["infer_volatility_high_ratio", "infer_vol_high_ratio"], 1.5))
        VOL_MD = float(_cfg_get(cfg, ["infer_volatility_med_ratio",  "infer_vol_med_ratio"],  0.9))

        # Volume — já padronizado
        VOLR_HI = float(_cfg_get(cfg, ["infer_volume_high_ratio"], 1.5))
        VOLR_MD = float(_cfg_get(cfg, ["infer_volume_med_ratio"], 1.05))

        FLAT = float(_cfg_get(cfg, ["trend_flat_threshold"], 0.0007))
        BB_BW_FACTOR = float(_cfg_get(cfg, ["squeeze_bandwidth_factor"], 0.005))

        # Requisitos mínimos
        if len(candles) < max(MIN, W + 1):
            return contexto
        if "close" not in candles.columns or "timestamp" not in candles.columns:
            return contexto

        df = candles.copy()
        ts = _to_utc_series(df["timestamp"])
        if ts.isna().all():
            return contexto

        # Horário (UTC) → manha / tarde / noite / madrugada
        try:
            hora = ts.iloc[-1].time()
            if time(6, 0) <= hora < time(12, 0):
                contexto["horario"] = "manha"
            elif time(12, 0) <= hora < time(18, 0):
                contexto["horario"] = "tarde"
            elif time(18, 0) <= hora <= time(23, 59, 59, 999999):
                contexto["horario"] = "noite"
            else:
                contexto["horario"] = "madrugada"
        except Exception:
            contexto["horario"] = "indefinido"

        # Fechamentos sanitizados
        closes = pd.to_numeric(df["close"], errors="coerce").dropna()
        closes = closes.tail(max(WREF + 1, W + 1, 35)).astype(float)
        if len(closes) < (W + 1):
            return contexto

        # Volatilidade (retornos % ou log)
        if USE_LOG:
            rets = np.diff(np.log(closes.values))
        else:
            arr = closes.values
            prev = arr[:-1]
            rets = (arr[1:] - prev) / np.where(prev == 0, 1e-12, prev)
        rets[~np.isfinite(rets)] = 0.0

        std_curto = float(np.std(rets[-W:], ddof=1)) if len(rets) >= W else float(np.std(rets, ddof=1))
        if len(rets) >= max(WREF, W):
            std_roll = pd.Series(rets[-WREF:]).rolling(window=W).std(ddof=1)
            std_ref_mediana = float(std_roll.median(skipna=True))
        else:
            std_ref_mediana = float(pd.Series(rets).rolling(window=W).std(ddof=1).median(skipna=True))

        ratio_vol = (std_curto / std_ref_mediana) if (std_ref_mediana and np.isfinite(std_ref_mediana) and std_ref_mediana > 0) else 1.0
        if ratio_vol > VOL_HI:
            contexto["volatilidade"] = "alta"
        elif ratio_vol > VOL_MD:
            contexto["volatilidade"] = "media"
        else:
            contexto["volatilidade"] = "baixa"

        # Volume (razão último vs média WVOL)
        vol_col = _pick_volume_col(df)
        if vol_col:
            vols = pd.to_numeric(df[vol_col], errors="coerce").dropna().tail(WVOL + 1).astype(float)
            if len(vols) >= 2:
                v_last = float(vols.iloc[-1])
                v_mean = float(vols.iloc[:-1].mean()) if len(vols) > 1 else float(vols.mean())
                ratio_v = (v_last / v_mean) if v_mean > 0 else 1.0
                if ratio_v > VOLR_HI:
                    contexto["volume"] = "alto"
                elif ratio_v > VOLR_MD:
                    contexto["volume"] = "medio"
                else:
                    contexto["volume"] = "baixo"

        # Tendência (delta % na janela)
        first = float(closes.iloc[0]); last = float(closes.iloc[-1])
        delta_pct = (last - first) / (abs(first) if first != 0 else 1e-12)
        if abs(delta_pct) < FLAT:
            contexto["tendencia"] = "lateral"
        elif delta_pct > 0:
            contexto["tendencia"] = "alta"
        else:
            contexto["tendencia"] = "baixa"

        # Squeeze (bandwidth relativa de Bollinger)
        if "bb_high" in df.columns and "bb_low" in df.columns:
            bbh = pd.to_numeric(df["bb_high"], errors="coerce").tail(W)
            bbl = pd.to_numeric(df["bb_low"], errors="coerce").tail(W)
            c_last = last
            if len(bbh) > 0 and len(bbl) > 0 and np.isfinite(c_last) and c_last != 0 and pd.notna(bbh.iloc[-1]) and pd.notna(bbl.iloc[-1]):
                bw = float((bbh.iloc[-1] - bbl.iloc[-1]) / abs(c_last))
                contexto["squeeze"] = bool(bw < BB_BW_FACTOR)
            else:
                contexto["squeeze"] = False
        else:
            s = closes.rolling(window=W, min_periods=W).mean()
            st = closes.rolling(window=W, min_periods=W).std(ddof=1)
            if not pd.isna(s.iloc[-1]) and not pd.isna(st.iloc[-1]) and last != 0:
                upper = float(s.iloc[-1] + 2 * st.iloc[-1])
                lower = float(s.iloc[-1] - 2 * st.iloc[-1])
                bw = (upper - lower) / abs(last)
                contexto["squeeze"] = bool(bw < BB_BW_FACTOR)
            else:
                contexto["squeeze"] = False

        log_event(
            f"[INFERÊNCIA CONTEXTUAL] ativo={ativo} horario={contexto['horario']} "
            f"vol={contexto['volatilidade']} vol_ratio={ratio_vol:.3f} "
            f"volume={contexto['volume']} trend={contexto['tendencia']} squeeze={contexto['squeeze']}",
            level="info",
            modulo="inferencia_contextual"
        )
        return contexto

    except Exception as e:
        log_event(f"[INFERÊNCIA CONTEXTUAL] Erro: {e}", level="error", modulo="inferencia_contextual")
        return contexto
