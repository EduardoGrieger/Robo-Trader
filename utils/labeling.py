# utils/labeling.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

try:
    from utils.debug_logger import log_event
except Exception:
    def log_event(msg, level="info"):
        print(f"[{level.upper()}] {msg}")

def has_ohlc(df: pd.DataFrame) -> bool:
    cols = set(c.lower() for c in df.columns)
    return "close" in cols

def _col(df: pd.DataFrame, name: str):
    for c in df.columns:
        if c.lower() == name.lower():
            return df[c]
    raise KeyError(name)

def _label_fixed_horizon(df: pd.DataFrame, tp_pips: float, sl_pips: float, janela: int, pip_factor: float) -> pd.Series:
    n = len(df)
    close = _col(df, "close").values.astype(float)
    high = _col(df, "high").values.astype(float) if "high".lower() in [c.lower() for c in df.columns] else None
    low  = _col(df, "low").values.astype(float)  if "low".lower()  in [c.lower() for c in df.columns] else None
    tp = tp_pips * float(pip_factor)
    sl = sl_pips * float(pip_factor)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        c0 = close[i]
        up = c0 + tp; dn = c0 - sl
        j_end = min(n, i + 1 + int(janela))
        if high is not None and low is not None:
            hit_up = np.any(high[i+1:j_end] >= up)
            hit_dn = np.any(low[i+1:j_end]  <= dn)
        else:
            seg = close[i+1:j_end]
            hit_up = np.any(seg >= up)
            hit_dn = np.any(seg <= dn)
        if hit_up and hit_dn:
            seg_up = np.argmax((high if high is not None else close)[i+1:j_end] >= up) if hit_up else janela+1
            seg_dn = np.argmax((low  if low  is not None else close)[i+1:j_end] <= dn) if hit_dn else janela+1
            y[i] = 1 if seg_up <= seg_dn else -1
        elif hit_up:
            y[i] = 1
        elif hit_dn:
            y[i] = -1
        else:
            y[i] = 0
    return pd.Series(y, index=df.index, name="sinal")

def relabel_profile(df: pd.DataFrame, profile: str = "B", pip_factor: float = 0.0001,
                    params_A=None, params_B=None) -> pd.DataFrame:
    if not has_ohlc(df):
        log_event("[LABEL] OHLC ausente; manter rótulos originais.", level="warning")
        return df.copy()
    params_A = params_A or {"tp_pips": 40, "sl_pips": 20, "janela": 20}
    params_B = params_B or {"tp_pips": 40, "sl_pips": 40, "janela": 12}
    profile = str(profile).upper().strip()
    if profile not in ("A", "B"): profile = "B"
    if profile == "A":
        if "sinal" in df.columns:
            return df.copy()
        params = params_A
    else:
        params = params_B
    y = _label_fixed_horizon(df, tp_pips=float(params["tp_pips"]), sl_pips=float(params["sl_pips"]),
                             janela=int(params["janela"]), pip_factor=pip_factor)
    out = df.copy(); out["sinal"] = y
    log_event(f"[LABEL] perfil={profile} | tp={params['tp_pips']} sl={params['sl_pips']} janela={params['janela']} | pip_factor={pip_factor}", "info")
    return out
