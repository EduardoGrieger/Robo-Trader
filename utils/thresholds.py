try:
    from utils.debug_logger import log_event
except Exception as e:
    def log_event(msg, level="info"):
        try:
            print(f"[{level.upper()}] {msg}")
        except Exception as e:
            pass


# utils/thresholds.py
# -*- coding: utf-8 -*-
import json, os, time

def carregar_thresholds(path="logs/walkforward_summary.json", default_tau=0.50, default_delta=0.10):
    """
    Lê tau_star e delta_star do resumo do walk-forward.
    """
    try:
        log_event(f"[THR] load | tentando carregar {path}", level="info")
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        tau = float(j.get("tau_star", default_tau)); delta = float(j.get("delta_star", default_delta))
        log_event(f"[THR] ok | tau={tau:.3f} | delta={delta:.3f}", level="info")
        return tau, delta
    except Exception as e:
        log_event(f"[THR] fallback | usando default tau={default_tau} delta={default_delta} | erro={e}", level="warning")
        return default_tau, default_delta

def arquivo_stale(path="logs/walkforward_summary.json", dias=7):
    """
    Retorna True se o arquivo está mais velho que N dias.
    """
    try:
        mtime = os.path.getmtime(path)
        stale = (time.time() - mtime) > (dias*86400)
        if stale:
            log_event(f"[THR] stale | {path} com mais de {dias} dias", level="warning")
        return stale
    except Exception:
        return True
