# utils/thresholds.py
# -*- coding: utf-8 -*-
try:
    from utils.debug_logger import log_event
except Exception as e:
    def log_event(msg, level="info"):
        try:
            print(f"[{level.upper()}] {msg}")
        except Exception:
            pass

import json, os, time

def carregar_thresholds(path="logs/walkforward_summary.json", default_tau=0.50, default_delta=0.10):
    """Carrega tau/delta do summary. Prefere *_meta se existir. Aplica clamp em delta."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        tau = float(j.get("tau_star_meta", j.get("tau_star", default_tau)))
        delta = float(j.get("delta_star_meta", j.get("delta_star", default_delta)))
        # clamp delta mínimo para evitar neutro crônico
        if delta < 0.02:
            log_event(f"[THR] delta<{0.02} ajustado para 0.05", level="warning")
            delta = 0.05
        log_event(f"[THR] ok | tau={tau:.3f} | delta={delta:.3f}", level="info")
        return tau, delta
    except Exception as e:
        log_event(f"[THR] fallback | usando default tau={default_tau} delta={default_delta} | erro={e}", level="warning")
        return default_tau, default_delta

def arquivo_stale(path="logs/walkforward_summary.json", dias=7):
    """Retorna True se o arquivo está mais velho que N dias."""
    try:
        mtime = os.path.getmtime(path)
        stale = (time.time() - mtime) > (dias*86400)
        if stale:
            log_event(f"[THR] stale | {path} com mais de {dias} dias", level="warning")
        return stale
    except Exception:
        return True

def carregar_tau_por_regime(path="logs/walkforward_summary.json"):
    """Retorna dict regime->tau, se existir no summary."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        return j.get("tau_by_regime", None)
    except Exception as e:
        log_event(f"[THR] tau_by_regime indisponível: {e}", level="warning")
        return None

def carregar_pesos_ensemble_por_regime(path="logs/walkforward_summary.json"):
    """Retorna dict regime->{model_key: weight}, se existir."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        # Aceita várias chaves comuns
        for k in ("ensemble_weights_by_regime", "weights_by_regime"):
            if k in j:
                return j[k]
        return None
    except Exception as e:
        log_event(f"[THR] weights_by_regime indisponível: {e}", level="warning")
        return None

# ========================= ADITIVOS COMPATÍVEIS =========================

def obter_tau_delta(cfg: dict | None = None,
                    summary_path: str = "logs/walkforward_summary.json",
                    default_tau: float = 0.50,
                    default_delta: float = 0.10) -> tuple[float, float, str]:
    """
    Wrapper seguro: respeita config ('usar_thresholds_walkforward', 'stale_thresholds_dias')
    e cai para defaults/config em caso de arquivo obsoleto.
    Retorna (tau, delta, fonte) onde fonte ∈ {'walkforward','config','default'}.
    """
    cfg = cfg or {}
    usar_wf = bool(cfg.get("usar_thresholds_walkforward", True))
    dias_stale = int(cfg.get("stale_thresholds_dias", 7))

    if usar_wf and not arquivo_stale(summary_path, dias=dias_stale):
        tau, delta = carregar_thresholds(summary_path, default_tau, default_delta)
        return float(tau), float(delta), "walkforward"

    # fallback para config se houver algo equivalente
    tau_cfg = cfg.get("ensemble_min_conf", None)
    delta_cfg = cfg.get("anti_neutro_margin", None)
    if tau_cfg is not None or delta_cfg is not None:
        tau = float(tau_cfg if tau_cfg is not None else default_tau)
        delta = float(delta_cfg if delta_cfg is not None else default_delta)
        if delta < 0.02:
            delta = 0.05
        log_event(f"[THR] usando config | tau={tau:.3f} delta={delta:.3f}", level="info")
        return tau, delta, "config"

    log_event(f"[THR] usando defaults | tau={default_tau:.3f} delta={default_delta:.3f}", level="warning")
    return float(default_tau), float(default_delta), "default"

def aplicar_cap_neutro(prob: dict[int, float], cap_neutro: float = 0.60) -> dict[int, float]:
    """
    Limita a probabilidade do neutro para evitar dominância.
    É compatível com dicts {-1: p_down, 0: p_neu, 1: p_up}.
    """
    p = dict(prob or {})
    try:
        if 0 in p:
            p[0] = min(max(float(p[0]), 0.0), float(cap_neutro))
    except Exception:
        pass
    return p
