
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import os, json, time

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

from utils.debug_logger import log_event

# -------- Telegram safe sender --------
def _enviar_telegram_safe(msg: str) -> None:
    for mod_path in ("utils.telegram_utils", "utils.telegram_bot", "telegram_utils"):
        try:
            mod = __import__(mod_path, fromlist=["enviar_telegram"])
            enviar_telegram = getattr(mod, "enviar_telegram", None)
            if callable(enviar_telegram):
                try:
                    enviar_telegram(msg)
                except Exception:
                    pass
                return
        except Exception:
            continue
    try:
        from builtins import enviar_telegram  # type: ignore
        if callable(enviar_telegram):
            try:
                enviar_telegram(msg)  # type: ignore
            except Exception:
                pass
    except Exception:
        pass

# -------- Health state debounce store --------
_HEALTH_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "health_state.json")

def _load_health() -> dict:
    try:
        if os.path.exists(_HEALTH_PATH):
            with open(_HEALTH_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_health(d: dict) -> None:
    try:
        with open(_HEALTH_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _should_send(key: str, cooldown_min: int) -> bool:
    d = _load_health()
    last = float(d.get(key, 0.0) or 0.0)
    now = time.time()
    if now - last >= cooldown_min * 60.0:
        d[key] = now
        _save_health(d)
        return True
    return False

def _clear_flag(key: str) -> None:
    d = _load_health()
    if key in d:
        del d[key]
        _save_health(d)

# -------- Helpers --------
def _carregar_config() -> dict:
    base = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(base, "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _now() -> float:
    return time.time()

# -------- Monitores --------
def monitor_tick_silence(cfg: Optional[dict] = None) -> bool:
    """
    Verifica se algum símbolo ficou sem tick por muito tempo.
    Retorna True se emitiu alerta.
    """
    cfg = cfg or _carregar_config()
    hc = cfg.get("healthcheck", {}) or {}
    symbols = hc.get("symbols_check") or cfg.get("ativos") or []
    max_silence_sec = int(hc.get("max_tick_silencio_min", 5)) * 60
    cooldown_min = int(hc.get("telegram_cooldown_min", 10))

    if mt5 is None or not symbols:
        return False

    alerted = False
    now = _now()
    stale: List[str] = []
    for sym in symbols:
        try:
            tk = mt5.symbol_info_tick(sym)
            t = getattr(tk, "time", None) if tk else None
            if not t:
                # sem tick conhecido: considerar estagnado
                stale.append(sym)
                continue
            age = now - float(t)
            if age >= max_silence_sec:
                stale.append(sym)
        except Exception:
            stale.append(sym)

    if stale and _should_send("tick_silence", cooldown_min):
        msg = "🚨 *Healthcheck*: Sem tick há muito tempo em: " + ", ".join(stale) + f" (>{max_silence_sec//60} min)."
        try:
            _enviar_telegram_safe(msg)
        except Exception:
            pass
        log_event(f"[HEALTH] Tick silence em {stale}", level="warning")
        alerted = True
    elif not stale:
        # tudo bem — limpa flag para permitir próximo alerta se voltar a ocorrer
        _clear_flag("tick_silence")
    return alerted

def monitor_order_queue(cfg: Optional[dict] = None) -> bool:
    """
    Verifica ordens pendentes presas por muito tempo.
    Retorna True se emitiu alerta.
    """
    cfg = cfg or _carregar_config()
    hc = cfg.get("healthcheck", {}) or {}
    max_pending_sec = int(hc.get("pending_order_timeout_sec", 60))
    cooldown_min = int(hc.get("telegram_cooldown_min", 10))

    if mt5 is None:
        return False

    try:
        ords = mt5.orders_get() or []
    except Exception:
        ords = []

    stuck: List[int] = []
    now = _now()
    for o in ords:
        try:
            t0 = getattr(o, "time_setup", None)
            if t0 is None:
                t0 = getattr(o, "time_done", None)
            if t0 is None:
                # sem timestamp — considera stuck se estiver pendente
                stuck.append(int(getattr(o, "ticket", 0) or 0))
                continue
            age = now - float(t0)
            if age >= max_pending_sec:
                stuck.append(int(getattr(o, "ticket", 0) or 0))
        except Exception:
            pass

    if stuck and _should_send("pending_orders", cooldown_min):
        msg = f"🚨 *Healthcheck*: Ordens pendentes presas há >{max_pending_sec}s: " + ", ".join(map(str, stuck[:20]))
        try:
            _enviar_telegram_safe(msg)
        except Exception:
            pass
        log_event(f"[HEALTH] Pending orders stuck: {stuck}", level="warning")
        return True
    elif not stuck:
        _clear_flag("pending_orders")
    return False

def run_healthcheck_once(cfg: Optional[dict] = None) -> None:
    cfg = cfg or _carregar_config()
    try:
        monitor_tick_silence(cfg)
    except Exception as e:
        log_event(f"[HEALTH] monitor_tick_silence falhou: {e}", level="warning")
    try:
        monitor_order_queue(cfg)
    except Exception as e:
        log_event(f"[HEALTH] monitor_order_queue falhou: {e}", level="warning")
