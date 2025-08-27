
from __future__ import annotations
from typing import Dict, Any, Optional, List
import os, json
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DADOS_DIR = os.path.join(BASE_DIR, "dados")
os.makedirs(DADOS_DIR, exist_ok=True)
STATE_PATH = os.path.join(DADOS_DIR, "estado_execucao.json")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _load_raw() -> dict:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_raw(st: dict) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)

# -------- API pública --------
def carregar_estado() -> dict:
    return _load_raw()

def salvar_estado(st: dict) -> None:
    _save_raw(st)

def is_bloqueado(st: Optional[dict] = None) -> bool:
    st = st if isinstance(st, dict) else _load_raw()
    return bool(st.get("bloqueado"))

def tempo_restante_minutos(st: Optional[dict] = None) -> Optional[int]:
    st = st if isinstance(st, dict) else _load_raw()
    ate = st.get("blocked_until") or st.get("fim")
    if not ate:
        return None
    try:
        dt_ate = datetime.fromisoformat(ate)
        now = datetime.now(timezone.utc)
        delta = dt_ate - now
        return max(0, int(delta.total_seconds() // 60))
    except Exception:
        return None

def set_bloqueio(motivo: str, cooldown_min: int, tickets: Optional[list] = None, meta: Optional[dict] = None) -> dict:
    st = _load_raw()
    now_iso = _now_utc_iso()
    ate = (datetime.now(timezone.utc) + timedelta(minutes=int(cooldown_min or 0))).isoformat()
    st.update({
        "bloqueado": True,
        "motivo": motivo,
        "blocked_reason": motivo,
        "blocked_since": now_iso,
        "blocked_until": ate,
        "inicio": now_iso,
        "fim": ate,
        "tickets": list(tickets or []),
        "tickets_afetados": list(tickets or []),
        "meta": meta or {},
        "last_heartbeat": None,
    })
    _save_raw(st)
    return st

def limpar_bloqueio() -> dict:
    st = _load_raw()
    st.update({
        "bloqueado": False,
        "motivo": None,
        "blocked_reason": None,
        "inicio": None,
        "fim": None,
        "blocked_since": None,
        "blocked_until": None,
        "tickets": [],
        "tickets_afetados": [],
    })
    _save_raw(st)
    return st

def registrar_heartbeat() -> None:
    st = _load_raw()
    st["last_heartbeat"] = _now_utc_iso()
    _save_raw(st)

def deve_emitir_heartbeat(periodo_min: int = 60) -> bool:
    st = _load_raw()
    last = st.get("last_heartbeat")
    try:
        if not last:
            st["last_heartbeat"] = _now_utc_iso()
            _save_raw(st)
            return True
        last_dt = datetime.fromisoformat(last)
        delta = datetime.now(timezone.utc) - last_dt
        if delta.total_seconds() >= periodo_min * 60:
            st["last_heartbeat"] = _now_utc_iso()
            _save_raw(st)
            return True
        return False
    except Exception:
        st["last_heartbeat"] = _now_utc_iso()
        _save_raw(st)
        return True


def tick_estado() -> dict:
    """
    Retorna o estado atual sem alterar bloqueio, útil para heartbeat/telemetry.
    Se arquivo não existir, cria estrutura mínima.
    """
    st = _load_raw()
    if not isinstance(st, dict):
        st = {}
    # normaliza campos
    st.setdefault("bloqueado", False)
    st.setdefault("last_heartbeat", None)
    _save_raw(st)
    return st


def deve_notificar_cooldown(min_intervalo_min: int = 60) -> bool:
    """
    Debounce de notificações de cooldown/bloqueio via Telegram.
    Retorna True se já passou do intervalo mínimo desde a última notificação.
    """
    st = _load_raw()
    key = "last_cooldown_notify"
    last = st.get(key)
    now = datetime.now(timezone.utc)
    if not last:
        st[key] = now.isoformat()
        _save_raw(st)
        return True
    try:
        last_dt = datetime.fromisoformat(last)
    except Exception:
        st[key] = now.isoformat()
        _save_raw(st)
        return True
    delta = now - last_dt
    if delta.total_seconds() >= max(1, int(min_intervalo_min)) * 60:
        st[key] = now.isoformat()
        _save_raw(st)
        return True
    return False


def registrar_notificacao_cooldown() -> None:
    """Marca o instante da última notificação de cooldown."""
    st = _load_raw()
    st["last_cooldown_notify"] = datetime.now(timezone.utc).isoformat()
    _save_raw(st)
