
from __future__ import annotations
from typing import Dict, Any, List
import os, json

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

from utils.estado_execucao import carregar_estado, tempo_restante_minutos
from utils.debug_logger import log_event

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

def _carregar_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _resumo_conta() -> dict:
    if mt5 is None:
        return {}
    try:
        ai = mt5.account_info()
        if ai is None:
            return {}
        return {
            "login": getattr(ai, "login", None),
            "balance": float(getattr(ai, "balance", 0.0) or 0.0),
            "equity": float(getattr(ai, "equity", 0.0) or 0.0),
            "margin": float(getattr(ai, "margin", 0.0) or 0.0),
            "margin_free": float(getattr(ai, "margin_free", 0.0) or 0.0),
            "margin_level": float(getattr(ai, "margin_level", 0.0) or 0.0),
            "leverage": int(getattr(ai, "leverage", 0) or 0),
            "currency": getattr(ai, "currency", None),
        }
    except Exception as e:
        log_event(f"[STATUS] Erro ao consultar account_info: {e}", level="warning")
        return {}

def _resumo_posicoes() -> tuple[list[dict], dict]:
    itens: List[dict] = []
    total = {"count": 0, "volume_lotes": 0.0, "pnl_aberto": 0.0}
    if mt5 is None:
        return itens, total
    try:
        poss = mt5.positions_get() or []
        por_symbol: dict[str, dict] = {}
        for p in poss:
            sym = getattr(p, "symbol", "") or ""
            d = por_symbol.setdefault(sym, {"symbol": sym, "count": 0, "volume_lotes": 0.0, "pnl_aberto": 0.0})
            d["count"] += 1
            d["volume_lotes"] += float(getattr(p, "volume", 0.0) or 0.0)
            d["pnl_aberto"] += float(getattr(p, "profit", 0.0) or 0.0)
        itens = list(por_symbol.values())
        for d in itens:
            total["count"] += d["count"]
            total["volume_lotes"] += d["volume_lotes"]
            total["pnl_aberto"] += d["pnl_aberto"]
    except Exception as e:
        log_event(f"[STATUS] Erro ao coletar posições: {e}", level="warning")
    return itens, total

def snapshot_status() -> Dict[str, Any]:
    cfg = _carregar_config()
    st = carregar_estado() or {}
    pos, tot = _resumo_posicoes()
    conta = _resumo_conta()
    blocked = bool(st.get("bloqueado"))
    restante = None
    if blocked:
        restante = tempo_restante_minutos(st) if callable(tempo_restante_minutos) else None

    snap = {
        "bloqueado": blocked,
        "bloqueio_motivo": st.get("motivo") or st.get("blocked_reason"),
        "blocked_since": st.get("blocked_since") or st.get("inicio"),
        "blocked_until": st.get("blocked_until") or st.get("fim"),
        "cooldown_rest_min": restante,
        "tickets_afetados": st.get("tickets") or st.get("tickets_afetados") or [],
        "posicoes": pos,
        "posicoes_total": tot,
        "conta": conta,
        "config_protecao": cfg.get("protecao", {}),
    }
    return snap

def formatar_status(snap: dict | None = None, compacto: bool = False) -> str:
    if snap is None:
        snap = snapshot_status()
    linhas: List[str] = []
    if snap.get("bloqueado"):
        rest = snap.get("cooldown_rest_min")
        rest_str = f" | resta ~{int(rest)} min" if rest is not None else ""
        linhas.append(f"⏸️ BLOQUEADO — motivo: {snap.get('bloqueio_motivo')}{rest_str}")
        # Tickets afetados (se houver)
        tickets = snap.get("tickets_afetados") or []
        if tickets:
            prev = ", ".join(str(t) for t in tickets[:20])
            if len(tickets) > 20:
                prev += "..."
            linhas.append(f"🎫 Tickets afetados: {prev}")
    else:
        linhas.append("✅ Livre para operar")

    tot = snap.get("posicoes_total", {})
    linhas.append(f"📈 Posições abertas: {tot.get('count', 0)} | Volume: {tot.get('volume_lotes', 0.0):.2f} lot | PnL aberto: {tot.get('pnl_aberto', 0.0):+.2f}")
    if not compacto:
        for it in snap.get("posicoes", [])[:10]:
            linhas.append(f"• {it['symbol']}: {it['count']} pos | {it['volume_lotes']:.2f} lot | PnL {it['pnl_aberto']:+.2f}")
    conta = snap.get("conta") or {}
    if conta:
        linhas.append(f"💼 Equity: {conta.get('equity')} | Margin: {conta.get('margin')} | Nível: {conta.get('margin_level')} | Alav.: {conta.get('leverage')}")
    return "\n".join(linhas)
