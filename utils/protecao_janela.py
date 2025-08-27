
from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import os, json, time

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

from utils.estado_execucao import set_bloqueio, carregar_estado
from utils.debug_logger import log_event

# ----------------- Utilidades de envio -----------------
def _enviar_telegram_safe(msg: str) -> None:
    # Tenta achar uma função enviar_telegram no projeto
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
    # último recurso: se existir uma global no escopo do main_loop
    try:
        from builtins import enviar_telegram  # type: ignore
        if callable(enviar_telegram):
            try:
                enviar_telegram(msg)  # type: ignore
            except Exception:
                pass
    except Exception:
        pass

# ----------------- Snapshot de posições -----------------
def _snapshot_posicoes_mt5():
    itens = []
    total = {"count": 0, "volume_lotes": 0.0, "pnl_aberto": 0.0}
    if mt5 is None:
        return itens, total
    try:
        poss = mt5.positions_get() or []
        for p in poss:
            d = {
                "ticket": int(getattr(p, "ticket", 0) or 0),
                "symbol": getattr(p, "symbol", "") or "",
                "volume": float(getattr(p, "volume", 0.0) or 0.0),
                "profit": float(getattr(p, "profit", 0.0) or 0.0),
                "type": int(getattr(p, "type", 0) or 0),
            }
            itens.append(d)
            total["count"] += 1
            total["volume_lotes"] += d["volume"]
            total["pnl_aberto"] += d["profit"]
    except Exception as e:
        log_event(f"[PROTECAO] Falha ao capturar snapshot de posições: {e}", level="warning")
    return itens, total

# ----------------- Fechamento de posições -----------------
def _fechar_todas_posicoes_fallback() -> Tuple[bool, List[int]]:
    """Fecha posições diretamente via MT5 quando utilitário do projeto não estiver disponível."""
    tickets: List[int] = []
    if mt5 is None:
        return False, tickets
    try:
        poss = mt5.positions_get() or []
        for p in poss:
            symbol = getattr(p, "symbol", "")
            vol = float(getattr(p, "volume", 0.0) or 0.0)
            pos_type = int(getattr(p, "type", 0) or 0)  # 0 buy, 1 sell
            ti = mt5.symbol_info_tick(symbol)
            if ti is None or vol <= 0.0:
                continue
            if pos_type == 0:  # buy -> close by SELL
                price = ti.bid
                order_type = mt5.ORDER_TYPE_SELL
            else:  # sell -> close by BUY
                price = ti.ask
                order_type = mt5.ORDER_TYPE_BUY
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": vol,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "position": int(getattr(p, "ticket", 0) or 0),
                "comment": "proteção_pnl",
            }
            r = mt5.order_send(req)
            if r is not None and getattr(r, "retcode", 0) in (getattr(mt5, "TRADE_RETCODE_DONE", 10009), getattr(mt5, "TRADE_RETCODE_PLACED", 10008)):
                tickets.append(int(getattr(p, "ticket", 0) or 0))
        # sucesso se não há mais posições
        rem = mt5.positions_get() or []
        ok = len(rem) == 0
        return ok, tickets
    except Exception as e:
        log_event(f"[PROTECAO] Falha no fechamento fallback: {e}", level="error")
        return False, tickets

def _fechar_todas_posicoes_retry(max_tentativas: int = 5, espera_s: float = 2.0) -> tuple[bool, list[int], dict]:
    """Fecha todas posições com até N tentativas. Retorna (ok, tickets_fechados, falhas_por_ticket)."""
    tickets_ok: List[int] = []
    falhas: Dict[int, str] = {}

    if mt5 is None:
        return False, tickets_ok, falhas

    for tentativa in range(1, max_tentativas + 1):
        try:
            poss = mt5.positions_get() or []
        except Exception as e:
            log_event(f"[PROTECAO] positions_get falhou: {e}", level="error")
            break

        if not poss:
            return True, tickets_ok, falhas

        for p in poss:
            try:
                ticket = int(getattr(p, "ticket", 0) or 0)
                symbol = getattr(p, "symbol", "")
                vol = float(getattr(p, "volume", 0.0) or 0.0)
                pos_type = int(getattr(p, "type", 0) or 0)  # 0 buy, 1 sell
                ti = mt5.symbol_info_tick(symbol)
                if ti is None or vol <= 0.0:
                    falhas[ticket] = "tick/volume inválido"
                    continue
                if pos_type == 0:
                    price = ti.bid
                    order_type = mt5.ORDER_TYPE_SELL
                else:
                    price = ti.ask
                    order_type = mt5.ORDER_TYPE_BUY
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": vol,
                    "type": order_type,
                    "price": price,
                    "deviation": 20,
                    "position": ticket,
                    "comment": f"proteção_pnl_retry_t{tentativa}",
                }
                r = mt5.order_send(req)
                rc = getattr(r, "retcode", None) if r is not None else None
                if rc in (getattr(mt5, "TRADE_RETCODE_DONE", 10009), getattr(mt5, "TRADE_RETCODE_PLACED", 10008)):
                    tickets_ok.append(ticket)
                    if ticket in falhas:
                        del falhas[ticket]
                else:
                    falhas[ticket] = f"retcode={rc}"
            except Exception as e:
                falhas[int(getattr(p, "ticket", 0) or 0)] = f"exceção: {e}"

        # verifica novamente
        try:
            rem = mt5.positions_get() or []
            if not rem:
                return True, tickets_ok, falhas
        except Exception:
            pass

        time.sleep(max(0.5, espera_s))

    # terminou tentativas, verifica estado
    try:
        rem = mt5.positions_get() or []
        if not rem:
            return True, tickets_ok, falhas
    except Exception:
        pass
    return False, tickets_ok, falhas

def _fechar_todas_posicoes() -> Tuple[bool, List[int]]:
    # Prioriza utilitário do projeto, se existir
    try:
        from mt5.ordens import fechar_todas_posicoes  # type: ignore
        res = fechar_todas_posicoes()
        # aceitar (ok, tickets) ou apenas lista de tickets
        if isinstance(res, tuple) and len(res) == 2:
            return bool(res[0]), list(res[1] or [])
        elif isinstance(res, list):
            return True, list(res)
        else:
            # se retornou algo diferente, checa posições restantes
            if mt5 is not None:
                rem = mt5.positions_get() or []
                return len(rem) == 0, []
            return False, []
    except Exception:
        pass
    # fallback
    return _fechar_todas_posicoes_fallback()

# ----------------- Config -----------------
def _carregar_config() -> dict:
    base = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(base, "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_event(f"[PROTECAO] Falha ao carregar config: {e}", level="warning")
        return {}

# ----------------- Formatação -----------------
def _formatar_msg_bloqueio(snapshot_total: dict, gatilho: float, cooldown_min: int, tickets: Optional[List[int]] = None) -> str:
    tickets = tickets or []
    linhas = []
    linhas.append("🛑 PROTEÇÃO ACIONADA (PnL aberto)")
    try:
        linhas.append(f"PnL aberto somado: {snapshot_total.get('pnl_aberto', 0.0):+.2f}  |  Gatilho: {gatilho:+.2f}")
    except Exception:
        pass
    try:
        linhas.append(f"Fechamento forçado de posições. Cooldown: {cooldown_min} min")
    except Exception:
        pass
    if tickets:
        try:
            preview = ", ".join(str(t) for t in tickets[:20])
            linhas.append(f"Tickets afetados: {preview}{'...' if len(tickets)>20 else ''}")
        except Exception:
            pass
    return "\n".join(linhas)

# ----------------- Buffer de PnL (janela deslizante) -----------------
_PNLBUF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "protecao_pnl_buffer.json")

def _load_pnlbuf() -> list:
    try:
        if os.path.exists(_PNLBUF_PATH):
            with open(_PNLBUF_PATH, "r", encoding="utf-8") as f:
                return json.load(f) or []
    except Exception:
        pass
    return []

def _save_pnlbuf(buf: list) -> None:
    try:
        with open(_PNLBUF_PATH, "w", encoding="utf-8") as f:
            json.dump(buf, f, ensure_ascii=False)
    except Exception:
        pass

def _push_pnl_sample(valor: float) -> None:
    buf = _load_pnlbuf()
    now = time.time()
    buf.append([now, float(valor)])
    # mantem no max 24h pra segurança
    cutoff = now - 24*3600
    buf = [x for x in buf if x and x[0] >= cutoff]
    _save_pnlbuf(buf)

def _window_below_threshold(gatilho: float, janela_min: int, ratio_ok: float = 0.6) -> bool:
    """True se, na janela 'janela_min', >= ratio_ok das amostras ficaram <= gatilho."""
    buf = _load_pnlbuf()
    if not buf:
        return False
    now = time.time()
    win_from = now - max(60, janela_min * 60)
    win_vals = [v for (t, v) in buf if t >= win_from]
    if not win_vals:
        return False
    # Cobertura mínima: 50% da janela
    # (evita disparar com poucas amostras após restart)
    min_cov = 0.5 * (janela_min * 60)
    cobertura = min(now - win_from, janela_min*60)
    if cobertura < min_cov:
        return False
    abaixo = sum(1 for v in win_vals if v <= gatilho)
    frac = abaixo / float(len(win_vals))
    return frac >= ratio_ok

# ----------------- Regra de proteção principal -----------------
def check_e_acionar_protecao() -> bool:
    """
    Dispara proteção se PnL aberto ficar abaixo do gatilho de forma persistente
    pela janela de minutos configurada.
    """
    cfg = _carregar_config()
    prot = cfg.get("protecao", {}) or {}
    gatilho = float(prot.get("gatilho_pnl_aberto", -250.0))
    cooldown_min = int(prot.get("cooldown_minutos", 45))
    janela_min = int(prot.get("janela_minutos", 10))

    # Já bloqueado? não reavisa aqui
    st = carregar_estado()
    if st.get("bloqueado"):
        return False

    if mt5 is None:
        log_event("[PROTECAO] MT5 indisponível; proteção não pode avaliar PnL.", level="warning")
        return False

    try:
        poss = mt5.positions_get() or []
        pnl_aberto = sum(float(getattr(p, "profit", 0.0) or 0.0) for p in poss)
    except Exception as e:
        log_event(f"[PROTECAO] Falha ao computar PnL aberto: {e}", level="warning")
        return False

    # Amostra atual no buffer
    _push_pnl_sample(pnl_aberto)

    # Janela deslizante (60% das amostras <= gatilho)
    if _window_below_threshold(gatilho, janela_min, ratio_ok=0.6):
        log_event("[PROTECAO] Gatilho persistente atingido na janela. Fechando posições...", level="warning")

        # Snapshot antes do fechamento
        _, snap_tot = _snapshot_posicoes_mt5()

        ok, tickets, falhas = _fechar_todas_posicoes_retry(max_tentativas=5, espera_s=2.0)
        if ok:
            # Bloqueio + mensagem detalhada
            set_bloqueio("pnl_window_trigger", minutos=cooldown_min, tickets=tickets)
            try:
                msg = _formatar_msg_bloqueio(snap_tot, gatilho, cooldown_min, tickets)
                if falhas:
                    prev = ", ".join(f"{k}:{v}" for k, v in list(falhas.items())[:10])
                    if len(falhas) > 10:
                        prev += "..."
                    msg += f"\nAlgumas falhas durante o fechamento: {prev}"
                _enviar_telegram_safe(msg)
            except Exception:
                pass
            return True
        else:
            try:
                prev = ", ".join(f"{k}:{v}" for k, v in list(falhas.items())[:10])
                if len(falhas) > 10:
                    prev += "..."
                _enviar_telegram_safe(f"⚠️ PROTEÇÃO: não foi possível zerar todas as posições. Falhas: {prev}")
            except Exception:
                pass
            log_event(f"[PROTECAO] Não zerou posições após retries. Falhas: {falhas}", level="error")
            return False
    return False


# ----------------- Buffer deslizante de PnL -----------------
from collections import deque
from datetime import datetime, timezone, timedelta

class _PnLBuffer:
    def __init__(self, janela_min: int = 10):
        self.janela = int(max(1, janela_min))
        self._q = deque()  # (ts, pnl_aberto_total)
    def add(self, pnl_total: float) -> None:
        now = datetime.now(timezone.utc)
        self._q.append((now, float(pnl_total or 0.0)))
        self._trim()
    def _trim(self):
        now = datetime.now(timezone.utc)
        limite = now - timedelta(minutes=self.janela)
        while self._q and self._q[0][0] < limite:
            self._q.popleft()
    def media(self) -> float:
        self._trim()
        if not self._q:
            return 0.0
        return sum(v for _, v in self._q) / len(self._q)
    def min(self) -> float:
        self._trim()
        if not self._q:
            return 0.0
        return min(v for _, v in self._q)
    def max(self) -> float:
        self._trim()
        if not self._q:
            return 0.0
        return max(v for _, v in self._q)
_pnl_buffer = _PnLBuffer(10)
