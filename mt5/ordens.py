
from __future__ import annotations
from typing import List, Tuple, Dict

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

from utils.debug_logger import log_event

def fechar_todas_posicoes(max_tentativas: int = 5, espera_s: float = 2.0) -> Tuple[bool, List[int]]:
    """
    Fecha todas as posições com retry. Retorna (ok, tickets_fechados).
    """
    if mt5 is None:
        return False, []

    tickets_ok: List[int] = []
    for tentativa in range(1, max_tentativas + 1):
        try:
            poss = mt5.positions_get() or []
        except Exception as e:
            log_event(f"[ORDENS] positions_get falhou: {e}", level="error")
            break

        if not poss:
            return True, tickets_ok

        for p in poss:
            try:
                ticket = int(getattr(p, "ticket", 0) or 0)
                symbol = getattr(p, "symbol", "")
                vol = float(getattr(p, "volume", 0.0) or 0.0)
                pos_type = int(getattr(p, "type", 0) or 0)
                ti = mt5.symbol_info_tick(symbol)
                if ti is None or vol <= 0.0:
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
                    "comment": f"ordens_close_retry_t{tentativa}",
                }
                r = mt5.order_send(req)
                rc = getattr(r, "retcode", None) if r is not None else None
                if rc in (getattr(mt5, "TRADE_RETCODE_DONE", 10009), getattr(mt5, "TRADE_RETCODE_PLACED", 10008)):
                    tickets_ok.append(ticket)
            except Exception as e:
                log_event(f"[ORDENS] Falha ao fechar ticket {int(getattr(p, 'ticket', 0) or 0)}: {e}", level="error")

        try:
            rem = mt5.positions_get() or []
            if not rem:
                return True, tickets_ok
        except Exception:
            pass
    # estado final
    try:
        rem = mt5.positions_get() or []
        return len(rem) == 0, tickets_ok
    except Exception:
        return False, tickets_ok


def fechar_todas_posicoes_detalhado(max_tentativas: int = 5, espera_s: float = 2.0) -> tuple[bool, list[int], dict]:
    """
    Versão detalhada: retorna (ok, tickets_fechados, falhas_por_ticket).
    Não quebra compatibilidade com fechar_todas_posicoes().
    """
    tickets_ok: list[int] = []
    falhas: dict[int, str] = {}
    if mt5 is None:
        return False, tickets_ok, falhas

    for tentativa in range(1, max_tentativas + 1):
        try:
            poss = mt5.positions_get() or []
        except Exception as e:
            log_event(f"[ORDENS] positions_get falhou: {e}", level="error")
            break

        if not poss:
            return True, tickets_ok, falhas

        for p in poss:
            try:
                ticket = int(getattr(p, "ticket", 0) or 0)
                symbol = getattr(p, "symbol", "")
                vol = float(getattr(p, "volume", 0.0) or 0.0)
                ptype = int(getattr(p, "type", 0) or 0)
                ti = mt5.symbol_info_tick(symbol)
                if ti is None or vol <= 0.0:
                    falhas[ticket] = "tick/volume inválido"
                    continue
                if ptype == 0:
                    price = ti.bid
                    otype = mt5.ORDER_TYPE_SELL
                else:
                    price = ti.ask
                    otype = mt5.ORDER_TYPE_BUY
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": vol,
                    "type": otype,
                    "price": price,
                    "deviation": 20,
                    "position": ticket,
                    "comment": f"ordens_close_retry_t{tentativa}",
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

        try:
            rem = mt5.positions_get() or []
            if not rem:
                return True, tickets_ok, falhas
        except Exception:
            pass

        import time as _t
        _t.sleep(max(0.5, espera_s))

    try:
        rem = mt5.positions_get() or []
        return len(rem) == 0, tickets_ok, falhas
    except Exception:
        return False, tickets_ok, falhas
