# gestao/fechar_todas_ordens.py
# Fechamento seguro de todas as ordens com logs, retcodes fortes e avisos Telegram.

from __future__ import annotations
import time
import os
import importlib
from typing import Any, Dict, Iterable, List, Optional

# --------- utilidades de integração opcionais (sem quebrar IDE) ----------
def _load_func(module_names: Iterable[str], func_name: str):
    for m in module_names:
        try:
            mod = importlib.import_module(m)
            fn = getattr(mod, func_name, None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None

def _carregar_enviar_telegram():
    send = _load_func(
        ["comunicacao.telegram_alertas", "telegram_alertas"],
        "enviar_telegram"
    )
    return send or (lambda *a, **k: False)

enviar_telegram = _carregar_enviar_telegram()

# Helpers esperados no projeto (import flexível / stubs)
log_event = _load_func(
    ["utils.debug_logger", "debug_logger"],
    "log_event"
) or (lambda msg, level="info": print(f"[{level.upper()}] {msg}"))

obter_ordens_abertas_mt5 = _load_func(
    ["gestao.gestao_posicoes", "gestao_posicoes"],
    "obter_ordens_abertas_mt5"
) or (lambda *a, **k: [])

fechar_ordem = _load_func(
    ["ordens.executar_ordem", "ordens", "trading.ordens", "trading"],
    "fechar_ordem"
) or (lambda *a, **k: {"retcode": 0, "retcode_name": "NOIMPL"})

atualizar_operacao = _load_func(
    ["dados.operacoes_store", "operacoes_store", "storage.operacoes_store"],
    "atualizar_operacao"
) or (lambda **kwargs: None)

# Códigos MT5 de sucesso (retorno da trade server)
SUCCESS_CODES = {10008, 10009}  # TRADE_RETCODE_DONE, TRADE_RETCODE_PLACED

def _to_dict(ordem) -> Dict[str, Any]:
    if isinstance(ordem, dict):
        return ordem
    if hasattr(ordem, "_asdict"):
        return ordem._asdict()
    # fallback genérico
    d = {}
    for attr in ("ticket", "order", "position", "id", "symbol", "ativo", "price_open", "volume"):
        if hasattr(ordem, attr):
            d[attr] = getattr(ordem, attr)
    return d

def _extrair_ticket_symbol(od: Dict[str, Any]):
    # ticket
    ticket = od.get("ticket") or od.get("order") or od.get("position") or od.get("id")
    # ativo/símbolo
    symbol = od.get("symbol") or od.get("ativo") or od.get("ativo_nome")
    return ticket, symbol

def fechar_todas_ordens(
    motivo: str = "encerramento_sistema",
    delay_seg: float = 0.0,
    ativos: Optional[List[str]] = None
):
    """
    Fecha todas as ordens abertas (opcionalmente filtrando por 'ativos').
    Retorna (total_fechadas, total_erros).
    """
    try:
        ordens_raw = obter_ordens_abertas_mt5()
    except Exception as e:
        log_event(f"[FECHAMENTO] Falha ao obter ordens: {e}", level="error")
        return 0, 0

    ordens = [_to_dict(o) for o in (ordens_raw or [])]
    if ativos:
        ativos_set = set(ativos)
        ordens = [o for o in ordens if (o.get("symbol") or o.get("ativo")) in ativos_set]

    if not ordens:
        log_event("[FECHAMENTO] Nenhuma ordem aberta encontrada.", level="info")
        return 0, 0

    total_fechadas = 0
    erros: List[Any] = []

    for ordem in ordens:
        ticket, symbol = _extrair_ticket_symbol(ordem)
        if not ticket or not symbol:
            log_event(f"[FECHAMENTO] Ordem sem ticket/symbol: {ordem}", level="error")
            erros.append(str(ordem))
            continue

        try:
            result = fechar_ordem(ticket, symbol) or {}
            # retcode robusto
            try:
                retcode = int(result.get("retcode", 0))
            except Exception:
                retcode = 0
            rc_name = str(result.get("retcode_name") or "")
            preco_fechamento = result.get("preco_fechamento")
            lucro = result.get("lucro")

            if retcode in SUCCESS_CODES:
                try:
                    atualizar_operacao(
                        ticket=ticket,
                        preco_fechamento=preco_fechamento,
                        lucro=lucro,
                        motivo_fechamento=motivo
                    )
                except Exception as e:
                    log_event(f"[FECHAMENTO] Não foi possível atualizar operação {ticket}: {e}", level="warning")
                log_event(f"[FECHAMENTO] OK ticket={ticket} {symbol} retcode={retcode} {rc_name}", level="info")
                total_fechadas += 1
            else:
                log_event(f"[FECHAMENTO] FALHA ticket={ticket} {symbol} retcode={retcode} {rc_name}", level="warning")
                erros.append(ticket)
        except Exception as e:
            log_event(f"[FECHAMENTO] Erro ao fechar ticket={ticket} {symbol}: {e}", level="error")
            erros.append(ticket)

        if delay_seg and delay_seg > 0:
            time.sleep(delay_seg)

    msg = f"[FECHAMENTO] {total_fechadas}/{len(ordens)} ordens fechadas. Motivo: {motivo}"
    lvl = "warning" if total_fechadas < len(ordens) else "info"
    log_event(msg, level=lvl)
    try:
        enviar_telegram(msg)
        if erros:
            enviar_telegram(f"[FECHAMENTO] Falhas: {erros}")
    except Exception as e:
        log_event(f"[FECHAMENTO] Falha no alerta Telegram: {e}", level="warning")

    if erros:
        log_event(f"[FECHAMENTO] Erros ao fechar ordens: {erros}", level="error")
    return total_fechadas, len(erros)

if __name__ == "__main__":
    fechar_todas_ordens()
