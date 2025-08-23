# gestao/gestao_posicoes.py
# -*- coding: utf-8 -*-
"""
Gestão de posições no MetaTrader 5

Funções expostas (compatíveis com main_loop.py):
- listar_posicoes_ativas(symbol: str | None = None) -> list
- valor_investido(symbol: str | None = None, usar_preco_ultimo: bool = True) -> float
- calcular_valor_investido = valor_investido (alias de compatibilidade)
- posicao_liquida_por_ativo() -> dict[str, float]
- resumo_posicoes() -> dict
- lucro_aberto(ativo: str) -> float
- lucro_aberto_total(symbol: str | None = None) -> float   # novo
- lucro_fechado(start=None, end=None, symbol=None, incluir_comissoes=True, incluir_swaps=True) -> float
- lucro_total(start=None, end=None, symbol=None, incluir_aberto=True, incluir_comissoes=True, incluir_swaps=True) -> float   # novo
- exposicao_ftmo(ativo: str) -> float
- saldo_bruto() -> float
- risco_aberto_ftmo(ativo: str | None = None) -> float

Observação importante:
- ESTE MÓDULO **NÃO** inicializa nem finaliza o MT5. Ele supõe que o processo principal
  já chamou `mt5.initialize()` e manterá a sessão aberta. Se o MT5 não estiver importável
  ou não estiver inicializado, as funções retornam valores neutros (0.0 / {} / []).

Isso evita conflitos de sessão com o seu main_loop.py.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
import datetime as dt
import importlib

# ========= Logging (compatível com seu projeto) =========
def _load_func(module_names, func_name):
    for m in module_names:
        try:
            mod = importlib.import_module(m)
            fn = getattr(mod, func_name, None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None

log_event = _load_func(["utils.debug_logger", "debug_logger"], "log_event") \
    or (lambda msg, level="info": None)  # silencioso se não houver logger

# ========= Acesso seguro ao MT5 (sem initialize/shutdown aqui) =========
def _get_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
        return mt5
    except Exception:
        return None

def _mt5_ok(mt5) -> bool:
    try:
        return (mt5 is not None) and hasattr(mt5, "account_info") and (mt5.account_info() is not None)
    except Exception:
        return False

# ========= Helpers =========
def _mid_price(mt5, symbol: str) -> Optional[float]:
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        last = float(getattr(tick, "last", 0.0) or 0.0)
        return last if last > 0 else None
    except Exception:
        return None

def _contract_size(mt5, symbol: str) -> float:
    try:
        info = mt5.symbol_info(symbol)
        if info and getattr(info, "trade_contract_size", 0):
            cs = float(info.trade_contract_size)
            if cs > 0:
                return cs
    except Exception:
        pass
    # Fallback comum para Forex
    return 100_000.0

def _as_dict_list(posicoes) -> List[Dict[str, Any]]:
    if not posicoes:
        return []
    out = []
    for p in posicoes:
        if isinstance(p, dict):
            out.append(p)
        elif hasattr(p, "_asdict"):
            out.append(p._asdict())
        else:
            d = {}
            for attr in ("ticket", "symbol", "type", "volume", "price_open", "sl", "tp"):
                if hasattr(p, attr):
                    d[attr] = getattr(p, attr)
            out.append(d)
    return out

def _simbolos_posicoes(mt5) -> Set[str]:
    try:
        posicoes = mt5.positions_get() or []
    except Exception:
        posicoes = []
    symbols: Set[str] = set()
    for p in posicoes:
        try:
            sym = getattr(p, "symbol", None)
            if sym:
                symbols.add(sym)
        except Exception:
            continue
    return symbols

def _position_type_buy(mt5) -> int:
    # Nas posições, o campo type usa POSITION_TYPE_BUY=0 e POSITION_TYPE_SELL=1.
    return int(getattr(mt5, "POSITION_TYPE_BUY", 0))

# ========= API pública =========
def listar_posicoes_ativas(symbol: Optional[str] = None):
    """
    Retorna lista de posições ativas (objetos MT5). Use obter_ordens_abertas_mt5 para dicionários.
    """
    mt5 = _get_mt5()
    if not _mt5_ok(mt5):
        return []
    try:
        return list(mt5.positions_get(symbol=symbol)) if symbol else list(mt5.positions_get() or [])
    except Exception:
        return []

def obter_ordens_abertas_mt5(ativo: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retorna lista de posições abertas em formato de dicionário.
    """
    mt5 = _get_mt5()
    if not _mt5_ok(mt5):
        return []
    try:
        posicoes = mt5.positions_get(symbol=ativo) if ativo else mt5.positions_get()
    except Exception:
        posicoes = None
    return _as_dict_list(posicoes)

def valor_investido(symbol: Optional[str] = None, usar_preco_ultimo: bool = True) -> float:
    """
    Exposição nocional total das posições abertas.
    Fórmula: sum( abs(lotes) * contract_size(symbol) * preço ), onde preço é mid (bid/ask) se disponível.
    """
    mt5 = _get_mt5()
    if not _mt5_ok(mt5):
        return 0.0
    try:
        posicoes = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    except Exception:
        posicoes = None
    if not posicoes:
        return 0.0

    total = 0.0
    for p in posicoes:
        try:
            sym = getattr(p, "symbol", None)
            vol = float(getattr(p, "volume", 0.0) or 0.0)
            po  = float(getattr(p, "price_open", 0.0) or 0.0)
            if not sym or vol <= 0 or po <= 0:
                continue
            cs = _contract_size(mt5, sym)
            preco = _mid_price(mt5, sym) if usar_preco_ultimo else po
            if preco is None or preco <= 0:
                preco = po
            total += abs(vol) * cs * float(preco)
        except Exception:
            continue
    return float(total)

# Alias de compatibilidade
calcular_valor_investido = valor_investido

def posicao_liquida_por_ativo() -> Dict[str, float]:
    """
    Volume líquido (em lotes) por ativo: >0 comprado, <0 vendido.
    """
    mt5 = _get_mt5()
    if not _mt5_ok(mt5):
        return {}
    try:
        posicoes = mt5.positions_get() or []
    except Exception:
        posicoes = []
    acc: Dict[str, float] = {}
    buy_const = _position_type_buy(mt5)
    for p in posicoes:
        try:
            sym = getattr(p, "symbol", None)
            vol = float(getattr(p, "volume", 0.0) or 0.0)
            typ = int(getattr(p, "type", 0) or 0)  # 0=buy,1=sell
            if not sym or vol <= 0:
                continue
            sinal = 1.0 if typ == buy_const else -1.0
            acc[sym] = acc.get(sym, 0.0) + sinal * vol
        except Exception:
            continue
    return acc

def resumo_posicoes() -> Dict[str, Any]:
    """
    Resumo simples: contagem de posições, valor investido (nocional),
    posição líquida por ativo e risco aberto aproximado.
    """
    return {
        "qtd_posicoes": len(listar_posicoes_ativas()),
        "valor_investido": valor_investido(),
        "liquido_por_ativo": posicao_liquida_por_ativo(),
        "risco_aberto": risco_aberto_ftmo(),
    }

def lucro_aberto(ativo: str) -> float:
    """
    PnL aberto aproximado do ativo (moeda da conta).
    Compra (type=0) usa BID; Venda (type=1) usa ASK.
    Aproximação nocional: contract_size * delta_preco * lotes.
    """
    mt5 = _get_mt5()
    if not _mt5_ok(mt5):
        return 0.0

    try:
        posicoes = mt5.positions_get(symbol=ativo) or []
        tick = mt5.symbol_info_tick(ativo)
        if not tick or not posicoes:
            return 0.0
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
    except Exception:
        return 0.0

    pnl = 0.0
    buy_const = _position_type_buy(mt5)
    for p in posicoes:
        try:
            vol = float(getattr(p, "volume", 0.0) or 0.0)
            po  = float(getattr(p, "price_open", 0.0) or 0.0)
            typ = int(getattr(p, "type", 0) or 0)  # 0=buy,1=sell
            sym = getattr(p, "symbol", ativo)
            if vol <= 0 or po <= 0:
                continue
            cs = _contract_size(mt5, sym)
            if typ == buy_const:
                pnl += (bid - po) * vol * cs
            else:
                pnl += (po - ask) * vol * cs
        except Exception:
            continue
    return float(pnl)

def lucro_aberto_total(symbol: Optional[str] = None) -> float:
    """
    PnL aberto total. Se `symbol` for fornecido, restringe a ele; caso contrário,
    soma aberto de todos os símbolos com posições.
    """
    mt5 = _get_mt5()
    if not _mt5_ok(mt5):
        return 0.0

    if symbol:
        return lucro_aberto(symbol)

    total = 0.0
    for sym in _simbolos_posicoes(mt5):
        try:
            total += lucro_aberto(sym)
        except Exception:
            continue
    return float(total)

def lucro_fechado(start: Optional[dt.datetime] = None,
                  end: Optional[dt.datetime] = None,
                  symbol: Optional[str] = None,
                  incluir_comissoes: bool = True,
                  incluir_swaps: bool = True) -> float:
    """
    Lucro líquido FECHADO no histórico (somatório de deals) no intervalo [start, end].
    Usa mt5.history_deals_get(). Soma: profit (+ commission, + swap) conforme flags.
    Default: hoje 00:00 até agora.
    """
    mt5 = _get_mt5()
    if not _mt5_ok(mt5):
        return 0.0

    now = dt.datetime.now()
    if start is None:
        start = dt.datetime(now.year, now.month, now.day)  # hoje 00:00
    if end is None:
        end = now

    try:
        deals = mt5.history_deals_get(start, end)
    except Exception:
        deals = None
    if not deals:
        return 0.0

    total = 0.0
    for d in deals:
        try:
            sym = getattr(d, "symbol", None)
            if symbol and sym != symbol:
                continue
            lucro = float(getattr(d, "profit", 0.0) or 0.0)
            if incluir_comissoes:
                lucro += float(getattr(d, "commission", 0.0) or 0.0)
            if incluir_swaps:
                lucro += float(getattr(d, "swap", 0.0) or 0.0)
            total += lucro
        except Exception:
            continue
    return float(total)

def lucro_total(start: Optional[dt.datetime] = None,
                end: Optional[dt.datetime] = None,
                symbol: Optional[str] = None,
                incluir_aberto: bool = True,
                incluir_comissoes: bool = True,
                incluir_swaps: bool = True) -> float:
    """
    Retorna o lucro total (fechado +, opcionalmente, aberto) no intervalo solicitado.
    - Se `symbol` for informado, restringe a ele.
    - Fechado: via `lucro_fechado(...)`.
    - Aberto: soma PnL aberto atual das posições (em tempo real) se `incluir_aberto=True`.
    """
    fechado = lucro_fechado(start=start, end=end, symbol=symbol,
                            incluir_comissoes=incluir_comissoes,
                            incluir_swaps=incluir_swaps)
    if not incluir_aberto:
        return float(fechado)
    aberto = lucro_aberto_total(symbol=symbol)
    return float(fechado + aberto)

def exposicao_ftmo(ativo: str) -> float:
    """
    Exposição aproximada (notional) do ativo.
    """
    mt5 = _get_mt5()
    if not _mt5_ok(mt5):
        return 0.0
    try:
        pos = mt5.positions_get(symbol=ativo)
    except Exception:
        pos = None
    if not pos:
        return 0.0

    total = 0.0
    for p in pos:
        try:
            vol = float(getattr(p, "volume", 0.0) or 0.0)
            po  = float(getattr(p, "price_open", 0.0) or 0.0)
            sym = getattr(p, "symbol", ativo)
            if vol > 0 and po > 0:
                cs = _contract_size(mt5, sym)
                total += vol * cs * po
        except Exception:
            continue
    return float(total)

def saldo_bruto() -> float:
    """
    Retorna o saldo (balance) atual da conta. 0.0 se indisponível.
    """
    mt5 = _get_mt5()
    if not _mt5_ok(mt5):
        return 0.0
    try:
        acc = mt5.account_info()
        return float(getattr(acc, "balance", 0.0)) if acc else 0.0
    except Exception:
        return 0.0

def risco_aberto_ftmo(ativo: Optional[str] = None) -> float:
    """
    Soma de risco aproximado com base no SL: distância até preço de abertura * notional.
    """
    mt5 = _get_mt5()
    if not _mt5_ok(mt5):
        return 0.0
    try:
        posicoes = mt5.positions_get(symbol=ativo) if ativo else mt5.positions_get()
    except Exception:
        posicoes = None
    if not posicoes:
        return 0.0

    total_risco = 0.0
    buy_const = _position_type_buy(mt5)
    for p in posicoes:
        try:
            vol = float(getattr(p, "volume", 0.0) or 0.0)
            po  = float(getattr(p, "price_open", 0.0) or 0.0)
            sl  = float(getattr(p, "sl", 0.0) or 0.0)
            sym = getattr(p, "symbol", None)
            typ = int(getattr(p, "type", 0) or 0)  # 0=buy,1=sell
            if not sym or vol <= 0 or po <= 0 or sl <= 0:
                continue
            cs = _contract_size(mt5, sym)
            if typ == buy_const:  # buy
                risco = max(0.0, (po - sl) * vol * cs)
            else:                 # sell
                risco = max(0.0, (sl - po) * vol * cs)
            total_risco += risco
        except Exception:
            continue
    return float(total_risco)

if __name__ == "__main__":
    # Sanity checks leves — não falham execução do robô:
    try:
        print("Resumo posições:", resumo_posicoes())
        print("Lucro fechado hoje (todos):", lucro_fechado())
        print("Lucro TOTAL (inclui aberto):", lucro_total())
    except Exception:
        pass

# ======== Métricas percentuais / de período ========
def _inicio_do_dia(dt_in: Optional[dt.datetime] = None) -> dt.datetime:
    now = dt.datetime.now() if dt_in is None else dt_in
    return dt.datetime(now.year, now.month, now.day)

def lucro_dia(symbol: Optional[str] = None,
              incluir_aberto: bool = True,
              incluir_comissoes: bool = True,
              incluir_swaps: bool = True) -> float:
    """
    Convenience: lucro TOTAL de hoje (00:00 -> agora).
    """
    start = _inicio_do_dia()
    return lucro_total(start=start, end=None, symbol=symbol,
                       incluir_aberto=incluir_aberto,
                       incluir_comissoes=incluir_comissoes,
                       incluir_swaps=incluir_swaps)

def percentual_lucro(start: Optional[dt.datetime] = None,
                     end: Optional[dt.datetime] = None,
                     symbol: Optional[str] = None,
                     incluir_aberto: bool = True,
                     incluir_comissoes: bool = True,
                     incluir_swaps: bool = True,
                     base: str = "saldo_atual",
                     base_val: Optional[float] = None) -> float:
    """
    Percentual de lucro no intervalo.
    - `base` pode ser:
        - "saldo_atual": usa saldo_bruto() atual
        - "investido": usa valor_investido() (fallback para saldo se zero)
        - qualquer outro: cai para saldo atual
    - `base_val`: se informado, usa esse valor explicitamente.
    Retorna 0.0 se a base for <= 0.
    """
    lucro = lucro_total(start=start, end=end, symbol=symbol,
                        incluir_aberto=incluir_aberto,
                        incluir_comissoes=incluir_comissoes,
                        incluir_swaps=incluir_swaps)

    if base_val is not None:
        denom = float(base_val)
    else:
        if base == "investido":
            denom = float(valor_investido())
            if denom <= 0:
                denom = float(saldo_bruto())
        else:  # saldo_atual (default)
            denom = float(saldo_bruto())

    if denom <= 0:
        return 0.0
    return float((lucro / denom) * 100.0)

def percentual_lucro_dia(symbol: Optional[str] = None,
                         incluir_aberto: bool = True,
                         incluir_comissoes: bool = True,
                         incluir_swaps: bool = True,
                         base: str = "saldo_atual",
                         base_val: Optional[float] = None) -> float:
    """
    Percentual de lucro de hoje (00:00 -> agora).
    """
    start = _inicio_do_dia()
    return percentual_lucro(start=start, end=None, symbol=symbol,
                            incluir_aberto=incluir_aberto,
                            incluir_comissoes=incluir_comissoes,
                            incluir_swaps=incluir_swaps,
                            base=base, base_val=base_val)
