# mt5/executar_ordem_mt5.py
import MetaTrader5 as mt5
from utils.utils import carregar_config
from utils.debug_logger import log_event
from datetime import datetime
import numpy as np

# =========================
# Integração com lote centralizado
# =========================
def _motivo_lote_from_return(res: dict) -> str:
    try:
        partes = []
        if res.get("aplicou_piso"):
            partes.append("piso_config")
        if res.get("aplicou_teto"):
            partes.append("teto_config")
        if not partes:
            partes.append("lote_teorico")
        return " / ".join(partes)
    except Exception:
        return "lote_teorico"

try:
    # Usa o cálculo institucional centralizado (com piso/teto + logs)
    from utils.lote_adaptativo import calcular_lote_adaptativo as _calc_lote_central

    def calcular_lote_adaptativo(ativo, contexto=None):
        res = _calc_lote_central(ativo=ativo, contexto=contexto, return_all=True)
        volume = float(res["lote_utilizado"])
        motivo = _motivo_lote_from_return(res)
        log_event(f"[LOTE-EXECUTOR] ativo={ativo} volume={volume} motivo={motivo} | detalhes={res}", level="info")
        return volume, motivo

except Exception as e:
    # Fallback: usa volume do config se não conseguir importar o módulo central
    log_event(f"[LOTE-EXECUTOR] Fallback para volume do config (erro import lote_adaptativo: {e})", level="warning")
    def calcular_lote_adaptativo(ativo, contexto=None):
        cfg = carregar_config()
        volume = float(cfg.get("volumes", {}).get(ativo, cfg.get("volume_padrao", 0.01)))
        return volume, "fallback_config_volume"


# =========================
# Sanity & helpers
# =========================
def sanity_check_dict(d, campos):
    problemas = []
    for k in campos:
        v = d.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and str(v).strip() == ""):
            problemas.append(k)
    if problemas:
        log_event(f"[SANITY CHECK] Campos vazios/invalidos: {problemas} | Dados: {d}", level="warning")

def _retcode_sucesso(ret):
    try:
        if isinstance(ret, str):
            return ret.startswith("100")
        return int(ret) in (10009, 10008, 10004)  # DONE, PLACED, DONE_PARTIAL
    except Exception:
        return False

def _retcode_info(ret):
    """Tradução amigável dos principais retcodes do MT5 (cobertura prática)."""
    try:
        r = int(ret)
    except Exception:
        return {"codigo": ret, "categoria": "desconhecido", "motivo": "retcode não numérico"}

    tabela = {
        10009: ("sucesso", "Ordem executada (DONE)."),
        10008: ("sucesso", "Ordem colocada (PLACED)."),
        10004: ("sucesso", "Execução parcial (DONE_PARTIAL)."),

        10012: ("erro", "Erro genérico na negociação."),
        10013: ("invalido", "Parâmetros inválidos."),
        10014: ("preco_invalido", "Preço inválido (PRICE_INVALID)."),
        10015: ("ticket_invalido", "Ticket inválido."),
        10016: ("volume_invalido", "Volume inválido ou fora do step/min/max."),
        10017: ("stop_invalido", "SL/TP inválidos (STOP_INVALID / muito perto do preço ou stops_level)."),
        10018: ("tempo_invalido", "Expiração inválida."),
        10019: ("mercado_fechado", "Mercado fechado."),
        10020: ("sem_margem", "Margem insuficiente (NO_MONEY)."),
        10021: ("preco_mudou", "Preço mudou (PRICE_CHANGED)."),
        10022: ("preco_off", "Preço off (PRICE_OFF / sem cotação válida)."),
        10023: ("expiracao_invalida", "Expiração inválida."),
        10024: ("ordem_modificada", "Ordem alterada pelo servidor."),
        10025: ("muitas_requisicoes", "Muitas requisições (TOO_MANY_REQUESTS)."),
        10026: ("sem_mudancas", "Sem mudanças."),
        10027: ("server_bloqueia", "Servidor bloqueia a negociação."),
        10028: ("cliente_bloqueia", "Cliente bloqueia a negociação."),
        10029: ("travada", "Negociação travada (LOCKED)."),
        10030: ("frozen/requote", "Preço congelado ou requote."),
        10031: ("repetir", "Tente novamente (RETRY)."),
        10032: ("colisao", "Colisão/conflito."),
        10033: ("modo_long_only", "Símbolo em LONG_ONLY/SHORT_ONLY."),
    }
    cat, mot = tabela.get(r, ("desconhecido", "Código não mapeado. Consulte os logs do terminal."))
    return {"codigo": r, "categoria": cat, "motivo": mot}

def _symbol_info_str(si) -> str:
    try:
        return (f"digits={getattr(si,'digits',None)} point={getattr(si,'point',None)} "
                f"stops_level={getattr(si,'trade_stops_level',None)} "
                f"freeze_level={getattr(si,'trade_freeze_level',None)} "
                f"fill_mode={getattr(si,'trade_fill_mode',None)} "
                f"vol(min={getattr(si,'volume_min',None)}, step={getattr(si,'volume_step',None)}, "
                f"max={getattr(si,'volume_max',None)})")
    except Exception:
        return "symbol_info=N/A"

def _ajustar_volume_para_simbolo(volume, si):
    try:
        step = float(getattr(si, "volume_step", 0.01) or 0.01)
        vmin = float(getattr(si, "volume_min", step) or step)
        vmax = float(getattr(si, "volume_max", 100.0) or 100.0)
        passos = int(float(volume) / step)
        v = passos * step
        if v < vmin: v = vmin
        if v > vmax: v = vmax
        # Arredonda conforme nº de casas do step
        s = f"{step:.10f}".rstrip("0")
        casas = len(s.split(".")[1]) if "." in s else 0
        v = float(f"{v:.{casas}f}")
        return v
    except Exception:
        return float(volume)

def _pip_factor(ativo, cfg):
    pf = (cfg.get("pip_factors", {}) or {}).get(ativo, 0.0001)
    if pf == 0.0001 and "JPY" in (ativo or ""):
        pf = 0.01
    return float(pf)

def _ler_tp_sl(cfg, ativo):
    lp = cfg.get("label_params", {}) or {}
    tp = int(cfg.get("tp_pips", lp.get("tp_pips", 40)))
    sl = int(cfg.get("sl_pips", lp.get("sl_pips", 20)))
    if isinstance(lp.get(ativo, {}), dict):
        tp = int(lp[ativo].get("tp_pips", tp))
        sl = int(lp[ativo].get("sl_pips", sl))
    return sl, tp

def _choose_fill_type(si):
    try:
        fm = int(getattr(si, "trade_fill_mode", 0) or 0)
        if hasattr(mt5, "SYMBOL_FILLING_FOK") and (fm & mt5.SYMBOL_FILLING_FOK):
            return mt5.ORDER_FILLING_FOK
        if hasattr(mt5, "SYMBOL_FILLING_IOC") and (fm & mt5.SYMBOL_FILLING_IOC):
            return mt5.ORDER_FILLING_IOC
    except Exception:
        pass
    return mt5.ORDER_FILLING_IOC

def _calc_sl_tp_prices(tipo, entry, sl_pips, tp_pips, pip_factor, si):
    sl_dist = sl_pips * pip_factor
    tp_dist = tp_pips * pip_factor
    if tipo == "buy":
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist

    point  = float(getattr(si, "point", 0.00001) or 0.00001)
    digits = int(getattr(si, "digits", 5) or 5)
    stops_level_pts = int(getattr(si, "trade_stops_level", 0) or 0)
    stops_level = stops_level_pts * point

    # Ajusta para respeitar stops_level
    if abs(entry - sl) < stops_level:
        delta = (stops_level - abs(entry - sl)) + 2 * point
        sl = sl - delta if tipo == "buy" else sl + delta
    if abs(tp - entry) < stops_level:
        delta = (stops_level - abs(tp - entry)) + 2 * point
        tp = tp + delta if tipo == "buy" else tp - delta

    sl = round(sl, digits)
    tp = round(tp, digits)
    return sl, tp


# =========================
# ENVIAR ORDEM
# =========================
def sanity_check_ordem(d):
    sanity_check_dict(d, ["retcode", "order", "preco"])

def enviar_ordem(tipo, ativo, timestamp, volume=None, sl_pips=None, tp_pips=None, contexto=None):
    cfg = carregar_config()
    simulado = cfg.get("usar_modo_simulado", True)
    deviation = int(cfg.get("mt5_deviation", 10))
    magic = int(cfg.get("mt5_magic", 123456))

    lado = str(tipo).strip().lower()
    if lado in ("compra", "buy", "long"):
        lado = "buy"
    elif lado in ("venda", "sell", "short"):
        lado = "sell"
    else:
        log_event(f"[ENVIO ORDEM] Tipo inválido: {tipo}", level="error")
        out = {"retcode": "erro", "order": None, "preco": None, "comment": "tipo_invalido", "motivo": "tipo inválido"}
        sanity_check_ordem(out)
        return out

    # Volume: se não vier definido, calcula com módulo central (piso/teto + logs)
    if volume is None:
        volume, motivo_lote = calcular_lote_adaptativo(ativo, contexto)
    else:
        motivo_lote = "Lote recebido externamente."

    # SL/TP em pips
    if sl_pips is None or tp_pips is None:
        sl_cfg, tp_cfg = _ler_tp_sl(cfg, ativo)
        sl_pips = sl_cfg if sl_pips is None else sl_pips
        tp_pips = tp_cfg if tp_pips is None else tp_pips

    log_event(f"[ENVIO ORDEM] tipo={lado} ativo={ativo} volume={volume} timestamp={timestamp} "
              f"SL={sl_pips} TP={tp_pips} motivo_lote={motivo_lote}")

    # Simulado
    if simulado:
        tick_price = 1.2345
        result = {
            "retcode": "simulada",
            "order": "simulado",
            "preco": tick_price,
            "comment": motivo_lote,
            "motivo": "modo_simulado",
            "diagnostico": {"last_error": None, "symbol_info": None},
        }
        sanity_check_ordem(result)
        log_event(f"[ORDEM SIMULADA] {result}")
        return result

    # Real
    if not mt5.initialize():
        log_event("erro_conexao_mt5", level="error")
        result = {
            "retcode": "erro",
            "order": None,
            "preco": None,
            "comment": "init_fail",
            "motivo": "falha_initialize",
            "diagnostico": {"last_error": mt5.last_error(), "symbol_info": None},
        }
        sanity_check_ordem(result)
        return result

    try:
        si = mt5.symbol_info(ativo)
        if not si:
            raise Exception(f"Ativo {ativo} não encontrado no MT5.")
        if not si.visible:
            mt5.symbol_select(ativo, True)

        # Ajusta volume conforme símbolo
        volume_aj = _ajustar_volume_para_simbolo(volume, si)
        if abs(volume_aj - float(volume)) > 1e-9:
            log_event(
                f"[ENVIO ORDEM] Ajuste de volume pelo símbolo | {ativo}: {volume} -> {volume_aj} | {_symbol_info_str(si)}",
                level="warning",
            )
        volume = volume_aj

        # Preço
        tick = mt5.symbol_info_tick(ativo)
        if not tick:
            saida = {
                "retcode": "erro",
                "order": None,
                "preco": None,
                "comment": "tick_indisponivel",
                "motivo": "tick indisponível",
                "diagnostico": {"last_error": mt5.last_error(), "symbol_info": None},
            }
            sanity_check_ordem(saida)
            log_event(f"[ERRO ENVIO ORDEM] Tick indisponível para {ativo}", level="error")
            return saida
        price = tick.ask if lado == "buy" else tick.bid

        # Stops
        pip_factor = _pip_factor(ativo, cfg)
        sl, tp = _calc_sl_tp_prices(lado, price, sl_pips, tp_pips, pip_factor, si)

        # Filling compatível
        type_filling = _choose_fill_type(si)

        ordem = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": ativo,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if lado == "buy" else mt5.ORDER_TYPE_SELL,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": deviation,
            "magic": magic,
            "comment": f"Lote: {motivo_lote}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": type_filling,
        }

        result = mt5.order_send(ordem)
        ticket = getattr(result, "order", 0) or getattr(result, "ticket", 0)
        retcode = getattr(result, "retcode", "erro")
        comment = getattr(result, "comment", "")

        # Diagnóstico detalhado
        diag = {}
        try:
            diag = {
                "last_error": mt5.last_error(),
                "symbol_info": {
                    "digits": getattr(si, "digits", None),
                    "point": getattr(si, "point", None),
                    "stops_level": getattr(si, "trade_stops_level", None),
                    "freeze_level": getattr(si, "trade_freeze_level", None),
                    "fill_mode": getattr(si, "trade_fill_mode", None),
                    "volume_min": getattr(si, "volume_min", None),
                    "volume_step": getattr(si, "volume_step", None),
                    "volume_max": getattr(si, "volume_max", None),
                },
            }
        except Exception:
            pass

        info = _retcode_info(retcode)
        saida = {
            "retcode": retcode,
            "order": ticket,
            "preco": float(price),
            "comment": comment,
            "motivo": info["motivo"],
            "diagnostico": diag,
        }
        sanity_check_ordem(saida)
        level = "info" if _retcode_sucesso(retcode) and (ticket and int(ticket) > 0) else "warning"
        log_event(f"[ORDEM ENVIADA] {saida} | {_symbol_info_str(si)}", level=level)
        return saida

    except Exception as e:
        log_event(f"[ERRO ENVIO ORDEM] {str(e)}", level="error")
        saida = {
            "retcode": "erro",
            "order": None,
            "preco": None,
            "comment": str(e),
            "motivo": "excecao_python",
            "diagnostico": {"last_error": mt5.last_error(), "symbol_info": None},
        }
        sanity_check_ordem(saida)
        return saida
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


# =========================
# FECHAR TODAS / FECHAR UMA / INFO AUX
# =========================
def fechar_todas_ordens_abertas(ativo, motivo):
    cfg = carregar_config()
    simulado = cfg.get("usar_modo_simulado", True)

    if simulado:
        log_event(f"[FECHAR TODAS] Simulado ON — nada a fechar para {ativo}.")
        return []

    if not mt5.initialize():
        log_event("erro_conexao_mt5_fechar_todas", level="error")
        return []

    try:
        posicoes = mt5.positions_get(symbol=ativo)
        fechamentos = []

        if not posicoes:
            log_event(f"nenhuma_ordem_aberta_para_fechar | ativo:{ativo}")
            return []

        for pos in posicoes:
            ticket = pos.ticket
            ordem_tipo = pos.type
            volume = pos.volume
            tick = mt5.symbol_info_tick(ativo)
            if not tick:
                log_event("[FECHAR ORDEM] Sem tick disponível", level="error")
                continue
            preco_fechamento = tick.bid if ordem_tipo == mt5.ORDER_TYPE_BUY else tick.ask
            tipo_fechamento = mt5.ORDER_TYPE_SELL if ordem_tipo == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": ativo,
                "volume": volume,
                "type": tipo_fechamento,
                "position": ticket,
                "price": preco_fechamento,
                "deviation": 10,
                "magic": int(cfg.get("mt5_magic", 123456)),
                "comment": motivo,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            retcode = getattr(result, "retcode", "erro")
            lucro = getattr(pos, "profit", 0)

            saida = {
                "ticket": ticket,
                "retcode": retcode,
                "preco_fechamento": preco_fechamento,
                "lucro": lucro,
                "motivo_fechamento": motivo,
                "data_fechamento": datetime.now(),
            }
            sanity_check_dict(saida, ["ticket", "retcode", "preco_fechamento"])
            log_event(f"[FECHAMENTO ORDEM] {saida}")
            fechamentos.append(saida)

        return fechamentos

    except Exception as e:
        log_event(f"[ERRO FECHAR TODAS ORDENS] {str(e)}", level="error")
        return []
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def fechar_ordem(ticket, ativo):
    cfg = carregar_config()
    simulado = cfg.get("usar_modo_simulado", True)

    if simulado:
        saida = {"retcode": "simulada", "ticket": ticket, "preco_fechamento": 0.0, "lucro": 0.0}
        log_event(f"[FECHAR ORDEM SIMULADA] {saida}")
        return saida

    if not mt5.initialize():
        log_event("erro_conexao_mt5_fechar", level="error")
        return {"retcode": "erro", "ticket": ticket}

    try:
        posicoes = mt5.positions_get(symbol=ativo)
        pos = None
        if posicoes:
            for p in posicoes:
                if int(getattr(p, "ticket", -1)) == int(ticket):
                    pos = p
                    break
        if not pos:
            return {"retcode": "nao_encontrada", "ticket": ticket}

        tick = mt5.symbol_info_tick(ativo)
        if not tick:
            return {"retcode": "erro", "ticket": ticket, "comment": "tick_indisponivel"}

        tipo_fechamento = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        preco_fech = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": ativo,
            "volume": pos.volume,
            "type": tipo_fechamento,
            "position": ticket,
            "price": preco_fech,
            "deviation": int(cfg.get("mt5_deviation", 10)),
            "magic": int(cfg.get("mt5_magic", 123456)),
            "comment": "close_by_api",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        retcode = getattr(result, "retcode", "erro")
        saida = {"retcode": retcode, "ticket": ticket, "preco_fechamento": preco_fech, "lucro": getattr(pos, "profit", 0)}
        log_event(f"[FECHAR ORDEM] {saida}", level="info" if _retcode_sucesso(retcode) else "error")
        return saida

    except Exception as e:
        log_event(f"[ERRO FECHAR ORDEM] {str(e)}", level="error")
        return {"retcode": "erro", "ticket": ticket, "comment": str(e)}
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def saldo_bruto_mt5(ativo=None):
    try:
        if not mt5.initialize():
            return {"status": "erro_init"}
        ai = mt5.account_info()
        out = {"balance": getattr(ai, "balance", None), "equity": getattr(ai, "equity", None)}
        return out
    except Exception as e:
        return {"status": "erro", "comment": str(e)}
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

def info_ordens_abertas_mt5(ativo=None):
    try:
        if not mt5.initialize():
            return []
        pos = mt5.positions_get(symbol=ativo) if ativo else mt5.positions_get()
        lst = []
        if pos:
            for p in pos:
                lst.append({
                    "ticket": getattr(p, "ticket", None),
                    "symbol": getattr(p, "symbol", None),
                    "type": getattr(p, "type", None),
                    "volume": getattr(p, "volume", None),
                    "price_open": getattr(p, "price_open", None),
                    "profit": getattr(p, "profit", None),
                    "time": getattr(p, "time", None),
                })
        return lst
    except Exception:
        return []
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


__all__ = [
    "enviar_ordem", "fechar_ordem", "saldo_bruto_mt5", "info_ordens_abertas_mt5",
    "fechar_todas_ordens_abertas", "calcular_lote_adaptativo"
]
