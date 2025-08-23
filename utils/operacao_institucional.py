# utils/operacao_institucional.py

from datetime import datetime
from mt5.executar_ordem_mt5 import enviar_ordem, fechar_ordem
from utils.registrar_operacao import registrar_operacao, atualizar_operacao
from utils.sinal_utils import normalizar_sinal

# Logs
try:
    from utils.debug_logger import log_event
except Exception:
    def log_event(msg, level="info"):
        print(f"[{level.upper()}] {msg}")

# MT5 (usado aqui só para diagnóstico e pré-checagens)
try:
    import MetaTrader5 as mt5
    MT5_OK = True
except Exception:
    MT5_OK = False


# -------------------------
# Helpers
# -------------------------
def _ajustar_volume_para_simbolo(volume: float, si) -> float:
    """Ajusta o volume para step/min/max do símbolo (trunca para baixo; arredonda pela precisão do step)."""
    try:
        step = float(getattr(si, "volume_step", 0.01) or 0.01)
        vmin = float(getattr(si, "volume_min", step) or step)
        vmax = float(getattr(si, "volume_max", 100.0) or 100.0)

        passos = int(float(volume) / step)
        v = passos * step
        if v < vmin: v = vmin
        if v > vmax: v = vmax

        s = f"{step:.10f}".rstrip("0")
        casas = len(s.split(".")[1]) if "." in s else 0
        return float(f"{v:.{casas}f}")
    except Exception:
        return float(volume)


def _symbol_info_str(si) -> str:
    try:
        return (
            f"digits={getattr(si,'digits',None)} point={getattr(si,'point',None)} "
            f"stops_level={getattr(si,'trade_stops_level',None)} "
            f"freeze_level={getattr(si,'trade_freeze_level',None)} "
            f"fill_mode={getattr(si,'trade_fill_mode',None)} "
            f"vol(min={getattr(si,'volume_min',None)}, step={getattr(si,'volume_step',None)}, "
            f"max={getattr(si,'volume_max',None)})"
        )
    except Exception:
        return "symbol_info=N/A"


def _diagnosticar_envio(ativo: str, tipo: str, volume: float, resultado: dict):
    """Coleta informações úteis do MT5 após um envio (especialmente quando falhou)."""
    if not MT5_OK:
        return {"mt5": "indisponivel"}
    diag = {}
    try:
        si = mt5.symbol_info(ativo)
        if si is None or not getattr(si, "visible", True):
            try:
                mt5.symbol_select(ativo, True)
                si = mt5.symbol_info(ativo)
            except Exception:
                pass

        le = None
        try:
            le = mt5.last_error()  # (code, description)
        except Exception:
            le = None

        diag = {
            "retcode": resultado.get("retcode"),
            "comment": resultado.get("comment", ""),
            "last_error": le,
            "symbol_info": {
                "digits": getattr(si, "digits", None) if si else None,
                "point": getattr(si, "point", None) if si else None,
                "stops_level": getattr(si, "trade_stops_level", None) if si else None,
                "freeze_level": getattr(si, "trade_freeze_level", None) if si else None,
                "fill_mode": getattr(si, "trade_fill_mode", None) if si else None,
                "volume_min": getattr(si, "volume_min", None) if si else None,
                "volume_step": getattr(si, "volume_step", None) if si else None,
                "volume_max": getattr(si, "volume_max", None) if si else None,
            },
            "tipo": tipo,
            "volume_enviado": volume,
        }
        lvl = "info" if str(diag.get("retcode", "")).startswith(("10009","10008","10004","100")) else "error"
        log_event(
            f"[MT5/DIAG] ativo={ativo} tipo={tipo} volume={volume} retcode={diag.get('retcode')} "
            f"comment={diag.get('comment')} last_error={diag.get('last_error')} | {_symbol_info_str(si)}",
            level=lvl
        )
    except Exception as e:
        diag = {"erro_diag": str(e)}
    return diag


def _retcode_sucesso(ret) -> bool:
    """Sucesso se ordem foi executada/colocada: 10009 (DONE), 10008 (PLACED), 10004 (DONE_PARTIAL)."""
    try:
        if isinstance(ret, str):
            return ret.startswith("100")
        return int(ret) in (10009, 10008, 10004)
    except Exception:
        return False


# -------------------------
# API principal
# -------------------------
def abrir_ordem_e_registrar(
    ativo: str,
    tipo: str,
    volume: float,
    timestamp=None,
    preco_abertura=None,
    contexto="",
    observacao="",
):
    """Envia ordem via executor e registra no banco quando sucesso. Sempre devolve 'motivo' e 'diagnostico'."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ajuste prévio de volume pelo símbolo (log se houve ajuste)
    volume_original = float(volume)
    if MT5_OK:
        try:
            si = mt5.symbol_info(ativo)
            if si is None or not getattr(si, "visible", True):
                try:
                    mt5.symbol_select(ativo, True)
                    si = mt5.symbol_info(ativo)
                except Exception:
                    si = None
            if si is not None:
                volume_aj = _ajustar_volume_para_simbolo(volume_original, si)
                if abs(volume_aj - volume_original) > 1e-12:
                    log_event(
                        f"[ENVIO ORDEM] Ajuste de volume pelo símbolo | {ativo}: {volume_original} -> {volume_aj} | {_symbol_info_str(si)}",
                        level="warning"
                    )
                volume = volume_aj
        except Exception as e:
            log_event(f"[ENVIO ORDEM] Falha ao ajustar volume pelo símbolo | {ativo}: {e}", level="warning")

    # Envio — o executor já retorna 'motivo' e 'diagnostico'
    resultado = enviar_ordem(tipo, ativo, timestamp, volume=volume)

    ticket = resultado.get("order")
    retcode = resultado.get("retcode")
    preco_exec = resultado.get("preco") or preco_abertura

    # Diagnóstico adicional: MERGE (não sobrescreve o que já veio do executor)
    diag_extra = _diagnosticar_envio(ativo, tipo, volume, resultado)
    if isinstance(resultado, dict):
        if isinstance(resultado.get("diagnostico"), dict):
            resultado["diagnostico"].update({k: v for k, v in diag_extra.items() if k not in resultado["diagnostico"]})
        else:
            resultado["diagnostico"] = diag_extra

    # Sucesso → registra
    if ticket and str(ticket).isdigit() and int(ticket) > 0 and _retcode_sucesso(retcode):
        try:
            registrar_operacao(
                ativo=ativo,
                ticket=ticket,
                padrao="",
                regime="",
                contexto=contexto if isinstance(contexto, str) else str(contexto),
                hora=datetime.now().strftime("%H:%M:%S"),
                motivo_fechamento="",
                retcode=retcode,
                volume=volume,
                observacao=observacao,
                timestamp=timestamp,
                preco_abertura=preco_exec,
                preco_fechamento=None,
                preco_saida=None,
                lucro=0,
                resultado=None,
                sinal=1 if tipo.lower() in ("compra","buy","long") else -1 if tipo.lower() in ("venda","sell","short") else 0
            )
        except Exception as e:
            log_event(f"[ENVIO ORDEM] Falha ao registrar operação {ativo}/{ticket}: {e}", level="error")
        return resultado

    # Falha → já tem motivo + diagnostico no resultado
    log_event(
        f"[ENVIO ORDEM] NÃO REGISTRADA | ativo={ativo} tipo={tipo} volume={volume} "
        f"retcode={retcode} ticket={ticket} motivo={resultado.get('motivo')} diag={resultado.get('diagnostico')}",
        level="warning"
    )
    return resultado


def fechar_ordem_e_registrar(
    ticket: int,
    ativo: str,
    preco_fechamento: float = None,
    timestamp=None,
    motivo_fechamento="fechamento_manual",
    lucro: float = None,
    resultado=None
):
    """Fecha ordem no MT5 e atualiza banco. Só atualiza se o MT5 aceitou (ou simulado)."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        resultado_envio = fechar_ordem(ticket, ativo)
        retcode = resultado_envio.get("retcode")

        preco_fech = preco_fechamento or resultado_envio.get("preco_fechamento")
        lucro_real = lucro if lucro is not None else resultado_envio.get("lucro", 0)

        if _retcode_sucesso(retcode) or retcode == "simulada":
            try:
                atualizar_operacao(
                    ticket=ticket,
                    preco_fechamento=preco_fech,
                    lucro=lucro_real,
                    motivo_fechamento=motivo_fechamento
                )
            except Exception as e:
                log_event(f"[FECHAR ORDEM] Falha ao atualizar operação {ativo}/{ticket}: {e}", level="error")
        elif retcode == "nao_encontrada":
            log_event(f"[FECHAR ORDEM] Ordem não encontrada {ativo}/{ticket}", level="warning")
        else:
            log_event(f"[FECHAR ORDEM] Falha ao fechar ordem {ativo}/{ticket} | retcode={retcode}", level="error")

    except Exception as e:
        log_event(f"[FECHAR ORDEM] Exceção no fechamento {ativo}/{ticket}: {e}", level="error")
        resultado_envio = {"retcode": "erro"}

    return resultado_envio
