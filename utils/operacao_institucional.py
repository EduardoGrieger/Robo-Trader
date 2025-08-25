# utils/operacao_institucional.py

from datetime import datetime
from mt5.executar_ordem_mt5 import enviar_ordem, fechar_ordem
from utils.registrar_operacao import registrar_operacao, atualizar_operacao
from utils.sinal_utils import normalizar_sinal
from utils.utils import carregar_config

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

# Adaptativo (opcional)
try:
    from utils.lote_adaptativo import calcular_lote_adaptativo  # pode não existir
except Exception:
    calcular_lote_adaptativo = None  # type: ignore


# -------------------------
# Helpers
# -------------------------
def _sf(x, default=None):
    """Safe float: lida com None e strings com vírgula/ponto."""
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return default


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
        return {"mt5": "indisponivel", "volume_enviado": volume}
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
        lvl = "info" if str(diag.get("retcode", "")).startswith(("10009","10008","10004","100")) else "warning"
        log_event(
            f"[MT5/DIAG] ativo={ativo} tipo={tipo} volume={volume} retcode={diag.get('retcode')} "
            f"comment={diag.get('comment')} last_error={diag.get('last_error')} | {_symbol_info_str(si)}",
            level=lvl
        )
    except Exception as e:
        diag = {"erro_diag": str(e), "volume_enviado": volume}
    return diag


def _retcode_sucesso(ret) -> bool:
    """Sucesso se ordem foi executada/colocada: 10009 (DONE), 10008 (PLACED), 10004 (DONE_PARTIAL) ou 100xx genérico."""
    try:
        if isinstance(ret, str):
            return ret.startswith("100")
        return int(ret) in (10009, 10008, 10004)
    except Exception:
        return False


def _resolver_volume(ativo: str, volume_param, contexto) -> float:
    """
    Resolve o volume a enviar:
      - se volume_param vier válido (>0), usa-o;
      - senão tenta calcular_lote_adaptativo(…);
      - senão usa config.volumes[ativo] ou volume_padrao;
      - por fim, ajusta ao step/min/max do símbolo.
    """
    cfg = carregar_config()
    origem = "param"
    vol = _sf(volume_param, None)
    detalhes_lote = None

    if not vol or vol <= 0:
        origem = "adaptativo"
        vol = None
        if calcular_lote_adaptativo:
            try:
                # Compatível com diferentes assinaturas: não passamos kwargs desconhecidas.
                # Preferimos return_all=True; se não suportar, caímos para retorno escalar.
                try:
                    res = calcular_lote_adaptativo(ativo=ativo, contexto=contexto, return_all=True)  # type: ignore
                    detalhes_lote = res
                    vol = _sf(res.get("lote_utilizado"), None)
                    if vol is None:
                        vol = _sf(res.get("lote_bruto_teorico"), None)
                except TypeError:
                    vol = _sf(calcular_lote_adaptativo(ativo=ativo, contexto=contexto), None)  # type: ignore
            except Exception as e:
                log_event(f"[ENVIO ORDEM] calcular_lote_adaptativo falhou: {e}", level="warning")
                vol = None
        if not vol or vol <= 0:
            origem = "config"
            vol = cfg.get("volumes", {}).get(ativo, cfg.get("volume_padrao", 0.01))
            vol = _sf(vol, 0.01)

    # Ajuste ao step/min/max do símbolo
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
                vol_adj = _ajustar_volume_para_simbolo(vol, si)
                if abs(vol_adj - vol) > 1e-12:
                    log_event(
                        f"[ENVIO ORDEM] Ajuste de volume pelo símbolo | {ativo}: {vol} -> {vol_adj} | {_symbol_info_str(si)}",
                        level="warning"
                    )
                vol = vol_adj
        except Exception as e:
            log_event(f"[ENVIO ORDEM] Falha ao ajustar volume pelo símbolo | {ativo}: {e}", level="warning")

    # Log detalhado da origem e detalhes (se houver)
    if detalhes_lote:
        log_event(f"[ENVIO ORDEM] Volume resolvido={vol} (origem={origem}) | detalhes_lote={detalhes_lote}", level="info")
    else:
        log_event(f"[ENVIO ORDEM] Volume resolvido={vol} (origem={origem})", level="info")

    return float(vol)


# -------------------------
# API principal
# -------------------------
def abrir_ordem_e_registrar(
    ativo: str,
    tipo: str,
    volume=None,                  # <- aceita None
    timestamp=None,
    preco_abertura=None,          # mantido p/ compat
    contexto="",
    observacao="",
):
    """Envia ordem via executor e registra no banco quando sucesso. Sempre devolve 'motivo' e 'diagnostico'."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Resolve volume (evita float(None))
    ctx_dict = contexto if isinstance(contexto, dict) else {}
    vol_final = _resolver_volume(ativo, volume, ctx_dict)

    # Envio — propagamos o contexto para o executor (ele loga e pode usar se for calcular lote)
    resultado = enviar_ordem(tipo, ativo, timestamp, volume=vol_final, contexto=ctx_dict)

    ticket = resultado.get("order")
    retcode = resultado.get("retcode")
    preco_exec = resultado.get("preco") or preco_abertura

    # Diagnóstico adicional: MERGE (não sobrescreve o que já veio do executor)
    diag_extra = _diagnosticar_envio(ativo, tipo, vol_final, resultado)
    if isinstance(resultado, dict):
        if isinstance(resultado.get("diagnostico"), dict):
            # não sobrescreve chaves existentes
            for k, v in diag_extra.items():
                resultado["diagnostico"].setdefault(k, v)
        else:
            resultado["diagnostico"] = diag_extra

    # Sucesso → registra
    if ticket and str(ticket).isdigit() and int(ticket) > 0 and _retcode_sucesso(retcode):
        try:
            registrar_operacao(
                ativo=ativo,
                ticket=ticket,
                padrao="",  # o main_loop registra contexto/padrão detalhado em outras rotinas
                regime="",
                contexto=contexto if isinstance(contexto, str) else str(contexto),
                hora=datetime.now().strftime("%H:%M:%S"),
                motivo_fechamento="",
                retcode=retcode,
                volume=vol_final,
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
        f"[ENVIO ORDEM] NÃO REGISTRADA | ativo={ativo} tipo={tipo} volume={vol_final} "
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
