# inteligencia/validar_tp_sl.py
import pandas as pd
from utils.debug_logger import log_event

def validar_tp_sl_historico(
    df_candles, preco_entrada, tipo_ordem, tp_pips, sl_pips, ponto_pip,
    prioridade_intracandle: str = "SL"  # "SL" | "TP" | "ordem"
):
    """
    Retorna: 'win' (TP primeiro), 'loss' (SL primeiro), 'neutro' (nenhum).
    prioridade_intracandle:
      - "SL": assume SL primeiro se high>=TP e low<=SL no mesmo candle (conservador)
      - "TP": assume TP primeiro (otimista)
      - "ordem": decide pelo nível MAIS PRÓXIMO do open do candle (heurística)
    Obs.: Sem dados intrabar, a sequência real é ambígua — por isso a prioridade configurável.
    """
    try:
        if df_candles is None or df_candles.empty:
            log_event("[TP/SL] DF futuro vazio.", level="warning"); return "neutro"
        for col in ("high", "low"):
            if col not in df_candles.columns:
                log_event(f"[TP/SL] Coluna ausente: {col}", level="error"); return "neutro"

        open_series = df_candles["open"] if "open" in df_candles.columns else None

        if tipo_ordem == "compra":
            preco_tp = float(preco_entrada) + tp_pips * ponto_pip
            preco_sl = float(preco_entrada) - sl_pips * ponto_pip
            for _, c in df_candles.iterrows():
                hi = float(c["high"]); lo = float(c["low"])
                hit_tp = hi >= preco_tp
                hit_sl = lo <= preco_sl
                if hit_tp and hit_sl:
                    if prioridade_intracandle.upper() == "TP":
                        log_event("[TP/SL] (buy) TP&SL no mesmo candle → TP por prioridade.", level="info")
                        return "win"
                    elif prioridade_intracandle.lower() == "ordem" and open_series is not None:
                        o = float(c["open"])
                        d_tp = abs(preco_tp - o)
                        d_sl = abs(o - preco_sl)
                        if d_tp < d_sl:
                            return "win"
                        elif d_sl < d_tp:
                            return "loss"
                        else:
                            return "loss"  # empate → conservador
                    else:
                        log_event("[TP/SL] (buy) TP&SL no mesmo candle → SL por prioridade.", level="info")
                        return "loss"
                if hit_sl:
                    return "loss"
                if hit_tp:
                    return "win"

        elif tipo_ordem == "venda":
            preco_tp = float(preco_entrada) - tp_pips * ponto_pip
            preco_sl = float(preco_entrada) + sl_pips * ponto_pip
            for _, c in df_candles.iterrows():
                hi = float(c["high"]); lo = float(c["low"])
                hit_tp = lo <= preco_tp
                hit_sl = hi >= preco_sl
                if hit_tp and hit_sl:
                    if prioridade_intracandle.upper() == "TP":
                        log_event("[TP/SL] (sell) TP&SL no mesmo candle → TP por prioridade.", level="info")
                        return "win"
                    elif prioridade_intracandle.lower() == "ordem" and "open" in df_candles.columns:
                        o = float(c["open"])
                        d_tp = abs(o - preco_tp)
                        d_sl = abs(preco_sl - o)
                        if d_tp < d_sl:
                            return "win"
                        elif d_sl < d_tp:
                            return "loss"
                        else:
                            return "loss"
                    else:
                        log_event("[TP/SL] (sell) TP&SL no mesmo candle → SL por prioridade.", level="info")
                        return "loss"
                if hit_sl:
                    return "loss"
                if hit_tp:
                    return "win"

        log_event("[TP/SL] Nenhum alvo atingido.", level="info")
        return "neutro"
    except Exception as e:
        log_event(f"[TP/SL] Erro: {e}", level="error")
        return "neutro"
