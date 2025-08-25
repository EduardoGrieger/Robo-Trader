# utils/lote_adaptativo.py
import json
from typing import Any, Dict, Optional
from utils.debug_logger import log_event

def carregar_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

# --------- Helpers ---------
def _as_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(str(x).replace(",", "."))
    except Exception:
        return float(default)

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _round2(x: float) -> float:
    try:
        return round(float(x), 2)
    except Exception:
        return 0.01

def _volume_cfg(cfg: dict, ativo: str, volume_padrao: float) -> float:
    try:
        return float((cfg.get("volumes", {}) or {}).get(ativo, volume_padrao))
    except Exception:
        return volume_padrao

def _ctx_num(contexto: Dict[str, Any], *keys, default=None) -> Optional[float]:
    """Tenta ler um float do contexto por várias chaves (ex.: 'volatilidade','std20','bb_bandwidth')."""
    for k in keys:
        if k in contexto:
            v = _as_float(contexto.get(k), None)
            if v is not None:
                return v
    return default

# --------- Núcleo da lógica ---------
def _calc_core(
    *,
    cfg: dict,
    saldo: float,
    volatilidade: float,
    drawdown_pct: float,
    meta_dia_atingida: bool,
    performance_recente: float,
    score_conf: float,
    ativo: str
) -> Dict[str, Any]:
    """Implementa a lógica de cálculo e retorna um dicionário detalhado."""
    saldo = _as_float(saldo, cfg.get("capital_conta", 50000.0))
    risco_pct = _as_float(cfg.get("risco_por_trade_percentual", 1.0), 1.0)  # %
    stop_pts = int(_as_float(cfg.get("sl_pips", 20), 20))
    volume_padrao = _as_float(cfg.get("volume_padrao", 0.01), 0.01)

    # Valor configurado para o ativo (piso e, opcionalmente, teto)
    vol_cfg = _volume_cfg(cfg, ativo, volume_padrao)

    usar_piso = bool(cfg.get("usar_volume_config_como_piso", True))
    usar_teto = bool(cfg.get("usar_volume_config_como_teto", True))

    max_drawdown = _as_float(cfg.get("max_drawdown_diario", 0.05), 0.05)  # fração (0.05 = 5%)

    # 1) Risco absoluto em USD
    risco_usd = float(saldo) * (risco_pct / 100.0) * float(score_conf)

    # 2) Ajustes (sem incluir vol_cfg aqui para evitar “dupla capagem”)
    volatilidade = _as_float(volatilidade, 1.0)
    ajuste_vol = max(0.2, min(2.0, 1.0 / max(0.01, float(volatilidade))))

    drawdown_pct = _as_float(drawdown_pct, 0.0)  # já em %
    ajuste_dd = 1.0
    if drawdown_pct > 100.0 * max_drawdown:      # > 5% diário por default
        ajuste_dd = 0.7
    if drawdown_pct > 2 * 100.0 * max_drawdown:  # > 10% diário por default
        ajuste_dd = 0.4
    if meta_dia_atingida:
        ajuste_dd *= 0.3
    if _as_float(performance_recente, 0.0) < 0:
        ajuste_dd *= 0.7

    # 3) Lote teórico
    valor_ponto = 10.0  # padrão FX
    if stop_pts * valor_ponto == 0:
        lote_bruto = 0.01
    else:
        lote_bruto = risco_usd / (stop_pts * valor_ponto)

    ajuste_final = min(ajuste_vol, ajuste_dd)
    lote_calc = max(0.0, lote_bruto * ajuste_final)

    # 4) Piso/Teto via config do ativo
    lote_util = float(lote_calc)
    aplicou_piso = False
    aplicou_teto = False

    if usar_piso and lote_util < vol_cfg:
        lote_util = vol_cfg
        aplicou_piso = True

    if usar_teto and lote_util > vol_cfg:
        lote_util = vol_cfg
        aplicou_teto = True

    lote_util = _round2(lote_util)

    return {
        "lote_bruto_teorico": round(float(lote_calc), 4),
        "lote_utilizado": float(lote_util),
        "volume_cfg": float(vol_cfg),
        "aplicou_piso": bool(aplicou_piso),
        "aplicou_teto": bool(aplicou_teto),
        "origem": {
            "ajuste_volatilidade": float(ajuste_vol),
            "ajuste_drawdown": float(ajuste_dd),
            "score_conf": float(score_conf),
        },
        "saldo": float(saldo),
        "risco_pct": float(risco_pct),
        "stop_pts": int(stop_pts),
        "valor_ponto": float(valor_ponto),
    }

# --------- Função pública compatível (Nova + Legada) ---------
def calcular_lote_adaptativo(
    # --- Assinatura “nova” (institucional)
    ativo: Optional[str] = None,
    contexto: Optional[Dict[str, Any]] = None,
    return_all: bool = False,
    # --- kwargs diversos (aceita 'confianca', 'preco', etc.)
    **kwargs
):
    """
    Compatível com:
      1) NOVO: calcular_lote_adaptativo(ativo="EURUSD", contexto={...}, return_all=True, confianca=0.8, ...)
      2) LEGADO: calcular_lote_adaptativo(saldo=..., volatilidade=..., drawdown_pct=..., score_conf=..., ativo="EURUSD")

    Notas:
      - Se vier 'confianca' (0..1) e 'score_conf' não vier, mapeamos 'confianca' -> score_conf.
      - Em modo NOVO, usamos proxies de vol do contexto: 'volatilidade' numérica, ou 'std20'/'bb_bandwidth'.
    """
    cfg = carregar_config()

    # Detecta “modo legado” (sem 'ativo' explícito e com parâmetros legados no kwargs)
    modo_legado = False
    if ativo is None and any(k in kwargs for k in ("saldo", "volatilidade", "drawdown_pct", "score_conf")):
        modo_legado = True

    if modo_legado:
        # --------- MODO LEGADO ---------
        ativo = kwargs.get("ativo", "EURUSD")
        saldo = kwargs.get("saldo", cfg.get("capital_conta", 50000.0))
        volatilidade = kwargs.get("volatilidade", 1.0)
        drawdown_pct = kwargs.get("drawdown_pct", 0.0)
        meta_dia_atingida = kwargs.get("meta_dia_atingida", False)
        performance_recente = kwargs.get("performance_recente", 0.0)
        score_conf = kwargs.get("score_conf", None)
        if score_conf is None:
            score_conf = kwargs.get("confianca", 1.0)  # <-- mapeia confianca -> score_conf no legado
        score_conf = _clamp(_as_float(score_conf, 1.0), 0.0, 1.0)

        log_event(
            f"[LOTE-DEBUG] (LEGADO) saldo={saldo}, volatilidade={volatilidade}, drawdown_pct={drawdown_pct}, "
            f"meta_dia_atingida={meta_dia_atingida}, performance_recente={performance_recente}, "
            f"score_conf={score_conf}, ativo={ativo}",
            level="debug"
        )

        try:
            det = _calc_core(
                cfg=cfg,
                saldo=_as_float(saldo, cfg.get("capital_conta", 50000.0)),
                volatilidade=_as_float(volatilidade, 1.0),
                drawdown_pct=_as_float(drawdown_pct, 0.0),
                meta_dia_atingida=bool(meta_dia_atingida),
                performance_recente=_as_float(performance_recente, 0.0),
                score_conf=score_conf,
                ativo=str(ativo),
            )
            msg = (f"[LOTE] (LEGADO) {ativo} | lote_teorico={det['lote_bruto_teorico']:.4f} | "
                   f"lote_utilizado={det['lote_utilizado']:.2f} | volume_cfg={det['volume_cfg']:.2f} | "
                   f"ajustes=vol:{det['origem']['ajuste_volatilidade']:.3f}, "
                   f"dd:{det['origem']['ajuste_drawdown']:.3f}, score:{det['origem']['score_conf']:.2f}")
            if det["aplicou_piso"]:
                msg += " | 🟰 Piso do config aplicado"
            if det["aplicou_teto"]:
                msg += " | ⚠️ Teto do config aplicado"
            log_event(msg, level="info")
            return det if return_all else det["lote_utilizado"]
        except Exception as e:
            log_event(f"[LOTE] Erro (LEGADO) para {ativo}: {e}", level="error")
            return {"lote_utilizado": 0.01} if return_all else 0.01

    # --------- MODO NOVO (institucional) ---------
    contexto = contexto or {}
    ativo = ativo or kwargs.get("ativo", "EURUSD")

    # Mapeia confiança nova → score_conf legado
    confianca = kwargs.get("confianca", contexto.get("confianca", 1.0))
    score_conf = _clamp(_as_float(confianca, 1.0), 0.0, 1.0)

    # Valores padrão quando chamados no fluxo novo
    saldo = cfg.get("capital_conta", 50000.0)

    # volatilidade: tenta numérica direta; senão proxies (std20/bb_bandwidth); senão 1.0
    vol_num = None
    if isinstance(contexto, dict):
        vol_num = _ctx_num(contexto, "volatilidade", "std20", "bb_bandwidth", default=None)
    volatilidade = _as_float(vol_num, 1.0)

    # extras do contexto (opcionais)
    drawdown_pct = _as_float((contexto.get("drawdown_pct") if isinstance(contexto, dict) else None), 0.0)
    meta_dia_atingida = bool((contexto.get("meta_dia_atingida") if isinstance(contexto, dict) else False))
    performance_recente = _as_float((contexto.get("performance_recente") if isinstance(contexto, dict) else None), 0.0)

    log_event(
        f"[LOTE-DEBUG] (NOVO) ativo={ativo} contexto_keys={list(contexto.keys()) if isinstance(contexto, dict) else 'N/A'} "
        f"confianca={score_conf}",
        level="debug"
    )

    try:
        det = _calc_core(
            cfg=cfg,
            saldo=_as_float(saldo, cfg.get("capital_conta", 50000.0)),
            volatilidade=_as_float(volatilidade, 1.0),
            drawdown_pct=_as_float(drawdown_pct, 0.0),
            meta_dia_atingida=bool(meta_dia_atingida),
            performance_recente=_as_float(performance_recente, 0.0),
            score_conf=score_conf,
            ativo=str(ativo),
        )
        msg = (f"[LOTE] (NOVO) {ativo} | lote_teorico={det['lote_bruto_teorico']:.4f} | "
               f"lote_utilizado={det['lote_utilizado']:.2f} | volume_cfg={det['volume_cfg']:.2f} | "
               f"ajustes=vol:{det['origem']['ajuste_volatilidade']:.3f}, "
               f"dd:{det['origem']['ajuste_drawdown']:.3f}, score:{det['origem']['score_conf']:.2f} | "
               f"contexto={contexto}")
        if det["aplicou_piso"]:
            msg += " | 🟰 Piso do config aplicado"
        if det["aplicou_teto"]:
            msg += " | ⚠️ Teto do config aplicado"
        log_event(msg, level="info")
        return det if return_all else det["lote_utilizado"]
    except Exception as e:
        log_event(f"[LOTE] Erro (NOVO) para {ativo}: {e}", level="error")
        return {"lote_utilizado": 0.01} if return_all else 0.01

__all__ = ["calcular_lote_adaptativo"]
