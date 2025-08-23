# inteligencia/clusterizar_regimes.py
import numpy as np
import pandas as pd
from utils.utils import carregar_config
from utils.debug_logger import log_event

# Estado simples para histerese (não persiste entre reinícios)
_LAST_REGIME = None
_STREAK = 0

def _calc_retornos(closes: np.ndarray, use_log: bool) -> np.ndarray:
    if use_log:
        return np.diff(np.log(closes))
    # retornos percentuais (mais estáveis que diffs absolutas)
    prev = closes[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (closes[1:] - prev) / np.where(prev == 0, 1e-12, prev)
    r[np.isnan(r) | np.isinf(r)] = 0.0
    return r

def _classificar_por_centroide(media: float, vol: float, pesos: tuple[float, float]) -> tuple[str, dict]:
    """
    Distância Euclidiana ponderada por pesos (media, vol).
    Centróides podem ser ajustados no futuro se quiser.
    """
    centroides = {
        "lateral":   (0.0,   0.0002),
        "tendencia": (0.001, 0.0010),
        "explosao":  (0.0,   0.0050),
    }
    wm, wv = pesos
    feats = np.array([media, vol], dtype=float)
    dists = {}
    for reg, cent in centroides.items():
        c = np.array(cent, dtype=float)
        diff = feats - c
        # distância euclidiana com pesos (equivalente a reescalar features)
        d = np.sqrt((wm * diff[0] ** 2) + (wv * diff[1] ** 2))
        dists[reg] = float(d)
    regime = min(dists, key=dists.get)
    return regime, dists

def _aplicar_histerese(regime_atual: str, min_streak: int) -> str:
    """
    Evita flip de regime em um único candle.
    Só confirma troca após 'min_streak' leituras consecutivas.
    """
    global _LAST_REGIME, _STREAK
    if _LAST_REGIME is None:
        _LAST_REGIME = regime_atual
        _STREAK = 1
        return regime_atual

    if regime_atual == _LAST_REGIME:
        _STREAK = min(_STREAK + 1, 1_000_000)
        return _LAST_REGIME

    # candidato a troca
    _STREAK -= 1
    if _STREAK <= -min_streak:
        _LAST_REGIME = regime_atual
        _STREAK = 1
        return regime_atual
    # ainda não confirmou a troca
    return _LAST_REGIME

def detectar_regime(candles=None, retorno_completo=False):
    """
    Detecta o regime do mercado com base em candles recentes.
    Regimes: 'lateral', 'tendencia', 'explosao', 'indefinido'.
    Sempre retorna algo — nunca quebra.
    """
    try:
        cfg = carregar_config()
        janela = int(cfg.get("janela_regime", 30))
        use_log = bool(cfg.get("regime_use_logret", False))
        pesos_cfg = cfg.get("regime_weights", {"media": 1.0, "vol": 3.0})
        pesos = (float(pesos_cfg.get("media", 1.0)), float(pesos_cfg.get("vol", 3.0)))
        min_streak = max(1, int(cfg.get("regime_hysterese_min_streak", 2)))

        if candles is None or not hasattr(candles, "shape"):
            log_event("[REGIME] Nenhum DataFrame válido recebido.", level="warning")
            regime = "indefinido"; media = volatilidade = None
            return (regime, media, volatilidade) if retorno_completo else regime

        # Aceita DataFrame ou dict-like com colunas
        if not hasattr(candles, "columns") or "close" not in candles.columns:
            log_event("[REGIME] Coluna 'close' ausente.", level="warning")
            regime = "indefinido"; media = volatilidade = None
            return (regime, media, volatilidade) if retorno_completo else regime

        qtd = len(candles)
        if qtd < max(janela, 3):
            log_event(f"[REGIME] Dados insuficientes ({qtd}) para janela={janela}.", level="warning")
            regime = "indefinido"; media = volatilidade = None
            return (regime, media, volatilidade) if retorno_completo else regime

        closes = np.asarray(candles["close"].tail(janela), dtype=float)
        retornos = _calc_retornos(closes, use_log=use_log)

        # estatísticas simples (poderia usar EWMA no futuro)
        media = float(np.mean(retornos))
        volatilidade = float(np.std(retornos, ddof=1))

        regime_bruto, dists = _classificar_por_centroide(media, volatilidade, pesos)
        regime_final = _aplicar_histerese(regime_bruto, min_streak=min_streak)

        log_event(
            f"[REGIME] bruto={regime_bruto} → final={regime_final} | "
            f"media={media:.5f} vol={volatilidade:.5f} | pesos={pesos} | dists={dists}",
            level="info"
        )

        if retorno_completo:
            return regime_final, media, volatilidade
        return regime_final

    except Exception as e:
        log_event(f"[REGIME] Erro ao detectar regime: {e}", level="error")
        return ("indefinido", None, None) if retorno_completo else "indefinido"
