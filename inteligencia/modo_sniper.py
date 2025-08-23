# inteligencia/modo_sniper.py
import numpy as np
from utils.debug_logger import log_event
from utils.utils import carregar_config

def _rsi(prices, periodo=14):
    if len(prices) < periodo + 1:
        return 50.0
    deltas = np.diff(prices)
    up = deltas.clip(min=0).mean()
    down = (-deltas.clip(max=0)).mean()
    rs = up / (down if down != 0 else 1e-12)
    return 100. - 100. / (1. + rs)

def _vol_satisfaz(require: str, atual: str) -> bool:
    """Comparação ordenada: baixa < media < alta."""
    order = {"baixa": 0, "media": 1, "alta": 2}
    if require is None or require == "" or require.lower() == "any":
        return True
    if atual is None:
        return True  # sem contexto → não bloqueia
    return order.get(atual, -1) >= order.get(require, 1)  # default 'media'

def detectar_sniper(candles, ativo, contexto=None, regime=None):
    """
    Dispara sniper quando há confluência configurável:
      - RSI extremo (<= low OU >= high)
      - Regime requerido (ex.: 'tendencia', ou 'any' para ignorar)
      - Squeeze: por padrão exige NÃO estar em squeeze (pode desligar)
      - Volatilidade mínima (baixa|media|alta), default 'media'
    """
    try:
        cfg = carregar_config()
        rsi_low  = float(cfg.get("sniper_rsi_low", 30))
        rsi_high = float(cfg.get("sniper_rsi_high", 70))
        rsi_per  = int(cfg.get("sniper_rsi_period", 14))

        require_regime = str(cfg.get("sniper_require_regime", "tendencia")).lower()  # 'tendencia' | 'any'
        require_no_squeeze = bool(cfg.get("sniper_requires_no_squeeze", True))
        min_vol = str(cfg.get("sniper_min_volatility", "media")).lower()            # 'baixa'|'media'|'alta'|'any'

        if candles is None or "close" not in candles.columns:
            log_event(f"[SNIPER] Candles inválidos para {ativo}.", level="warning")
            return False

        rsi_val = _rsi(candles["close"].astype(float).values, periodo=rsi_per)
        ok_rsi = (rsi_val <= rsi_low) or (rsi_val >= rsi_high)

        if require_regime in ("", "any", "none"):
            ok_regime = True
        else:
            ok_regime = (str(regime).lower() == require_regime)

        if contexto:
            sqz = bool(contexto.get("squeeze", False))
            vol = str(contexto.get("volatilidade", "")).lower()
        else:
            sqz = False
            vol = None

        ok_squeeze = (not require_no_squeeze) or (not sqz)
        ok_vol = _vol_satisfaz(min_vol, vol)

        sniper = bool(ok_rsi and ok_regime and ok_squeeze and ok_vol)
        log_event(
            f"[SNIPER] {ativo} rsi={rsi_val:.1f} "
            f"regime={regime} (need={require_regime}) "
            f"squeeze={sqz} (need_no={require_no_squeeze}) "
            f"vol={vol} (min={min_vol}) -> {sniper}",
            level="info"
        )
        return sniper
    except Exception as e:
        log_event(f"[SNIPER] Erro em detectar_sniper({ativo}): {e}", level="error")
        return False
