# utils/sinal_utils.py
from __future__ import annotations
from typing import Dict, Tuple

# ========= SUAS FUNÇÕES (mantidas) =========

def normalizar_sinal(sinal):
    """
    Converte sinal de qualquer formato para int padrão: 1 (compra), -1 (venda), 0 (neutro)
    """
    if isinstance(sinal, int):
        if sinal in [1, -1, 0]:
            return sinal
    s = str(sinal).strip().lower()
    if s in ['1', 'compra', 'comprar', 'buy', 'long']:
        return 1
    if s in ['-1', 'venda', 'vender', 'sell', 'short']:
        return -1
    if s in ['0', 'neutro', 'neutral', 'none', 'nan']:
        return 0
    # Default fallback:
    return 0

def sinal_to_str(sinal):
    """
    Converte sinal int para string amigável (ex: para logs)
    """
    s = normalizar_sinal(sinal)
    if s == 1:
        return "compra"
    if s == -1:
        return "venda"
    return "neutro"

# ========= ADITIVOS COMPATÍVEIS =========

def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def escolher_sinal_por_maior_conf(prob: Dict[int, float]) -> Tuple[int, float]:
    """
    prob: dict {-1: p_down, 0: p_neu, 1: p_up} (qualquer escala [0..1]).
    Retorna (sinal, confianca). Nunca lança exceção.
    """
    if not isinstance(prob, dict) or not prob:
        return 0, 0.0
    p_dn = _clamp01(prob.get(-1, 0.0)); p_neu = _clamp01(prob.get(0, 0.0)); p_up = _clamp01(prob.get(1, 0.0))
    vencedor = max([(-1, p_dn), (0, p_neu), (1, p_up)], key=lambda t: t[1])
    return vencedor[0], float(vencedor[1])

def aplicar_cap_neutro(prob: Dict[int, float], cap_neutro: float = 0.60) -> Dict[int, float]:
    """
    Limita a probabilidade do neutro para evitar dominância.
    """
    p = dict(prob or {})
    if 0 in p:
        p[0] = min(_clamp01(p[0]), float(cap_neutro))
    return p
