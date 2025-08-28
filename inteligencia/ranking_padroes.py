# inteligencia/ranking_padroes.py
# -*- coding: utf-8 -*-

"""
MÓDULO COMBINADO: Detecção + Ranking de Padrões

O que há aqui:
- DETECÇÃO: rankear(row) -> (padrao: str, score: float em [0..1])
  Aliases: rankear_padroes, detectar_padrao, calcular_padrao
  -> Tolerante a valores faltantes; nunca levanta exceção; sempre retorna uma tupla.

- RANKING (placar): atualizar_ranking, exibir_top_padroes, obter_score_padrao, obter_score_e_n
  -> Compatível com sua versão: 
     * obter_score_padrao(padrao, ativo=None) -> float  (mantido)
     * obter_score_padrao(df: DataFrame) -> (padrao, score)  (NOVO)

Motivação:
- Evitar o erro de "unpack" quando quem chama faz `padrao, score = obter_score_padrao(df)`.
"""

from __future__ import annotations

import os
import math
from typing import Any, Mapping, Tuple, List

import pandas as pd

try:
    import numpy as np  # usado apenas no helper de DF
except Exception:
    np = None

try:
    from utils.debug_logger import log_event
except Exception:
    # Fallback simples se o util não estiver disponível
    import logging
    _fallback_logger = logging.getLogger("ranking_padroes")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    def log_event(msg: str, level: str = "info"):
        lvl = getattr(_fallback_logger, level if hasattr(_fallback_logger, level) else "info")
        lvl(msg)

# =====================================================================
# ------------------------- DETECÇÃO DE PADRÕES -----------------------
# =====================================================================

def _safe_get(m: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        if m is None:
            return default
        if hasattr(m, "get"):
            return m.get(key, default)
        return m[key]
    except Exception:
        return default

def _as_float(v: Any, default: float = math.nan) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        if math.isnan(f):
            return default
        return f
    except Exception:
        try:
            # tenta trocar vírgula por ponto
            f = float(str(v).replace(",", "."))
            return f if not math.isnan(f) else default
        except Exception:
            return default

def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "t", "y", "yes", "sim"}

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x is None or math.isnan(x):
        return lo
    return max(lo, min(hi, float(x)))

def rankear(row: Mapping[str, Any]) -> Tuple[str, float]:
    """
    Analisa a última linha de dados e retorna (padrao, score∈[0..1]).
    - Nunca levanta exceção para o chamador.
    - Sempre retorna uma TUPLA.
    - Se nada bater, retorna ("sem_padrao", 0.0).
    """
    try:
        if row is None:
            log_event("[PADRAO] Linha NULA recebida. Retornando fallback.", level="warning")
            return "sem_padrao", 0.0

        # OHLC
        o = _as_float(_safe_get(row, "open"))
        h = _as_float(_safe_get(row, "high"))
        l = _as_float(_safe_get(row, "low"))
        c = _as_float(_safe_get(row, "close"))

        if any(math.isnan(x) for x in (o, h, l, c)) or l > h:
            log_event(f"[PADRAO] OHLC inválido (o={o} h={h} l={l} c={c}). Fallback.", level="warning")
            return "sem_padrao", 0.0

        rng = max(h - l, 1e-9)
        body = abs(c - o)
        body_ratio = body / rng

        # Features auxiliares (podem não existir)
        squeeze = _as_bool(_safe_get(row, "squeeze"))
        bb_bw = _as_float(_safe_get(row, "bb_bandwidth"), default=math.nan)
        up_shadow = _as_float(_safe_get(row, "upper_shadow_ratio"), default=math.nan)
        low_shadow = _as_float(_safe_get(row, "lower_shadow_ratio"), default=math.nan)
        body_pos = _as_float(_safe_get(row, "body_position"), default=math.nan)
        breakout_flag = _as_bool(_safe_get(row, "breakout_adaptive"))
        bull_eng = _as_bool(_safe_get(row, "bullish_engulfing"))
        bear_eng = _as_bool(_safe_get(row, "bearish_engulfing"))
        regime = _safe_get(row, "regime_kmeans", None)

        log_event(
            "[PADRAO] Base OHLC: "
            f"o={o:.5f} h={h:.5f} l={l:.5f} c={c:.5f} | rng={rng:.6f} body_ratio={body_ratio:.3f} "
            f"squeeze={squeeze} bb_bw={'nan' if math.isnan(bb_bw) else f'{bb_bw:.6f}'} "
            f"up={'nan' if math.isnan(up_shadow) else f'{up_shadow:.3f}'} "
            f"low={'nan' if math.isnan(low_shadow) else f'{low_shadow:.3f}'} "
            f"pos={'nan' if math.isnan(body_pos) else f'{body_pos:.3f}'} reg={regime}",
            level="debug",
        )

        # Candidatos: (nome, score, tag_debug)
        candidatos: List[Tuple[str, float, str]] = []

        # 1) Engulfings se já pré-computados nos features
        if bull_eng:
            sc = 0.65
            if squeeze: sc += 0.10
            if not math.isnan(bb_bw): sc += _clamp((0.002 - bb_bw) / 0.002, 0.0, 0.15)
            candidatos.append(("bullish_engulfing", _clamp(sc), "flag+squeeze+bw"))

        if bear_eng:
            sc = 0.65
            if squeeze: sc += 0.10
            if not math.isnan(bb_bw): sc += _clamp((0.002 - bb_bw) / 0.002, 0.0, 0.15)
            candidatos.append(("bearish_engulfing", _clamp(sc), "flag+squeeze+bw"))

        # 2) Inside bar (aproximação: corpo pequeno + compressão)
        if body_ratio < 0.25 and (squeeze or (not math.isnan(bb_bw) and bb_bw < 0.0012)):
            sc = 0.55
            if not math.isnan(bb_bw): sc += _clamp((0.0012 - bb_bw) / 0.0012, 0.0, 0.25)
            candidatos.append(("inside_bar", _clamp(sc), "small_body+compressao"))

        # 3) Breakout adaptativo
        if breakout_flag:
            sc = 0.60
            if not math.isnan(bb_bw): sc += _clamp((0.0015 - bb_bw) / 0.0015, 0.0, 0.20)
            candidatos.append(("breakout_adaptive", _clamp(sc), "flag+bw"))

        # 4) Sombra longa — martelo / estrela cadente
        if not math.isnan(up_shadow) and not math.isnan(low_shadow):
            if body_ratio < 0.35 and low_shadow > 1.8 and up_shadow < 0.6:
                sc = 0.58 + (0.07 if squeeze else 0.0)
                candidatos.append(("hammer", _clamp(sc), "shadow_asym"))
            if body_ratio < 0.35 and up_shadow > 1.8 and low_shadow < 0.6:
                sc = 0.58 + (0.07 if squeeze else 0.0)
                candidatos.append(("shooting_star", _clamp(sc), "shadow_asym"))

        # 5) Compressão geral (squeeze/bandwidth)
        if squeeze or (not math.isnan(bb_bw) and bb_bw < 0.0009):
            sc = 0.50
            if not math.isnan(bb_bw): sc += _clamp((0.0010 - bb_bw) / 0.0010, 0.0, 0.20)
            candidatos.append(("compressao", _clamp(sc), "squeeze/bw"))

        # Ajuste por regime (quando numérico: 0=lateral, 1=tendência, 2=explosão)
        try:
            reg_i = int(regime) if regime is not None else None
        except Exception:
            reg_i = None

        if candidatos and reg_i is not None:
            for i, (nome, sc, tag) in enumerate(candidatos):
                if nome in ("inside_bar", "compressao") and reg_i == 0:
                    sc = _clamp(sc + 0.05)
                if nome in ("breakout_adaptive",) and reg_i in (1, 2):
                    sc = _clamp(sc + 0.05)
                candidatos[i] = (nome, sc, f"{tag}+reg{reg_i}")

        # Seleção
        if candidatos:
            candidatos.sort(key=lambda t: t[1], reverse=True)
            melhor = candidatos[0]
            log_event(
                f"[PADRAO] Selecionado: {melhor[0]} (score={melhor[1]:.3f}) | "
                f"candidatos={[(n, round(s,3)) for n, s, _ in candidatos]}",
                level="info",
            )
            return melhor[0], melhor[1]

        log_event("[PADRAO] Nenhum padrão elegível. Retornando ('sem_padrao', 0.0).", level="debug")
        return "sem_padrao", 0.0

    except Exception as e:
        # Segurança: nunca propagar
        log_event(f"[PADRAO] Erro inesperado em rankear(): {e}. Fallback ('sem_padrao', 0.0).", level="error")
        return "sem_padrao", 0.0

# Aliases para compatibilidade
rankear_padroes = rankear
detectar_padrao = rankear
calcular_padrao = rankear

# =====================================================================
# ----------------- Helper NOVO para DataFrame inteiro ----------------
# =====================================================================

def _inferir_padrao_df(df: "pd.DataFrame") -> Tuple[str, float]:
    """
    Heurística leve e estável usando os últimos closes:
    - 'tend_alta': inclinação > limiar
    - 'tend_baixa': inclinação < -limiar
    - 'range': caso contrário
    Score ∈ [0,1] pela força da tendência; em 'range' penaliza volatilidade.
    """
    try:
        if df is None or df.empty or "close" not in df.columns:
            return "sem_padrao", 0.0
        if np is None:
            return "sem_padrao", 0.0

        s = df["close"].tail(100).astype(float).dropna()
        if s.shape[0] < 5:
            return "sem_padrao", 0.0

        y = s.values
        x = np.arange(len(y), dtype=float)
        # regressão linear simples
        slope = float(np.polyfit(x, y, 1)[0])
        std = float(np.std(y))
        if std <= 0:
            return "sem_padrao", 0.0

        strength = abs(slope) / (std + 1e-12)
        LIM = 0.20

        if slope > 0 and strength >= LIM:
            padrao = "tend_alta"
            score = _clamp(0.5 + min(0.5, strength))
        elif slope < 0 and strength >= LIM:
            padrao = "tend_baixa"
            score = _clamp(0.5 + min(0.5, strength))
        else:
            padrao = "range"
            rng = float(np.max(y) - np.min(y)) / (float(np.mean(y)) + 1e-12)
            score = _clamp(0.7 - min(0.3, rng * 3.0))

        return padrao, float(score)
    except Exception:
        return "sem_padrao", 0.0

# =====================================================================
# --------------------------- RANKING (CSV) ---------------------------
# =====================================================================

RANKING_PATH = "dados/ranking_padroes.csv"
COLS = ["ativo", "padrao", "acertos", "erros", "neutros", "total"]

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Garante colunas esperadas e tipos básicos."""
    if df is None or df.empty:
        return pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df.columns:
            df[c] = 0 if c in ("acertos", "erros", "neutros", "total") else ""
    for c in ("acertos", "erros", "neutros", "total"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["ativo"] = df["ativo"].astype(str)
    df["padrao"] = df["padrao"].astype(str)
    return df[COLS]

def _read_df() -> pd.DataFrame:
    """Lê o CSV do ranking (se não existir, retorna df vazio com schema)."""
    try:
        if not os.path.exists(RANKING_PATH) or os.stat(RANKING_PATH).st_size == 0:
            return _ensure_schema(pd.DataFrame(columns=COLS))
        df = pd.read_csv(RANKING_PATH)
        return _ensure_schema(df)
    except Exception as e:
        log_event(f"[RANKING] Falha ao ler CSV: {e}", level="error")
        return _ensure_schema(pd.DataFrame(columns=COLS))

def _safe_write_df(df: pd.DataFrame, path: str):
    """Escrita atômica: grava em .tmp e faz replace."""
    try:
        _ensure_dir(path)
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception as e:
        log_event(f"[RANKING] Falha ao gravar CSV: {e}", level="error")

def atualizar_ranking(ativo, row: dict):
    """
    Atualiza o ranking de padrões vencedores/perdedores.
    - Se 'row' trouxer 'padrao', usa-o diretamente.
    - Caso contrário, recompõe com (sinal, regime, contexto).
    - acerto/erro/neutro com base em 'lucro' (>0, <0, ==0).
    """
    try:
        padrao_in = row.get("padrao")
        sinal = str(row.get("sinal", "sem_sinal"))
        regime = str(row.get("regime", "indefinido"))
        contexto = str(row.get("contexto", "sem_contexto"))

        padrao = str(padrao_in) if padrao_in not in (None, "", "None") else f"{sinal}_{regime}_{contexto}"

        lucro_raw = row.get("lucro", 0)
        try:
            lucro = float(lucro_raw)
        except Exception:
            try:
                lucro = float(str(lucro_raw).replace(",", "."))
            except Exception:
                lucro = 0.0

        acerto = 1 if lucro > 0 else 0
        erro = 1 if lucro < 0 else 0
        neutro = 1 if lucro == 0 else 0

        df = _read_df()
        cond = (df["ativo"] == str(ativo)) & (df["padrao"] == padrao)

        if df[cond].empty:
            novo = pd.DataFrame([{
                "ativo": str(ativo),
                "padrao": padrao,
                "acertos": acerto,
                "erros": erro,
                "neutros": neutro,
                "total": 1
            }])
            df = pd.concat([df, novo], ignore_index=True)
        else:
            idx = df[cond].index[0]
            df.at[idx, "acertos"] += acerto
            df.at[idx, "erros"] += erro
            df.at[idx, "neutros"] += neutro
            df.at[idx, "total"] += 1

        df = _ensure_schema(df)
        _safe_write_df(df, RANKING_PATH)

        log_event(
            f"[RANKING] Atualizado | ativo={ativo} padrao='{padrao}' "
            f"(+a={acerto}, +e={erro}, +n={neutro})",
            level="info"
        )
    except Exception as e:
        log_event(f"[RANKING] Erro ao atualizar ranking: {e}", level="error")

def exibir_top_padroes(top=10):
    """Apenas logging: top vencedores e perdedores por taxa."""
    try:
        df = _read_df()
        if df.empty:
            log_event("[RANKING] Ranking vazio.", level="warning")
            return
        df["taxa_acerto"] = df["acertos"] / df["total"].replace(0, 1)
        df["taxa_erro"] = df["erros"] / df["total"].replace(0, 1)
        top_win = df.sort_values(by=["taxa_acerto", "total"], ascending=[False, False]).head(top)
        top_lose = df.sort_values(by=["taxa_erro", "total"], ascending=[False, False]).head(top)
        log_event(f"[RANKING] Top vencedores: {top_win.to_dict(orient='records')}", level="info")
        log_event(f"[RANKING] Top perdedores: {top_lose.to_dict(orient='records')}", level="info")
    except Exception as e:
        log_event(f"[RANKING] Erro ao exibir ranking: {e}", level="error")

def obter_score_padrao(padrao_ou_df, ativo=None):
    """
    Sobrecarga compatível:
      - obter_score_padrao(padrao: str, ativo: str|None) -> float      (MODO ANTIGO)
      - obter_score_padrao(df: pandas.DataFrame) -> (padrao: str, score: float)  (NOVO)
    """
    # NOVO: modo DataFrame → (padrao, score)
    try:
        if isinstance(padrao_ou_df, pd.DataFrame):
            return _inferir_padrao_df(padrao_ou_df)
    except Exception:
        pass

    # MODO ANTIGO: (padrao, ativo) → float com base no CSV
    try:
        score, _n = obter_score_e_n(str(padrao_ou_df), ativo=ativo)
        return score
    except Exception as e:
        log_event(f"[RANKING] Erro em obter_score_padrao('{padrao_ou_df}'): {e}", level="error")
        return 0.0

def obter_score_e_n(padrao, ativo=None):
    """
    Retorna (score, N) do padrão.
      - score ∈ [-1, +1]  com score = (acertos - erros) / total
      - N = total de ocorrências
    Se 'ativo' for fornecido, filtra por ativo; senão, usa o consolidado.
    """
    try:
        df = _read_df()
        if df.empty:
            return 0.0, 0

        if ativo is not None:
            dfp = df[(df["padrao"] == str(padrao)) & (df["ativo"] == str(ativo))]
        else:
            dfp = df[df["padrao"] == str(padrao)]

        if dfp.empty:
            return 0.0, 0

        acertos = int(dfp["acertos"].sum())
        erros = int(dfp["erros"].sum())
        total = int(dfp["total"].sum())
        total = max(total, 0)

        if total == 0:
            return 0.0, 0

        score = (acertos - erros) / total
        score = max(-1.0, min(1.0, float(score)))  # clamp
        log_event(f"[RANKING] obter_score_e_n('{padrao}', ativo={ativo}) -> score={score:.2f}, N={total}", level="info")
        return score, total

    except Exception as e:
        log_event(f"[RANKING] Erro em obter_score_e_n('{padrao}'): {e}", level="error")
        return 0.0, 0

__all__ = [
    # detecção
    "rankear", "rankear_padroes", "detectar_padrao", "calcular_padrao",
    # ranking
    "atualizar_ranking", "exibir_top_padroes", "obter_score_padrao", "obter_score_e_n",
]
