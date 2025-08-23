# inteligencia/ranking_padroes.py
import os
import pandas as pd
from utils.debug_logger import log_event

RANKING_PATH = "dados/ranking_padroes.csv"
COLS = ["ativo", "padrao", "acertos", "erros", "neutros", "total"]

# =============== utilitários internos ===============

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
    # Tipos numéricos
    for c in ("acertos", "erros", "neutros", "total"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    # Strings
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

# =============== API pública ===============

def atualizar_ranking(ativo, row: dict):
    """
    Atualiza o ranking de padrões vencedores/perdedores.
    - Se 'row' trouxer 'padrao', usa-o diretamente.
    - Caso contrário, recompõe com (sinal, regime, contexto).
    - acerto/erro/neutro com base em 'lucro' (>0, <0, ==0).
    """
    try:
        # Extrai campos usuais (com fallback)
        padrao_in = row.get("padrao")
        sinal = str(row.get("sinal", "sem_sinal"))
        regime = str(row.get("regime", "indefinido"))
        contexto = str(row.get("contexto", "sem_contexto"))

        # Use o padrao explícito se existir; senão, reconstrói
        padrao = str(padrao_in) if padrao_in not in (None, "", "None") else f"{sinal}_{regime}_{contexto}"

        lucro_raw = row.get("lucro", 0)
        try:
            lucro = float(lucro_raw)
        except Exception:
            # se vier string tipo "0.00" ou "None"
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
    """
    Apenas logging: top vencedores e perdedores por taxa.
    """
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

def obter_score_padrao(padrao, ativo=None) -> float:
    """
    Retorna score adaptativo do padrão (-1 a +1).
    Score = (acertos - erros) / total
    (mantido por compatibilidade)
    """
    try:
        score, _n = obter_score_e_n(padrao, ativo=ativo)
        return score
    except Exception as e:
        log_event(f"[RANKING] Erro em obter_score_padrao('{padrao}'): {e}", level="error")
        return 0.0

def obter_score_e_n(padrao, ativo=None):
    """
    NOVO: retorna (score, N) do padrão.
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
    "atualizar_ranking",
    "exibir_top_padroes",
    "obter_score_padrao",
    "obter_score_e_n",
]
