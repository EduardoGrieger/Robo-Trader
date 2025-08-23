# inteligencia/memoria_adaptativa.py
import os
import pandas as pd
from utils.debug_logger import log_event

MEMORIA_PATH = "dados/memoria_adaptativa.csv"
COLS = ["ativo", "padrao", "score", "acertos", "erros", "total"]

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _safe_write_df(df: pd.DataFrame, path: str):
    _ensure_dir(path)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df.columns:
            df[c] = 0 if c in ("score","acertos","erros","total") else ""
    df["ativo"] = df["ativo"].astype(str)
    df["padrao"] = df["padrao"].astype(str)
    for c in ("score","acertos","erros","total"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df[COLS]

def _load_df() -> pd.DataFrame:
    try:
        if not os.path.exists(MEMORIA_PATH) or os.stat(MEMORIA_PATH).st_size == 0:
            return _ensure_schema(pd.DataFrame(columns=COLS))
        return _ensure_schema(pd.read_csv(MEMORIA_PATH))
    except Exception as e:
        log_event(f"[MEMORIA] Falha ao ler CSV: {e}", level="error")
        return _ensure_schema(pd.DataFrame(columns=COLS))

def _resultado_para_score(resultado) -> int:
    s = str(resultado).strip().lower()
    if s in {"1","win","sucesso","acerto","10009","ok"}:
        return 1
    if s in {"-1","0","loss","falha","erro","stop"}:
        return -1
    return 0

def reforcar_memoria(ativo, row, resultado):
    """Atualiza memória adaptativa de forma atômica e à prova de schema."""
    try:
        ativo = str(ativo)
        padrao = str(row.get("padrao", "desconhecido"))
        score = _resultado_para_score(resultado)
        acerto = int(score > 0)
        erro = int(score < 0)

        df = _load_df()
        cond = (df["ativo"] == ativo) & (df["padrao"] == padrao)

        if df[cond].empty:
            novo = pd.DataFrame([{
                "ativo": ativo, "padrao": padrao,
                "score": score, "acertos": acerto, "erros": erro, "total": 1
            }])
            df = pd.concat([df, novo], ignore_index=True)
            idx = df.index[-1]
        else:
            idx = df[cond].index[0]
            df.at[idx, "score"] += score
            df.at[idx, "acertos"] += acerto
            df.at[idx, "erros"] += erro
            df.at[idx, "total"] += 1

        df = _ensure_schema(df)
        _safe_write_df(df, MEMORIA_PATH)
        log_event(f"[MEMORIA] '{padrao}'/{ativo}: score={df.at[idx,'score']} total={df.at[idx,'total']}", level="info")
    except Exception as e:
        log_event(f"[MEMORIA] Erro ao reforçar memória: {e}", level="error")

def obter_score_memoria(ativo, padrao) -> int:
    try:
        df = _load_df()
        cond = (df["ativo"] == str(ativo)) & (df["padrao"] == str(padrao))
        if df[cond].empty:
            return 0
        return int(df.loc[cond, "score"].values[0])
    except Exception as e:
        log_event(f"[MEMORIA] Erro ao obter score: {e}", level="error")
        return 0

def exibir_memoria(top=10):
    try:
        df = _load_df()
        if df.empty:
            log_event("[MEMORIA] Memória vazia.", level="warning"); return
        top_pos = df.sort_values("score", ascending=False).head(top)
        top_neg = df.sort_values("score", ascending=True).head(top)
        log_event(f"[MEMORIA] ++ {top_pos.to_dict(orient='records')}", level="info")
        log_event(f"[MEMORIA] -- {top_neg.to_dict(orient='records')}", level="info")
    except Exception as e:
        log_event(f"[MEMORIA] Erro ao exibir memória: {e}", level="error")

__all__ = ["reforcar_memoria","obter_score_memoria","exibir_memoria"]
