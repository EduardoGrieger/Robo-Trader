# inteligencia/contexto_ordens.py
import os
import pandas as pd
from utils.debug_logger import log_event

CONTEXTOS_PATH = "dados/contextos_ordens.csv"

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _safe_append_csv(df_new: pd.DataFrame, path: str) -> None:
    """
    Append atômico: une colunas (antigo ∪ novo), reindexa, grava em .tmp e faz replace.
    Evita corromper CSV em quedas/brutal kill.
    """
    _ensure_dir(path)
    if os.path.exists(path):
        try:
            df_old = pd.read_csv(path)
        except Exception as e:
            log_event(f"[CONTEXTO] Falha ao ler CSV existente: {e} — recriando arquivo.", level="warning")
            df_old = pd.DataFrame()
        # união de colunas
        cols = sorted(set(df_old.columns).union(df_new.columns))
        df_cat = pd.concat(
            [df_old.reindex(columns=cols), df_new.reindex(columns=cols)],
            ignore_index=True
        )
    else:
        df_cat = df_new.copy()

    tmp = path + ".tmp"
    df_cat.to_csv(tmp, index=False)
    os.replace(tmp, path)

def registrar_contexto_ordem(ordem_info: dict, contexto_info: dict):
    """
    Salva contexto completo da ordem (merge plano de ordem_info e contexto_info) em CSV atômico.
    - Aceita chaves adicionais sem quebrar o CSV (schema se expande automaticamente).
    - Nunca lança exceção para cima: loga e retorna silenciosamente em caso de erro.
    """
    try:
        if not isinstance(ordem_info, dict) or not isinstance(contexto_info, dict):
            log_event("[CONTEXTO] registrar_contexto_ordem: parâmetros inválidos (esperado dict).", level="warning")
            return
        registro = {**ordem_info, **contexto_info}
        df = pd.DataFrame([registro])
        _safe_append_csv(df, CONTEXTOS_PATH)
        log_event(f"[CONTEXTO] Contexto de ordem salvo: chaves={list(registro.keys())}", level="info")
    except Exception as e:
        log_event(f"[CONTEXTO] Falha ao registrar contexto da ordem: {e}", level="error")

def ranking_padroes_vencedores(top: int = 10):
    """
    Gera (e loga) ranking dos padrões mais lucrativos e mais perdedores.
    Mantém compat com uso anterior, mas privilegia logs em vez de print.
    """
    try:
        if not os.path.exists(CONTEXTOS_PATH):
            log_event("[CONTEXTO] Nenhum contexto registrado ainda.", level="warning")
            return

        df = pd.read_csv(CONTEXTOS_PATH)
        # validações mínimas
        if "padrao" not in df.columns or "lucro" not in df.columns:
            log_event("[CONTEXTO] Colunas 'padrao' e 'lucro' são necessárias para ranking.", level="warning")
            return

        df["lucro"] = pd.to_numeric(df["lucro"], errors="coerce").fillna(0.0)
        rank_sum = df.groupby("padrao", as_index=False)["lucro"].sum()
        top_ganha = rank_sum.sort_values(by="lucro", ascending=False).head(top)
        top_perde = rank_sum.sort_values(by="lucro", ascending=True).head(top)

        log_event(f"[CONTEXTO] === PADRÕES MAIS LUCRATIVOS === {top_ganha.to_dict(orient='records')}", level="info")
        log_event(f"[CONTEXTO] === PADRÕES MAIS PERDEDORES === {top_perde.to_dict(orient='records')}", level="info")

    except Exception as e:
        log_event(f"[CONTEXTO] Erro ao gerar ranking: {e}", level="error")

__all__ = ["registrar_contexto_ordem", "ranking_padroes_vencedores"]
