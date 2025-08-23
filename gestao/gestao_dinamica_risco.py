# gestao/gestao_dinamica_risco.py
# Ajuste dinâmico de risco com persistência robusta e suavização bayesiana.

from __future__ import annotations
import os
import pandas as pd
from typing import Any, Dict

import importlib

def _load_func(module_names, func_name):
    for m in module_names:
        try:
            mod = importlib.import_module(m)
            fn = getattr(mod, func_name, None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None

log_event = _load_func(["utils.debug_logger", "debug_logger"], "log_event") \
    or (lambda msg, level="info": print(f"[{level.upper()}] {msg}"))

HISTORICO_PATH = os.environ.get("HIST_RISCO_PATH", os.path.join("dados", "historico_operacoes.csv"))

def salvar_performance(resultado: Dict[str, Any]):
    """
    Salva 1 linha em CSV de forma resistente a concorrência (append atômico).
    Campo recomendado: {timestamp (UTC), ativo, resultado (win/loss/1/0), ...}
    """
    try:
        os.makedirs(os.path.dirname(HISTORICO_PATH), exist_ok=True)
        df = pd.DataFrame([resultado])
        tmp = HISTORICO_PATH + ".tmp"
        header_needed = not os.path.exists(HISTORICO_PATH)
        # escreve temporário
        df.to_csv(tmp, mode="w", index=False, header=header_needed)
        # append atômico
        with open(tmp, "rb") as src, open(HISTORICO_PATH, "ab") as dst:
            if not header_needed:
                # pular header do tmp
                next(src)
            dst.write(src.read())
        os.remove(tmp)
        log_event(f"[RISCO DINÂMICO] Resultado salvo: {resultado}", level="info")
    except Exception as e:
        log_event(f"[RISCO DINÂMICO] Erro ao salvar resultado: {e}", level="error")

def _norm_result(v) -> int:
    """
    Normaliza resultado para {1=win, 0=loss}.
    Aceita strings ('win'/'loss') e números (>0 -> win).
    """
    try:
        s = str(v).strip().lower()
        if s in {"win", "sucesso", "true", "1"}:
            return 1
        if s in {"loss", "falha", "false", "0", "-1"}:
            return 0
        f = float(v)
        return 1 if f > 0 else 0
    except Exception:
        return 0

def calcular_fator_risco(
    ativo: str,
    janela_minutos: int = 240,
    n_min: int = 10,
    suav: float = 0.5,
    fator_min: float = 0.3,
    fator_max: float = 1.5,
    prior: float = 0.5
) -> float:
    """
    Calcula fator multiplicador de risco ∈ [fator_min, fator_max].
    - Suavização p/ amostras pequenas: p' = (1-suav)*p + suav*prior (se n < n_min).
    - Usa timestamps em UTC.
    """
    try:
        if not os.path.exists(HISTORICO_PATH):
            return 1.0

        df = pd.read_csv(HISTORICO_PATH, parse_dates=["timestamp"], infer_datetime_format=True)
        if df.empty:
            return 1.0

        # Timestamps -> UTC naive para comparação segura
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
        agora = pd.Timestamp.utcnow().tz_localize(None)
        limite = agora - pd.Timedelta(minutes=janela_minutos)

        if "ativo" in df.columns:
            df = df[df["ativo"] == ativo]
        df = df[df["timestamp"] >= limite].copy()

        if df.empty:
            return 1.0

        df["ok"] = df.get("resultado", 0).apply(_norm_result)
        n = int(df.shape[0])
        taxa = float(df["ok"].mean()) if n else prior

        taxa_eff = ((1 - suav) * taxa + suav * prior) if n < n_min else taxa
        fator = max(fator_min, min(fator_max, fator_max * taxa_eff))

        log_event(
            f"[RISCO DINÂMICO] {ativo}: n={n}, taxa={taxa:.2f}, taxa_eff={taxa_eff:.2f}, fator={fator:.2f}",
            level="info"
        )
        return float(fator)
    except Exception as e:
        log_event(f"[RISCO DINÂMICO] Erro ao calcular fator risco: {e}", level="error")
        return 1.0

def ajustar_volume_base(volume: float, fator_risco: float, minimo: float = 0.01, maximo: float = 100.0) -> float:
    """
    Aplica fator de risco ao volume base e clampa.
    """
    try:
        v = float(volume) * float(fator_risco)
        return float(max(minimo, min(maximo, v)))
    except Exception:
        return volume

if __name__ == "__main__":
    # Exemplo rápido
    salvar_performance({
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "ativo": "EURUSD",
        "resultado": "win"
    })
    f = calcular_fator_risco("EURUSD")
    print("fator_risco(EURUSD) =", f)
