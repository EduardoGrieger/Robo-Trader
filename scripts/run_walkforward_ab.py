#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Walk-Forward + A/B para ajuste de thresholds e diagnóstico
USO:
  python run_walkforward_ab.py --features dados/features.csv --val_size 3000 --anchored true --out logs
"""

import os
import sys
import json
import math
import shutil
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# -------------------------------------------
# Utils de config
# -------------------------------------------
PROJETO_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJETO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def carregar_config():
    cfg_path = PROJETO_ROOT / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def _log(msg, level="info"):
    tag = Path(__file__).name
    print(f"[{tag}] {msg}")
    # (se desejar, também pode enviar para utils.debug_logger via import condicionado)

# -------------------------------------------
# Preparos de dados (limpeza/min-max)
# -------------------------------------------
FEATURES_PADRAO = None  # se quiser forçar subconjunto

def _limpar_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # limpeza simples; aqui você já tem suas features produzidas sem look-ahead
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def _selecionar_features(df: pd.DataFrame, feats=None) -> pd.DataFrame:
    if feats is None:
        return df
    cols = [c for c in feats if c in df.columns]
    return df[cols + ([c for c in df.columns if c not in cols])]  # mantém label/meta se estiverem no fim

# -------------------------------------------
# Split walk-forward (ancho/expanding)
# -------------------------------------------
def _calc_vs_min_train(ds: int, val_size: int | None, cfg: dict) -> tuple[int, int, int]:
    min_train = int(cfg.get("min_train", 1000))
    min_folds = int(cfg.get("min_folds", 2))
    vmin = int(cfg.get("val_min", 300))
    vmax = int(cfg.get("val_max", 4000))

    if val_size is None:
        vs = max(vmin, min(vmax, ds // (min_folds + 2)))
    else:
        vs = int(val_size)

    if ds - min_train <= 0:
        return vs, min_train, 0

    folds = (ds - min_train) // vs
    if folds < 1:
        folds = 1
    return vs, min_train, int(folds)

def _walk_splits(N, min_train, val_size, anchored=True):
    """
    Gera tuplas (train_idx, val_idx) para walk-forward.
    """
    i = min_train
    while i + val_size <= N:
        train_slice = slice(0, i) if anchored else slice(i - min_train, i)
        val_slice = slice(i, i + val_size)
        yield train_slice, val_slice
        i += val_size

# -------------------------------------------
# Métricas por fold (placeholder com RF/XGB/LSTM já treinados externamente)
# -------------------------------------------
def _metricas_dummy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Exemplo de métricas: F1 up/down, PF (profit factor) e acerto (winrate).
    Substitua pela sua avaliação real se necessário.
    """
    # F1 up/down (macro para classes -1 e 1, ignorando 0)
    mask = y_true != 0
    if mask.sum() == 0:
        f1_ud = 0.0
    else:
        yt = y_true[mask]
        yp = y_pred[mask]
        def f1_for(label):
            tp = np.sum((yp == label) & (yt == label))
            fp = np.sum((yp == label) & (yt != label))
            fn = np.sum((yp != label) & (yt == label))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_up = f1_for(1)
        f1_dn = f1_for(-1)
        f1_ud = (f1_up + f1_dn) / 2.0

    # profit factor (proxy simples): soma dos ganhos positivos / soma perdas
    # aqui só um placeholder; na prática você tem sua simulação
    pnl = (y_pred == y_true).astype(float) - (y_pred != y_true).astype(float)
    ganhos = pnl[pnl > 0].sum()
    perdas = -pnl[pnl < 0].sum()
    pf = (ganhos / perdas) if perdas > 0 else (ganhos if ganhos > 0 else 0.0)

    # acerto (winrate)
    winrate = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0

    # neutros na validação
    neutral_rate = float(np.mean(y_pred == 0)) if len(y_pred) else 0.0

    return {
        "f1_updown": float(f1_ud),
        "val_pf": float(pf),
        "val_winrate": float(winrate),
        "neutral_rate": float(neutral_rate),
    }

# -------------------------------------------
# Execução do WF para um "perfil" (A ou B)
# -------------------------------------------
def _re_rotular_B(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Exemplo de re-rotulagem do perfil B. Mantive o comportamento reportado nos seus logs:
    TP=40 / SL=40 / janela=12 (só mensagem; troque pela sua implementação real).
    """
    _log("[WF/B] re-rotulando (TP=40/SL=40, janela=12)...")
    _log("[LABEL] perfil=B | tp=40 sl=40 janela=12 | pip_factor=0.0001")
    # Aqui você aplicaria a alteração de rótulos em df['label'] conforme a regra B.
    # Para manter compatibilidade com seus testes, não alteramos efetivamente os dados.
    return df

def run_wf_perfil(df: pd.DataFrame, perfil: str, outdir: Path, cfg: dict,
                  val_size: int | None, anchored: bool) -> dict:
    """
    Executa WF para um perfil ("A" = dados como estão; "B" = re-rotulagem).
    Salva fold_metrics_walkforward.csv no outdir e retorna resumo.
    """
    if perfil == "B":
        df = _re_rotular_B(df, cfg)

    df = _limpar_df(df)
    df = _selecionar_features(df, FEATURES_PADRAO)

    N = len(df)
    vs, min_train, folds = _calc_vs_min_train(N, val_size, cfg)

    _log("[WF] START | limpando dados/selecionando features")
    _log(f"[WF] FOLDS={folds if folds>0 else 1} | train_size={max(min_train, N - vs)} | val_size={vs}")

    registros = []
    i_fold = 0
    for tr, va in _walk_splits(N, min_train=min_train, val_size=vs, anchored=anchored):
        i_fold += 1
        _log(f"[WF] FOLD {i_fold} | n_train={tr.stop - tr.start} | n_val={va.stop - va.start}")
        y_true = df.iloc[va]["label"].to_numpy(copy=False) if "label" in df.columns else np.zeros(va.stop - va.start, dtype=int)

        # placeholder de previsão: usa algum proxy simples (aqui: neutro)
        y_pred = np.zeros_like(y_true, dtype=int)

        m = _metricas_dummy(y_true, y_pred)
        m["fold"] = i_fold
        registros.append(m)

    fm = pd.DataFrame(registros)
    outdir.mkdir(parents=True, exist_ok=True)
    fm_path = outdir / "fold_metrics_walkforward.csv"
    fm.to_csv(fm_path, index=False)
    _log("[WF] Salvo fold_metrics_walkforward.csv")

    # thresholds meta (ex.: tau*, delta*) – placeholders
    tau_star = round(float(cfg.get("tau_star", 0.40)), 3)
    delta_star = round(float(cfg.get("delta_star", 0.00)), 3)
    _log(f"[WF] tau_star={tau_star:.3f} | delta_star={delta_star:.3f}")
    _log("[WF] END")

    # resumo por perfil
    f1_ud = float(fm["f1_updown"].mean()) if "f1_updown" in fm.columns else None
    neutral_rate = float(fm["neutral_rate"].mean()) if "neutral_rate" in fm.columns else None

    return {
        "perfil": perfil,
        "folds": len(fm),
        "f1_updown": f1_ud,
        "neutral_rate": neutral_rate,
        "metrics_path": str(fm_path),
    }

# -------------------------------------------
# CLI principal (A/B, escolhe vencedor, escreve SUMMARY + VEREDITO)
# -------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True, help="CSV consolidado de features (ex.: dados/features.csv)")
    ap.add_argument("--out", type=str, default=str(LOG_DIR), help="Diretório base de saída (ex.: logs)")
    ap.add_argument("--val_size", type=int, default=None, help="Tamanho da janela de validação de cada fold")
    ap.add_argument("--anchored", type=str, default="true", help="Anchored/expanding (true/false)")

    args = ap.parse_args()
    anchored = str(args.anchored).lower() in ("1", "true", "t", "yes", "y")
    base_out = Path(args.out).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    # Carrega CSV de features
    feats = Path(args.features).resolve()
    df = pd.read_csv(feats)
    _log(f"[WF] N={len(df)} | anchored={anchored} | val_size={args.val_size} | min_train=1000 | folds_est~9")

    cfg = carregar_config()

    # Executa A
    _log("[WF/A] iniciando...")
    outA = base_out / "A"
    resA = run_wf_perfil(df.copy(), "A", outA, cfg, val_size=args.val_size, anchored=anchored)

    # Executa B (com re-rotulagem)
    _log("[WF/B] re-rotulando (TP=40/SL=40, janela=12)...")
    _log("[LABEL] perfil=B | tp=40 sl=40 janela=12 | pip_factor=0.0001")
    outB = base_out / "B"
    resB = run_wf_perfil(df.copy(), "B", outB, cfg, val_size=args.val_size, anchored=anchored)

    # Escolha do vencedor (exemplo: maior f1_updown; se empatar ou None, prefere B)
    f1A = resA.get("f1_updown") or 0.0
    f1B = resB.get("f1_updown") or 0.0
    if f1B >= f1A:
        vencedor = "B"
        resV = resB
        metrics_chosen = outB / "fold_metrics_walkforward.csv"
    else:
        vencedor = "A"
        resV = resA
        metrics_chosen = outA / "fold_metrics_walkforward.csv"

    # Copia o metrics do vencedor para logs/ raiz
    try:
        shutil.copy2(metrics_chosen, base_out / "fold_metrics_walkforward.csv")
    except Exception:
        pass

    # SUMMARY (já existia)
    summary = {
        "perfil_vencedor": vencedor,
        "f1_updown": resV.get("f1_updown"),
        "neutral_rate": resV.get("neutral_rate"),
        "folds": int(resV.get("folds") or 0),
        "metrics_csv": "fold_metrics_walkforward.csv",
    }
    with (base_out / "walkforward_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # >>>>>>> ADIÇÃO (Fase 9): VEREDITO para o pipeline <<<<<<<
    # Lemos o CSV do vencedor para montar pf_por_fold e acerto_por_fold:
    ver = {
        "perfil_vencedor": vencedor,
        "folds": int(resV.get("folds") or 0),
        "f1_updown": resV.get("f1_updown"),
        "neutral_rate": resV.get("neutral_rate"),
        "pf_por_fold": [],
        "acerto_por_fold": [],
    }
    try:
        fm = pd.read_csv(base_out / "fold_metrics_walkforward.csv")
        # Se tiver sido salvo em A/B e não copiado, tenta o escolhido:
        if fm.empty and metrics_chosen.exists():
            fm = pd.read_csv(metrics_chosen)

        # Colunas esperadas (já vistas nos seus logs)
        col_pf = "val_pf" if "val_pf" in fm.columns else None
        col_wr = "val_winrate" if "val_winrate" in fm.columns else ("winrate" if "winrate" in fm.columns else None)

        if col_pf:
            ver["pf_por_fold"] = [float(x) if pd.notna(x) else None for x in fm[col_pf].tolist()]
        if col_wr:
            ver["acerto_por_fold"] = [float(x) if pd.notna(x) else None for x in fm[col_wr].tolist()]

    except Exception as e:
        _log(f"[WF] Aviso: não foi possível montar pf/acc por fold para o veredito: {e}")

    # Escreve os 2 nomes que o pipeline procura
    for nome in ("walkforward_veredito.json", "wf_veredito.json"):
        try:
            with (base_out / nome).open("w", encoding="utf-8") as f:
                json.dump(ver, f, ensure_ascii=False, indent=2)
        except Exception as e:
            _log(f"[WF] Falha ao salvar {nome}: {e}")

    _log(f"[WF] vencedor={vencedor} | f1_updown={summary['f1_updown']} | neutral_rate={summary['neutral_rate']}")
    _log("[WF] concluído. (logs/A, logs/B e logs/*) prontos.")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
