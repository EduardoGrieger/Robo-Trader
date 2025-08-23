# scripts/relatorio_walkforward.py
# -*- coding: utf-8 -*-
import os, json, time, argparse
import pandas as pd

try:
    from utils.debug_logger import log_event
except Exception:
    def log_event(msg, level="info"):
        print(f"[{level.upper()}] {msg}")

def arquivo_stale(path: str, dias: int = 7) -> bool:
    try:
        mtime = os.path.getmtime(path)
        return (time.time() - mtime) > (dias*86400)
    except Exception:
        return True

def _print_report(csv_path, json_path):
    df = pd.read_csv(csv_path)
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    num_cols = [c for c in ["accuracy","mcc","f1_macro","f1_updown","neutral_rate","ev_mean_pips","sharpe_approx"] if c in df.columns]
    agg = df[num_cols].mean().to_dict() if num_cols else {}
    if "tau" in df.columns and "delta" in df.columns:
        top = (df.groupby(["tau","delta"]).size().reset_index(name="freq").sort_values("freq", ascending=False).head(5))
    else:
        top = pd.DataFrame(columns=["tau","delta","freq"])

    print("\n========== RELATÓRIO WALK-FORWARD ==========")
    print(f"Folds: {int(summary.get('n_folds', len(df)))} | Modelo: {summary.get('model','?')} | Anchored: {summary.get('anchored', True)}")
    print("Classes:", summary.get("classes", []))
    if "tau_star" in summary and "delta_star" in summary:
        print(f"τ* (tau_star): {summary['tau_star']:.3f}  |  Δ* (delta_star): {summary['delta_star']:.3f}")
    print("\n-- MÉDIAS (por fold) --")
    for k in ["f1_updown","mcc","accuracy","neutral_rate","ev_mean_pips","sharpe_approx"]:
        if k in agg: print(f"{k:>15}: {agg[k]:.4f}")
    print("\n-- TOP (tau, delta) por frequência --")
    if not top.empty:
        for _,row in top.iterrows():
            print(f"(tau={row['tau']:.3f}, delta={row['delta']:.3f}) → freq={int(row['freq'])}")
    else:
        print("Sem dados de (tau, delta) por fold.")
    print("\nArquivos:"); print(f"- {csv_path}"); print(f"- {json_path}")
    print("============================================\n")

def main():
    ap = argparse.ArgumentParser(description="Relatório do walk-forward promovido.")
    ap.add_argument("--base", "-b", default="logs", help="Diretório base onde estão os arquivos (default: logs)")
    args = ap.parse_args()
    base = args.base

    csv_path = os.path.join(base, "fold_metrics_walkforward.csv")
    json_path = os.path.join(base, "walkforward_summary.json")
    if os.path.exists(csv_path) and os.path.exists(json_path):
        if arquivo_stale(json_path, dias=7):
            log_event("[REL] Atenção: thresholds walk-forward parecem desatualizados (>7 dias).", level="warning")
        _print_report(csv_path, json_path); return

    cand = []
    for sub in ["A", "B"]:
        c = os.path.join(base, sub, "fold_metrics_walkforward.csv")
        j = os.path.join(base, sub, "walkforward_summary.json")
        if os.path.exists(c) and os.path.exists(j):
            cand.append((c, j))
    if cand:
        print("[INFO] Promovidos não encontrados; exibindo o mais recente entre A/B.")
        cand.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
        c, j = cand[0]; _print_report(c, j)
        print("Dica: rode 'python scripts/relatorio_labeling_ab.py' para ver A vs B.")
        return

    log_event(f"[REL] Arquivos não encontrados em '{base}'. Esperado: fold_metrics_walkforward.csv e walkforward_summary.json", level="error")

if __name__ == "__main__":
    main()
