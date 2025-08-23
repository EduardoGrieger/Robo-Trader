# scripts/relatorio_labeling_ab.py
# -*- coding: utf-8 -*-
import os, json, pandas as pd

try:
    from utils.debug_logger import log_event
except Exception:
    def log_event(msg, level="info"):
        print(f"[{level.upper()}] {msg}")

def _read_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_event(f"[REL-AB] falha ao ler {p}: {e}", level="error"); return {}

def _fmt(m: dict, k: str):
    v = m.get(k, None)
    try: return f"{float(v):.4f}"
    except Exception: return "-" if v is None else str(v)

def main(base="logs"):
    sumA = _read_json(os.path.join(base, "A", "walkforward_summary.json"))
    sumB = _read_json(os.path.join(base, "B", "walkforward_summary.json"))
    if not sumA and not sumB:
        log_event("[REL-AB] Não há summaries de A/B. Rode o treino/avaliador com A/B.", level="error"); return
    metA = sumA.get("metrics_mean", {}); metB = sumB.get("metrics_mean", {})
    def row(label, key): return f"{label:>14} | { _fmt(metA, key):>10} | { _fmt(metB, key):>10}"
    print("\n=========== COMPARATIVO LABELING A vs B ===========")
    print(f"Arquivos base: {base}")
    print("                 |      A     |      B    ")
    for k in ["f1_updown","mcc","accuracy","neutral_rate","ev_mean_pips","sharpe_approx"]:
        print(row(k, k))
    def key(m): return (float(m.get("f1_updown", float("-inf"))), float(m.get("mcc", float("-inf"))), -float(m.get("neutral_rate", float("inf"))))
    escolha = "A" if key(metA) > key(metB) else "B"
    print(f"\n➡️  Vencedor (heurística): {escolha}")
    if "tau_star" in sumA or "delta_star" in sumA: print(f"A: tau*={sumA.get('tau_star')} | delta*={sumA.get('delta_star')}")
    if "tau_star" in sumB or "delta_star" in sumB: print(f"B: tau*={sumB.get('tau_star')} | delta*={sumB.get('delta_star')}")
    print(f"Folds A: {sumA.get('n_folds','?')} | Folds B: {sumB.get('n_folds','?')}")
    print("\nArquivos:"); print(f"- {os.path.join(base, 'A', 'walkforward_summary.json')}"); print(f"- {os.path.join(base, 'B', 'walkforward_summary.json')}")
    promoted = os.path.join(base, 'walkforward_summary.json')
    if os.path.exists(promoted): print(f"- {promoted}  (promovido)")
    print("====================================================\n")

if __name__ == "__main__":
    main()
