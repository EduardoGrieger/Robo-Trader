# scripts/run_walkforward_ab.py
# -*- coding: utf-8 -*-
import os, sys, json, argparse, inspect
import pandas as pd
import numpy as np

# garante acesso a utils/ e afins
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from utils.debug_logger import log_event
except Exception:
    def log_event(msg, level="info"):
        print(f"[{level.upper()}] {msg}")

# importa sua função existente (não vamos tocar nela)
from utils.walkforward import run_walk_forward_df

# relabel opcional (se existir); se não existir, segue sem B
try:
    from utils.labeling import relabel_profile
except Exception:
    relabel_profile = None

# -------------------- helpers de compatibilidade --------------------
def _alias_inject(dest: dict, params: set, value, *candidates) -> bool:
    """Se algum nome candidato existir na assinatura, injeta value nele e retorna True."""
    for name in candidates:
        if name in params:
            dest[name] = value
            return True
    return False

def _call_run_wf(func, *, df=None, features_csv=None, label_col="sinal",
                 val_size=None, train_size=None, anchored=True, outdir=None, model=None, tag=None):
    """
    Chama run_walk_forward_df mapeando ALIASES para a assinatura real, para ficar
    compatível com qualquer variação (val_size/valid_size/val_len, out/out_dir/outdir, etc).
    """
    sig = inspect.signature(func)
    params = set(sig.parameters.keys())
    kwargs = {}

    # df / dataframe / features_df
    if df is not None:
        _alias_inject(kwargs, params, df, "df", "data", "features_df", "dataframe")

    # features path (se for o caso)
    if features_csv is not None:
        _alias_inject(kwargs, params, features_csv, "features_csv", "features_path", "features", "csv_path", "path")

    # label
    _alias_inject(kwargs, params, label_col, "label_col", "label", "target", "target_col", "y_col", "yname")

    # val_size (muitos nomes possíveis)
    if val_size is not None:
        _alias_inject(kwargs, params, val_size,
                      "val_size", "valid_size", "val_len", "validation_size", "test_size",
                      "val_window", "window_val")

    # train_size (se quiser usar)
    if train_size is not None:
        _alias_inject(kwargs, params, train_size, "train_size", "train_len", "train_window")

    # anchored / expanding
    _alias_inject(kwargs, params, anchored, "anchored", "expanding", "anchor")

    # diretório de saída
    if outdir is not None:
        _alias_inject(kwargs, params, outdir, "outdir", "out_dir", "out", "save_dir", "logdir", "log_dir")

    # modelo (se a função aceitar)
    if model is not None:
        _alias_inject(kwargs, params, model, "model", "model_name")

    # tag (se aceitar)
    if tag is not None:
        _alias_inject(kwargs, params, tag, "tag", "profile_tag")

    # chamada final
    try:
        return func(**kwargs)
    except TypeError as e:
        # loga para depuração e tenta um fallback ainda mais minimalista
        log_event(f"[WF] assinatura incompatível, tentando fallback minimalista ({e})", "warning")
        mini = {}
        if "df" in params and df is not None: mini["df"] = df
        if "label_col" in params: mini["label_col"] = label_col
        if "anchored" in params: mini["anchored"] = anchored
        if outdir is not None:
            for k in ("outdir","out_dir","out"):
                if k in params: mini[k] = outdir; break
        return func(**mini)

def carregar_config(path="config.json"):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log_event(f"[CFG] falha ao carregar {path}: {e}", "warning")
    return {}

def _pick_best_labeling(sumA: dict, sumB: dict) -> str:
    def _key(s):
        m = s.get("metrics_mean", {}) if isinstance(s, dict) else {}
        return (
            float(m.get("f1_updown", float("-inf"))),
            float(m.get("mcc", float("-inf"))),
            -float(m.get("neutral_rate", float("inf"))),
        )
    return "A" if _key(sumA) > _key(sumB) else "B"

# -------------------- util para garantir folds --------------------
def _autoshrink_val_size(ds:int, val_size:int|None, anchored:bool, cfg:dict):
    """
    Ajusta val_size para garantir pelo menos min_folds; se necessário,
    informa o ajuste via log_event.
    """
    min_train = int(cfg.get("train_min", 1000))
    min_folds = int(cfg.get("min_folds", 2))
    vmin = int(cfg.get("val_min", 300))
    vmax = int(cfg.get("val_max", 4000))

    if val_size is None:
        # chute inicial: 1/(min_folds+2) do dataset
        vs = max(vmin, min(vmax, ds // (min_folds + 2)))
    else:
        vs = int(val_size)

    if anchored:
        # nº de folds (aprox) = floor((N - min_train)/val_size)
        if ds - min_train <= 0:
            return vs, min_train, 0
        folds = (ds - min_train) // max(1, vs)
        if folds < min_folds:
            new_vs = max(vmin, (ds - min_train) // max(1, min_folds))
            if new_vs != vs:
                log_event(f"[WF] val_size ajustado {vs} -> {new_vs} para obter >= {min_folds} folds (N={ds}, min_train={min_train}).", "warning")
                vs = new_vs
            folds = (ds - min_train) // max(1, vs)
        return vs, min_train, int(folds)
    else:
        # janelas desancoradas tendem a produzir mais folds; mantemos o vs
        folds = max(1, (ds // max(1, vs)) - 1)
        return vs, min_train, int(folds)

# -------------------- tau meta --------------------
def _optimize_tau_meta(scores, labels):
    """Otimiza tau maximizando MCC com penalização de cobertura."""
    import numpy as np
    try:
        from scipy.optimize import minimize_scalar as _ms
    except Exception:
        _ms = None
    from sklearn.metrics import matthews_corrcoef

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    # map 0/1/2 -> -1/0/1 if needed
    labels = np.where(labels==2, 1, np.where(labels==0, 0, np.where(labels==-1, -1, labels)))

    def objective(tau):
        preds = np.where(scores > tau, 1, np.where(scores < -tau, -1, 0))
        mcc = matthews_corrcoef(labels, preds)
        cov = np.mean(preds != 0)
        penalty = max(0, 0.30 - cov)*5 + max(0, cov - 0.65)*3
        return -(mcc - penalty)

    if _ms is None:
        grid = np.linspace(0.45, 0.70, 26)
        best = max(grid, key=lambda t: -objective(t))
        return float(best)
    res = _ms(objective, bounds=(0.45, 0.70), method='bounded')
    return float(res.x)

def _try_add_tau_meta(summary_dict, fm_path):
    try:
        import pandas as pd, numpy as np, os
        if not os.path.exists(fm_path):
            return summary_dict
        dfm = pd.read_csv(fm_path)
        cand_cols_score = [c for c in dfm.columns if 'score' in c.lower()]
        cand_cols_label = [c for c in dfm.columns if 'label' in c.lower()]
        if not cand_cols_score or not cand_cols_label:
            return summary_dict
        scores = dfm[cand_cols_score[0]].values
        labels = dfm[cand_cols_label[0]].values
        tau_meta = _optimize_tau_meta(scores, labels)
        summary_dict['tau_star_meta'] = float(tau_meta)
        # clamp delta
        d = float(summary_dict.get('delta_star', 0.0))
        summary_dict['delta_star_meta'] = 0.05 if d < 0.03 else d
    except Exception as e:
        log_event(f"[WF] não foi possível calcular tau_meta: {e}", "warning")
    return summary_dict


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Walk-Forward A/B e promoção do vencedor (sem mexer no treino).")
    ap.add_argument("--features", "-f", default="dados/features.csv", help="CSV de features.")
    ap.add_argument("--out", "-o", default="logs", help="Diretório de saída (default: logs).")
    ap.add_argument("--val_size", type=int, default=None, help="Tamanho da janela de validação por fold (ex.: 3000).")
    ap.add_argument("--anchored", type=lambda x: str(x).lower() != "false", default=True, help="Anchored expanding (default: True).")
    args = ap.parse_args()

    cfg = carregar_config("config.json")

    if not os.path.exists(args.features):
        log_event(f"[WF] features não encontradas: {args.features}", "error")
        sys.exit(2)

    # Carrega features
    df = pd.read_csv(args.features)
    if "sinal" not in df.columns:
        log_event("[WF] coluna 'sinal' ausente no features.csv", "error")
        sys.exit(2)

    ds = len(df)

    # Determina val_size (se não veio) e garante nº de folds
    val_size = args.val_size
    val_size, min_train, folds_est = _autoshrink_val_size(ds, val_size, args.anchored, cfg)
    log_event(f"[WF] N={ds} | anchored={args.anchored} | val_size={val_size} | min_train={min_train} | folds_est~{folds_est}", "info")

    # cria pastas
    outA = os.path.join(args.out, "A")
    outB = os.path.join(args.out, "B")
    os.makedirs(outA, exist_ok=True)
    os.makedirs(outB, exist_ok=True)

    # ---------- A) baseline usando 'sinal' já existente ----------
    log_event("[WF/A] iniciando...", "info")
    try:
        metA, sumA = _call_run_wf(
            run_walk_forward_df,
            df=df, label_col="sinal",
            val_size=val_size, anchored=args.anchored,
            outdir=outA, model="rf", tag="A"
        )
    except RuntimeError as e:
        msg = str(e)
        log_event(f"[WF/A] falhou: {msg} — tentando reduzir val_size e repetir.", "warning")
        # tenta uma redução adicional de 25% e reexecuta uma vez
        new_vs = max(200, int(val_size * 0.75))
        metA, sumA = _call_run_wf(
            run_walk_forward_df,
            df=df, label_col="sinal",
            val_size=new_vs, anchored=args.anchored,
            outdir=outA, model="rf", tag="A"
        )
        log_event(f"[WF/A] recuperado com val_size={new_vs}.", "info")

    # ---------- B) re-rotular simétrico (se houver labeling disponível) ----------
    metB, sumB = metA, sumA  # fallback
    if relabel_profile is not None:
        log_event("[WF/B] re-rotulando (TP=40/SL=40, janela=12)...", "info")
        try:
            dfB = relabel_profile(
                df, profile="B", pip_factor=0.0001,
                params_A={"tp_pips": int(cfg.get("tp_pips", 40)),
                          "sl_pips": int(cfg.get("sl_pips", 20)),
                          "janela": 20},
                params_B={"tp_pips": int(cfg.get("tp_pips", 40)),
                          "sl_pips": int(cfg.get("tp_pips", 40)),
                          "janela": 12}
            )
            try:
                metB, sumB = _call_run_wf(
                    run_walk_forward_df,
                    df=dfB, label_col="sinal",
                    val_size=val_size, anchored=args.anchored,
                    outdir=outB, model="rf", tag="B"
                )
            except RuntimeError as e:
                log_event(f"[WF/B] falhou: {e} — reduzindo val_size e repetindo.", "warning")
                new_vs = max(200, int(val_size * 0.75))
                metB, sumB = _call_run_wf(
                    run_walk_forward_df,
                    df=dfB, label_col="sinal",
                    val_size=new_vs, anchored=args.anchored,
                    outdir=outB, model="rf", tag="B"
                )
                log_event(f"[WF/B] recuperado com val_size={new_vs}.", "info")
        except Exception as e:
            log_event(f"[WF/B] falhou ao re-rotular/rodar: {e}", "warning")
    else:
        log_event("[WF/B] utils.labeling.relabel_profile indisponível; usando apenas perfil A.", "warning")

    # ---------- Escolha e promoção ----------
    escolha = _pick_best_labeling(sumA, sumB)
    vencedor = sumA if escolha == "A" else sumB
    vm = vencedor.get('metrics_mean', {})
    log_event(f"[WF] vencedor={escolha} | f1_updown={vm.get('f1_updown')} | neutral_rate={vm.get('neutral_rate')}", "info")

    try:
        # salvar summaries A e B
        with open(os.path.join(outA, "walkforward_summary.json"), "w", encoding="utf-8") as fA:
            json.dump(sumA, fA, ensure_ascii=False, indent=2)
        with open(os.path.join(outB, "walkforward_summary.json"), "w", encoding="utf-8") as fB:
            json.dump(sumB, fB, ensure_ascii=False, indent=2)

        # tentar calcular tau_meta a partir do CSV de métricas do vencedor
        fm_srcs = [
            os.path.join(args.out, escolha, 'fold_metrics_walkforward.csv'),
            os.path.join(args.out, escolha, 'fold_metrics.csv'),
        ]
        fm_winner = next((p for p in fm_srcs if os.path.exists(p)), None)
        if isinstance(vencedor, dict) and fm_winner:
            vencedor = _try_add_tau_meta(vencedor, fm_winner)
        # clamp delta_star
        try:
            if float(vencedor.get('delta_star', 0.0)) < 0.03:
                vencedor['delta_star'] = 0.05
        except Exception:
            pass
        # promover vencedor para raiz (logs/)
        with open(os.path.join(args.out, "walkforward_summary.json"), "w", encoding="utf-8") as fp:
            json.dump(vencedor, fp, ensure_ascii=False, indent=2)

        # tentar copiar o CSV de métricas do vencedor (se existir)
        fm_srcs = [
            os.path.join(args.out, escolha, "fold_metrics_walkforward.csv"),
            os.path.join(args.out, escolha, "fold_metrics.csv"),
        ]
        fm_dst = os.path.join(args.out, "fold_metrics_walkforward.csv")
        copied = False
        for fm_src in fm_srcs:
            if os.path.exists(fm_src):
                import shutil; shutil.copyfile(fm_src, fm_dst)
                copied = True
                break
        if not copied:
            log_event(f"[WF] fold_metrics do vencedor não encontrado em: {fm_srcs}", "warning")
    except Exception as e:
        log_event(f"[WF] falhou ao promover arquivos: {e}", "warning")

    log_event("[WF] concluído. (logs/A, logs/B e logs/*) prontos.", "info")

if __name__ == "__main__":
    main()
