# scripts/run_walkforward_ab.py
# -*- coding: utf-8 -*-
import os, sys, json, argparse, inspect
import pandas as pd

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

# -------------------- main --------------------
def main():
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

    # Determina val_size (se não veio)
    val_size = args.val_size
    if val_size is None:
        # fallback conservador baseado no tamanho do dataset
        ds = len(df)
        val_size = max(1500, min(4000, ds // 8))
        # se houver hint no config, usa como teto
        hint = int(cfg.get("janela_candles", 3000))
        val_size = min(val_size, max(1000, hint))
        log_event(f"[WF] val_size inferido: {val_size}", "info")

    # cria pastas
    outA = os.path.join(args.out, "A")
    outB = os.path.join(args.out, "B")
    os.makedirs(outA, exist_ok=True)
    os.makedirs(outB, exist_ok=True)

    # ---------- A) baseline usando 'sinal' já existente ----------
    log_event("[WF/A] iniciando...", "info")
    metA, sumA = _call_run_wf(
        run_walk_forward_df,
        df=df, label_col="sinal",
        val_size=val_size, anchored=args.anchored,
        outdir=outA, model="rf", tag="A"
    )

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
            metB, sumB = _call_run_wf(
                run_walk_forward_df,
                df=dfB, label_col="sinal",
                val_size=val_size, anchored=args.anchored,
                outdir=outB, model="rf", tag="B"
            )
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
