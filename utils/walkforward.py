
# utils/walkforward.py
# -*- coding: utf-8 -*-
import os, json, warnings, numpy as np, pandas as pd
from collections import Counter
warnings.filterwarnings("ignore")

try:
    from utils.debug_logger import log_event as _log
except Exception:
    def _log(msg, level="info"):
        try:
            print(f"[{level.upper()}] {msg}")
        except Exception:
            pass


# Modelos
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
except Exception:
    RandomForestClassifier = None
    CalibratedClassifierCV = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef

LABEL_CANDS = ["label", "y", "sinal_ideal", "sinal_n1", "sinal", "target"]
NON_FEATURES = {"datahora","datetime","time","timestamp","ts","id","ativo","symbol","pair","sym","instrument"}

def detect_label(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in LABEL_CANDS:
            return c
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce").dropna().unique().tolist()
        if set(s).issubset({-1,0,1}) or set(s).issubset({0,1}):
            return c
    raise RuntimeError("Coluna de rótulo não encontrada (ex.: 'label' ou 'sinal_ideal').")

def select_features(df: pd.DataFrame, ycol: str):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if ycol in num: num.remove(ycol)
    feats = [c for c in num if c.lower() not in NON_FEATURES]
    if not feats:
        raise RuntimeError("Nenhuma feature numérica válida após filtros.")
    return feats

def class_weights(y, balanced=True):
    y = pd.Series(y)
    vc = y.value_counts().to_dict()
    if not balanced: return {c:1.0 for c in vc}
    tot = sum(vc.values())
    return {c: tot/(len(vc)*vc[c]) for c in vc}

def map_probs_multiclass(proba: np.ndarray, classes, tau=0.45, delta=0.10):
    neutral = 0 if 0 in classes else None
    top1_idx = proba.argmax(axis=1)
    top1 = proba[np.arange(len(proba)), top1_idx]
    part = np.partition(-proba, 1, axis=1)
    top2 = -part[:,1]
    margin = top1 - top2
    preds = []
    for i in range(len(proba)):
        if neutral is not None and (top1[i] < tau or margin[i] < delta):
            preds.append(neutral)
        else:
            preds.append(classes[top1_idx[i]])
    return np.array(preds, dtype=int)

def f1_updown(y_true, y_pred):
    mask = np.isin(y_true, [-1,1])
    if mask.sum() == 0: return np.nan
    return f1_score(y_true[mask], y_pred[mask], average="macro")

def metrics_all(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_updown": f1_updown(y_true, y_pred),
        "cm_(-1,0,1)": confusion_matrix(y_true, y_pred, labels=[-1,0,1]).tolist()
    }

def walk_indices(n, train_size, val_size, step=None, anchored=True):
    if step is None: step = val_size
    start_tr, end_tr = 0, train_size
    while True:
        start_va, end_va = end_tr, end_tr + val_size
        if end_va > n: break
        yield np.arange(start_tr, end_tr), np.arange(start_va, end_va)
        if anchored:
            end_tr += step
        else:
            start_tr += step; end_tr += step

def fit_model(X_tr, y_tr, model="rf", cw=None, seed=42):
    if model == "rf":
        if RandomForestClassifier is None or CalibratedClassifierCV is None:
            raise RuntimeError("sklearn indisponível.")
        base = RandomForestClassifier(
            n_estimators=400, min_samples_split=4, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        )
        base.fit(X_tr, y_tr)
        calib = CalibratedClassifierCV(base, method="isotonic", cv=3)
        calib.fit(X_tr, y_tr)
        return calib
    elif model == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost indisponível.")
        num_classes = len(np.unique(y_tr))
        xgb = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", num_class=num_classes,
            tree_method="hist", eval_metric="mlogloss",
            random_state=seed, n_jobs=-1,
        )
        sw = None
        if cw:
            sw = np.array([cw[c] for c in y_tr])
        xgb.fit(X_tr, y_tr, sample_weight=sw)
        calib = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
        calib.fit(X_tr, y_tr)
        return calib
    else:
        raise ValueError("model deve ser 'rf' ou 'xgb'.")

def evaluate_fold(model, X_va, y_va, classes, tau_grid, delta_grid):
    proba = model.predict_proba(X_va)
    if hasattr(model, "classes_"):
        order = [np.where(model.classes_==c)[0][0] for c in classes]
        proba = proba[:, order]
    best = {"f1_updown": -1, "tau": None, "delta": None, "metrics": None}
    for tau in tau_grid:
        for delta in delta_grid:
            y_pred = map_probs_multiclass(proba, classes, tau=tau, delta=delta)
            m = metrics_all(y_va, y_pred)
            score = m["f1_updown"]
            score = -np.inf if np.isnan(score) else score
            if score > best["f1_updown"]:
                best = {"f1_updown": score, "tau": tau, "delta": delta, "metrics": m}
    return best["metrics"], {"tau": best["tau"], "delta": best["delta"]}

def run_walk_forward_df(
    df, label_col=None, features=None, model="rf",
    train_size=25000, val_size=3000, step=None, anchored=True,
    tau_grid=None, delta_grid=None, outdir="logs"
):
    os.makedirs(outdir, exist_ok=True)
    if label_col is None:
        # detecta automaticamente coluna de rótulo
        label_col = detect_label(df)
    if features is None:
        features = select_features(df, label_col)

    _log("[WF] START | limpando dados/selecionando features", level="info")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    y = df[label_col].astype(int).values
    X = df[features].values
    classes = sorted(pd.Series(y).unique().tolist())

    tau_grid = tau_grid or np.linspace(0.40, 0.60, 5)
    delta_grid = delta_grid or np.linspace(0.00, 0.20, 5)

    folds = list(walk_indices(len(df), train_size, val_size, step=step, anchored=anchored))
    _log(f"[WF] FOLDS={len(folds)} | train_size={train_size} | val_size={val_size}", level="info")
    if not folds: raise RuntimeError("Dados insuficientes para os parâmetros informados.")

    rows, best_count = [], Counter()
    for i,(tr,va) in enumerate(folds,1):
        _log(f"[WF] FOLD {i} | n_train={len(tr)} | n_val={len(va)}", level="info")
        Xtr, Ytr = X[tr], y[tr]
        Xva, Yva = X[va], y[va]
        cw = class_weights(Ytr, balanced=True)
        mdl = fit_model(Xtr, Ytr, model=model, cw=cw)
        met, thr = evaluate_fold(mdl, Xva, Yva, classes, tau_grid, delta_grid)
        rows.append({
            "fold": i,
            "train_start": int(tr[0]), "train_end": int(tr[-1]),
            "val_start": int(va[0]), "val_end": int(va[-1]),
            "n_train": int(len(tr)), "n_val": int(len(va)),
            "tau": thr["tau"], "delta": thr["delta"], **met
        })
        best_count[(thr["tau"], thr["delta"])] += 1

    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(outdir, "fold_metrics_walkforward.csv"), index=False, encoding="utf-8")
    _log("[WF] Salvo fold_metrics_walkforward.csv", level="info")

    (tau_star, delta_star), _ = best_count.most_common(1)[0]
    summary = {
        "model": model, "train_size": train_size, "val_size": val_size, "anchored": anchored,
        "n_folds": len(folds), "classes": classes,
        "tau_star": float(tau_star), "delta_star": float(delta_star),
        "metrics_mean": df_res.drop(columns=["cm_(-1,0,1)"]).select_dtypes(include=[np.number]).mean().to_dict()
    }
    _log(f"[WF] tau_star={float(tau_star):.3f} | delta_star={float(delta_star):.3f}", level="info")
    with open(os.path.join(outdir, "walkforward_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _log("[WF] END", level="info")
    return df_res, summary
