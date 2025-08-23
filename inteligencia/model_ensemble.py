# inteligencia/model_ensemble.py
# --- silencia logs do TF (0=all, 1=info, 2=warning, 3=error-only) ---
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.debug_logger import log_event

# ================================
# Dependências opcionais (CatBoost, Keras/TF)
# ================================
try:
    from catboost import CatBoostClassifier  # type: ignore[import-not-found]
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False
    CatBoostClassifier = None  # type: ignore[assignment]

# Preferir tensorflow.keras; fallback para keras standalone.
try:
    from tensorflow.keras.models import Sequential  # type: ignore[import-not-found]
    from tensorflow.keras.layers import Dense, LSTM, Input  # type: ignore[import-not-found]
    from tensorflow.keras.optimizers import Adam  # type: ignore[import-not-found]
    HAS_KERAS = True
except Exception:
    try:
        from keras.models import Sequential  # type: ignore[import-not-found]
        from keras.layers import Dense, LSTM, Input  # type: ignore[import-not-found]
        from keras.optimizers import Adam  # type: ignore[import-not-found]
        HAS_KERAS = True
    except Exception:
        HAS_KERAS = False
        Sequential = Dense = LSTM = Input = Adam = None  # type: ignore[assignment]

# ================================
# Utilidades
# ================================
def _to_classes_for_loss(y: np.ndarray):
    """
    Mapeia labels originais (ex.: {-1,0,1}) -> {0..n-1} para softmax.
    Retorna: (y_mapped, inv_map, n_classes, class_order)
    """
    uniq = sorted(set(int(v) for v in y))
    mapping = {u: i for i, u in enumerate(uniq)}
    inv = {i: u for u, i in mapping.items()}
    y_m = np.array([mapping[int(v)] for v in y], dtype=int)
    return y_m, inv, len(uniq), uniq  # uniq é nossa ordem-base canônica

def _align_proba(proba: np.ndarray, model_classes, class_order):
    """
    Reordena/expande colunas de 'proba' para seguir 'class_order'.
    Se o modelo não tiver alguma classe, insere coluna zero.
    """
    model_classes = list(map(int, model_classes))
    cols = []
    for c in class_order:
        if c in model_classes:
            cols.append(proba[:, model_classes.index(c)])
        else:
            cols.append(np.zeros((proba.shape[0],), dtype=float))
    return np.column_stack(cols)

def _mk_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """Cria janelas (N, lookback, F) e rótulos (N,) para LSTM."""
    if len(X) <= lookback:
        return np.empty((0, lookback, X.shape[1])), np.empty((0,), dtype=int)
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)

# ================================
# Random Forest
# ================================
def treinar_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    log_event(f"[RF] Treinado. acc_train={model.score(X, y):.3f}")
    return model

def prever_random_forest(model, X):
    return model.predict(X), model.predict_proba(X), model.classes_

# ================================
# CatBoost (opcional)
# ================================
def treinar_catboost(X, y):
    if not HAS_CATBOOST:
        log_event("[CB] CatBoost não instalado; pulando.", level="warning")
        return None
    model = CatBoostClassifier(iterations=200, verbose=0, random_seed=42)
    model.fit(X, y)
    log_event(f"[CB] Treinado. acc_train={model.score(X, y):.3f}")
    return model

def prever_catboost(model, X):
    if model is None:
        return None, None, None
    pred = model.predict(X)
    proba = model.predict_proba(X)
    if isinstance(proba, list):
        proba = np.asarray(proba)
    if proba.ndim == 1:
        # binário em 1D → força 2 colunas
        proba = np.column_stack([1 - proba, proba])
    classes_ = getattr(model, "classes_", None)
    if classes_ is None:
        classes_ = sorted(set(map(int, pred)))
    return pred, proba, classes_

# ================================
# LSTM (opcional)
# ================================
def treinar_lstm(X, y, lookback=10, epochs=8, batch_size=64):
    if not HAS_KERAS:
        log_event("[LSTM] TensorFlow/Keras não disponível; pulando.", level="warning")
        return None, {"lookback": lookback, "inv": {}, "n": 0, "class_order": []}

    y_mapped, inv_map, n_classes, class_order = _to_classes_for_loss(y)
    X_seq, y_seq = _mk_sequences(X, y_mapped, lookback)
    if len(X_seq) == 0:
        log_event("[LSTM] Dados insuficientes para janelas; pulando.", level="warning")
        return None, {"lookback": lookback, "inv": inv_map, "n": n_classes, "class_order": class_order}

    # ✅ usa Input(...) como primeira camada (elimina o warning)
    model = Sequential([
        Input(shape=(lookback, X.shape[1])),
        LSTM(64),
        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(0.001), metrics=["accuracy"])
    hist = model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=0)
    log_event(f"[LSTM] Treinado. loss_final={hist.history['loss'][-1]:.3f}")
    return model, {"lookback": lookback, "inv": inv_map, "n": n_classes, "class_order": class_order}

def prever_lstm(model, meta, X):
    if model is None:
        return None, None, None, None
    lookback = int(meta["lookback"])
    inv_map = meta["inv"]
    class_order = meta["class_order"]

    X_seq = []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
    X_seq = np.asarray(X_seq)
    if len(X_seq) == 0:
        return None, None, None, None

    proba = model.predict(X_seq, verbose=0)  # (N, n_classes_mapeadas)
    pred_idx = np.argmax(proba, axis=1)
    pred = np.array([inv_map[int(i)] for i in pred_idx], dtype=int)
    model_classes = [inv_map[i] for i in range(proba.shape[1])]
    return pred, proba, model_classes, lookback

# ================================
# Stacking
# ================================
def treinar_stacking(probas_list, y_aligned):
    X_stack = np.column_stack(probas_list)
    meta = LogisticRegression(max_iter=200, random_state=42)
    meta.fit(X_stack, y_aligned)
    log_event("[STACK] Meta-LogReg treinado.")
    return meta

def prever_stacking(meta, probas_list):
    X_stack = np.column_stack(probas_list)
    return meta.predict(X_stack), meta.predict_proba(X_stack)

# ================================
# Pipeline completo
# ================================
def pipeline_ensemble(df, features, label):
    # limpa NaN no label
    y = df[label].values
    mask = ~pd.isnull(y)
    X = df.loc[mask, features].values
    y = y[mask].astype(int)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Ordem-base de classes (consistente na pipeline)
    _, _, _, class_order = _to_classes_for_loss(y_train)

    # === RF ===
    rf = treinar_random_forest(X_train, y_train)
    rf_pred, rf_proba, rf_classes = prever_random_forest(rf, X_test)
    rf_proba = _align_proba(rf_proba, rf_classes, class_order)

    # === CatBoost (se disponível) ===
    cb = treinar_catboost(X_train, y_train)
    cb_pred, cb_proba, cb_classes = prever_catboost(cb, X_test)
    if cb_proba is not None and cb_classes is not None:
        cb_proba = _align_proba(cb_proba, cb_classes, class_order)

    # === LSTM (opcional) ===
    lstm, lstm_meta = treinar_lstm(X_train, y_train, lookback=10, epochs=8, batch_size=64)
    lstm_pred, lstm_proba, lstm_classes, lb = prever_lstm(lstm, lstm_meta, X_test)
    if lstm_proba is not None and lstm_classes is not None:
        lstm_proba = _align_proba(lstm_proba, lstm_classes, class_order)

    # === Juntar probabilidades e alinhar comprimentos ===
    probas, lens = [], []
    probas.append(rf_proba); lens.append(len(rf_proba))
    if cb_proba is not None: probas.append(cb_proba); lens.append(len(cb_proba))
    if lstm_proba is not None: probas.append(lstm_proba); lens.append(len(lstm_proba))

    L = min(lens)
    probas = [p[-L:] for p in probas]
    y_align = y_test[-L:]

    # === Meta-modelo (stacking)
    meta = treinar_stacking(probas, y_align)
    stack_pred, _stack_proba = prever_stacking(meta, probas)

    # === Métricas
    try:
        rf_acc = accuracy_score(y_test, rf_pred)
    except Exception:
        rf_acc = np.nan
    try:
        cb_acc = accuracy_score(y_test, cb_pred) if cb_pred is not None else np.nan
    except Exception:
        cb_acc = np.nan
    try:
        lstm_acc = accuracy_score(y_test[-len(lstm_pred):], lstm_pred) if lstm_pred is not None else np.nan
    except Exception:
        lstm_acc = np.nan
    stack_acc = accuracy_score(y_align, stack_pred)

    log_event(f"[RESULT] RF={rf_acc:.3f} | CB={cb_acc:.3f} | LSTM={lstm_acc:.3f} | STACK={stack_acc:.3f}", level="info")

    return {
        "rf_model": rf, "cb_model": cb,
        "lstm_model": lstm, "lstm_meta": lstm_meta,
        "meta_model": meta, "stack_acc": float(stack_acc),
        "class_order": class_order
    }
