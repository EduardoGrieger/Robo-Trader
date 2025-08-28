import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import random

# Global config placeholder (overwritten later in main)
cfg = {}

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# XGBoost
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# LSTM
try:
    from tensorflow import keras
    import tensorflow as tf
except ImportError:
    keras = None
    tf = None

# Oversampling e Undersampling (opcional; não usamos SMOTE pra evitar vazamento)
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    SMOTE = None
    RandomOverSampler = None
    RandomUnderSampler = None

from utils.debug_logger import log_event


# ========================= SEED / DETERMINISMO =========================

def set_global_seed(seed: int = 42):
    """Define seeds em random, numpy e tensorflow (se disponível)."""
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    if tf is not None:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass
    log_event(f"[SEED] Global seed configurado para {seed}", level="info")


# ========================= UTIL & DIAGNÓSTICOS =========================

def salvar_relatorio(report_str, tag=""):
    os.makedirs("logs", exist_ok=True)
    rel_path = os.path.join("logs", f"classification_report_{tag}.txt")
    with open(rel_path, "a", encoding="utf-8") as f:
        f.write(report_str + "\n" + "="*50 + "\n")


def diagnosticar_features(df, tag=""):
    for col in df.columns:
        if df[col].isnull().any():
            log_event(f"[{tag}] {col} contém NaN ({df[col].isnull().sum()})", level="warning")
        if df[col].nunique() <= 1:
            log_event(f"[{tag}] {col} valor único", level="warning")
        if not np.issubdtype(df[col].dtype, np.number):
            log_event(f"[{tag}] {col} não-numérica ({df[col].dtype})", level="warning")


def filtrar_apenas_numericas(df):
    numericas = df.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    removidas = list(set(df.columns) - set(numericas))
    if removidas:
        log_event(f"[FILTRAR_NUMERICAS] Removidas colunas não-numéricas: {removidas}", level="warning")
    return df[numericas]


def carregar_dados(features_path):
    if not os.path.exists(features_path):
        log_event(f"❌ Arquivo {features_path} não encontrado.", level="error")
        return None
    df = pd.read_csv(features_path)
    diagnosticar_features(df, "[TREINO ANTES]")
    # Mantemos um fillna inicial leve (compat), mas faremos IMPUTAÇÃO PÓS-SPLIT com medianas do treino.
    df = df.fillna(df.median(numeric_only=True))
    if "sinal" not in df.columns:
        log_event("❌ Coluna 'sinal' não encontrada no features.csv.", level="error")
        return None
    return df


# ========================= META-FEATURES DEFENSIVAS =========================

def adicionar_meta_features(df):
    """
    Adiciona meta-features SE houver OHLC. É defensivo: se não tiver colunas,
    segue sem quebrar.
    """
    df = df.copy()
    ok_close = "close" in df.columns
    ok_h = "high" in df.columns
    ok_l = "low" in df.columns

    if ok_close:
        returns = df["close"].pct_change()
        df["volatility_20"] = returns.rolling(20).std()
        df["volatility_50"] = returns.rolling(50).std()
        df["trend_strength"] = df["close"].rolling(20).apply(
            lambda x: abs(np.polyfit(range(len(x)), x, 1)[0]) / np.std(x) if np.std(x) > 0 else 0,
            raw=False
        )
    if ok_close and ok_h and ok_l:
        roll_mean = df["close"].rolling(20).mean()
        df["range_ratio"] = (df["high"] - df["low"]) / (roll_mean.replace(0, np.nan))

    return df


# ========================= BALANCEAMENTO =========================
# (mantemos a versão antiga por compatibilidade, mas NÃO usamos antes do split)

def balancear_labels(X, y, perc_neutro=0.20, min_amostra=150):
    """
    (Compat) Não usar antes do split. Mantido para não quebrar importações antigas.
    """
    rng = np.random.default_rng(42)
    is_pd_X = hasattr(X, "iloc"); is_pd_y = hasattr(y, "iloc")
    dfx = X.reset_index(drop=True) if is_pd_X else pd.DataFrame(X)
    dfy = y.reset_index(drop=True) if is_pd_y else pd.Series(y)

    idx_neu = dfy[dfy == 0].index.values
    idx_dn  = dfy[dfy == -1].index.values
    idx_up  = dfy[dfy ==  1].index.values

    n = len(dfy)
    max_neu = int(perc_neutro * n)

    if len(idx_neu) > max_neu and max_neu > 0:
        idx_neu = rng.choice(idx_neu, size=max_neu, replace=False)

    target = max(len(idx_neu), len(idx_dn), len(idx_up), int(min_amostra))

    def _oversample(idxs, tgt):
        if len(idxs) == 0:
            return np.array([], dtype=int)
        if len(idxs) >= tgt:
            return rng.choice(idxs, size=tgt, replace=False)
        extra = rng.choice(idxs, size=tgt - len(idxs), replace=True)
        return np.concatenate([idxs, extra])

    idx_dn2 = _oversample(idx_dn, target)
    idx_up2 = _oversample(idx_up, target)

    keep = np.concatenate([idx_neu, idx_dn2, idx_up2])
    keep.sort(kind="mergesort")
    Xb = dfx.loc[keep].reset_index(drop=True)
    yb = dfy.loc[keep].reset_index(drop=True)
    log_event(f"[BALANCEAMENTO] Distribuição pós-balanceamento (compat): {yb.value_counts().to_dict()}")
    return Xb if is_pd_X else Xb.values, yb if is_pd_y else yb.values


def balancear_labels_avancado(X, y, max_neutro_ratio=0.25, min_amostras_classe=150, random_state=42):
    """
    Balanceia APENAS o conjunto de TREINO:
      - Corta neutros para no máx. max_neutro_ratio
      - Oversample em {-1, +1} até min_amostras_classe
    Não usar antes do split para evitar vazamento entre conjuntos.
    """
    from sklearn.utils import resample
    rng = np.random.RandomState(random_state)

    Xp = X.reset_index(drop=True) if hasattr(X, "iloc") else pd.DataFrame(X)
    yp = y.reset_index(drop=True) if hasattr(y, "iloc") else pd.Series(y)

    counts = yp.value_counts().to_dict()
    n_total = len(yp)

    target_neu = int(min(counts.get(0, 0), max_neutro_ratio * n_total))
    target_dn  = max(min_amostras_classe, counts.get(-1, 0))
    target_up  = max(min_amostras_classe, counts.get( 1, 0))

    def _undersample_neutro():
        mask = (yp == 0)
        X0, y0 = Xp[mask], yp[mask]
        n = len(y0)
        if n <= target_neu:
            return X0, y0
        idx = rng.choice(n, size=target_neu, replace=False)
        return X0.iloc[idx], y0.iloc[idx]

    def _oversample(classe, target):
        mask = (yp == classe)
        Xc, yc = Xp[mask], yp[mask]
        n = len(yc)
        if n == 0:
            return Xc.iloc[:0], yc.iloc[:0]
        if n >= target:
            idx = rng.choice(n, size=target, replace=False)
            return Xc.iloc[idx], yc.iloc[idx]
        add_idx = rng.choice(n, size=target - n, replace=True)
        return pd.concat([Xc, Xc.iloc[add_idx]]), pd.concat([yc, yc.iloc[add_idx]])

    X0, y0 = _undersample_neutro()
    Xdn, ydn = _oversample(-1, target_dn)
    Xup, yup = _oversample( 1, target_up)

    Xb = pd.concat([X0, Xdn, Xup], axis=0).reset_index(drop=True)
    yb = pd.concat([y0, ydn, yup], axis=0).reset_index(drop=True)

    idx = rng.permutation(len(Xb))
    Xb, yb = Xb.iloc[idx], yb.iloc[idx]

    log_event(f"[BAL_AVANÇADO] Pós-balanceamento treino: {yb.value_counts().to_dict()}")
    return Xb, yb


# ========================= LABELING/XGB MAPS =========================

def mapear_label_xgb(y):
    return np.where(y == -1, 0, np.where(y == 0, 1, 2))


def desmapear_label_xgb(y_xgb):
    return np.where(y_xgb == 0, -1, np.where(y_xgb == 1, 0, 1))


def _salvar_listas_features_compat(prefixo_curto, lista_cols):
    """
    Salva as listas de features com nomes legacy e novos (compatíveis com a inferência).
    RF:
      - modelos/features_treinadas_rf.pkl
      - modelos/features_treinadas_random_forest.pkl
    XGB:
      - modelos/features_treinadas_xgb.pkl
      - modelos/features_treinadas_xgboost.pkl
    """
    os.makedirs("modelos", exist_ok=True)
    if prefixo_curto == "rf":
        joblib.dump(list(lista_cols), "modelos/features_treinadas_rf.pkl")
        joblib.dump(list(lista_cols), "modelos/features_treinadas_random_forest.pkl")
        log_event("[TREINO] Listas de features RF salvas (rf.pkl e random_forest.pkl).", level="info")
    elif prefixo_curto == "xgb":
        joblib.dump(list(lista_cols), "modelos/features_treinadas_xgb.pkl")
        joblib.dump(list(lista_cols), "modelos/features_treinadas_xgboost.pkl")
        log_event("[TREINO] Listas de features XGB salvas (xgb.pkl e xgboost.pkl).", level="info")
    elif prefixo_curto == "lstm":
        joblib.dump(list(lista_cols), "modelos/features_treinadas_lstm.pkl")
        log_event("[TREINO] Lista de features LSTM salva (lstm.pkl).", level="info")


# ========================= TREINOS =========================

def treinar_rf(X_train, y_train):
    """
    Random Forest com pesos anti-neutro e regularização leve.
    """
    classes = np.array([-1, 0, 1])
    base_w = compute_class_weight('balanced', classes=classes, y=y_train)
    cw = {int(c): float(w) for c, w in zip(classes, base_w)}

    # reforços (podem ser ajustados em config.json):
    updown_scale = float(cfg.get("rf_peso_updown_scale", 1.7))
    neutro_scale = float(cfg.get("rf_peso_neutro_scale", 0.6))
    cw[-1] *= updown_scale
    cw[ 1] *= updown_scale
    cw[ 0] *= neutro_scale

    modelo = RandomForestClassifier(
        n_estimators=int(cfg.get("rf_n_estimators", 400)),
        max_depth=int(cfg.get("rf_max_depth", 12)),
        min_samples_leaf=int(cfg.get("rf_min_samples_leaf", 10)),
        n_jobs=-1,
        random_state=42,
        class_weight=cw,
        max_features=cfg.get("rf_max_features", "sqrt"),
        bootstrap=True
    )
    modelo.fit(X_train, y_train)
    _salvar_listas_features_compat("rf", list(X_train.columns))
    return modelo


def treinar_xgb(X_train, y_train, X_val=None, y_val=None):
    if not xgb:
        raise ImportError("XGBoost não instalado.")
    modelo = xgb.XGBClassifier(
        n_estimators=int(cfg.get("xgb_n_estimators", 600)),
        max_depth=int(cfg.get("xgb_max_depth", 6)),
        subsample=float(cfg.get("xgb_subsample", 0.9)),
        colsample_bytree=float(cfg.get("xgb_colsample_bytree", 0.9)),
        reg_lambda=float(cfg.get("xgb_reg_lambda", 1.0)),
        random_state=42,
        eval_metric='mlogloss',
        tree_method=cfg.get("xgb_tree_method", "hist")
    )
    # y_train (0/1/2)
    classes_xgb = np.array([0, 1, 2])
    cw = compute_class_weight('balanced', classes=classes_xgb, y=y_train)
    wmap = {int(c): float(w) for c, w in zip(classes_xgb, cw)}
    sample_weight = np.vectorize(wmap.get)(y_train)

    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))

    try:
        modelo.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            early_stopping_rounds=int(cfg.get("xgb_es_rounds", 50)),
            verbose=False
        )
    except TypeError:
        modelo.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=False
        )
    _salvar_listas_features_compat("xgb", list(X_train.columns))
    return modelo


def treinar_lstm(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    if not keras:
        raise ImportError("Tensorflow/Keras não instalado.")
    # Corrigir labels para 0,1,2
    y_train = np.where(y_train == -1, 0, np.where(y_train == 0, 1, 2))
    y_val = np.where(y_val == -1, 0, np.where(y_val == 0, 1, 2))
    n_features = X_train.shape[1]
    X_train_lstm = np.expand_dims(X_train, axis=1)
    X_val_lstm = np.expand_dims(X_val, axis=1)
    model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(1, n_features)),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(32),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    model.fit(X_train_lstm, y_train, validation_data=(X_val_lstm, y_val),
              epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
    _salvar_listas_features_compat("lstm", list(X_train.columns))
    return model


# ========================= AVALIAÇÃO & PREPARO =========================

def avaliar_modelo(modelo, X_test, y_test, tipo="rf"):
    if tipo == "lstm":
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        y_pred_prob = modelo.predict(X_test_lstm)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_test_lstm = np.where(y_test == -1, 0, np.where(y_test == 0, 1, 2))
        y_pred_final = np.where(y_pred == 0, -1, np.where(y_pred == 1, 0, 1))
        acc = accuracy_score(y_test, y_pred_final)
        report = classification_report(y_test, y_pred_final, digits=4)
        conf_matrix = confusion_matrix(y_test, y_pred_final)
        return acc, report, conf_matrix, y_pred_final
    elif tipo == "xgb":
        y_pred = modelo.predict(X_test)
        y_pred = desmapear_label_xgb(y_pred)
        y_test_ = desmapear_label_xgb(y_test)
        acc = accuracy_score(y_test_, y_pred)
        report = classification_report(y_test_, y_pred, digits=4)
        conf_matrix = confusion_matrix(y_test_, y_pred)
        return acc, report, conf_matrix, y_pred
    else:
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return acc, report, conf_matrix, y_pred


def preparar_para_previsao(df, features):
    """ Aceita lista de features (preferido) ou path para pkl legado. """
    if isinstance(features, str):
        features_treinadas = joblib.load(features)
    else:
        features_treinadas = features
    df_num = df.select_dtypes(include=[np.number, 'bool'])
    df_alinhado = df_num.reindex(columns=features_treinadas).fillna(0)
    return df_alinhado


def _salvar_avaliacao_ensemble(metrics_dict):
    os.makedirs("modelos", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    path_modelos = os.path.join("modelos", "avaliacao_ensemble.json")
    path_logs = os.path.join("logs", "avaliacao_ensemble.json")
    with open(path_modelos, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    with open(path_logs, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    log_event(f"[AVALIACAO] Arquivo salvo em {path_modelos} e {path_logs}", level="info")


# ========================= IMPUTAÇÃO PÓS-SPLIT =========================

def imputar_pos_split(X_train, X_val, X_test):
    """
    Calcula medianas **somente no treino** e aplica em val/test para evitar vazamento.
    Mantém DataFrames e colunas originais.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    med_train = X_train.median(numeric_only=True)
    num_cols = med_train.index.tolist()

    X_train[num_cols] = X_train[num_cols].fillna(med_train)
    X_val[num_cols] = X_val[num_cols].fillna(med_train)
    X_test[num_cols] = X_test[num_cols].fillna(med_train)

    return X_train, X_val, X_test, med_train.to_dict()


# ========================= LÓGICA DE PROMOÇÃO =========================

def _promover_melhor_modelo(avaliacao, modelos, features_ref):
    """
    Escolhe o melhor modelo entre RF/XGB (compatível com joblib) e salva em
    'modelos/cerebro_mestre.joblib'. LSTM é ignorado por compatibilidade.
    """
    permitir_promocao = bool(cfg.get("promover_melhor_modelo", True))
    permitir_lstm = bool(cfg.get("permitir_promocao_lstm", False))  # por padrão, False
    escolhido = {"tipo": "rf", "acc": None}

    if not permitir_promocao:
        # Compat: mantém RF como 'cerebro_mestre'
        try:
            joblib.dump(modelos.get("rf"), "modelos/cerebro_mestre.joblib")
            log_event("[PROMOCAO] Promoção desabilitada. RF salvo como cerebro_mestre.", level="info")
        except Exception as e:
            log_event(f"[PROMOCAO] Falha ao salvar RF como cerebro_mestre: {e}", level="error")
        return {"tipo": "rf", "acc": avaliacao.get("random_forest", {}).get("acc")}

    # Monta ranking por acc
    candidatos = []
    if "random_forest" in avaliacao and modelos.get("rf") is not None:
        candidatos.append(("rf", float(avaliacao["random_forest"].get("acc") or -1)))
    if "xgboost" in avaliacao and modelos.get("xgb") is not None and avaliacao["xgboost"].get("acc") is not None:
        candidatos.append(("xgb", float(avaliacao["xgboost"]["acc"])))

    # LSTM não é joblib; só promove se explicitamente habilitado
    if permitir_lstm and "lstm" in avaliacao and modelos.get("lstm") is not None and avaliacao["lstm"].get("acc") is not None:
        candidatos.append(("lstm", float(avaliacao["lstm"]["acc"])))

    if not candidatos:
        log_event("[PROMOCAO] Nenhum candidato válido. Mantendo RF como cerebro_mestre.", level="warning")
        try:
            joblib.dump(modelos.get("rf"), "modelos/cerebro_mestre.joblib")
        except Exception:
            pass
        return {"tipo": "rf", "acc": avaliacao.get("random_forest", {}).get("acc")}

    candidatos.sort(key=lambda t: t[1], reverse=True)
    best_tipo, best_acc = candidatos[0]
    escolhido = {"tipo": best_tipo, "acc": best_acc}

    if best_tipo == "rf":
        joblib.dump(modelos["rf"], "modelos/cerebro_mestre.joblib")
        log_event(f"[PROMOCAO] RF promovido a cerebro_mestre (acc={best_acc:.4f}).", level="info")
    elif best_tipo == "xgb":
        # tentar salvar em joblib para manter compat no main_loop
        try:
            joblib.dump(modelos["xgb"], "modelos/cerebro_mestre.joblib")
            log_event(f"[PROMOCAO] XGB promovido a cerebro_mestre (acc={best_acc:.4f}).", level="info")
        except Exception as e:
            log_event(f"[PROMOCAO] Falha ao salvar XGB em joblib ({e}). Mantendo RF como cerebro_mestre.", level="warning")
            joblib.dump(modelos["rf"], "modelos/cerebro_mestre.joblib")
            escolhido = {"tipo": "rf", "acc": avaliacao.get("random_forest", {}).get("acc")}
    else:
        # LSTM: incompatível com joblib → não substitui o cerebro_mestre por padrão
        log_event(f"[PROMOCAO] LSTM teve melhor acc ({best_acc:.4f}) mas não será promovido (compat).", level="warning")
        joblib.dump(modelos["rf"], "modelos/cerebro_mestre.joblib")
        escolhido = {"tipo": "rf", "acc": avaliacao.get("random_forest", {}).get("acc")}

    # Persistir uma “ficha” do ensemble
    ensemble_cfg = {
        "cerebro_mestre_tipo": escolhido["tipo"],
        "acc": escolhido["acc"],
        "features_ref": list(features_ref) if features_ref is not None else None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("modelos/ensemble_config.json", "w", encoding="utf-8") as f:
        json.dump(ensemble_cfg, f, ensure_ascii=False, indent=2)

    return escolhido


# ========================= MAIN =========================

def main():
    global cfg

    set_global_seed(42)

    features_path = "dados/features.csv"
    os.makedirs("modelos", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # carrega config
    config_path = "config.json"
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    treinar_lstm_flag = cfg.get("treinar_lstm", True)
    acc_min_promocao = cfg.get("acc_min_promocao", 0.55)  # limiar p/ 'promover'
    usar_meta_features = bool(cfg.get("usar_meta_features", True))

    log_event("🚀 Iniciando treino dos modelos ENSEMBLE (RF, XGBoost, LSTM)", level="info")

    # ---- load
    df = carregar_dados(features_path)
    if df is None:
        return

    # ---- meta-features (opcional)
    if usar_meta_features:
        df = adicionar_meta_features(df)
        # preencher NaNs criados pelas janelas (pré-split, leve)
        df = df.fillna(df.median(numeric_only=True))

    # ---- drops de colunas não-úteis p/ modeling
    drop_cols = ["timestamp", "data_hora", "ativo", "regime", "regime_nome", "regime_hmm"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    diagnosticar_features(df, "[TREINO POS DROPS]")

    if "sinal" not in df.columns:
        log_event("❌ Coluna 'sinal' não encontrada após drop de cols.", level="error")
        return

    y = df["sinal"].copy()
    X = df.drop(columns=["sinal"])

    log_event(f"Distribuição original: {y.value_counts().to_dict()}")

    X = filtrar_apenas_numericas(X)

    if len(X) != len(y):
        log_event(f"❌ X e y desalinhados! X.shape={X.shape}, y.shape={y.shape}", level="error")
        return
    if X.empty or y.empty:
        log_event(f"❌ X ou y vazio após filtragem!", level="error")
        return

    # === NÃO balancear aqui (evita vazamento). Fazemos após split. ===

    # Split temporal
    n = len(X)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size

    X_train, y_train = X.iloc[:train_size].copy(), y.iloc[:train_size].copy()
    X_val,   y_val   = X.iloc[train_size:train_size+val_size].copy(), y.iloc[train_size:train_size+val_size].copy()
    X_test,  y_test  = X.iloc[train_size+val_size:].copy(), y.iloc[train_size+val_size:].copy()

    # === IMPUTAÇÃO PÓS-SPLIT (sem vazamento) ===
    X_train, X_val, X_test, med_train_dict = imputar_pos_split(X_train, X_val, X_test)

    # === Balanceamento avançado APENAS no treino ===
    max_neu = float(cfg.get("max_neutro_ratio_treino", 0.25))
    min_cls = int(cfg.get("min_amostras_classe_treino", 150))
    X_train, y_train = balancear_labels_avancado(
        X_train, y_train,
        max_neutro_ratio=max_neu,
        min_amostras_classe=min_cls
    )

    log_event(f"Shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    avaliacao = {
        "timestamp": timestamp,
        "acc_min_promocao": acc_min_promocao,
        "imputacao_pos_split": True
    }

    modelos = {"rf": None, "xgb": None, "lstm": None}
    features_ref = list(X_train.columns)

    # =================== RandomForest =====================
    modelo_rf = treinar_rf(X_train, y_train)
    modelos["rf"] = modelo_rf
    X_test_rf = preparar_para_previsao(X_test, "modelos/features_treinadas_rf.pkl")
    acc_rf, report_rf, cm_rf, y_pred_rf = avaliar_modelo(modelo_rf, X_test_rf, y_test, "rf")
    # salvas (RF sempre salvo — compat)
    joblib.dump(modelo_rf, "modelos/cerebro_mestre.joblib")
    joblib.dump(modelo_rf, f"modelos/cerebro_mestre_{timestamp}.joblib")
    log_event(f"RandomForest treinado. Acc={acc_rf:.4f}", level="info")
    salvar_relatorio("[RF]\n" + report_rf + "\nConfMatrix:\n" + str(cm_rf), "rf")

    train_acc_rf = accuracy_score(y_train, modelo_rf.predict(X_train))
    log_event(f"RF - Acurácia Treino: {train_acc_rf:.4f} | Teste: {acc_rf:.4f} | Dif: {train_acc_rf-acc_rf:.4f}")

    importances = modelo_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    log_event("Feature Importance (RF):")
    for i in indices[:10]:
        log_event(f"{X_train.columns[i]}: {importances[i]:.4f}")

    avaliacao["random_forest"] = {
        "acc": float(acc_rf),
        "promover": bool(acc_rf >= acc_min_promocao)
    }

    # =================== XGBoost ==========================
    if xgb:
        try:
            y_train_xgb = mapear_label_xgb(y_train)
            y_val_xgb = mapear_label_xgb(y_val)
            y_test_xgb = mapear_label_xgb(y_test)
            modelo_xgb = treinar_xgb(X_train, y_train_xgb, X_val, y_val_xgb)
            modelos["xgb"] = modelo_xgb
            X_test_xgb = preparar_para_previsao(X_test, "modelos/features_treinadas_xgb.pkl")
            if X_test_xgb.shape[1] != X_train.shape[1]:
                log_event(f"[XGB] Shape mismatch: X_test {X_test_xgb.shape}, X_train {X_train.shape}", level="error")
            acc_xgb, report_xgb, cm_xgb, y_pred_xgb = avaliar_modelo(modelo_xgb, X_test_xgb, y_test_xgb, "xgb")
            try:
                modelo_xgb.save_model("modelos/xgb_cerebro.json")
                modelo_xgb.save_model(f"modelos/xgb_cerebro_{timestamp}.json")
            except Exception:
                joblib.dump(modelo_xgb, "modelos/xgb_cerebro.joblib")
                joblib.dump(modelo_xgb, f"modelos/xgb_cerebro_{timestamp}.joblib")
            log_event(f"XGBoost treinado. Acc={acc_xgb:.4f}", level="info")
            salvar_relatorio("[XGB]\n" + report_xgb + "\nConfMatrix:\n" + str(cm_xgb), "xgb")
            train_acc_xgb = accuracy_score(y_train_xgb, modelo_xgb.predict(X_train))
            log_event(f"XGB - Acurácia Treino: {train_acc_xgb:.4f} | Teste: {acc_xgb:.4f} | Dif: {train_acc_xgb-acc_xgb:.4f}")
            avaliacao["xgboost"] = {
                "acc": float(acc_xgb),
                "promover": bool(acc_xgb >= acc_min_promocao)
            }
        except Exception as e:
            log_event(f"Erro treino XGBoost: {e}", level="error")
            avaliacao["xgboost"] = {"acc": None, "promover": False, "erro": str(e)}
    else:
        log_event("XGBoost não instalado. Pulando treino XGBoost.", level="warning")
        avaliacao["xgboost"] = {"acc": None, "promover": False, "erro": "xgboost_nao_instalado"}

    # =================== LSTM (Deep Learning) =============
    if keras and treinar_lstm_flag:
        try:
            features_lstm = list(X_train.columns)
            X_train_lstm = X_train[features_lstm].values
            X_val_lstm = X_val[features_lstm].values
            X_test_lstm = preparar_para_previsao(X_test, features_lstm).values
            modelo_lstm = treinar_lstm(X_train[features_lstm], y_train.values, X_val[features_lstm], y_val.values)
            modelos["lstm"] = modelo_lstm
            acc_lstm, report_lstm, cm_lstm, y_pred_lstm = avaliar_modelo(modelo_lstm, X_test_lstm, y_test.values, "lstm")
            modelo_lstm.save("modelos/lstm_cerebro.h5")
            modelo_lstm.save(f"modelos/lstm_cerebro_{timestamp}.h5")
            log_event(f"LSTM treinado. Acc={acc_lstm:.4f}", level="info")
            salvar_relatorio("[LSTM]\n" + report_lstm + "\nConfMatrix:\n" + str(cm_lstm), "lstm")
            train_pred_lstm = np.argmax(modelo_lstm.predict(X_train_lstm.reshape((X_train_lstm.shape[0], 1, X_train_lstm.shape[1]))), axis=1)
            train_acc_lstm = accuracy_score(np.where(y_train.values == -1, 0, np.where(y_train.values == 0, 1, 2)), train_pred_lstm)
            log_event(f"LSTM - Acurácia Treino: {train_acc_lstm:.4f} | Teste: {acc_lstm:.4f} | Dif: {train_acc_lstm-acc_lstm:.4f}")
            if acc_lstm < 0.55:
                log_event(f"⚠️ LSTM não apresentou ganho relevante. Considere desabilitar ou ajustar features/janela.", level="warning")
            avaliacao["lstm"] = {
                "acc": float(acc_lstm),
                "promover": bool(acc_lstm >= acc_min_promocao)
            }
        except Exception as e:
            log_event(f"Erro treino LSTM: {e}", level="error")
            avaliacao["lstm"] = {"acc": None, "promover": False, "erro": str(e)}
    elif keras:
        log_event("Flag treinar_lstm=False. Treino LSTM ignorado.", level="info")
        avaliacao["lstm"] = {"acc": None, "promover": False, "erro": "treinar_lstm_false"}
    else:
        log_event("Tensorflow/Keras não instalado. Pulando treino LSTM.", level="warning")
        avaliacao["lstm"] = {"acc": None, "promover": False, "erro": "keras_nao_instalado"}

    # === promoção do melhor modelo (compat joblib) ===
    escolhido = _promover_melhor_modelo(avaliacao, modelos, features_ref)
    avaliacao["cerebro_mestre_escolhido"] = escolhido

    # === grava avaliação para o pipeline de promoção ===
    _salvar_avaliacao_ensemble(avaliacao)

    # Persistir também as medianas do treino para referência/uso no pipeline
    try:
        with open("modelos/imputacao_medianas_treino.json", "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in ({} if 'med_train_dict' not in locals() else med_train_dict).items()}, f, indent=2, ensure_ascii=False)
        log_event("[IMPUTAÇÃO] Medianas do treino salvas em modelos/imputacao_medianas_treino.json", level="info")
    except Exception as e:
        log_event(f"[IMPUTAÇÃO] Falha ao salvar medianas do treino: {e}", level="warning")

    log_event("🏁 Treino dos modelos ensemble finalizado!", level="info")


def _compute_sample_weight(y, w_updown=3.0):
    import numpy as np
    y = np.asarray(y)
    w = np.ones_like(y, dtype=float)
    w[(y==1) | (y==-1)] = float(w_updown)
    return w

if __name__ == "__main__":
    main()


def _pick_best_labeling(sumA: dict, sumB: dict) -> str:
    """Escolhe 'A' ou 'B' com base em f1_updown, depois mcc, depois neutral_rate (menor é melhor)."""
    def _key(s):
        m = s.get("metrics_mean", {})
        return (
            float(m.get("f1_updown", float("-inf"))),
            float(m.get("mcc", float("-inf"))),
            -float(m.get("neutral_rate", float("inf"))),
        )
    kA, kB = _key(sumA), _key(sumB)
    return "A" if kA > kB else "B"
