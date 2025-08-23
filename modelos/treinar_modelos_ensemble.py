import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Global config placeholder (overwritten later in main)
cfg = {}


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# XGBoost
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# LSTM
try:
    from tensorflow import keras
except ImportError:
    keras = None

# Oversampling e Undersampling
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    SMOTE = None
    RandomOverSampler = None
    RandomUnderSampler = None

from utils.debug_logger import log_event


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
    df = df.fillna(df.median(numeric_only=True))
    if "sinal" not in df.columns:
        log_event("❌ Coluna 'sinal' não encontrada no features.csv.", level="error")
        return None
    return df


def balancear_labels(X, y, perc_neutro=0.20, min_amostra=150):
    from collections import Counter
    counts = Counter(y)
    n_total = len(y)
    n_neutro = counts.get(0, 0)
    n_venda = counts.get(-1, 0)
    n_compra = counts.get(1, 0)

    n_neutro_limite = int(perc_neutro * n_total)
    n_venda_limite = max(min_amostra, n_venda)
    n_compra_limite = max(min_amostra, n_compra)

    # Neutros: undersample se necessário
    if n_neutro > n_neutro_limite:
        idx_neutros = y[y == 0].sample(n=n_neutro_limite, random_state=42).index
    else:
        idx_neutros = y[y == 0].index

    # Venda: oversample se necessário
    idx_venda = y[y == -1].index
    falta_venda = n_venda_limite - len(idx_venda)
    if falta_venda > 0 and len(idx_venda) > 0:
        idx_venda = idx_venda.append(pd.Index(np.random.choice(idx_venda, falta_venda, replace=True)))

    # Compra: oversample se necessário
    idx_compra = y[y == 1].index
    falta_compra = n_compra_limite - len(idx_compra)
    if falta_compra > 0 and len(idx_compra) > 0:
        idx_compra = idx_compra.append(pd.Index(np.random.choice(idx_compra, falta_compra, replace=True)))

    idx_usar = idx_neutros.union(idx_venda).union(idx_compra)
    Xb = X.loc[idx_usar]
    yb = y.loc[idx_usar]
    log_event(f"[BALANCEAMENTO] Distribuição pós-balanceamento: {yb.value_counts().to_dict()}")
    return Xb, yb


def validar_features(X, expected_features):
    missing = set(expected_features) - set(X.columns)
    extra = set(X.columns) - set(expected_features)
    if missing:
        log_event(f"Features faltando: {missing}", level="error")
        return False
    if extra:
        log_event(f"Features extras: {extra}", level="warning")
    return True


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


def treinar_rf(X_train, y_train):
    modelo = RandomForestClassifier(class_weight='balanced_subsample', n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    _salvar_listas_features_compat("rf", list(X_train.columns))
    return modelo


def treinar_xgb(X_train, y_train):
    if not xgb:
        raise ImportError("XGBoost não instalado.")
    modelo = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    modelo.fit(X_train, y_train, sample_weight=_compute_sample_weight(y_train, w_updown=cfg.get('class_weight_updown', 3.0)))
    _salvar_listas_features_compat("xgb", list(X_train.columns))
    return modelo


def treinar_lstm(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    if not keras:
        raise ImportError("Tensorflow/Keras não instalado.")
    # Corrigir labels para 0, 1, 2
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
    """
    Salva modelos/avaliacao_ensemble.json e logs/avaliacao_ensemble.json
    no formato:
    {
      "timestamp": "...",
      "acc_min_promocao": 0.55,
      "random_forest": {"acc": 0.61, "promover": true},
      "xgboost": {"acc": 0.58, "promover": true},
      "lstm": {"acc": 0.52, "promover": false}
    }
    """
    os.makedirs("modelos", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    path_modelos = os.path.join("modelos", "avaliacao_ensemble.json")
    path_logs = os.path.join("logs", "avaliacao_ensemble.json")
    with open(path_modelos, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    with open(path_logs, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    log_event(f"[AVALIACAO] Arquivo salvo em {path_modelos} e {path_logs}", level="info")


def main():
    features_path = "dados/features.csv"
    os.makedirs("modelos", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    config_path = "config.json"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    treinar_lstm_flag = config.get("treinar_lstm", True)
    acc_min_promocao = config.get("acc_min_promocao", 0.55)  # limiar para 'promover'

    log_event("🚀 Iniciando treino dos modelos ENSEMBLE (RF, XGBoost, LSTM)", level="info")

    df = carregar_dados(features_path)
    if df is None:
        return

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
    X = X.fillna(0)

    if len(X) != len(y):
        log_event(f"❌ X e y desalinhados! X.shape={X.shape}, y.shape={y.shape}", level="error")
        return
    if X.empty or y.empty:
        log_event(f"❌ X ou y vazio após filtragem!", level="error")
        return

    X, y = balancear_labels(X, y, perc_neutro=0.20, min_amostra=150)
    log_event(f"Distribuição balanceada: {y.value_counts().to_dict()}")

    n = len(X)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size

    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:train_size+val_size], y.iloc[train_size:train_size+val_size]
    X_test, y_test = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]

    log_event(f"Shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    avaliacao = {
        "timestamp": timestamp,
        "acc_min_promocao": acc_min_promocao
    }

    # =================== RandomForest =====================
    modelo_rf = treinar_rf(X_train, y_train)
    X_test_rf = preparar_para_previsao(X_test, "modelos/features_treinadas_rf.pkl")
    acc_rf, report_rf, cm_rf, y_pred_rf = avaliar_modelo(modelo_rf, X_test_rf, y_test, "rf")
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
        log_event(f"{X.columns[i]}: {importances[i]:.4f}")

    avaliacao["random_forest"] = {
        "acc": float(acc_rf),
        "promover": bool(acc_rf >= acc_min_promocao)
    }

    # =================== XGBoost ==========================
    if xgb:
        try:
            y_train_xgb = mapear_label_xgb(y_train)
            y_test_xgb = mapear_label_xgb(y_test)
            modelo_xgb = treinar_xgb(X_train, y_train_xgb)
            X_test_xgb = preparar_para_previsao(X_test, "modelos/features_treinadas_xgb.pkl")
            if X_test_xgb.shape[1] != X_train.shape[1]:
                log_event(f"[XGB] Shape mismatch: X_test {X_test_xgb.shape}, X_train {X_train.shape}", level="error")
            acc_xgb, report_xgb, cm_xgb, y_pred_xgb = avaliar_modelo(modelo_xgb, X_test_xgb, y_test_xgb, "xgb")
            modelo_xgb.save_model("modelos/xgb_cerebro.json")
            modelo_xgb.save_model(f"modelos/xgb_cerebro_{timestamp}.json")
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

    # === grava avaliação para o pipeline de promoção ===
    _salvar_avaliacao_ensemble(avaliacao)

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
